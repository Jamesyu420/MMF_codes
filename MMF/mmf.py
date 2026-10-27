"""MMF: Pyro SVI with empirical Bayes and nested program backfitting.

Model: Poisson counts, Gaussian spatial activities, nonnegative gene loadings.
Training: program guide -> loading -> EB variance, then shared effects and pi.
"""
from contextlib import contextmanager
from dataclasses import dataclass, fields
from functools import partial
from typing import Optional

import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceMeanField_ELBO
from torch.distributions import constraints


@dataclass
class FitConfig:
    kernel: str = "nystrom"
    lengthscale: float = 1.0
    n_landmarks: int = 128
    max_sweeps: int = 20
    within_type_sweeps: int = 1
    q_steps: int = 5
    w_steps: int = 5
    lr: float = 0.01
    batch_size: Optional[int] = None
    num_particles: int = 2
    nuisance_steps: int = 5
    fit_pi: bool = True
    pi_steps: int = 5
    pi_floor: float = 1e-6
    root_steps: int = 80  # Legacy input compatibility; unused by the SVI backend.
    variance_floor: float = 1e-6
    seed: int = 0
    dtype: str = "float64"
    device: str = "cuda"
    evaluation_particles: int = 8
    tolerance: float = 1e-4
    patience: int = 3
    warm_start_sweeps: int = 1
    jitter: float = 1e-6
    chunk_size: int = 256


def fit_config_from_mapping(mapping, **overrides):
    """Build a FitConfig and reject a low-rank coefficient guide."""
    allowed = {item.name for item in fields(FitConfig)}
    guide_rank = mapping.get("guide_rank", 0)
    if guide_rank not in (0, None) and int(guide_rank) != 0:
        raise ValueError("Low-rank coefficient guides are not part of this model")
    data = {key: value for key, value in mapping.items() if key in allowed}
    data.update(overrides)
    return FitConfig(**data)


def _array(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x)


def compute_reference_mean(X, labels, cell_types=None, library_size=None):
    """Return the exact cell-type mean of X[j, g] / T[j], without smoothing.

    Compute exposures on the common gene universe before subsetting genes.
    Zero-library cells are removed. Explicit cell_types sets the output order.
    """
    X = np.asarray(_array(X), dtype=float)
    labels = np.asarray(labels)
    lib = X.sum(1) if library_size is None else np.asarray(_array(library_size), float)
    keep = lib > 0
    X, labels, lib = X[keep], labels[keep], lib[keep]
    types = list(dict.fromkeys(labels.tolist())) if cell_types is None else list(cell_types)
    normalized = X / lib[:, None]
    return np.stack([normalized[labels == c].mean(0) for c in types]), types


class MMF:
    """Cell-type-specific programs with a fixed reference and Gaussian guide."""
    inference_backend = "pyro_svi_trace_mean_field_elbo"
    frozen_programs = ()
    compute_reference_mean = staticmethod(compute_reference_mean)

    def __init__(self, Y, eta0, pi, coords=None, library_size=None,
                 programs_per_type=2, config=None):
        self.config = config or FitConfig()
        self.device = torch.device(self.config.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = getattr(torch, self.config.dtype)
        y = np.asarray(_array(Y), dtype=float)
        self.Y, self.eta0 = self._tensor(y), self._tensor(_array(eta0))
        self.N, self.G = self.Y.shape
        self.C = self.eta0.shape[0]
        pi = np.asarray(_array(pi), dtype=float)
        lib = y.sum(1) if library_size is None else _array(library_size)
        self.Kc = np.broadcast_to(programs_per_type, (self.C,)).tolist()
        self.programs = [(c, k) for c in range(self.C) for k in range(self.Kc[c])]
        self.library_size = self._tensor(lib)
        self.support = self.eta0 > 0
        self.support_indices = [self.support[c].nonzero().flatten() for c in range(self.C)]
        self._initial_pi = self._tensor(pi / pi.sum(1, keepdims=True))
        if self.config.fit_pi:
            self._initial_pi = self._initial_pi.clamp_min(self.config.pi_floor)
            self._initial_pi /= self._initial_pi.sum(1, keepdim=True)
        self.log_eta0, self.log_N = self.eta0.log(), self.library_size.log()
        self.v_alpha, self.v_gamma = self._tensor(1.0), self._tensor(1.0)
        self._param_state = {"params": {}, "constraints": {}}
        self.group_names, self.optimizers, self.svis, self.v = {}, {}, {}, {}
        self._training_elbo = TraceMeanField_ELBO(num_particles=self.config.num_particles)
        self._evaluation_elbo = TraceMeanField_ELBO(num_particles=self.config.evaluation_particles)
        for elbo in (self._training_elbo, self._evaluation_elbo):
            elbo.vectorize_particles = True
            elbo.max_plate_nesting = 1
        self.history, self._warm_started = [], False
        self._cuda_devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=self._cuda_devices):
            self._seed_rng(self.config.seed)
            self._make_basis(coords)
            with self._scope():
                self._initialize_parameters()
            self._rng_state = torch.get_rng_state()
            self._cuda_rng_state = (torch.cuda.get_rng_state(self.device)
                                    if self._cuda_devices else None)

    def _tensor(self, x):
        return torch.as_tensor(x, dtype=self.dtype, device=self.device)

    def _scope(self):
        return pyro.get_param_store().scope(state=self._param_state)

    def _seed_rng(self, seed):
        # Seed only the RNG streams protected by our fork, including on multi-GPU hosts.
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        if self._cuda_devices:
            state = torch.Generator(device=self.device).manual_seed(seed).get_state()
            torch.cuda.set_rng_state(state, self.device)

    @contextmanager
    def _rng_scope(self):
        with torch.random.fork_rng(devices=self._cuda_devices):
            torch.set_rng_state(self._rng_state)
            if self._cuda_devices:
                torch.cuda.set_rng_state(self._cuda_rng_state, self.device)
            try:
                yield
            finally:
                self._rng_state = torch.get_rng_state()
                if self._cuda_devices:
                    self._cuda_rng_state = torch.cuda.get_rng_state(self.device)

    def _cholesky(self, K):
        eye = torch.eye(len(K), dtype=self.dtype, device=self.device)
        for jitter in [0.0] + [self.config.jitter * 10**j for j in range(6)]:
            L, info = torch.linalg.cholesky_ex(K + jitter * eye)
            if int(info.max()) == 0:
                return L, jitter
        raise ValueError("kernel factorization failed; inspect coordinates and lengthscale")

    def _make_basis(self, coords):
        """Compute the spatial basis in float64, then use the fitting dtype."""
        fitting_dtype = self.dtype
        self.dtype = torch.float64
        self._build_basis(coords)
        self.dtype = fitting_dtype
        if self.basis is not None:
            self.basis = self.basis.to(fitting_dtype)
        self.basis_metadata["basis_compute_dtype"] = "float64"

    def _build_basis(self, coords):
        mode = self.config.kernel
        if mode not in {"identity", "full", "nystrom"}:
            raise ValueError("kernel must be identity, full or nystrom")
        self.basis = None
        metadata = {"kernel": mode, "lengthscale": self.config.lengthscale, "jitter": 0.0}
        if mode == "identity":
            self.latent_dim = self.N
        else:
            x = self._tensor(_array(coords))
            kernel = lambda a, b: torch.exp(-torch.cdist(a, b).square() / (2 * self.config.lengthscale**2))
            if mode == "full":
                self.basis, metadata["jitter"] = self._cholesky(kernel(x, x))
            else:
                d = min(self.config.n_landmarks, self.N)
                selected, nearest = [], torch.full((self.N,), float("inf"), device=self.device, dtype=self.dtype)
                point = int((x - x.mean(0)).square().sum(1).argmax())
                for _ in range(d):
                    selected.append(point)
                    nearest = torch.minimum(nearest, (x - x[point]).square().sum(1))
                    nearest[selected] = -1
                    point = int(nearest.argmax())
                landmarks = x[selected]
                L, metadata["jitter"] = self._cholesky(kernel(landmarks, landmarks))
                cross = kernel(x, landmarks)  # N x d, no dense N x N kernel
                self.basis = torch.linalg.solve_triangular(L, cross.T, upper=False).T
                metadata["landmark_indices"] = selected
            self.latent_dim = self.basis.shape[1]
        metadata["latent_dim"] = self.latent_dim
        self.basis_metadata = metadata

    def _initialize_parameters(self):
        for name, size in (("alpha", self.N), ("gamma", self.G)):
            pyro.param(f"{name}_mean", self._tensor(np.zeros(size)))
            pyro.param(f"{name}_var", self._tensor(np.full(size, 0.01)),
                       constraint=constraints.positive)
            self._add_optimizer(name, [f"{name}_mean", f"{name}_var"])
        if self.config.fit_pi:
            pyro.param("pi_logits", self._initial_pi.log())
            self._add_optimizer("pi", ["pi_logits"])
        d = self.latent_dim
        for c, k in self.programs:
            b = f"{c}_{k}"
            weights = torch.randn(len(self.support_indices[c]), device=self.device,
                                  dtype=self.dtype).softmax(0)
            pyro.param(f"w_{b}", weights, constraint=constraints.simplex)
            pyro.param(f"mu_{b}", 0.01 * torch.randn(d, device=self.device, dtype=self.dtype))
            pyro.param(f"dvar_{b}", self._tensor(np.ones(d)), constraint=constraints.positive)
            self.v[c, k] = self._tensor(1.0)
            self._add_optimizer((c, k, "q"), [f"mu_{b}", f"dvar_{b}"])
            self._add_optimizer((c, k, "w"), [f"w_{b}"])

    def _add_optimizer(self, key, names):
        """Expose only this block's parameters to SVI, never hide sample sites."""
        self.group_names[key] = tuple(names)
        active_names = frozenset(names)
        optimizer = pyro.optim.Adam({"lr": self.config.lr})
        self.optimizers[key] = optimizer
        self.svis[key] = SVI(
            partial(self._model_for_block, active_names),
            partial(self._guide_for_block, active_names),
            optimizer,
            loss=self._training_elbo,
        )

    @staticmethod
    def _hide_parameter(active_names, site):
        return site["type"] == "param" and site["name"] not in active_names

    def _model_for_block(self, active_names, *args, **kwargs):
        with pyro.poutine.block(hide_fn=partial(self._hide_parameter, active_names)):
            return self._model(*args, **kwargs)

    def _guide_for_block(self, active_names, *args, **kwargs):
        with pyro.poutine.block(hide_fn=partial(self._hide_parameter, active_names)):
            return self._guide(*args, **kwargs)

    def set_optimization(self, lr=None, num_particles=None):
        """Update Pyro's current and future optimizers and its particle budget."""
        if lr is not None:
            self.config.lr = float(lr)
            for optimizer in self.optimizers.values():
                optimizer.pt_optim_args["lr"] = self.config.lr
                for state in optimizer._state_waiting_to_be_consumed.values():
                    for group in state.get("param_groups", []):
                        group["lr"] = self.config.lr
                for torch_optimizer in optimizer.optim_objs.values():
                    for group in torch_optimizer.param_groups:
                        group["lr"] = self.config.lr
        if num_particles is not None:
            self.config.num_particles = int(num_particles)
        self._training_elbo.num_particles = self.config.num_particles

    def _pi(self):
        return (pyro.param("pi_logits").softmax(-1) if self.config.fit_pi
                else self._initial_pi)

    def _log_pi(self):
        return (pyro.param("pi_logits").log_softmax(-1) if self.config.fit_pi
                else self._initial_pi.log())

    @property
    def pi(self):
        with self._scope():
            return self._pi().detach().clone()

    def _w(self, c, k):
        return torch.zeros_like(self.eta0[c]).scatter(
            0, self.support_indices[c], pyro.param(f"w_{c}_{k}"))

    def _q(self, c, k):
        b = f"{c}_{k}"
        return pyro.param(f"mu_{b}"), pyro.param(f"dvar_{b}")

    def _q_distribution(self, c, k):
        mu, dvar = self._q(c, k)
        return dist.Normal(mu, dvar.sqrt()).to_event(1)

    def _indexed_log_pi(self, idx):
        return self._log_pi()[idx]

    def _shared(self, name):
        return pyro.param(f"{name}_mean"), pyro.param(f"{name}_var")

    def _apply_A(self, z, idx):
        # Supports both single draws (d,) and particle arrays (S,d).
        return z[..., idx] if self.basis is None else z @ self.basis[idx].T

    def _log_mixture(self, idx, z_samples):
        active = [c for c in range(self.C) if self.Kc[c]]
        inactive = [c for c in range(self.C) if not self.Kc[c]]
        if not active or not inactive:
            terms, log_pi = [], self._log_pi()
            for c in range(self.C):
                term = log_pi[idx, c, None] + self.log_eta0[c, None, :]
                for k in range(self.Kc[c]):
                    h = self._apply_A(z_samples[c, k], idx)
                    term = term + h[..., None] * self._w(c, k)
                terms.append(term)
            return torch.logsumexp(torch.stack(terms), dim=0)
        log_pi = self._log_pi()[idx]
        background = log_pi[:, inactive, None] + self.log_eta0[inactive][None, :, :]
        # Avoid undefined gradients where every background reference value is zero.
        valid = torch.isfinite(background).any(dim=1)
        background = background.masked_fill(~valid[:, None, :], 0)
        background = torch.logsumexp(background, dim=1).masked_fill(~valid, -torch.inf)
        terms = [background]
        for c in active:
            term = log_pi[:, c, None] + self.log_eta0[c, None, :]
            for k in range(self.Kc[c]):
                h = self._apply_A(z_samples[c, k], idx)
                term = term + h[..., None] * self._w(c, k)
            terms.append(term)
        if len(terms) == 2:
            return torch.logaddexp(*terms)
        return torch.logsumexp(torch.stack(torch.broadcast_tensors(*terms)), dim=0)

    def _program_prior(self, c, k):
        zeros = self._tensor(np.zeros(self.latent_dim))
        variance = self.v[c, k].expand(self.latent_dim)
        return dist.Normal(zeros, variance.sqrt()).to_event(1)

    def _model(self, idx, reference_only=False):
        """Poisson mixture; only the observed minibatch is scaled by N/B."""
        alpha = pyro.sample("alpha", dist.Normal(torch.zeros_like(self.Y[:, 0]),
                                                self.v_alpha.sqrt()).to_event(1))
        gamma = pyro.sample("gamma", dist.Normal(torch.zeros_like(self.eta0[0]),
                                                self.v_gamma.sqrt()).to_event(1))
        z = {} if reference_only else {
            (c, k): pyro.sample(f"z_{c}_{k}", self._program_prior(c, k))
            for c, k in self.programs
        }
        log_mixture = (self._log_expected_mixture(idx, reference_only=True)
                       if reference_only else self._log_mixture(idx, z))
        log_rate = self.log_N[idx, None] + alpha[..., idx, None] + gamma[..., None, :] + log_mixture
        with pyro.poutine.scale(scale=self.N / len(idx)):
            pyro.sample("Y_0", dist.Poisson(log_rate.exp()).to_event(2), obs=self.Y[idx])

    def _guide(self, idx, reference_only=False):
        """Gaussian variational factors, sampled and replayed by Pyro itself."""
        for name in ("alpha", "gamma"):
            mean, variance = self._shared(name)
            pyro.sample(name, dist.Normal(mean, variance.sqrt()).to_event(1))
        if not reference_only:
            for c, k in self.programs:
                pyro.sample(f"z_{c}_{k}", self._q_distribution(c, k))

    def _cycle(self, within_sweeps):
        """Visit each type and program: guide SVI, loading SVI, then EB variance."""
        for c in range(self.C):
            for _ in range(within_sweeps):
                for k in range(self.Kc[c]):
                    if (c, k) in self.frozen_programs:
                        continue
                    for kind, steps in (("q", self.config.q_steps), ("w", self.config.w_steps)):
                        for _ in range(steps):
                            self._step_group((c, k, kind), self._batch())
                    self._update_variance(c, k)

    def _diagnostic_state(self):
        with torch.no_grad():
            variances = torch.stack([self.v_alpha, self.v_gamma] + [self.v[p] for p in self.programs])
            return {"rate_change": self._expected_counts(),
                    "pi_change": self._pi().clone(), "variance_change": variances}

    @staticmethod
    def _relative_change(new, old):
        return float(torch.linalg.vector_norm(new - old) / torch.linalg.vector_norm(old).clamp_min(1))

    def fit(self):
        """Observation noise is part of the fitted model; use PyroBackfitMMF."""
        raise RuntimeError("Use MMF.fit on a prepared npz. The base class has no observation noise.")

    def _moments(self, c, k, idx):
        mu, dvar = self._q(c, k)
        if self.basis is None:
            return mu[idx], dvar[idx]
        A = self.basis[idx]
        return A @ mu, A.square() @ dvar

    def _log_expected_mixture(self, idx, reference_only=False):
        terms, log_pi = [], self._log_pi()
        for c in range(self.C):
            term = log_pi[idx, c, None] + self.log_eta0[c, None, :]
            if not reference_only:
                for k in range(self.Kc[c]):
                    mean, variance = self._moments(c, k, idx)
                    w = self._w(c, k)
                    term = term + mean[:, None] * w + 0.5 * variance[:, None] * w.square()
            terms.append(term)
        return torch.logsumexp(torch.stack(terms), dim=0)

    def _log_expected_counts(self, idx, reference_only=False):
        a, s = self._shared("alpha")
        b, t = self._shared("gamma")
        return (self.log_N[idx, None] + a[idx, None] + s[idx, None] / 2 + b + t / 2
                + self._log_expected_mixture(idx, reference_only))

    def _batch(self):
        size = min(self.config.batch_size or self.N, self.N)
        return (torch.arange(self.N, device=self.device) if size == self.N
                else torch.randperm(self.N, device=self.device)[:size])

    def _chunks(self):
        for start in range(0, self.N, self.config.chunk_size):
            yield torch.arange(start, min(start + self.config.chunk_size, self.N), device=self.device)

    def _step_group(self, block, idx, reference_only=False):
        """Let Pyro update one block; keep all other parameters fixed."""
        active = self.group_names[block]
        parameters = list(pyro.get_param_store().named_parameters())
        flags = [p.requires_grad for _, p in parameters]
        try:
            for name, parameter in parameters:
                parameter.grad = None
                parameter.requires_grad_(name in active)
            return self.svis[block].step(idx, reference_only=reference_only)
        finally:
            for (_, parameter), flag in zip(parameters, flags):
                parameter.grad = None
                parameter.requires_grad_(flag)

    def _update_variance(self, c, k):
        with torch.no_grad():
            mu, dvar = self._q(c, k)
            moment = mu.square().sum() + dvar.sum()
            self.v[c, k].copy_((moment / self.latent_dim).clamp_min(self.config.variance_floor))

    def _update_shared_variances(self):
        with torch.no_grad():
            for name, v in (("alpha", self.v_alpha), ("gamma", self.v_gamma)):
                mean, variance = self._shared(name)
                v.copy_((mean.square() + variance).mean().clamp_min(self.config.variance_floor))

    def _set_shared(self, name, mean, variance):
        """Copy externally supplied Gaussian initial values; never an inference step."""
        with torch.no_grad():
            pyro.param(f"{name}_mean").unconstrained().copy_(mean)
            pyro.param(f"{name}_var").unconstrained().copy_(variance.log())

    def _update_nuisance(self, steps, reference_only=False):
        for _ in range(steps):
            self._step_group("alpha", self._batch(), reference_only=reference_only)
            self._step_group("gamma", self._batch(), reference_only=reference_only)

    def _expected_counts(self):
        result = torch.empty_like(self.Y)
        for idx in self._chunks():
            result[idx] = self._log_expected_counts(idx).exp()
        return result

    def _evaluate_elbo(self):
        # Same latent draws per chunk; B/N weights give the full KL weight one.
        with torch.no_grad(), torch.random.fork_rng(devices=self._cuda_devices):
            self._evaluation_elbo.num_particles = self.config.evaluation_particles
            value = 0.0
            for idx in self._chunks():
                self._seed_rng(self.config.seed + 100003)
                value -= len(idx) / self.N * float(
                    self._evaluation_elbo.loss(self._model, self._guide, idx))
        return value

    def posterior_summary(self):
        """Conditional Gaussian summaries and analytic expected counts (reporting only)."""
        with self._scope(), torch.no_grad():
            n_programs = len(self.programs)
            means = np.empty((n_programs, self.N), dtype=self.config.dtype)
            sds = np.empty_like(means)
            loadings = np.empty((n_programs, self.G), dtype=self.config.dtype)
            idx = torch.arange(self.N, device=self.device)
            for p, (c, k) in enumerate(self.programs):
                mean, variance = self._moments(c, k, idx)
                means[p], sds[p] = mean.cpu().numpy(), variance.sqrt().cpu().numpy()
                loadings[p] = self._w(c, k).cpu().numpy()
            a, s = self._shared("alpha")
            b, t = self._shared("gamma")
            arrays = dict(alpha_mean=a, alpha_sd=s.sqrt(), gamma_mean=b, gamma_sd=t.sqrt(),
                          alpha_prior_variance=self.v_alpha, gamma_prior_variance=self.v_gamma,
                          pi=self._pi(), eta0=self.eta0, reference_support=self.support,
                          library_size=self.library_size, expected_counts=self._expected_counts())
            result = {name: value.cpu().numpy().copy() for name, value in arrays.items()}
            result.update(h_mean=means, h_sd=sds, loadings=loadings,
                          prior_variance=np.array([float(self.v[p]) for p in self.programs]),
                          program_cell_type=np.array([p[0] for p in self.programs], dtype=int),
                          program_index=np.array([p[1] for p in self.programs], dtype=int))
            result.update(alpha=result["alpha_mean"].copy(), gamma=result["gamma_mean"].copy())
            return result
