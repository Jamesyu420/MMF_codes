"""Pyro backfitting, shared observation noise, and program selection.

log(lambda_ig) = log(lambda_structural_ig) + epsilon_ig with one shared
variance and no -v/2 term. q(z) is diagonal. Rank search adds and deletes
programs on private copies of this same model.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints, transform_to

from MMF.mmf import MMF


def _gaussian_epsilon_solution(log_a, y, variance, *, max_iterations=96,
                               tolerance=1e-11):
    """Float64 safeguarded root; also return the solved log expected rate.

    F is divided by v before evaluation to avoid multiplication by large v.
    m is reconstructed from x-log(a)-s/2, rather than v*(y-exp(x)); the latter
    loses all meaningful digits when y is large and exp(x) is close to y.
    """
    if max_iterations < 1 or tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError('positive root iteration budget and tolerance required')
    with torch.no_grad():
        la, yy = torch.broadcast_tensors(log_a.detach().double(), y.detach().double())
        vv = torch.as_tensor(variance, device=la.device).detach().double()
        vv = torch.broadcast_to(vv, la.shape)
        if torch.any(vv < 0) or not torch.isfinite(vv).all():
            raise ValueError('epsilon variances must be finite and nonnegative')
        if torch.any(yy < 0) or not torch.isfinite(yy).all():
            raise ValueError('counts must be finite and nonnegative')
        if torch.isposinf(la).any() or torch.isnan(la).any():
            raise FloatingPointError('nonfinite structural expected log-rate')
        if (torch.isneginf(la) & yy.gt(0)).any():
            raise FloatingPointError('positive observed count has structural rate zero')
        active = vv.gt(0) & torch.isfinite(la)
        v = torch.where(active, vv, torch.ones_like(vv))
        a_log = torch.where(active, la, torch.zeros_like(la))
        counts = torch.where(active, yy, torch.zeros_like(yy))
        precision = v.reciprocal()
        if not torch.isfinite(precision).all():
            raise FloatingPointError('positive epsilon variance is below representable precision')
        lo = torch.minimum(a_log, -v.log()) - 1
        log_y = counts.clamp_min(torch.finfo(torch.float64).tiny).log()
        log_y = torch.where(counts.gt(0), log_y, torch.full_like(counts, -torch.inf))
        hi = torch.maximum(torch.maximum(a_log + .5, log_y), torch.zeros_like(a_log))
        # A scale-aware start avoids hundreds of unit Newton steps if log(a)
        # is huge. This does not clip the rate or impose a variance bound.
        x = torch.logaddexp(log_y, precision.log()+torch.log1p(a_log.abs()))
        x = x.clamp(min=lo, max=hi)
        converged = ~active
        for _ in range(max_iterations):
            r = x.exp()
            denominator = precision+r
            fraction = precision/denominator
            value = precision*(x-a_log)+r-counts-.5*fraction
            derivative = precision+r+.5*fraction*(r/denominator)
            newton_step = value/derivative
            done = newton_step.abs().le(tolerance) | (hi-lo).le(tolerance)
            # Retain the final Newton improvement; never freeze the previous
            # x merely because its step first falls below tolerance.
            final = x-newton_step
            x = torch.where(done & torch.isfinite(final), final, x)
            converged |= done
            if bool(converged.all()):
                break
            lo = torch.where((value < 0) & ~converged, x, lo)
            hi = torch.where((value >= 0) & ~converged, x, hi)
            proposal = x-newton_step
            accepted = torch.isfinite(proposal) & proposal.gt(lo) & proposal.lt(hi)
            proposal = torch.where(accepted, proposal, .5*(lo+hi))
            x = torch.where(converged, x, proposal)
        if not bool(converged.all()):
            raise RuntimeError('local epsilon Gaussian VI root did not converge')
        r = x.exp()
        s = (precision+r).reciprocal()
        mean = x-a_log-.5*s
        # s/v = 1/(1+v*r). logaddexp avoids overflow in v*r.
        log_vr = v.log()+x
        log_one_plus_vr = torch.logaddexp(torch.zeros_like(x), log_vr)
        ratio = (-log_one_plus_vr).exp()
        kl = .5*(mean.square()/v + ratio-1 + log_one_plus_vr)
        mean = torch.where(active, mean, torch.zeros_like(mean))
        s = torch.where(active, s, vv)
        kl = torch.where(active, kl, torch.zeros_like(kl))
        x = torch.where(active, x, la)
        if not (torch.isfinite(mean).all() and torch.isfinite(s).all() and
                torch.isfinite(kl).all()):
            raise FloatingPointError('nonfinite local Gaussian epsilon solution')
        return mean, s, kl, x


def gaussian_epsilon_optimum(log_a, y, variance, *, max_iterations=96,
                             tolerance=1e-11):
    """Return detached m,s,KL at the unique local Gaussian VI optimum.

    a=E_q[structural count rate]; epsilon is uncentered N(0,v). Zero v is an
    exact deterministic boundary. No -v/2 correction is present. Solving the
    local factors before evaluating the loss enables envelope gradients for
    global parameters. The exact conditional posterior need not be Gaussian.
    """
    return _gaussian_epsilon_solution(log_a, y, variance,
        max_iterations=max_iterations, tolerance=tolerance)[:3]


def gaussian_epsilon_diagnostics(log_a, y, variance):
    """Small probe diagnostics, normalized in log-rate root units.

    Absolute v*(y-r) residuals are misleading at high counts because of
    floating-point subtraction. The residual divided by F' has units of a
    log-rate Newton correction. This function is read-only and detached.
    """
    m, s, kl, x = _gaussian_epsilon_solution(log_a, y, variance)
    with torch.no_grad():
        la, yy = torch.broadcast_tensors(log_a.detach().double(), y.detach().double())
        vv = torch.broadcast_to(torch.as_tensor(variance, device=la.device).double(), la.shape)
        active = vv.gt(0) & torch.isfinite(la)
        safe_v = torch.where(active, vv, torch.ones_like(vv))
        p, r = safe_v.reciprocal(), x.exp()
        denom = p+r
        f = p*(x-la)+r-yy-.5*p/denom
        derivative = p+r+.5*(p/denom)*(r/denom)
        correction = torch.where(active, (f/derivative).abs(), torch.zeros_like(x))
        consistency = torch.where(active, (la+m+.5*s-x).abs(), torch.zeros_like(x))
        return dict(maximum_log_rate_newton_correction=float(correction.max()),
            maximum_log_rate_reconstruction_error=float(consistency.max()),
            minimum_posterior_variance=float(s.min()), maximum_posterior_variance=float(s.max()),
            maximum_absolute_epsilon_mean=float(m.abs().max()),
            maximum_local_KL=float(kl.max()), entries=int(la.numel()))


class UncenteredEpsilonMMF(MMF):
    inference_backend = 'pyro_SVI_TraceMeanField_ELBO_analytic_moments_profiled_epsilon'
    epsilon_state_version = 2

    def __init__(self, *args, epsilon_variance_mode='shared',
                 epsilon_variance=.1, epsilon_variance_floor=1e-8, **kwargs):
        if epsilon_variance_mode != 'shared':
            raise ValueError('Only a single shared epsilon variance is part of the model')
        super().__init__(*args, **kwargs)
        self.block_optimizers = {}
        self.optimizer_calls = {}
        self.epsilon_variance_mode = 'shared'
        value = self._tensor(epsilon_variance).detach().clone()
        if value.numel() != 1:
            raise ValueError('shared epsilon variance must be scalar')
        value = value.reshape(())
        if torch.any(value < 0) or not torch.isfinite(value).all():
            raise ValueError('epsilon initial variance must be finite and nonnegative')
        self.v_eps = value
        self.epsilon_variance_floor = float(epsilon_variance_floor)
        if self.epsilon_variance_floor < 0 or not np.isfinite(self.epsilon_variance_floor):
            raise ValueError('epsilon_variance_floor must be finite and nonnegative')
        self.epsilon_updates = []

    def structural_log_expected_counts(self, idx, reference_only=False):
        """E_q[mu_structural], excluding the additional epsilon multiplier."""
        alpha, avar = self._shared('alpha')
        gamma, gvar = self._shared('gamma')
        return (self.log_N[idx, None].double()+alpha[idx, None].double()+gamma.double()
                +.5*(avar[idx, None].double()+gvar.double())
                +self._log_expected_mixture(idx, reference_only).double())

    def epsilon_posterior(self, idx=None, reference_only=False):
        if idx is None:
            idx = torch.arange(self.N, device=self.device)
        structural = self.structural_log_expected_counts(idx, reference_only)
        return gaussian_epsilon_optimum(structural, self.Y[idx], self.v_eps)

    def _log_expected_counts(self, idx, reference_only=False):
        """Posterior expected counts for fitted observations (uses training Y)."""
        structural = self.structural_log_expected_counts(idx, reference_only)
        return _gaussian_epsilon_solution(structural, self.Y[idx], self.v_eps)[3]

    def _expected_counts(self):
        # Parent allocates empty_like(Y), which breaks indexed assignment
        # when coefficient dtype is float32 and local profiling uses float64.
        result = torch.empty(self.Y.shape, device=self.device, dtype=torch.float64)
        for idx in self._chunks():
            result[idx] = self._log_expected_counts(idx).exp()
        return result

    def prior_predictive_log_expected_counts(self, idx, reference_only=False):
        """Fresh epsilon integrated out; differs from fitted epsilon posterior."""
        return self.structural_log_expected_counts(idx, reference_only).double()+.5*self.v_eps.double()

    def _model(self, idx, reference_only=False):
        # These sites retain the Gaussian KLs; their likelihood moments are exact.
        for name, prior in [('alpha', self.v_alpha), ('gamma', self.v_gamma)]:
            mean, _ = self._shared(name)
            pyro.sample(name, dist.Normal(torch.zeros_like(mean), prior.sqrt()).to_event(1))
        z = {} if reference_only else {
            p: pyro.sample(f'z_{p[0]}_{p[1]}', self._program_prior(*p))
            for p in self.programs}
        logmix = (self._log_expected_mixture(idx, True) if reference_only
                  else self._log_mixture(idx, z))
        a, _ = self._shared('alpha')
        b, _ = self._shared('gamma')
        structural = self.structural_log_expected_counts(idx, reference_only)
        m, s, kl, lograte = _gaussian_epsilon_solution(structural, self.Y[idx], self.v_eps)
        # Envelope derivative: optimized local moments stay fixed.
        rate = (structural + lograte - structural.detach()).exp()
        linear = self.log_N[idx, None].double() + a[idx, None].double() + b.double()
        y = self.Y[idx].double()
        value = (y * (linear + logmix.double() + m) - rate
                 - torch.lgamma(y + 1) - kl).sum((-2, -1))
        with pyro.poutine.scale(scale=self.N / len(idx)):
            pyro.factor('integrated_likelihood_and_local_KL', value)

    def _marginal_log_mixture(self, idx, programs, eps):
        """Monte Carlo E[log mixture] from the Gaussian marginal of each u_icg."""
        base = self._indexed_log_pi(idx).T[:, :, None] + self.log_eta0[:, None, :]
        qs = [self._q(*p) for p in programs]
        mu, dvar = [torch.stack([q[j] for q in qs]) for j in (0, 1)]
        weights = torch.stack([self._w(*p) for p in programs])
        if self.basis is None:
            mean, variance = mu[:, idx], dvar[:, idx]
        else:
            A = self.basis[idx]
            mean, variance = mu @ A.T, dvar @ A.square().T
        means, variances = [], []
        for c in range(self.C):
            members = [j for j, p in enumerate(programs) if p[0] == c]
            if members:
                W = weights[members]
                means.append(mean[members].T @ W)
                variances.append(variance[members].T @ W.square())
            else:
                zero = torch.zeros((len(idx), self.G), device=self.device, dtype=self.dtype)
                means.append(zero)
                variances.append(zero)
        M, V = torch.stack(means), torch.stack(variances)
        if eps.ndim != 4 or tuple(eps.shape[1:]) != (self.C, len(idx), self.G):
            raise ValueError('MC eps must have shape (draws, C, batch, G)')
        zero = V.eq(0)
        sigma = V.masked_fill(zero, 1).sqrt().masked_fill(zero, 0)
        return (base[None] + M[None] + sigma[None] * eps).logsumexp(1).mean(0)

    def _structural_objective_parts(self, idx, particles, reference_only, eps):
        """Diagonal-guide KL, marginal E[log mixture], and analytic log E[rate]."""
        alpha, avar = self._shared('alpha')
        gamma, gvar = self._shared('gamma')
        kl = .5 * (((alpha.double().square()+avar.double())/self.v_alpha.double()-1+
            self.v_alpha.double().log()-avar.double().log()).sum() +
            ((gamma.double().square()+gvar.double())/self.v_gamma.double()-1+
            self.v_gamma.double().log()-gvar.double().log()).sum())
        programs = [] if reference_only else self.programs
        if programs:
            qs = [self._q(*p) for p in programs]
            mu, dvar = [torch.stack([q[j] for q in qs]) for j in (0, 1)]
            prior = torch.stack([self.v[p] for p in programs]).double()
            md, dd = mu.double(), dvar.double()
            logdet, moment = dd.log().sum(1), (md.square()+dd).sum(1)
            kl = kl + .5*(moment/prior-self.latent_dim+self.latent_dim*prior.log()-logdet).sum()
            if eps is None:
                count = max(2, int(particles or self.config.num_particles))
                count += count % 2
                half = torch.randn((count // 2, self.C, len(idx), self.G),
                                   device=self.device, dtype=self.dtype)
                eps = torch.cat([half, -half], 0)
            log_mix = self._marginal_log_mixture(idx, programs, eps)
        else:
            log_mix = (self._indexed_log_pi(idx).T[:, :, None] + self.log_eta0[:, None, :]).logsumexp(0)
        loglinear = self.log_N[idx, None].double()+alpha[idx, None].double()+gamma.double()
        return (kl, loglinear+log_mix.double(),
                self.structural_log_expected_counts(idx, reference_only))

    def direct_loss(self, idx=None, particles=None, reference_only=False, eps=None, z_samples=None):
        if idx is None:
            idx = torch.arange(self.N, device=self.device)
        del z_samples
        kl_global, b, structural = self._structural_objective_parts(
            idx, particles, reference_only, eps)
        m, s, kl, log_rate = _gaussian_epsilon_solution(structural, self.Y[idx], self.v_eps)
        adjusted_log_rate = structural + (log_rate-structural.detach())
        yy = self.Y[idx].double()
        negative_likelihood_and_local_KL = (
            adjusted_log_rate.exp()-yy*(b+m)+torch.lgamma(yy+1)+kl)
        return kl_global + self.N/len(idx)*negative_likelihood_and_local_KL.sum()

    def _evaluate_elbo(self):
        # Marginal Monte Carlo for E[log mixture]; the rate and local residual stay analytic.
        value = 0.
        with torch.no_grad(), torch.random.fork_rng(devices=self._cuda_devices):
            for idx in self._chunks():
                self._seed_rng(self.config.seed+100003)
                value -= len(idx)/self.N*float(self.direct_loss(
                    idx, particles=self.config.evaluation_particles))
        return value

    def _diagnostic_state(self):
        state = super()._diagnostic_state()
        state['epsilon_variance_change'] = self.v_eps.clone()
        return state

    def update_epsilon_variance(self, *, record=True):
        """One exact full-data EB step, holding current Gaussian q_eps fixed.

        v_new = mean over spots and genes of m^2+s.
        q_eps is profiled using the old v throughout this E/M step. Fresh q at
        the next loss can only improve the exact local Gaussian objective.
        Zero-initialized components intentionally remain exactly zero.
        """
        with self._scope(), torch.no_grad():
            old = self.v_eps.detach().clone()
            total = torch.zeros(self.G, dtype=torch.float64, device=self.device)
            count = 0
            for idx in self._chunks():
                m, s, _ = self.epsilon_posterior(idx)
                total += (m.square()+s).sum(0)
                count += len(idx)
            proposed = (total/count).mean().clamp_min(self.epsilon_variance_floor)
            proposed = torch.where(old.eq(0), torch.zeros_like(proposed), proposed)
            if not torch.isfinite(proposed).all():
                raise FloatingPointError('nonfinite epsilon EB update')
            self.v_eps.copy_(proposed.to(self.v_eps.dtype))
            row = dict(full_data_rows=count, mode=self.epsilon_variance_mode,
                       old_min=float(old.min()), old_max=float(old.max()),
                       new_min=float(self.v_eps.min()), new_max=float(self.v_eps.max()),
                       update='exact full-data Gaussian second-moment EB; no mean correction')
            if record:
                self.epsilon_updates.append(row)
            return row

    def fit(self):
        """Native full-cycle entry point; external runners may own this schedule."""
        with self._scope(), self._rng_scope():
            self.set_optimization(lr=self.config.lr, num_particles=self.config.num_particles)
            if not self._warm_started:
                self._update_nuisance(self.config.nuisance_steps, reference_only=True)
                self._update_shared_variances()
                self._cycle(self.config.warm_start_sweeps)
                self.update_epsilon_variance()
                self._warm_started = True
            previous_elbo, previous = self._evaluate_elbo(), self._diagnostic_state()
            stable = 0
            for _ in range(self.config.max_sweeps):
                self._cycle(self.config.within_type_sweeps)
                self._update_nuisance(self.config.nuisance_steps)
                self._update_shared_variances()
                if self.config.fit_pi:
                    for _ in range(self.config.pi_steps):
                        self._step_group('pi', self._batch())
                self.update_epsilon_variance()
                current_elbo, current = self._evaluate_elbo(), self._diagnostic_state()
                changes = {name: self._relative_change(current[name], previous[name]) for name in current}
                changes['elbo_change'] = abs(current_elbo-previous_elbo)/max(1, abs(previous_elbo))
                self.history.append(dict(sweep=len(self.history)+1, elbo=current_elbo, **changes))
                stable = stable+1 if max(changes.values()) < self.config.tolerance else 0
                previous_elbo, previous = current_elbo, current
                if stable >= self.config.patience:
                    break
        return self

    def posterior_summary(self, *, include_epsilon_arrays=False):
        result = super().posterior_summary()
        result.update(epsilon_prior_variance=self.v_eps.detach().cpu().numpy().copy(),
                      epsilon_variance_mode=np.asarray(self.epsilon_variance_mode),
                      epsilon_mean_correction=np.asarray(False))
        if include_epsilon_arrays:
            mean = np.empty((self.N, self.G), dtype=self.config.dtype)
            sd = np.empty_like(mean)
            with self._scope(), torch.no_grad():
                for idx in self._chunks():
                    m, s, _ = self.epsilon_posterior(idx)
                    mean[idx.cpu().numpy()] = m.cpu().numpy()
                    sd[idx.cpu().numpy()] = s.sqrt().cpu().numpy()
            result.update(epsilon_mean=mean, epsilon_sd=sd)
        return result



class PyroBackfitMMF(UncenteredEpsilonMMF):
    """Coordinate SVI. Each step exposes one block to Pyro and holds the rest fixed."""

    def _step_group(self, block, idx, reference_only=False):
        if block == 'joint':
            raise ValueError('Use one coordinate block per SVI step')
        value = MMF._step_group(self, block, idx, reference_only)
        if not torch.isfinite(torch.as_tensor(value)):
            raise FloatingPointError('Nonfinite Pyro SVI loss')
        for name in self.group_names[block]:
            if not torch.isfinite(self._param_state['params'][name]).all():
                raise FloatingPointError('Nonfinite Pyro parameter: ' + name)
        self.optimizer_calls[str(block)] = self.optimizer_calls.get(str(block), 0) + 1
        return value

    def generative_model(self, idx, reference_only=False):
        a = pyro.sample('alpha', dist.Normal(torch.zeros(self.N, device=self.device,
            dtype=self.dtype), self.v_alpha.sqrt()).to_event(1))
        b = pyro.sample('gamma', dist.Normal(torch.zeros(self.G, device=self.device,
            dtype=self.dtype), self.v_gamma.sqrt()).to_event(1))
        z = {} if reference_only else {p: pyro.sample(f'z_{p[0]}_{p[1]}',
            self._program_prior(*p)) for p in self.programs}
        mix = (self._log_expected_mixture(idx, True) if reference_only
               else self._log_mixture(idx, z))
        with pyro.poutine.scale(scale=self.N / len(idx)):
            e = pyro.sample('epsilon', dist.Normal(torch.zeros_like(self.Y[idx]),
                self.v_eps.sqrt()).to_event(2))
            lograte = self.log_N[idx, None] + a[..., idx, None] + b[..., None, :] + mix + e
            pyro.sample('Y', dist.Poisson(lograte.exp()).to_event(2), obs=self.Y[idx])

    def full_guide(self, idx, reference_only=False):
        self._guide(idx, reference_only)
        m, s, _ = self.epsilon_posterior(idx, reference_only)
        with pyro.poutine.scale(scale=self.N / len(idx)):
            pyro.sample('epsilon', dist.Normal(m, s.sqrt()).to_event(2))


def epsilon_checkpoint_fields(model):
    return dict(epsilon_state_version=model.epsilon_state_version,
                epsilon_variance_mode=model.epsilon_variance_mode,
                epsilon_prior_variance=model.v_eps.detach().cpu().clone(),
                epsilon_variance_floor=model.epsilon_variance_floor,
                epsilon_updates=copy.deepcopy(model.epsilon_updates),
                epsilon_local_state='profiled Gaussian optimum from current structure, Y, and v_eps; no Adam state',
                epsilon_mean_correction=False)


def restore_epsilon_checkpoint_fields(model, state):
    if state.get('epsilon_mean_correction') is not False:
        raise ValueError('checkpoint does not explicitly certify uncentered epsilon')
    if state.get('epsilon_variance_mode') != model.epsilon_variance_mode:
        raise ValueError('epsilon variance mode mismatch')
    if state.get('epsilon_state_version') != model.epsilon_state_version:
        raise ValueError('unsupported epsilon checkpoint version')
    value = torch.as_tensor(state['epsilon_prior_variance'], device=model.device,
                            dtype=model.dtype)
    if value.shape != model.v_eps.shape:
        raise ValueError('epsilon variance shape mismatch')
    floor = float(state['epsilon_variance_floor'])
    if not torch.isfinite(value).all() or torch.any(value < 0):
        raise ValueError('checkpoint epsilon variance must be finite and nonnegative')
    if not np.isfinite(floor) or floor < 0:
        raise ValueError('checkpoint epsilon floor must be finite and nonnegative')
    model.v_eps.copy_(value)
    model.epsilon_variance_floor = floor
    model.epsilon_updates = copy.deepcopy(state.get('epsilon_updates', []))


def clone_epsilon_state(parent, child):
    """Required after legacy helpers' shallow __dict__ copy."""
    child.v_eps = parent.v_eps.detach().clone()
    child.epsilon_updates = copy.deepcopy(parent.epsilon_updates)
    return child


def load_epsilon_real_state(path, device, *, immutable_input=None, basis_path=None):
    """Restore the existing real-runner payload plus epsilon checkpoint fields.

    Optional path overrides support a deliberately reorganized canonical tree.
    This restores a phase boundary; the returned batch/monitor state must still
    be used by a runner that promises exact in-phase continuation.
    """
    from MMF.mmf import fit_config_from_mapping
    from MMF.fit_mmf import _restore_adam_step_device
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('epsilon_variance_mode', 'shared') != 'shared':
        raise ValueError('Only a shared epsilon variance can be restored')
    if any(str(name).startswith('F_') for name in state['parameters']):
        raise ValueError('Checkpoint contains a low-rank coefficient factor')
    source = state['immutable_input'] if immutable_input is None else immutable_input
    with np.load(source, allow_pickle=False) as z:
        data = {key: z[key] for key in ['Y', 'eta0', 'pi', 'coords', 'library_size']}
    config = fit_config_from_mapping(state['config'], device=device)
    model = PyroBackfitMMF(**data, programs_per_type=state['ranks'], config=config,
        epsilon_variance_mode='shared',
        epsilon_variance=state['epsilon_prior_variance'],
        epsilon_variance_floor=state['epsilon_variance_floor'])
    with model._scope(), torch.no_grad():
        for name, value in state['parameters'].items():
            model._param_state['params'][name].copy_(value.to(model.device))
        basis_source = state.get('basis_path') if basis_path is None else basis_path
        if model.basis is not None:
            model.basis.copy_(model._tensor(np.load(basis_source, allow_pickle=False)))
        model.basis_metadata = state['basis_metadata']
        model._initial_pi.copy_(state['initial_pi'].to(model.device))
        model.v_alpha.copy_(state['v_alpha'].to(model.device))
        model.v_gamma.copy_(state['v_gamma'].to(model.device))
        for program, value in state['program_variances'].items():
            model.v[program].copy_(value.to(model.device))
        restore_epsilon_checkpoint_fields(model, state)
    model.program_uids = state['program_uids']
    model._rng_state = state['cpu_rng']
    model._cuda_rng_state = state['cuda_rng']
    for key, value in state['block_optimizer_states'].items():
        model.block_optimizers[key] = torch.optim.Adam(
            [model._param_state['params'][name] for name in model.group_names[key]],
            lr=model.config.lr)
        model.block_optimizers[key].load_state_dict(value)
        _restore_adam_step_device(model.block_optimizers[key])
    model._warm_started = True
    return model, state


def paired_draw_count(requested):
    return max(2, int(requested) + int(requested) % 2)


def copy_subset(parent, retained=None, seed=None):
    retained = set(parent.programs if retained is None else retained)
    mapping = {}
    counts = [0]*parent.C
    for old in parent.programs:
        if old in retained:
            c = old[0]; mapping[old] = (c, counts[c]); counts[c] += 1
    model = object.__new__(parent.__class__)
    model.__dict__ = parent.__dict__.copy()
    model.config = replace(parent.config, seed=parent.config.seed if seed is None else seed)
    model.Kc = counts
    model.programs = list(mapping.values())
    old_uids = getattr(parent, 'program_uids', {p: f'initial_{p[0]}_{p[1]}' for p in parent.programs})
    model.program_uids = {new: old_uids[old] for old, new in mapping.items()}
    model._initial_pi = parent._initial_pi.detach().clone()
    model.v_alpha = parent.v_alpha.detach().clone()
    model.v_gamma = parent.v_gamma.detach().clone()
    model.v = {new: parent.v[old].detach().clone() for old, new in mapping.items()}
    model._rng_state = parent._rng_state.clone()
    model._cuda_rng_state = None if parent._cuda_rng_state is None else parent._cuda_rng_state.clone()
    model._training_elbo = copy.copy(parent._training_elbo)
    model._evaluation_elbo = copy.copy(parent._evaluation_elbo)
    old_program_names = {f'{prefix}_{c}_{k}' for c, k in parent.programs for prefix in ['w', 'mu', 'dvar']}
    renamed = {f'{prefix}_{c}_{k}': f'{prefix}_{new[0]}_{new[1]}'
        for (c, k), new in mapping.items() for prefix in ['w', 'mu', 'dvar']}
    params, cons = {}, {}
    for name, parameter in parent._param_state['params'].items():
        if name.startswith('F_') or (name in old_program_names and name not in renamed):
            continue
        target = renamed.get(name, name)
        params[target] = parameter.detach().clone().requires_grad_(parameter.requires_grad)
        cons[target] = parent._param_state['constraints'][name]
    model._param_state = dict(params=params, constraints=cons)
    model.group_names, model.optimizers, model.svis = {}, {}, {}
    model.block_optimizers, model.optimizer_calls = {}, {}
    for name in ['alpha', 'gamma', 'pi']:
        if name in parent.group_names:
            model._add_optimizer(name, parent.group_names[name])
    for c, k in model.programs:
        model._add_optimizer((c, k, 'q'), [f'mu_{c}_{k}', f'dvar_{c}_{k}'])
        model._add_optimizer((c, k, 'w'), [f'w_{c}_{k}'])
    model.history = []
    model.frozen_programs = ()
    model._warm_started = True
    if hasattr(parent, 'v_eps'):
        clone_epsilon_state(parent, model)
    return model, mapping


def add_program(parent, c, idx, seed):
    model, _ = copy_subset(parent, seed=seed)
    k = model.Kc[c]
    model.Kc[c] += 1
    model.programs.append((c, k))
    model.programs.sort()
    model.program_uids[c, k] = f'birth_{seed}_{c}_{k}'
    d = model.latent_dim
    with model._scope(), torch.random.fork_rng(devices=model._cuda_devices), torch.no_grad():
        model._seed_rng(seed)
        pyro.param(f'w_{c}_{k}', torch.ones(len(model.support_indices[c]), device=model.device, dtype=model.dtype)/len(model.support_indices[c]), constraint=constraints.simplex)
        pyro.param(f'mu_{c}_{k}', torch.zeros(d, device=model.device, dtype=model.dtype))
        pyro.param(f'dvar_{c}_{k}', torch.ones(d, device=model.device, dtype=model.dtype), constraint=constraints.positive)
        model.v[c, k] = model._tensor(1.)
        model._add_optimizer((c, k, 'q'), [f'mu_{c}_{k}', f'dvar_{c}_{k}'])
        model._add_optimizer((c, k, 'w'), [f'w_{c}_{k}'])
    model.expansion = seed_program(model, parent, c, k, idx, seed)
    return model


def seed_program(model, parent, c, k, idx, seed):
    """Responsibility-weighted residual SVD and a small Fisher system.

    The supplied deterministic training subset controls initialization cost only;
    subsequent fitting and ELBO comparisons use the unchanged whole-data model.
    """
    with parent._scope(), torch.no_grad():
        components = parent._indexed_log_pi(idx).T[:, :, None] + parent.log_eta0[:, None, :]
        for q in parent.programs:
            mean, var = parent._moments(*q, idx)
            w = parent._w(*q)
            components[q[0]] += mean[:, None]*w + .5*var[:, None]*w.square()
        responsibility = components.softmax(0)[c]
        rate = parent._log_expected_counts(idx).exp()
    with model._scope(), torch.no_grad(), torch.random.fork_rng(devices=model._cuda_devices):
        model._seed_rng(seed)
        ids = model.support_indices[c]
        residual = (model.Y[idx]-rate)/(rate+1).sqrt()
        attributed = (residual*responsibility.sqrt())[:, ids]
        _, singular_values, vectors = torch.svd_lowrank(attributed, q=min(3, *attributed.shape), niter=4)
        choices = []
        for j in range(vectors.shape[1]):
            for sign in [1, -1]:
                positive = (sign*vectors[:, j]).clamp_min(0)
                energy = positive.square().sum()
                if energy >= .25:
                    choices.append((float(positive.sum().square()/energy), j, sign, positive))
        effective, j, sign, direction = min(choices, key=lambda row: row[0])
        count = min(len(ids), max(8, min(50, int(np.ceil(np.sqrt(len(ids)))))))
        chosen = direction.topk(count).indices
        scale = model.N/len(idx)
        information = scale*(rate*responsibility.square()).sum(0)[ids]
        weights = torch.zeros_like(direction)
        weights[chosen] = direction[chosen]/information[chosen].clamp_min(1).sqrt()
        weights /= weights.sum()
        weights = .95*weights + .05/len(ids)
        pyro.param(f'w_{c}_{k}').unconstrained().copy_(transform_to(constraints.simplex).inv(weights))
        w = model._w(c, k)
        derivative = responsibility*w
        score = scale*((model.Y[idx]-rate)*derivative).sum(1)
        curvature = scale*(rate*derivative.square()).sum(1)
        prior = (.7/w.max()).square()
        if model.basis is None:
            # Identity-kernel simulation seeds use all training locations.
            assert len(idx) == model.N
            precision = curvature+1/prior
            mean, variance = score/precision, 1/precision
            activity = mean
        else:
            A = model.basis[idx].double()
            precision = A.T@(curvature.double()[:, None]*A)
            precision.diagonal().add_(1/prior.double())
            mean = torch.linalg.solve(precision, A.T@score.double()).to(model.dtype)
            variance = (1/precision.diagonal()).to(model.dtype)
            activity = model.basis@mean
        mean *= (1.5/(activity.abs().max()*w.max())).clamp_max(1)
        pyro.param(f'mu_{c}_{k}').unconstrained().copy_(mean)
        pyro.param(f'dvar_{c}_{k}').unconstrained().copy_(variance.log())
        model._update_variance(c, k)
    return dict(cell_type=c, program=k, training_seed_locations=len(idx),
        seed_genes=count, top_gene_indices=ids[chosen].cpu().tolist(),
        singular_direction=j, direction_sign=sign, singular_value=float(singular_values[j]), effective_genes=effective)


def raw_program_norms(model, programs=None):
    with model._scope(), torch.no_grad():
        programs = model.programs if programs is None else programs
        if not programs:
            return []
        means = torch.stack([model._q(*p)[0] for p in programs], 1)
        h = means if model.basis is None else model.basis@means
        weights = torch.stack([model._w(*p) for p in programs])
        rms = torch.linalg.vector_norm(h.double(), dim=0)*torch.linalg.vector_norm(weights.double(), dim=1)/np.sqrt(model.N*model.G)
        return [dict(cell_type=c, program=k, rms=float(rms[j])) for j, (c, k) in enumerate(programs)]


def fixed_training_score(model, idx, seed=91025, particles=8, batches=4):
    """Common marginal antithetic noise keyed by type, stable across ranks.

    Independent batches are the units for Monte Carlo standard errors; the
    positive and negative signs within each pair are not independent draws.
    Neither the model nor caller RNG stream is consumed.
    """
    count = paired_draw_count(particles)
    values = []
    with model._scope(), torch.no_grad():
        for batch in range(batches):
            types = []
            for c in range(model.C):
                generator = torch.Generator(device=model.device).manual_seed((seed+batch*1000003+c*15485863) % (2**63-1))
                half = torch.randn((count//2, len(idx), model.G), generator=generator, device=model.device, dtype=model.dtype)
                types.append(torch.cat([half, -half], 0))
            eps = torch.stack(types, 1)
            values.append(-float(model.direct_loss(idx, eps=eps, particles=count)))
    return dict(mean=float(np.mean(values)), elbo_batches=values,
        mc_standard_error=float(np.std(values, ddof=1)/np.sqrt(batches)) if batches > 1 else None,
        training_positions=len(idx), requested_particles_per_batch=particles,
        particles_per_batch=count, independent_antithetic_pairs_per_batch=count//2,
        batches=batches, seed=seed,
        common_noise='type-keyed location-gene Gaussian marginals, antithetic pairs',
        objective='unbiased marginal Monte Carlo training ELBO with exact full-coefficient Gaussian KL')


def ranks(model):
    return list(map(int, model.Kc))


def select_programs(parent, *, seed_idx, seed, output, fit, score, export, write,
                    tau=.065, k_max=3, forward_cycles=150, backfit_cycles=1000,
                    max_rounds=45, restore=None):
    """Select beneficial factors and profile weak-factor deletions.

    ``fit(model, phase, prefix, cycles, conditional_program=None)`` mutates only
    that private state and returns a JSON-compatible phase receipt. ``score``
    uses the same fixed Monte Carlo streams at every compared state. No truth,
    held-out observations, desired count vector, or type-specific threshold is
    an input to this function.
    """
    # Optional restore(path)->model keeps only one candidate on the GPU.
    # In this mode export(model,prefix) must create prefix+'.state.pt'.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(phase='selecting', threshold=tau, K_max=k_max,
        rank_selection='unknown per-type ranks', heldout_or_truth_read=False,
        proposal_gate='positive fixed-MC training ELBO gain; raw mean-effect RMS is screened only after whole-model backfitting',
        winner='highest training ELBO among eligible proposals; smaller type breaks ties',
        deletion_gate='mean-effect RMS <= tau AND positive training ELBO gain after deletion/refit',
        rounds=[], deletions=[], current_ranks=ranks(parent), selection_complete=False)

    def save():
        report['current_ranks'] = ranks(parent)
        write(output/'selection.json', report)

    def delete_weak(model, folder, final=False):
        """A small mean hW is a screen, never itself a deletion certificate."""
        checked = set()
        while True:
            weak = [r for r in raw_program_norms(model) if r['rms'] <= tau and
                    (r['cell_type'], r['program']) not in checked]
            if not weak:
                return model
            row = min(weak, key=lambda r: (r['rms'], r['cell_type'], r['program']))
            p = (row['cell_type'], row['program'])
            before = score(model)
            reduced, mapping = copy_subset(model, [q for q in model.programs if q != p])
            prefix = folder/f'delete_{len(report["deletions"]):02d}_T{p[0]}P{p[1]}'
            phase = fit(reduced, 'terminal' if final else 'backfit', prefix,
                        6000 if final else backfit_cycles)
            after = score(reduced)
            accepted = bool(after > before)
            receipt = dict(program=list(p), rms=row['rms'], baseline_elbo=before,
                deleted_refit_elbo=after, delta_elbo=after-before, accepted=accepted,
                phase_fit=phase, before_ranks=ranks(model), after_ranks=ranks(reduced),
                surviving_reindexing=[dict(old=list(a), new=list(b)) for a, b in mapping.items()])
            report['deletions'].append(receipt)
            if accepted:
                model = reduced
                checked = set()
                export(model, prefix/'accepted')
            else:
                checked.add(p)
            save()

    save()
    terminal_polished = False
    for round_index in range(1, max_rounds+1):
        baseline = score(parent)
        eligible_types = [c for c in range(parent.C) if parent.Kc[c] < k_max]
        round_path = output/f'round{round_index:02d}'
        entry = dict(round=round_index, parent_ranks=ranks(parent),
            parent_elbo=baseline, proposals=[], trials=[], accepted=False)
        report['rounds'].append(entry)
        candidates = []
        for c in eligible_types:
            trial_seed = int(seed+100003*round_index+1009*c)
            candidate = add_program(parent, c, seed_idx, trial_seed)
            p = (c, parent.Kc[c])
            prefix = round_path/f'type{c}'/'forward'
            phase = fit(candidate, 'forward', prefix, forward_cycles, conditional_program=p)
            elbo = score(candidate)
            norm = next(r['rms'] for r in raw_program_norms(candidate, programs=[p])
                        if (r['cell_type'], r['program']) == p)
            row = dict(cell_type=c, new_program=p[1], seed=trial_seed,
                training_elbo=elbo, delta_elbo=elbo-baseline, score_rms=norm,
                eligible=bool(elbo > baseline), fit=phase,
                initialization=candidate.expansion)
            entry['proposals'].append(row)
            if row['eligible']:
                if restore is not None:
                    # Persist only mutable model state. The runner's export
                    # must write this path without duplicating immutable Y.
                    checkpoint_prefix = prefix/'forward_candidate'
                    export(candidate, checkpoint_prefix)
                    candidates.append((row, str(checkpoint_prefix)+'.state.pt'))
                else:
                    candidates.append((row, candidate))
            # No GPU model reference survives to the next forward proposal
            # when disk-backed restore is enabled.
            if restore is not None:
                del candidate
            save()
        candidates.sort(key=lambda pair: (-pair[0]['training_elbo'], pair[0]['cell_type']))
        for row, candidate_state in candidates:
            candidate = restore(candidate_state) if restore is not None else candidate_state
            p = (row['cell_type'], row['new_program'])
            prefix = round_path/f'type{p[0]}'/'backfit'
            phase = fit(candidate, 'backfit', prefix, backfit_cycles)
            elbo = score(candidate)
            norm = next(r['rms'] for r in raw_program_norms(candidate, programs=[p])
                        if (r['cell_type'], r['program']) == p)
            accepted = bool(elbo > baseline and norm > tau)
            entry['trials'].append(dict(cell_type=p[0], new_program=p[1],
                training_elbo=elbo, delta_elbo=elbo-baseline, score_rms=norm,
                accepted=accepted, fit=phase))
            save()
            if not accepted:
                if restore is not None:
                    del candidate
                continue
            parent = delete_weak(candidate, round_path/'cleanup')
            entry['accepted'] = True
            entry['retained_ranks'] = ranks(parent)
            export(parent, round_path/'accepted')
            terminal_polished = False
            break
        if entry['accepted']:
            save()
            continue
        if not terminal_polished:
            report['phase'] = 'strict_terminal_polish'
            phase = fit(parent, 'terminal', round_path/'terminal', 6000)
            entry['terminal_fit'] = phase
            parent = delete_weak(parent, round_path/'terminal_cleanup', final=True)
            export(parent, round_path/'terminal'/'retained')
            terminal_polished = True
            save()
            # Re-test births at the actual strictly polished incumbent. An old
            # finite-search proposal cannot certify the final state's stop.
            continue
        report.update(phase='finished', selection_complete=True,
            stop_reason='no eligible beneficial admissible birth at the strictly polished incumbent under K_max',
            final_ranks=ranks(parent), final_training_elbo=score(parent),
            final_program_norms=raw_program_norms(parent),
            final_types_at_cap=[c for c in range(parent.C) if parent.Kc[c] >= k_max],
            final_admissible_types=eligible_types)
        export(parent, output/'final')
        save()
        return parent, report
    report.update(phase='incomplete', stop_reason='round guard', final_ranks=ranks(parent))
    export(parent, output/'incomplete_final')
    save()
    return parent, report
