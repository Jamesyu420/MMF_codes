#!/usr/bin/env python3
"""Fit MMF from arrays or an NPZ file; save posterior summaries and a JSON run record."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
from importlib.metadata import version
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import pyro
from torch.distributions import constraints, transform_to

from MMF.mmf import MMF, FitConfig, fit_config_from_mapping


def parse_programs(value: str) -> int | list[int]:
    """Accept one common nonnegative count or a count per cell type."""
    counts = [int(part) for part in value.split(",")]
    return counts[0] if len(counts) == 1 else counts


def _release_cuda(device: str) -> None:
    """Return this process's unused CUDA cache to the driver."""
    if torch.device(device).type != "cuda":
        return
    gc.collect()
    with torch.cuda.device(device):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def write_results(model: MMF, config: FitConfig, prefix: Path,
                  metadata: dict | None = None, summary: dict | None = None) -> tuple[Path, Path]:
    """Save MMF's numeric summary arrays and JSON-native run metadata."""
    if summary is None:
        summary = model.posterior_summary()
    record = {
        "config": asdict(config),
        "basis_metadata": model.basis_metadata,
        "history": model.history,
        "posterior_shapes": {key: list(value.shape) for key, value in summary.items()},
        "package_versions": {name: version(name) for name in ("numpy", "torch", "pyro-ppl")},
        "inference": {
            "backend": model.inference_backend,
            "implementation_version": "5.1",
            "interface": "pyro.infer.SVI",
            "objective": "pyro.infer.TraceMeanField_ELBO",
            "optimizer": "pyro.optim.Adam",
            "schedule": "nested_backfitting",
            "prior_variances": "closed_form_empirical_bayes",
            "likelihood_expectation": "reparameterized_monte_carlo",
            "gaussian_kl": "pyro_builtin_analytic",
        },
        "metadata": {"model_version": "4.0", **(metadata or {})},
    }
    json_text = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = Path(str(prefix) + ".npz")
    json_path = Path(str(prefix) + ".json")
    np.savez_compressed(npz_path, **summary)
    json_path.write_text(json_text + "\n", encoding="utf-8")
    return npz_path, json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path,
                        help="NPZ with Y, eta0, initial pi and optional coords/library_size")
    parser.add_argument("--output", required=True, type=Path, help="output prefix, without .npz/.json")
    parser.add_argument("--programs", type=parse_programs, default=2,
                        help="programs per type, e.g. 2 or 0,1; 0 is reference-only (default: 2)")
    parser.add_argument("--kernel", choices=("nystrom", "full", "identity"), default=None)
    integer_options = ("n-landmarks", "max-sweeps", "within-type-sweeps",
                       "warm-start-sweeps", "q-steps", "w-steps", "batch-size",
                       "num-particles", "pi-steps", "chunk-size", "seed",
                       "evaluation-particles", "patience")
    float_options = ("lengthscale", "lr", "pi-floor", "variance-floor", "jitter", "tolerance")
    for name in integer_options:
        parser.add_argument(f"--{name}", type=int)
    for name in float_options:
        parser.add_argument(f"--{name}", type=float)
    parser.add_argument("--nuisance-steps", type=int, default=None,
                        help="alternating SVI steps for q(alpha) and q(gamma) per sweep")
    parser.add_argument("--fixed-pi", dest="fit_pi", action="store_false", default=None,
                        help="keep input pi fixed instead of jointly estimating it")
    parser.add_argument("--root-steps", type=int, default=None,
                        help="legacy compatibility option; unused by the standard SVI backend")
    parser.add_argument("--dtype", choices=("float32", "float64"), default=None)
    parser.add_argument("--device", default="cuda", help="torch device, e.g. cuda or cpu")
    return parser


def fit_mmf(Y, eta0, pi, coords=None, library_size=None, output=None,
            programs: int | list[int] = 2, metadata: dict | None = None, **config):
    """Retired. This entry fitted a model without observation noise."""
    raise RuntimeError("Use MMF.fit on a prepared npz.")


def main() -> None:
    args = build_parser().parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        Y, eta0, pi = (data[key] for key in ("Y", "eta0", "pi"))
        coords = data["coords"] if "coords" in data else None
        library_size = data["library_size"] if "library_size" in data else None
    overrides = {key: value for key, value in vars(args).items()
                 if key not in {"input", "output", "programs"} and value is not None}
    fit_mmf(Y, eta0, pi, coords=coords, library_size=library_size,
            output=args.output, programs=args.programs,
            metadata={"input": str(args.input.resolve())}, **overrides)
    print(f"Saved {args.output}.npz and {args.output}.json", flush=True)


def initialize_from_reference_fit(model, baseline, seed=0, n_genes=None,
                                  background_mass=0.05,
                                  initial_log_effect=0.7,
                                  max_initial_log_effect=1.5):
    """Initialize a fresh MMF from a fitted K=0 model or its posterior summary.

    Responsibility-weighted Pearson residuals give a sparse, nonnegative gene
    direction. A Fisher-scoring step estimates signed activities in the model's
    existing spatial basis. All changes are starting values of existing model
    parameters: the likelihood, priors, constraints and ELBO are unchanged.

    ``n_genes=None`` retains ceil(sqrt(reference-supported genes)), bounded to
    8--50 genes. The small background loading mass keeps all supported genes
    available to subsequent optimization. Returned records describe the seeds.
    """
    summary = (baseline.posterior_summary() if hasattr(baseline, "posterior_summary")
               else baseline)
    records = []
    with model._scope(), torch.no_grad(), torch.random.fork_rng(devices=model._cuda_devices):
        model._seed_rng(seed)
        for name in ("alpha", "gamma"):
            model._set_shared(name, model._tensor(summary[name + "_mean"]),
                              model._tensor(summary[name + "_sd"]).square())
            getattr(model, "v_" + name).copy_(model._tensor(summary[name + "_prior_variance"]))
        pi = model._tensor(summary["pi"])
        if model.config.fit_pi:
            pyro.param("pi_logits").unconstrained().copy_(pi.log())
        else:
            model._initial_pi = pi.clone()
        base_rate = model._tensor(summary["expected_counts"])
        mixture = pi @ model.eta0
        A, d = model.basis, model.latent_dim

        for c in range(model.C):
            ids = model.support_indices[c]
            responsibility = pi[:, c, None] * model.eta0[c] / mixture
            type_rate = base_rate * responsibility
            rate = base_rate.clone()
            for k in range(model.Kc[c]):
                r = type_rate / rate
                pearson = (model.Y - rate) / (rate + 1).sqrt()
                attributed = (pearson * r.sqrt())[:, ids]
                rank = min(3, *attributed.shape)
                _, _, vectors = torch.svd_lowrank(attributed, q=rank, niter=4)
                direction = vectors[:, 0]
                if direction.clamp_min(0).square().sum() < (-direction).clamp_min(0).square().sum():
                    direction = -direction
                direction = direction.clamp_min(0)
                count = min(len(ids), n_genes or max(8, min(50, int(np.ceil(np.sqrt(len(ids)))))))
                chosen = direction.topk(count).indices
                # Undo the gene-dependent information scale of Pearson residuals.
                information = (rate * r.square()).sum(0)[ids]
                weights = torch.zeros_like(direction)
                weights[chosen] = direction[chosen] / information[chosen].clamp_min(1).sqrt()
                weights /= weights.sum()
                weights = (1 - background_mass) * weights + background_mass / len(ids)
                name = f"{c}_{k}"
                pyro.param("w_" + name).unconstrained().copy_(
                    transform_to(constraints.simplex).inv(weights))
                w = model._w(c, k)

                derivative = r * w
                score = ((model.Y - rate) * derivative).sum(1)
                curvature = (rate * derivative.square()).sum(1)
                v0 = (initial_log_effect / w.max()).square()
                if A is None:
                    precision = curvature + 1 / v0
                    mean, variance = score / precision, 1 / precision
                    activity = mean
                else:
                    # Form the small Fisher system in double precision so its
                    # ridge is retained for correlated spatial basis columns.
                    A64 = A.double()
                    precision = A64.T @ (curvature.double()[:, None] * A64)
                    precision.diagonal().add_(1 / v0.double())
                    mean = torch.linalg.solve(precision, A64.T @ score.double()).to(model.dtype)
                    variance = (1 / precision.diagonal()).to(model.dtype)
                    activity = A @ mean
                scale = (max_initial_log_effect / (activity.abs().max() * w.max())).clamp_max(1)
                mean, activity = mean * scale, activity * scale
                pyro.param("mu_" + name).unconstrained().copy_(mean)
                pyro.param("dvar_" + name).unconstrained().copy_(variance.log())
                model._update_variance(c, k)
                changed_type_rate = type_rate * torch.exp(activity[:, None] * w)
                rate = rate + changed_type_rate - type_rate
                type_rate = changed_type_rate
                records.append(dict(cell_type=c, program=k, seed_genes=count,
                                    top_gene_indices=ids[chosen].cpu().tolist(),
                                    max_absolute_log_effect=float(activity.abs().max() * w.max()),
                                    prior_variance=float(model.v[c, k])))
        model._warm_started = True
    return records

def initialize_fixed_loadings(model, baseline, source, rcc):
    """Fisher seed from core training counts; source gene loadings stay fixed."""
    records = []
    source_programs = np.flatnonzero(source['program_cell_type'] == rcc)
    with torch.random.fork_rng(devices=model._cuda_devices), model._scope(), torch.no_grad():
        model._seed_rng(model.config.seed)
        for name in ['alpha', 'gamma']:
            model._set_shared(name, model._tensor(baseline[name + '_mean']),
                              model._tensor(baseline[name + '_sd']).square())
            getattr(model, 'v_' + name).copy_(model._tensor(baseline[name + '_prior_variance']))
        pi = model._tensor(baseline['pi'])
        pyro.param('pi_logits').unconstrained().copy_(pi.log())
        rate = model._tensor(baseline['expected_counts'])
        responsibility = pi[:, rcc, None] * model.eta0[rcc] / (pi @ model.eta0)
        type_rate = rate * responsibility
        A = model.basis.double()
        ids = model.support_indices[rcc]
        for k, source_p in enumerate(source_programs):
            name = f'{rcc}_{k}'
            weights = model._tensor(source['loadings'][source_p])[ids]
            pyro.param('w_' + name).unconstrained().copy_(transform_to(constraints.simplex).inv(weights))
            w = model._w(rcc, k)
            r = type_rate / rate
            derivative = r * w
            score = ((model.Y - rate) * derivative).sum(1).double()
            curvature = (rate * derivative.square()).sum(1).double()
            v0 = (.7 / w.max()).square().double()
            precision = A.T @ (curvature[:, None] * A)
            precision.diagonal().add_(1 / v0)
            mean = torch.linalg.solve(precision, A.T @ score).to(model.dtype)
            variance = (1 / precision.diagonal()).to(model.dtype)
            activity = model.basis @ mean
            scale = (1.5 / (activity.abs().max() * w.max())).clamp_max(1)
            mean, activity = mean * scale, activity * scale
            pyro.param('mu_' + name).unconstrained().copy_(mean)
            pyro.param('dvar_' + name).unconstrained().copy_(variance.log())
            model._update_variance(rcc, k)
            changed = type_rate * torch.exp(activity[:, None] * w)
            rate, type_rate = rate + changed - type_rate, changed
            records.append(dict(program=k, max_absolute_log_effect=float(activity.abs().max() * w.max()),
                                prior_variance=float(model.v[rcc, k])))
        model._warm_started = True
    return records

def _restore_adam_step_device(optimizer):
    """Undo map_location for non-capturable Adam's CPU-only scalar step."""
    torch_optimizers = (optimizer.optim_objs.values()
                        if hasattr(optimizer, "optim_objs") else (optimizer,))
    for torch_optimizer in torch_optimizers:
        for group in torch_optimizer.param_groups:
            if not group.get("capturable", False) and not group.get("fused", False):
                for parameter in group["params"]:
                    state = torch_optimizer.state.get(parameter, {})
                    if torch.is_tensor(state.get("step")):
                        state["step"] = state["step"].cpu()
    # Pyro may have restored named state that no parameter has consumed yet.
    for saved in getattr(optimizer, "_state_waiting_to_be_consumed", {}).values():
        for group in saved["param_groups"]:
            if not group.get("capturable", False) and not group.get("fused", False):
                for parameter_id in group["params"]:
                    state = saved["state"].get(parameter_id, {})
                    if torch.is_tensor(state.get("step")):
                        state["step"] = state["step"].cpu()

def snapshot(model):
    """CPU snapshots; effects are gene-centred within each type's reference support."""
    s = model.posterior_summary()
    effects = {}
    for c in np.unique(s["program_cell_type"]):
        ids = s["program_cell_type"] == c
        support = s["reference_support"][c]
        w = s["loadings"][ids][:, support]
        effects[str(int(c))] = s["h_mean"][ids].T @ (w - w.mean(1, keepdims=True))
    with model._scope():
        elbo = model._evaluate_elbo()
    return dict(sweep=len(model.history), rate=s["expected_counts"], pi=s["pi"],
                variances=np.r_[s["alpha_prior_variance"], s["gamma_prior_variance"],
                                s["prior_variance"]].astype(float),
                variance_names=["alpha", "gamma"] +
                    [f"program_{c}_{k}" for c, k in model.programs],
                effects=effects, loadings=s["loadings"], activities=s["h_mean"],
                elbo=elbo)

def _relative_change(new, old):
    return float(np.linalg.norm(new - old) / max(np.linalg.norm(old), 1e-12))

def check_convergence(model, previous_snapshot, window=50):
    """Require the prespecified single-sweep and full-window criteria without relaxation."""
    current = snapshot(model)
    keys = ["rate_change", "pi_change", "variance_change", "elbo_change"]
    recent = model.history[-10:]
    recent_max = {k: max((float(row[k]) for row in recent), default=None) for k in keys}
    checks = {"ten_stable_sweeps": len(recent) == 10 and
              all(float(row[k]) < 1e-4 for row in recent for k in keys)}
    stat = dict(sweep=current["sweep"],
                effective_batch=min(model.config.batch_size or model.N, model.N),
                particles=model.config.num_particles,
                evaluation_particles=model.config.evaluation_particles,
                lr=float(model.config.lr), recent_max=recent_max)
    checks["window_complete"] = (previous_snapshot is not None and
                                current["sweep"] - previous_snapshot["sweep"] >= window)
    if previous_snapshot is not None:
        old = previous_snapshot
        variance_change = np.abs(current["variances"] - old["variances"]) / np.maximum(
            np.abs(old["variances"]), model.config.variance_floor)
        effect_drift = {c: float(np.sqrt(np.mean((effect - old["effects"][c]) ** 2)))
                        for c, effect in current["effects"].items()}
        values = dict(window_sweeps=current["sweep"] - old["sweep"],
                      window_rate_change=_relative_change(current["rate"], old["rate"]),
                      window_pi_change=_relative_change(current["pi"], old["pi"]),
                      window_variance_change=float(variance_change.max()),
                      variance_changes=dict(zip(current["variance_names"], variance_change.tolist())),
                      centered_effect_rms_drift=max(effect_drift.values(), default=0.0),
                      effect_rms_drift_by_type=effect_drift,
                      elbo_drift_per_entry=abs(current["elbo"] - old["elbo"]) / current["rate"].size,
                      elbo_gain_per_entry=(current["elbo"] - old["elbo"]) / current["rate"].size,
                      loading_relative_change=_relative_change(current["loadings"], old["loadings"]),
                      activity_relative_change=_relative_change(current["activities"], old["activities"]))
        stat.update(values)
        checks.update(rate=values["window_rate_change"] < 1e-3,
                      pi=values["window_pi_change"] < 1e-3,
                      variance=values["window_variance_change"] < 1e-3,
                      effect=values["centered_effect_rms_drift"] < 1e-3,
                      elbo=values["elbo_drift_per_entry"] < 1e-5)
    stat.update(checks=checks, converged=bool(all(checks.values())))
    return stat, current

def save_checkpoint(model, path):
    """Atomically save the full model, Adam state and random-number streams."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(model, temporary)
    temporary.replace(path)

def load_checkpoint(path, device="cuda"):
    """Load only trusted checkpoints created by save_checkpoint."""
    target = torch.device(device)
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    model = torch.load(path, map_location=target, weights_only=False)
    if (getattr(model, "inference_backend", "").startswith("pyro_svi")
            and not hasattr(model, "svis")):
        raise ValueError("Legacy custom-ELBO checkpoint cannot resume with the Pyro SVI "
                         "implementation. Start a new fit or explicitly transfer initial values.")
    model.device = target
    model.config.device = str(target)
    model._cuda_devices = [target.index] if target.type == "cuda" else []
    model._rng_state = model._rng_state.cpu()
    if model._cuda_rng_state is not None:
        model._cuda_rng_state = model._cuda_rng_state.cpu()
    elif target.type == "cuda":
        model._cuda_rng_state = torch.Generator(device=target).manual_seed(
            model.config.seed).get_state()
    for optimizer in model.optimizers.values():
        _restore_adam_step_device(optimizer)
    return model


def set_optimization(model, lr, particles):
    """Set learning rate and particle count in all Pyro optimizer blocks."""
    model.set_optimization(lr=float(lr), num_particles=int(particles))


def fit_to_convergence(model, checkpoint, *, max_sweeps=12000,
                       minimum_sweeps=None, window=50, schedule=None):
    """Monitor scheduled core SVI sweeps without external latent-variable updates.

    Two passing windows are required. The cap is an absolute history length;
    reaching it saves the current model and reports budget exhaustion.
    """
    checkpoint = Path(checkpoint)
    schedule = schedule or [(150, .02, 16), (float('inf'), .005, 32)]
    minimum_sweeps = (500 if model.programs else 100) if minimum_sweeps is None else minimum_sweeps
    protocol = 'pyro_svi_only_v1'
    if getattr(model, 'convergence_protocol', None) != protocol:
        model.converged = False
        model.convergence_history = []
        model.convergence_protocol = protocol
    if getattr(model, 'converged', False):
        return model
    previous = snapshot(model)
    started = time.perf_counter()
    diagnostics = model.convergence_history
    while len(model.history) < max_sweeps:
        start = len(model.history)
        end, lr, particles = next(stage for stage in schedule if start < stage[0])
        set_optimization(model, lr, particles)
        model.config.batch_size = model.N
        model.config.evaluation_particles = 64
        model.config.patience = window + 1
        model.config.max_sweeps = int(min(window, end - start, max_sweeps - start))
        model.fit()
        for row in model.history[start:]:
            row.update(learning_rate=lr, training_particles=particles)
        diagnostic, previous = check_convergence(model, previous, window=window)
        diagnostic.update(protocol=protocol, seconds=time.perf_counter() - started)
        diagnostics.append(diagnostic)
        model.converged = (len(model.history) >= minimum_sweeps and len(diagnostics) >= 2
                           and all(row['converged'] for row in diagnostics[-2:]))
        save_checkpoint(model, checkpoint)
        checkpoint.with_name(checkpoint.stem + '_convergence.json').write_text(
            json.dumps(diagnostics, indent=2) + '\n')
        print(f'{checkpoint.stem}: sweep={len(model.history)}, lr={lr:g}, '
              f'particles={particles}, converged={model.converged}', flush=True)
        if model.converged:
            break
    model.fit_status = 'converged' if model.converged else 'budget_exhausted'
    save_checkpoint(model, checkpoint)
    return model


def program_norms(summary):
    """Raw posterior-mean ||h w.T||_F and its spot-gene RMS, without centering."""
    h, w = summary['h_mean'], summary['loadings']
    scale = np.sqrt(h.shape[1] * w.shape[1])
    norms = np.linalg.norm(h, axis=1) * np.linalg.norm(w, axis=1)
    return [dict(cell_type=int(c), program=int(k), global_index=p,
                 frobenius=float(norms[p]), rms=float(norms[p]/scale),
                 h_mean=float(h[p].mean()), h_spatial_sd=float(h[p].std()))
            for p, (c,k) in enumerate(zip(summary['program_cell_type'], summary['program_index']))]


def _seed_added_program(model, summary, c, k, seed):
    """Initialize the added factor from full-model responsibility-weighted residuals."""
    with model._scope(), torch.no_grad(), torch.random.fork_rng(devices=model._cuda_devices):
        model._seed_rng(seed)
        pi = model._tensor(summary['pi'])
        components = pi.T.log()[:, :, None] + model.log_eta0[:, None, :]
        for p, old_c in enumerate(summary['program_cell_type']):
            w = model._tensor(summary['loadings'][p])
            h = model._tensor(summary['h_mean'][p])
            variance = model._tensor(summary['h_sd'][p]).square()
            components[int(old_c)] += h[:, None]*w + .5*variance[:, None]*w.square()
        responsibility = components.softmax(0)[c]
        rate, ids = model._tensor(summary['expected_counts']), model.support_indices[c]
        residual = (model.Y-rate)/(rate+1).sqrt()
        attributed = (residual*responsibility.sqrt())[:, ids]
        _, singular_values, vectors = torch.svd_lowrank(
            attributed, q=min(3,*attributed.shape), niter=4)
        # Prefer a compact nonnegative module over its dense negative complement.
        directions = []
        for j in range(vectors.shape[1]):
            for sign in (1,-1):
                positive = (sign*vectors[:,j]).clamp_min(0)
                energy = positive.square().sum()
                if energy >= .25:
                    effective_genes = positive.sum().square()/energy
                    directions.append((float(effective_genes),j,sign,positive))
        effective_genes,j,sign,direction = min(directions,key=lambda row:row[0])
        count = min(len(ids), max(8,min(50,int(np.ceil(np.sqrt(len(ids)))))))
        chosen = direction.topk(count).indices
        information = (rate*responsibility.square()).sum(0)[ids]
        weights = torch.zeros_like(direction)
        weights[chosen] = direction[chosen]/information[chosen].clamp_min(1).sqrt()
        weights /= weights.sum()
        weights = .95*weights + .05/len(ids)
        name = f'{c}_{k}'
        pyro.param('w_'+name).unconstrained().copy_(transform_to(constraints.simplex).inv(weights))
        w = model._w(c,k)
        derivative = responsibility*w
        score = ((model.Y-rate)*derivative).sum(1)
        curvature = (rate*derivative.square()).sum(1)
        prior = (.7/w.max()).square()
        if model.basis is None:
            precision = curvature+1/prior
            mean, variance = score/precision, 1/precision
            activity = mean
        else:
            A = model.basis.double()
            precision = A.T@(curvature.double()[:,None]*A)
            precision.diagonal().add_(1/prior.double())
            mean = torch.linalg.solve(precision,A.T@score.double()).to(model.dtype)
            variance = (1/precision.diagonal()).to(model.dtype)
            activity = model.basis@mean
        mean *= (1.5/(activity.abs().max()*w.max())).clamp_max(1)
        pyro.param('mu_'+name).unconstrained().copy_(mean)
        pyro.param('dvar_'+name).unconstrained().copy_(variance.log())
        model._update_variance(c,k)
    return dict(cell_type=c,program=k,seed_genes=count,
                top_gene_indices=ids[chosen].cpu().tolist(),
                singular_direction=j, singular_value=float(singular_values[j]),
                direction_sign=sign, effective_genes=effective_genes)


def expand_program(parent, cell_type, coords, *, seed=None):
    """Warm independent candidate: one new factor, preserved old factors, fresh Adam."""
    from dataclasses import replace
    c = int(cell_type)
    counts = list(parent.Kc)
    k = counts[c]
    counts[c] += 1
    seed = parent.config.seed if seed is None else int(seed)
    summary = parent.posterior_summary()
    model = MMF(parent.Y,parent.eta0,summary['pi'],coords=coords,
                library_size=parent.library_size,programs_per_type=counts,
                config=replace(parent.config,seed=seed))
    if parent.basis is not None:
        model.basis.copy_(parent.basis)
    with torch.no_grad():
        for name, parameter in parent._param_state['params'].items():
            model._param_state['params'][name].copy_(parameter)
        model._initial_pi = parent._initial_pi.detach().clone()
        model.v_alpha.copy_(parent.v_alpha)
        model.v_gamma.copy_(parent.v_gamma)
        for program in parent.programs:
            model.v[program].copy_(parent.v[program])
    model.frozen_programs = tuple(parent.programs)
    model.expansion = _seed_added_program(model,summary,c,k,seed)
    model.expansion['global_index'] = model.programs.index((c,k))
    model._warm_started = True
    return model


def fit_progressive_phase(model, prefix, *, phase, max_sweeps=1000, particles=32):
    """Pure SVI forward/backfit phase: .02 for 150 sweeps, then .005; hard cap."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    model.fit_phase = phase
    if phase != 'forward':
        model.frozen_programs = ()
    model.history, model.convergence_history = [], []
    model.converged = False
    current = snapshot(model)
    initial_elbo = current['elbo']
    norm_history = [dict(sweep=0,programs=program_norms(model.posterior_summary()))]
    started = time.perf_counter()
    while len(model.history) < max_sweeps:
        start = len(model.history)
        lr, end = (.02,150) if start < 150 else (.005,max_sweeps)
        set_optimization(model,lr,particles)
        model.config.batch_size = model.N
        model.config.evaluation_particles = 64
        model.config.patience = 51
        model.config.max_sweeps = min(50,end-start,max_sweeps-start)
        model.fit()
        for row in model.history[start:]:
            row.update(learning_rate=lr,training_particles=particles)
        diagnostic,current = check_convergence(model,current,window=50)
        model.convergence_history.append(diagnostic)
        norm_history.append(dict(sweep=len(model.history),programs=program_norms(model.posterior_summary())))
        model.converged = (phase != 'forward' and len(model.history)>=500 and
            len(model.convergence_history)>=2 and
            all(row['converged'] for row in model.convergence_history[-2:]))
        save_warm_state(model,prefix)
        if model.converged:
            break
    summary = model.posterior_summary()
    added = getattr(model,'expansion',None)
    new_norm = next((row for row in program_norms(summary) if added and
        (row['cell_type'],row['program']) == (added['cell_type'],added['program'])),None)
    record = dict(phase=phase,initial_elbo=initial_elbo,final_elbo=current['elbo'],
        sweeps=len(model.history),max_sweeps=max_sweeps,seconds=time.perf_counter()-started,
        status='converged' if model.converged else 'budget_exhausted',
        converged=model.converged,ranks=list(model.Kc),added_program=added,
        new_program_norm=new_norm,norm_history=norm_history,
        convergence=model.convergence_history,shared_corrections=False,
        activity='raw posterior-mean h; no centering or standardization')
    write_results(model,model.config,prefix,metadata=record,summary=summary)
    save_warm_state(model,prefix)
    print(f'{prefix}: {phase}, sweeps={record["sweeps"]}, '
          f'ELBO={record["final_elbo"]:.1f}, new RMS={None if new_norm is None else new_norm["rms"]}',flush=True)
    return summary,record


def _assign_worker_device(queue):
    global _worker_device
    _worker_device = queue.get()


def _fit_addition(parent_path, coords_path, prefix, cell_type, device,
                  forward_sweeps, backfit_sweeps, particles):
    """Worker used by progressive_fit; GPU choice changes no selection rule."""
    from contextlib import redirect_stdout, redirect_stderr
    device = globals().get('_worker_device',device)
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    with prefix.with_suffix('.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
        parent = load_warm_state(parent_path,coords_path,device=device)
        coords = np.load(coords_path)['coords']
        model = expand_program(parent,cell_type,coords)
        del parent
        _, forward = fit_progressive_phase(model,prefix.parent/'forward',
            phase='forward',max_sweeps=forward_sweeps,particles=particles)
        _, backfit = fit_progressive_phase(model,prefix.parent/'backfit',
            phase='backfit',max_sweeps=backfit_sweeps,particles=particles)
        row = dict(cell_type=int(cell_type),score_rms=backfit['new_program_norm']['rms'],
            forward_score_rms=forward['new_program_norm']['rms'],
            seconds=forward['seconds']+backfit['seconds'],
            sweeps=forward['sweeps']+backfit['sweeps'],
            final_elbo=backfit['final_elbo'],
            status=backfit['status'],converged=backfit['converged'])
        del model
        _release_cuda(device)
    return row


def progressive_fit(data_path, output, *, threshold_rms, baseline=None,
                    devices=('cuda:0',), forward_sweeps=150, backfit_sweeps=1000,
                    particles=32, max_additions=15):
    """Select additions by raw ||h w.T||_F/sqrt(NG), after equal-budget backfitting.

    Every type is tested from the same incumbent; all latent means/variances use
    Pyro SVI. Candidates rank by training ELBO; acceptance uses one fixed product-norm threshold.
    Truth and held-out data are not inputs. max_additions is a safety guard, not
    evidence of norm stopping. Independent proposals can run on separate GPUs.
    """
    raise RuntimeError("Use MMF.fit on a prepared npz.")




def save_warm_state(model, prefix):
    """Save numeric warm starts as prefix.warm.npz/json, without pickle or Adam."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f'param_{name}': value.detach().cpu().numpy().copy()
              for name, value in model._param_state['params'].items()}
    arrays.update(initial_pi=model._initial_pi.detach().cpu().numpy(),
                  v_alpha=model.v_alpha.detach().cpu().numpy(),
                  v_gamma=model.v_gamma.detach().cpu().numpy(),
                  program_variances=np.array([float(model.v[p]) for p in model.programs],
                                             dtype=model.config.dtype))
    if model.basis is not None:
        arrays['basis'] = model.basis.detach().cpu().numpy()
    record = dict(format='MMF numeric warm state v1', config=asdict(model.config),
                  ranks=list(model.Kc), basis_metadata=model.basis_metadata,
                  parameters=list(model._param_state['params']),
                  frozen_programs=list(model.frozen_programs))
    npz_path, json_path = prefix.with_suffix('.warm.npz'), prefix.with_suffix('.warm.json')
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(record, indent=2) + '\n')
    return npz_path, json_path


def load_warm_state(prefix, data_path, device):
    """Rebuild MMF from safe numeric arrays; no torch.load or pickle is used."""
    prefix = Path(prefix)
    record = json.loads(prefix.with_suffix('.warm.json').read_text())
    config = fit_config_from_mapping(record['config'], device=str(device))
    with np.load(data_path, allow_pickle=False) as data:
        model = MMF(data['Y'], data['eta0'], data['pi'],
                    coords=data['coords'] if 'coords' in data else None,
                    library_size=data['library_size'] if 'library_size' in data else None,
                    programs_per_type=record['ranks'], config=config)
    with np.load(prefix.with_suffix('.warm.npz'), allow_pickle=False) as arrays, torch.no_grad():
        if 'basis' in arrays:
            model.basis.copy_(model._tensor(arrays['basis']))
        model.basis_metadata = {key: value for key, value in record['basis_metadata'].items()
                                if key != 'guide_rank'}
        for name in record['parameters']:
            if str(name).startswith('F_'):
                raise ValueError('Warm state contains a low-rank coefficient factor')
            model._param_state['params'][name].copy_(model._tensor(arrays[f'param_{name}']))
        model._initial_pi.copy_(model._tensor(arrays['initial_pi']))
        model.v_alpha.copy_(model._tensor(arrays['v_alpha']))
        model.v_gamma.copy_(model._tensor(arrays['v_gamma']))
        for p, variance in zip(model.programs, arrays['program_variances']):
            model.v[p].copy_(model._tensor(variance))
    model.frozen_programs = tuple(map(tuple, record['frozen_programs']))
    model._warm_started = True
    return model


if __name__ == "__main__":
    main()
