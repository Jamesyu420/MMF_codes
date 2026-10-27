"""Fresh-K0 paired MMF simulation runner; all writes stay in --output.

The default command performs one explicitly requested fit. --interface-check
only exercises a small synthetic caller-owned input; it is not a study fit.
The fit is shared-epsilon Pyro backfitting with a diagonal coefficient guide.
No truth or heldout file is opened during optimization or automatic rank selection.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
import socket
from pathlib import Path
import sys
import time

import numpy as np
import torch
from MMF.backfit import (PyroBackfitMMF, add_program, fixed_training_score, raw_program_norms,
    select_programs, epsilon_checkpoint_fields, gaussian_epsilon_diagnostics)
from MMF.mmf import FitConfig

sys.dont_write_bytecode = True


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['pyro'], default='pyro')
    p.add_argument('--guide-rank', type=int, default=0)
    p.add_argument('--particles', type=int, default=8)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--variance-mode', choices=['shared'], default='shared')
    p.add_argument('--epsilon-initial-variance', type=float, default=.1)
    p.add_argument('--epsilon-variance-floor', type=float, default=1e-8)
    p.add_argument('--budget', type=float, default=21600)
    p.add_argument('--fixed-ranks', help='Comma-separated diagnostic ranks; automatic K0 is primary')
    p.add_argument('--stop-after-k0', action='store_true', help='Finish strict fresh K0 then stop before rank selection')
    p.add_argument('--interface-check', action='store_true')
    return p


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    names = ['MMF.mmf', 'MMF.fit_mmf', 'MMF.backfit']
    files = {Path(__file__).resolve()}
    for name in names:
        files.add(Path(importlib.import_module(name).__file__).resolve())
    return {str(path): sha(path) for path in sorted(files)}


def stable(history):
    if len(history) < 10:
        return False
    recent = history[-10:]
    first = np.mean([r['elbo_per_entry'] for r in recent[:5]])
    last = np.mean([r['elbo_per_entry'] for r in recent[5:]])
    keys = ['rate', 'pi', 'variance', 'effect']
    return bool(abs(first-last)/max(1., abs(first)) < .005 and all(
        max(r[key+'_change'] for key in keys) < .01 and
        r['epsilon_variance_change'] < .01 and r['epsilon_EB_last3_relative_change'] < .01 and
        r['epsilon_stationarity_relative'] < 1e-6 for r in recent[-3:]))


class SimulationCallbacks:
    def __init__(self, args):
        self.args = args
        self.output, self.deadline, self.seed = args.output, time.time()+args.budget-30, args.seed
        self.score_calls = 0

    def score(self, model):
        idx = torch.arange(model.N, device=model.device)
        row = fixed_training_score(model, idx, seed=91025, particles=16, batches=4)
        row.update(call=self.score_calls, ranks=list(model.Kc))
        write(self.output/'scores'/f'{self.score_calls:05d}.json', row)
        self.score_calls += 1
        return row['mean']

    def probe(self, model):
        with model._scope(), torch.no_grad():
            idx = torch.arange(model.N, device=model.device)
            elbo = fixed_training_score(model, idx, seed=51023, particles=16, batches=4)['mean']
            rate = model._log_expected_counts(idx).exp().cpu().numpy()
            pi = model._indexed_log_pi(idx).exp().cpu().numpy()
            variance = torch.stack([model.v_alpha, model.v_gamma]+[model.v[p] for p in model.programs]).reshape(-1)
            stationarity = 0.; epsilon_variance = np.zeros(1); epsilon_EB_last3 = 0.
            if hasattr(model, 'v_eps'):
                epsilon_variance=model.v_eps.reshape(-1).cpu().numpy().copy()
                diagnostics=gaussian_epsilon_diagnostics(model.structural_log_expected_counts(idx),model.Y[idx],model.v_eps)
                stationarity=max(diagnostics['maximum_log_rate_newton_correction'],
                                 diagnostics['maximum_log_rate_reconstruction_error'])
                updates=getattr(model,'epsilon_updates',[])[-3:]
                epsilon_EB_last3=max([row.get('true_relative_change',float('inf')) for row in updates]) if len(updates)==3 else float('inf')
            variance = variance.cpu().numpy()
            effect = torch.zeros((model.C, model.N, model.G), device=model.device, dtype=model.dtype)
            for c, k in model.programs:
                mean, _ = model._moments(c, k, idx)
                effect[c] += mean[:, None]*model._w(c, k)
            effect = effect.cpu().numpy()
        if not np.isfinite(elbo) or not all(np.isfinite(a).all() for a in [rate, pi, variance, effect]):
            raise FloatingPointError('Nonfinite MMF training diagnostic')
        if not np.isfinite(stationarity):
            raise FloatingPointError('Nonfinite local epsilon stationarity diagnostic')
        return dict(elbo_per_entry=elbo/(model.N*model.G), rate=rate, pi=pi,
                    variance=variance, effect=effect, epsilon_variance=epsilon_variance,
                    epsilon_EB_last3_relative_change=epsilon_EB_last3,
                    epsilon_stationarity_relative=stationarity)

    def update_epsilon(self,model):
        old=model.v_eps.detach().clone()
        model.update_epsilon_variance()
        change=float(torch.linalg.vector_norm((model.v_eps-old).reshape(-1))/
                     max(float(torch.linalg.vector_norm(old.reshape(-1))),model.epsilon_variance_floor,1e-300))
        model.epsilon_updates[-1]['true_relative_change']=change
        return change

    def epsilon_change(self,model,current,previous):
        return float(np.linalg.norm(current['epsilon_variance']-previous['epsilon_variance'])/
                     max(float(np.linalg.norm(previous['epsilon_variance'])),
                         getattr(model,'epsilon_variance_floor',1e-8),1e-300))

    def cycle(self, model, conditional_program=None):
        """One full-data cycle is exactly one simulation epoch."""
        idx = torch.arange(model.N, device=model.device)
        for p in model.programs if conditional_program is None else [conditional_program]:
            model._step_group((*p, 'q'), idx)
            model._step_group((*p, 'w'), idx)
            model._update_variance(*p)
        if conditional_program is None:
            for name in ['alpha', 'gamma']:
                model._step_group(name, idx)
            model._update_shared_variances()
            model._step_group('pi', idx)
            if hasattr(model, 'v_eps'):
                self.update_epsilon(model)

    def fit(self, model, phase, prefix, cycles, conditional_program=None):
        prefix = Path(prefix); prefix.mkdir(parents=True, exist_ok=True)
        began = time.time(); history = []; previous = None; status = 'phase_budget_completed'
        initial_epsilon_updates = len(getattr(model, 'epsilon_updates', []))
        exit_validation = None
        strict = phase in ['terminal', 'k0', 'fixed_k_terminal']
        if strict:
            previous = self.probe(model)
        with model._scope(), model._rng_scope():
            for step in range(1, cycles+1):
                if time.time() >= self.deadline:
                    model._rng_state = torch.get_rng_state()
                    if model._cuda_devices:
                        model._cuda_rng_state = torch.cuda.get_rng_state(model.device)
                    self.export(model, prefix/'deadline', summary=False)
                    raise TimeoutError('Simulation wall-clock budget; full state saved')
                lr = .02 if step <= 150 else .005 if step <= 1000 else .001 if step <= 3000 else .0002
                model.set_optimization(lr=lr, num_particles=8)
                self.cycle(model, conditional_program)
                if step % 50:
                    continue
                if strict:
                    current = self.probe(model)
                    row = dict(step=step, elbo_per_entry=current['elbo_per_entry'], seconds=time.time()-began,
                               epsilon_stationarity_relative=current['epsilon_stationarity_relative'],
                               epsilon_variance_change=self.epsilon_change(model,current,previous),
                               epsilon_EB_last3_relative_change=current['epsilon_EB_last3_relative_change'],
                               full_data_epsilon_EB_events=len(getattr(model, 'epsilon_updates', []))-initial_epsilon_updates)
                    for key in ['rate', 'pi', 'variance', 'effect']:
                        row[key+'_change'] = float(np.linalg.norm(current[key]-previous[key])/max(1., np.linalg.norm(previous[key])))
                    previous = current
                    history.append(row)
                    if step >= 500 and stable(history):
                        # Every cycle above is one complete-data epoch, so each
                        # 50-step monitor gap includes 50 genuine epsilon EB
                        # events, rather than repeated unchanged variance reads.
                        # Re-profile one additional complete EB step and probe
                        # the state that will actually be exported.
                        if hasattr(model, 'v_eps'):
                            self.update_epsilon(model)
                        verified = self.probe(model)
                        changes = {key:float(np.linalg.norm(verified[key]-current[key])/max(1.,np.linalg.norm(current[key])))
                                   for key in ['rate','pi','variance','effect']}
                        elbo_change=abs(verified['elbo_per_entry']-current['elbo_per_entry'])/max(1.,abs(current['elbo_per_entry']))
                        epsilon_change=self.epsilon_change(model,verified,current)
                        accepted=bool(max(changes.values()) < .01 and elbo_change < .005 and
                                      epsilon_change < .01 and verified['epsilon_EB_last3_relative_change'] < .01 and
                                      verified['epsilon_stationarity_relative'] < 1e-6)
                        exit_validation=dict(extra_complete_epsilon_EB_event=hasattr(model,'v_eps'),
                            relative_changes=changes,elbo_relative_change=elbo_change,
                            epsilon_variance_true_relative_change=epsilon_change,
                            epsilon_EB_last3_relative_change=verified['epsilon_EB_last3_relative_change'],
                            epsilon_stationarity_relative=verified['epsilon_stationarity_relative'],passed=accepted)
                        previous=verified
                        if accepted:status = 'training_probe_stable'
                else:
                    history.append(dict(step=step, seconds=time.time()-began))
                write(prefix/'phase.json', dict(phase=phase, status=status, steps=step,
                    history=history, ranks=list(model.Kc), seconds=time.time()-began))
                if step % 500 == 0:
                    # The context saves RNG on exit; capture it explicitly for
                    # an in-phase snapshot before serializing the model.
                    model._rng_state = torch.get_rng_state()
                    if model._cuda_devices:
                        model._cuda_rng_state = torch.cuda.get_rng_state(model.device)
                    self.export(model, prefix/'latest', summary=False)
                if status == 'training_probe_stable':
                    break
        record = dict(phase=phase, status=status, steps=step, seconds=time.time()-began,
            history=history, ranks=list(model.Kc),
            full_data_epsilon_EB_events=len(getattr(model, 'epsilon_updates', []))-initial_epsilon_updates,
            strict_exit_validation=exit_validation,
            conditional_program=None if conditional_program is None else list(conditional_program),
            monitor='Minimum500; two5 fixed-training ELBO windows relative<.005; last3 rate/Pi/structural-EBvariance/rawtypeeffect changes<.01; independent epsilon true relative norm changes<.01 and last3 genuine full-data EB events<.01; final full EB and re-probe; normalized log-root stationarity<1e-6',
            optimizer='q/W/EB in turn; shared and Pi only whole-model; epsilon full-data EB once per whole-model cycle; no mean correction')
        write(prefix/'phase.json', record)
        self.export(model, prefix/'checkpoint', summary=False)
        if strict and status != 'training_probe_stable':
            raise RuntimeError(f'Strict terminal criterion was not met in {cycles} cycles')
        return record

    def export(self, model, prefix, summary=True):
        prefix = Path(prefix); prefix.parent.mkdir(parents=True, exist_ok=True)
        with model._scope(), torch.no_grad():
            if summary:
                payload = model.posterior_summary(include_epsilon_arrays=True)
                idx = torch.arange(model.N, device=model.device)
                payload.update(structural_expected_counts=model.structural_log_expected_counts(idx).exp().cpu().numpy(),
                    same_entry_expected_counts=model._log_expected_counts(idx).exp().cpu().numpy(),
                    fresh_epsilon_expected_counts=model.prior_predictive_log_expected_counts(idx).exp().cpu().numpy())
                payload['program_uids'] = np.asarray([getattr(model, 'program_uids', {}).get(p, f'initial_{p[0]}_{p[1]}') for p in model.programs])
                np.savez_compressed(str(prefix)+'.npz', **payload)
        temporary = Path(str(prefix)+'.state.pt.tmp')
        torch.save(model, temporary)
        temporary.replace(Path(str(prefix)+'.state.pt'))
        metadata = dict(schema='MMF_eps_simulation_export_v1', variance_mode=self.args.variance_mode,
            epsilon_mean_correction=False, seed=self.seed, ranks=list(model.Kc),
            checkpoint='full pickled model including Adam and model RNG; phase-boundary continuation only; rank-search resume is not implemented',
            immutable_training_input=str(self.args.input.resolve()), input_sha256=sha(self.args.input),
            fitted_local_epsilon_uses='training Y only; no heldout counts',
            structural_expected_counts='integrate structural Gaussian VI only',
            same_entry_expected_counts='structural rate times exp(m_eps+s_eps/2)',
            fresh_epsilon_expected_counts='structural rate times exp(v_eps/2)',
            structural_program_effect='h_mean times W; epsilon is not a program')
        if hasattr(model, 'v_eps'):
            state = epsilon_checkpoint_fields(model)
            metadata.update(epsilon_prior_variance=model.v_eps.detach().cpu().numpy().tolist(),
                epsilon_state_version=state['epsilon_state_version'], epsilon_EB_updates=len(model.epsilon_updates),
                epsilon_variance_floor=model.epsilon_variance_floor)
        write(str(prefix)+'.metadata.json', metadata)
        return str(prefix)

    def restore(self, path):
        # Same-device candidate restore: preserve CPU RNG ByteTensor placement.
        model = torch.load(path, weights_only=False)
        model._rng_state = model._rng_state.cpu()
        if model._cuda_rng_state is not None:
            model._cuda_rng_state = model._cuda_rng_state.cpu()
        if str(model.device) != str(torch.device(self.args.device)):
            raise ValueError('Candidate restore must use the same device')
        return model


def make_model(args, data, cfg, ranks=0):
    return PyroBackfitMMF(**data, programs_per_type=ranks, config=cfg,
        epsilon_variance_mode='shared', epsilon_variance=args.epsilon_initial_variance,
        epsilon_variance_floor=args.epsilon_variance_floor)


def interface_check(args, data, cfg, callback):
    """Exercise the production callback on tiny data without claiming convergence."""
    model = make_model(args, data, cfg)
    for _ in range(2):
        with model._scope(), model._rng_scope():
            callback.cycle(model)
    child = add_program(model, 0, torch.arange(model.N, device=model.device), args.seed+1)
    before = {k: v.detach().clone() for k, v in child._param_state['params'].items()
              if k not in ['mu_0_0', 'dvar_0_0', 'w_0_0']}
    before_eps = None if not hasattr(child, 'v_eps') else child.v_eps.clone()
    with child._scope(), child._rng_scope():
        callback.cycle(child, (0, 0))
    assert all(torch.equal(v, child._param_state['params'][k]) for k, v in before.items())
    assert before_eps is None or torch.equal(before_eps, child.v_eps)
    callback.export(child, args.output/'interface')
    restored = callback.restore(args.output/'interface.state.pt')
    a, b = callback.score(child), callback.score(restored)
    assert a == b, (a, b)
    probe = callback.probe(restored)
    assert probe['epsilon_stationarity_relative'] < 1e-6
    write(args.output/'interface_check.json', dict(passed=True, formal_study_fit=False,
        variance_mode=args.variance_mode, ranks=list(restored.Kc), score=a,
        epsilon_stationarity_relative=probe['epsilon_stationarity_relative'],
        conditional_existing_parameters_frozen=True, conditional_epsilon_variance_frozen=True,
        checkpoint_fixed_MC_score_identical=True))


def main():
    args = parser().parse_args()
    if args.guide_rank != 0:
        raise ValueError('q(z) is diagonal; guide-rank must be 0')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    began = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'experiment_status.json').exists():
        raise FileExistsError('Refusing to overwrite existing experiment_status.json; use a new output')
    with np.load(args.input, allow_pickle=False) as z:
        data = {k: z[k] for k in ['Y', 'eta0', 'pi', 'coords', 'library_size']}
    if not np.array_equal(data['library_size'],data['Y'].sum(1)) or np.any(data['library_size']<=0):
        raise ValueError('library_size must equal the positive observed row sum of the fitted gene panel')
    n = data['Y'].shape[0]
    cfg = FitConfig(kernel='nystrom', lengthscale=.2, n_landmarks=min(48,n),
        dtype='float32', device=args.device, batch_size=n, chunk_size=n,
        num_particles=args.particles, evaluation_particles=64, q_steps=1, w_steps=1,
        nuisance_steps=1, pi_steps=1, warm_start_sweeps=0, lr=.02, seed=args.seed)
    cb = SimulationCallbacks(args)
    if args.interface_check:
        if n > 32 or data['Y'].shape[1] > 40:
            raise ValueError('Interface check requires a deliberately tiny input (N<=32,G<=40)')
        interface_check(args, data, cfg, cb)
        return
    source_hashes = sources()
    fixed = None if args.fixed_ranks is None else list(map(int,args.fixed_ranks.split(',')))
    if fixed is not None and (len(fixed) != data['eta0'].shape[0] or min(fixed) < 0):
        raise ValueError('Fixed ranks must contain one nonnegative count per type')
    status = dict(phase='running', backend=args.backend, guide_rank=args.guide_rank, started_unix=began, budget_seconds=args.budget,
        seed=args.seed, pid=os.getpid(), host=socket.gethostname(), device=args.device, input_sha256=sha(args.input),
        source_sha256=source_hashes, config=asdict(cfg), truth_or_heldout_read=False,
        automatic_rank=fixed is None, fixed_ranks=fixed, K_max=3, threshold=.065,
        fresh_K0=True, variance_mode=args.variance_mode, epsilon_mean_correction=False,
        epsilon_initial_variance=args.epsilon_initial_variance,
        epsilon_EB_schedule='full-data at each whole-model cycle; held fixed in conditional program proposals',
        likelihood_estimator='Pyro coordinate backfitting; diagonal q(z); profiled shared epsilon; marginal Monte Carlo score')
    write(args.output/'experiment_status.json', status)
    try:
        model = make_model(args, data, cfg)
        status['k0'] = cb.fit(model, 'k0', args.output/'k0', 6000)
        cb.export(model, args.output/'k0/final')
        if args.stop_after_k0:
            status.update(phase='k0_finished', selection_complete=False, selected_ranks=list(model.Kc),
                          stop_reason='explicit --stop-after-k0; full requested study fit is not complete')
            return
        if fixed is None:
            model, selected = select_programs(model, seed_idx=torch.arange(model.N, device=model.device),
                seed=args.seed, output=args.output, fit=cb.fit, score=cb.score,
                export=cb.export, write=write, tau=.065, k_max=3, restore=cb.restore)
        else:
            for c, count in enumerate(fixed):
                for k in range(count):
                    model = add_program(model, c, torch.arange(model.N, device=model.device),
                                        args.seed+100003*(sum(model.Kc)+1)+1009*c)
                    cb.fit(model, 'fixed_k_forward', args.output/f'fixed_initialization/T{c}P{k}', 150,
                           conditional_program=(c,k))
            status['fixed_k_terminal'] = cb.fit(model, 'fixed_k_terminal', args.output/'fixed_terminal', 6000)
            cb.export(model, args.output/'final')
            selected = dict(selection_complete=True, stop_reason='prespecified fixed-K diagnostic')
        status.update(phase='finished' if selected['selection_complete'] else 'incomplete',
            selection_complete=selected['selection_complete'], selected_ranks=list(model.Kc),
            stop_reason=selected['stop_reason'], final_program_norms=raw_program_norms(model))
    except Exception as error:
        status.update(phase='incomplete', error=repr(error), selection_complete=False)
        raise
    finally:
        status.update(elapsed_seconds=time.time()-began,
            source_unchanged=all(sha(path) == value for path, value in source_hashes.items()))
        write(args.output/'experiment_status.json', status)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
