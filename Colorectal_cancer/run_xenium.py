"""Single-type epithelial MMF, exact pi=1, lengthscale0.06; alpha/gamma still fitted."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--code', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--target-type')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--chunk-size', type=int, default=512)
    args = p.parse_args()
    sys.path.insert(0, str(args.code.resolve()))
    from MMF.backfit import PyroBackfitMMF
    from MMF.mmf import FitConfig
    from MMF.MMF_ctback import select_programs_ct
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = dict(status='loading', start_utc=datetime.now(timezone.utc).isoformat(),
                   host=socket.gethostname(), pid=os.getpid(), visible_gpu=os.getenv('CUDA_VISIBLE_DEVICES'),
                   input_path=str(args.input), input_sha256=sha(args.input), target_type=args.target_type)
    receipt['source_sha256'] = {str(x.relative_to(args.code)):sha(x) for x in sorted((args.code/'MMF').glob('*.py'))}
    receipt['runner_sha256'] = sha(__file__)
    write(args.output/'run_manifest.json', receipt)
    with np.load(args.input, allow_pickle=False) as z:
        arrays = {k:z[k] for k in ['Y','eta0','pi','coords','library_size']}
        meta = {k:z[k] for k in ['genes','cell_types','coords','spot_names','original_indices']}
    n,g = arrays['Y'].shape
    assert meta['cell_types'].tolist() == ['Epithelial'] and arrays['eta0'].shape == (1,g)
    np.testing.assert_array_equal(arrays['pi'], np.ones((n,1)))
    assert np.all(arrays['eta0'] > 0)
    caps = [3 if args.target_type is None or t == args.target_type else 0 for t in meta['cell_types']]
    if not any(caps): raise ValueError('Target type not found')
    config = FitConfig(kernel='nystrom',lengthscale=.06,n_landmarks=min(128,n),
                       dtype='float32',device='cuda:0',batch_size=n,chunk_size=args.chunk_size,
                       num_particles=8,seed=23,lr=.02,fit_pi=False)
    receipt.update(status='running', N=n,G=g,cell_types=meta['cell_types'].tolist(),
                   configuration=asdict(config), C=1, K_max=caps, particle_execution='K0 vectorized; program fits serial',
                   pi_policy='Exact epithelial pi=1, no pi optimizer; alpha/gamma still estimated',
                   training='Full-data SVI, eight particles. K0 vectorized; program fits serial to bound memory.',
                   scoring='All observations, weighted chunks, exactly one global KL.',
                   thresholds=dict(tau=.065,k0_steps=6000,forward_steps=100,backfit_steps=500,polish_steps=1000))
    write(args.output/'run_manifest.json', receipt)
    fit_started = time.monotonic()

    class Callbacks:
        score_calls = 0
        def check_fixed(self, model):
            assert model.C == 1 and not model.config.fit_pi
            assert 'pi' not in model.group_names and 'pi_logits' not in model._param_state['params']
            assert torch.equal(model.pi, torch.ones((n,1),device=model.device,dtype=model.dtype))

        def fit(self, model, phase, prefix, cycles, conditional_program=None):
            self.check_fixed(model)
            began = time.monotonic(); history=[]; idx=torch.arange(model.N,device=model.device)
            model._training_elbo.vectorize_particles = not bool(model.programs)
            with model._scope(),model._rng_scope():
                for first in range(1,cycles+1,model.config.program_steps):
                    step=min(first+model.config.program_steps-1,cycles)
                    lr=.02 if first<=150 else .005 if first<=1000 else .001 if first<=3000 else .0002
                    model.set_optimization(lr=lr,num_particles=8)
                    model.backfit_cycle([idx]*(step-first+1),conditional_program)
                    if conditional_program is None: model.update_epsilon_variance()
                    if step%50==0 or step==cycles:
                        self.check_fixed(model)
                        row=dict(step=step,seconds=time.monotonic()-began,epsilon_variance=float(model.v_eps),pi_max_abs_drift=0.)
                        history.append(row)
                        write(Path(prefix)/'phase.json',dict(phase=phase,status='running',history=history,ranks=list(model.Kc)))
                        print(json.dumps(dict(phase=phase,ranks=list(model.Kc),**row)),flush=True)
            record=dict(phase=phase,status='fixed_step_budget_completed',steps=cycles,seconds=time.monotonic()-began,
                        ranks=list(model.Kc),history=history)
            write(Path(prefix)/'phase.json',record)
            # Completed budgets do not certify numerical convergence.
            return record

        def score(self, model, batches=4):
            self.check_fixed(model)
            values=[]; began=time.monotonic()
            with model._scope(),torch.no_grad():
                for repeat in range(batches):
                    gens=[torch.Generator(device=model.device).manual_seed(91025+repeat*1000003+c*15485863) for c in range(model.C)]
                    total=0.
                    for idx in model._chunks():
                        eps=None
                        if model.programs:
                            noises=[]
                            for gen in gens:
                                half=torch.randn((4,len(idx),model.G),generator=gen,device=model.device,dtype=model.dtype)
                                noises.append(torch.cat([half,-half],0))
                            eps=torch.stack(noises,1); del noises,half
                        total-=float(model.direct_loss(idx,particles=8,eps=eps))*len(idx)/model.N
                        del eps
                    values.append(total)
            if not np.isfinite(values).all(): raise FloatingPointError('Nonfinite complete-data score')
            write(args.output/'scores'/f'{self.score_calls:05d}.json',dict(mean=float(np.mean(values)),
                  elbo_batches=values,rows=model.N,ranks=list(model.Kc),seconds=time.monotonic()-began))
            self.score_calls+=1
            return float(np.mean(values))

        def export(self, model, prefix):
            self.check_fixed(model)
            prefix=Path(prefix); prefix.parent.mkdir(parents=True,exist_ok=True)
            payload=model.posterior_summary(); payload.update(meta)
            for key,value in payload.items():
                if np.issubdtype(np.asarray(value).dtype,np.number) and not np.isfinite(value).all():
                    raise FloatingPointError('Nonfinite export: '+key)
            np.savez_compressed(str(prefix)+'.npz',**payload)
            torch.save(model,str(prefix)+'.state.pt')
            return str(prefix)

    try:
        ranks=[3 if c else 0 for c in caps] if args.preflight else [0]*len(caps)
        model=PyroBackfitMMF(**arrays,programs_per_type=ranks,config=config,
                            epsilon_variance=.1,epsilon_variance_floor=1e-8)
        model._training_elbo.vectorize_particles=False
        del arrays
        cb=Callbacks()
        cb.check_fixed(model)
        if args.preflight:
            # Largest allowed rank; exercise both nuisance and program gradient allocations.
            idx=torch.arange(n,device=model.device)
            with model._scope(),model._rng_scope():
                if model.programs: model._step_group((*model.programs[-1],'qw'),idx)
            from MMF.backfit import copy_subset, add_program
            clone,_=copy_subset(model); cb.check_fixed(clone)
            reduced,_=copy_subset(model,[]); cb.check_fixed(reduced)
            added=add_program(reduced,0,torch.arange(min(128,n),device=model.device),23); cb.check_fixed(added)
            del clone, reduced, added
            cb.score(model,batches=1)
            receipt.update(status='preflight_passed',tested_ranks=ranks,no_pi_optimizer=True,pi_max_abs_drift=0.)
        else:
            cb.fit(model,'k0',args.output/'k0',6000)
            model,selection=select_programs_ct(model,seed_idx=torch.arange(n,device=model.device),seed=23,
                output=args.output,fit=cb.fit,score=cb.score,export=cb.export,write=write,restore=None,
                tau=.065,k_max=caps,forward_cycles=100,backfit_cycles=500,polish_cycles=1000)
            receipt.update(status='completed',selection_complete=selection['selection_complete'],
                           final_ranks=selection['final_ranks'],final_npz_sha256=sha(args.output/'final.npz'),
                           convergence_verified=False,no_pi_optimizer=True,pi_max_abs_drift=0.)
    except Exception as error:
        receipt.update(status='failed',error=repr(error));raise
    finally:
        receipt.update(fit_seconds=time.monotonic()-fit_started,total_seconds=time.monotonic()-started,
                       finish_utc=datetime.now(timezone.utc).isoformat(),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                       source_unchanged=all(sha(args.code/k)==v for k,v in receipt['source_sha256'].items()))
        write(args.output/'run_manifest.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__': main()
