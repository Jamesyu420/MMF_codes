"""HD MMF_fixfix programs with RCTD proportions, spot scales and platform effects fixed."""
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
    p.add_argument('--effects',type=Path,required=True)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--chunk-size', type=int, default=512)
    p.add_argument('--k0-elbo-rtol', type=float, default=1e-2)
    args = p.parse_args()
    if not 0 < args.k0_elbo_rtol < 1:
        p.error('--k0-elbo-rtol must be between zero and one')
    sys.path.insert(0, str(args.code.resolve()))
    from Colorectal_cancer.fixed_input_adapter import make_fixed_model,assert_fixed
    from MMF.mmf import FitConfig
    from MMF.MMF_fixfix import select_programs_ct
    from MMF.backfit import gaussian_epsilon_diagnostics
    from Colorectal_cancer.stability import probe, changes, window_change, stable, update_epsilon
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = dict(status='loading', start_utc=datetime.now(timezone.utc).isoformat(),
                   host=socket.gethostname(), pid=os.getpid(), visible_gpu=os.getenv('CUDA_VISIBLE_DEVICES'),
                   input_path=str(args.input), input_sha256=sha(args.input), target_type=args.target_type)
    receipt['source_sha256'] = {str(x.relative_to(args.code)):sha(x) for x in sorted((args.code/'MMF').glob('*.py'))}
    receipt['runner_sha256'] = sha(__file__)
    receipt['execution_source_sha256']={str(x.relative_to(args.code)):sha(x) for x in Path(__file__).resolve().parent.glob('*.py') if not x.name.startswith('._')}
    write(args.output/'run_manifest.json', receipt)
    with np.load(args.input, allow_pickle=False) as z:
        arrays = {k:z[k] for k in ['Y','eta0','pi','coords','library_size']}
        meta = {k:z[k] for k in ['genes','cell_types','coords','spot_names','original_indices']}
    with np.load(args.effects,allow_pickle=False) as z:
        effects={k:z[k] for k in ['alpha','gamma','pi']}
        for k in ['genes','cell_types','spot_names']:np.testing.assert_array_equal(z[k],meta[k])
    np.testing.assert_array_equal(effects['pi'],arrays['pi'])
    receipt.update(effects_path=str(args.effects),effects_sha256=sha(args.effects),model_entry='MMF.MMF_fixfix',
                   shared_effect_policy='External RCTD pi/alpha/gamma; deterministic plug-in, no shared optimizers or variational distributions',
                   epsilon_policy='Original profiled epsilon and EB policy retained; K0 calibrates epsilon only')
    n,g = arrays['Y'].shape
    caps = [3 if args.target_type is None or t == args.target_type else 0 for t in meta['cell_types']]
    if not any(caps): raise ValueError('Target type not found')
    config = FitConfig(kernel='nystrom',lengthscale=.2,n_landmarks=min(128,n),
                       dtype='float32',device='cuda:0',batch_size=n,chunk_size=args.chunk_size,
                       num_particles=8,seed=23,lr=.02,fit_pi=False)
    receipt.update(status='running', N=n,G=g,cell_types=meta['cell_types'].tolist(),
                   configuration=asdict(config), K_max=caps, particle_execution='K0 vectorized; program fits serial',
                   pi_policy='Exact normalized RCTD full-mode proportions held fixed; no pi optimizer, no floor added',
                   fitted_quantities='Program Gaussian guides and W; program/epsilon EB variances and profiled epsilon only',
                   training='Full-data SVI, eight particles. K0 vectorized; program fits serial to bound memory.',
                   scoring='All observations, weighted chunks, exactly one global KL.',
                   thresholds=dict(tau=.065,k0_min_steps=500,k0_max_steps=6000,k0_elbo_relative_tolerance=args.k0_elbo_rtol,k0_exit_ELBO_relative_tolerance=args.k0_elbo_rtol,forward_steps=100,backfit_steps=500,polish_steps=1000),
                   k0_stopping=f'ELBO window and exit-check relative tolerance {args.k0_elbo_rtol:g}; other numerical stability criteria unchanged; fixed-pi change exactly zero')
    write(args.output/'run_manifest.json', receipt)
    fit_started = time.monotonic()

    class Callbacks:
        score_calls = 0
        k0_status = None

        def check_fixed(self,model):
            assert_fixed(model)
            assert not model.config.fit_pi
            assert 'pi' not in model.group_names and 'pi_logits' not in model._param_state['params']
            assert torch.equal(model.pi,fixed_pi_reference), 'Fixed RCTD proportions changed'
            for name in ['alpha','gamma']:
                assert torch.equal(model._shared(name)[0],fixed_shared_reference[name]), name+' changed'

        def fit_k0(self,model,prefix,cycles):
            began=time.monotonic();history=[];previous=probe(model,gaussian_epsilon_diagnostics)
            idx=torch.arange(model.N,device=model.device);status='max_steps_not_stable'
            for start_step in range(0,cycles,50):
                with model._scope(),model._rng_scope():
                    for first in range(start_step+1,min(start_step+50,cycles)+1,10):
                        lr=.02 if first<=150 else .005 if first<=1000 else .001 if first<=3000 else .0002
                        model.set_optimization(lr=lr,num_particles=8)
                        model.backfit_cycle([idx]*min(10,cycles-first+1));update_epsilon(model)
                step=min(start_step+50,cycles);self.check_fixed(model)
                current=probe(model,gaussian_epsilon_diagnostics)
                row=dict(step=step,elbo=current['elbo'],elbo_per_entry=current['elbo_per_entry'],
                         epsilon_variance=float(current['epsilon'][0]),stationarity=current['stationarity'],
                         epsilon_last3=current['epsilon_last3'],pi_max_abs_drift=0.,**changes(current,previous))
                history.append(row);row['ELBO_window_relative_change']=window_change(history)
                if stable(history,step,elbo_rtol=args.k0_elbo_rtol):
                    update_epsilon(model);verified=probe(model,gaussian_epsilon_diagnostics)
                    validation=changes(verified,current)
                    delta=abs(verified['elbo_per_entry']-current['elbo_per_entry'])/max(1.,abs(current['elbo_per_entry']))
                    accepted=bool(max(validation.values())<.01 and delta<args.k0_elbo_rtol and verified['epsilon_last3']<.01 and verified['stationarity']<1e-6)
                    row['exit_validation']=dict(passed=accepted,ELBO_relative_change=delta,**validation)
                    row.update(elbo=verified['elbo'],elbo_per_entry=verified['elbo_per_entry'],**changes(verified,previous))
                    current=verified
                    if accepted:status='stable'
                previous=current;row['seconds']=time.monotonic()-began
                record=dict(phase='k0',status=status if status=='stable' or step==cycles else 'running',
                            steps=step,ranks=list(model.Kc),seconds=row['seconds'],history=history,pi_fixed=True)
                write(Path(prefix)/'phase.json',record);print(json.dumps(dict(phase='k0',**row)),flush=True)
                if status=='stable':break
            self.k0_status=status
            self.export(model,Path(prefix)/'final')
            return record

        def fit(self, model, phase, prefix, cycles, conditional_program=None):
            began = time.monotonic(); history=[]; idx=torch.arange(model.N,device=model.device)
            model._training_elbo.vectorize_particles = not bool(model.programs)
            self.check_fixed(model)
            if phase=='k0':return self.fit_k0(model,prefix,cycles)
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
        model=make_fixed_model(arrays,config,effects['alpha'],effects['gamma'],programs_per_type=ranks)
        fixed_shared_reference={name:model._shared(name)[0].detach().clone() for name in ['alpha','gamma']}
        fixed_pi_reference=model.pi.detach().clone()
        np.testing.assert_allclose(fixed_pi_reference.cpu().numpy(),arrays['pi'],rtol=2e-7,atol=2e-8)
        receipt['pi_normalization_max_abs_difference']=float(np.max(np.abs(fixed_pi_reference.cpu().numpy()-arrays['pi'])))
        assert (fixed_pi_reference>0).all(), 'Audit exact zero proportions before fitting'
        model._training_elbo.vectorize_particles=False
        del arrays
        cb=Callbacks()
        if args.preflight:
            # Largest allowed rank; only program gradients are allowed.
            idx=torch.arange(n,device=model.device)
            with model._scope(),model._rng_scope():
                if model.programs: model._step_group((*model.programs[-1],'qw'),idx)
            cb.check_fixed(model)
            from MMF.backfit import copy_subset,add_program
            cloned,_=copy_subset(model)
            cb.check_fixed(cloned)
            del cloned
            reduced,_=copy_subset(model,[])
            cb.check_fixed(reduced)
            added=add_program(reduced,0,torch.arange(min(128,n),device=model.device),23)
            cb.check_fixed(added)
            import pyro
            with added._scope(),added._rng_scope():
                trace=pyro.poutine.trace(added._guide).get_trace(torch.arange(min(8,n),device=model.device))
            assert not any(name in trace.nodes for name in ['alpha','gamma'])
            del added,reduced
            cb.score(model,batches=1)
            receipt.update(status='preflight_passed',tested_ranks=ranks,no_pi_optimizer=True,pi_max_abs_drift=0.,shared_effects_max_abs_drift=0.,no_shared_optimizers=True)
        else:
            cb.fit(model,'k0',args.output/'k0',6000)
            model,selection=select_programs_ct(model,seed_idx=torch.arange(n,device=model.device),seed=23,
                output=args.output,fit=cb.fit,score=cb.score,export=cb.export,write=write,restore=None,
                tau=.065,k_max=caps,forward_cycles=100,backfit_cycles=500,polish_cycles=1000)
            receipt.update(status='completed',selection_complete=selection['selection_complete'],
                           final_ranks=selection['final_ranks'],final_npz_sha256=sha(args.output/'final.npz'),
                           convergence_verified=False,k0_status=cb.k0_status,pi_max_abs_drift=0.,no_pi_optimizer=True,shared_effects_max_abs_drift=0.,no_shared_optimizers=True)
    except Exception as error:
        receipt.update(status='failed',error=repr(error));raise
    finally:
        receipt.update(fit_seconds=time.monotonic()-fit_started,total_seconds=time.monotonic()-started,
                       finish_utc=datetime.now(timezone.utc).isoformat(),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                       source_unchanged=all(sha(args.code/k)==v for k,v in {**receipt['source_sha256'],**receipt['execution_source_sha256']}.items()))
        write(args.output/'run_manifest.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__': main()
