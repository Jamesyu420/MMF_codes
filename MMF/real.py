"""Real-data Pyro backfitting for shared-epsilon MMF.

Immutable counts and the spatial basis may be shared by sequential candidate
restores. Every mutable Gaussian, point, EB, Adam and RNG state is private.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import torch
from torch.distributions import constraints
from MMF.mmf import FitConfig, fit_config_from_mapping
from MMF.backfit import (PyroBackfitMMF, fixed_training_score, raw_program_norms,
    epsilon_checkpoint_fields, restore_epsilon_checkpoint_fields, gaussian_epsilon_diagnostics,
    add_program, select_programs)
from MMF.fit_mmf import _restore_adam_step_device


def relative_change(new, old):
    return float(np.linalg.norm(new - old) / max(1., np.linalg.norm(old)))


def stable_probe(history):
    """Relative stability of the training probe, not a proof of convergence."""
    if len(history) < 10:
        return False
    recent = history[-10:]
    first = np.mean([row['elbo_per_entry'] for row in recent[:5]])
    second = np.mean([row['elbo_per_entry'] for row in recent[5:]])
    elbo_stable = abs(second - first) / max(1., abs(first)) < .005
    state_stable = all(max(row[key] for key in [
        'rate_change', 'pi_change', 'variance_change', 'effect_change']) < .01
        for row in recent[-3:])
    return bool(elbo_stable and state_stable)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def epsilon_relative_change(new,old,floor=1e-8):
    return float(np.linalg.norm(np.asarray(new)-np.asarray(old))/max(floor,float(np.linalg.norm(old))))


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def validate_output(path, execution_root):
    path=Path(path).resolve();execution_root=Path(execution_root).resolve()
    if path==execution_root or not path.is_relative_to(execution_root):
        raise ValueError('Output must be a new directory inside the execution root')
    return path


def restore_checkpoint(path, device, template=None):
    """Restore a trusted phase checkpoint, optionally sharing immutable tensors."""
    state=torch.load(path,map_location='cpu',weights_only=False)
    requested=torch.device(device)
    if requested.type=='cuda' and requested.index is None:requested=torch.device('cuda',torch.cuda.current_device())
    device=str(requested)
    if any(str(name).startswith('F_') for name in state['parameters']):
        raise ValueError('Checkpoint contains a low-rank coefficient factor')
    if state.get('epsilon_variance_mode','shared')!='shared' or state.get('comparison_backend','pyro')!='pyro':
        raise ValueError('Only the shared-epsilon Pyro backfit can be restored')
    config=fit_config_from_mapping(state['config'], device=device)
    cls=PyroBackfitMMF
    if template is None:
        if sha(state['immutable_input'])!=state['input_sha256']:
            raise ValueError('Immutable checkpoint input hash changed')
        with np.load(state['immutable_input'],allow_pickle=False) as z:
            data={k:z[k] for k in ['Y','eta0','pi','coords','library_size']}
        kw=dict(epsilon_variance_mode='shared', epsilon_variance=state['epsilon_prior_variance'],
            epsilon_variance_floor=state['epsilon_variance_floor'])
        model=cls(**data,programs_per_type=state['ranks'],config=config,**kw)
        if model.basis is not None:
            if state.get('basis_sha256') and sha(state['basis_path'])!=state['basis_sha256']:
                raise ValueError('Checkpoint spatial basis hash changed')
            model.basis.copy_(model._tensor(np.load(state['basis_path'],allow_pickle=False)))
    else:
        if type(template) is not cls or str(template.device)!=str(torch.device(device)):
            raise ValueError('Template class/device differs from checkpoint')
        if getattr(torch,config.dtype)!=template.dtype:
            raise ValueError('Template dtype differs from checkpoint')
        if state['input_sha256']!=getattr(template,'_immutable_input_sha256',None):
            raise ValueError('Template input hash differs from checkpoint')
        if state.get('basis_sha256')!=getattr(template,'_immutable_basis_sha256',None):
            raise ValueError('Template basis hash differs from checkpoint')
        model=object.__new__(cls);model.__dict__=template.__dict__.copy()
        model.config=config
        model._training_elbo=copy.copy(template._training_elbo)
        model._evaluation_elbo=copy.copy(template._evaluation_elbo)
    model.Kc=list(state['ranks'])
    model.programs=[(c,k) for c,n in enumerate(model.Kc) for k in range(n)]
    model.basis_metadata=copy.deepcopy(state['basis_metadata'])
    model._immutable_input_sha256=state['input_sha256']
    model._immutable_basis_sha256=state.get('basis_sha256')
    model._param_state=dict(params={n:v.to(model.device).detach().clone().requires_grad_(True)
        for n,v in state['parameters'].items()},constraints=state['constraints'])
    model._initial_pi=state['initial_pi'].to(model.device).clone()
    model.v_alpha=state['v_alpha'].to(model.device).clone()
    model.v_gamma=state['v_gamma'].to(model.device).clone()
    model.v={p:v.to(model.device).clone() for p,v in state['program_variances'].items()}
    model.group_names={};model.optimizers={};model.svis={};model.block_optimizers={};model.optimizer_calls={}
    for key,names in state['group_names'].items(): model._add_optimizer(key,names)
    model._rng_state=state['cpu_rng'].clone()
    model._cuda_rng_state=None if state['cuda_rng'] is None else state['cuda_rng'].clone()
    model.program_uids=copy.deepcopy(state['program_uids'])
    model.history=[];model.frozen_programs=();model._warm_started=True
    restore_epsilon_checkpoint_fields(model,state)
    for key,value in state['block_optimizer_states'].items():
        opt=torch.optim.Adam([model._param_state['params'][n] for n in model.group_names[key]],lr=config.lr)
        opt.load_state_dict(value);_restore_adam_step_device(opt);model.block_optimizers[key]=opt
    for key,value in state.get('pyro_optimizer_states',{}).items():
        model.optimizers[key].set_state(value)
    model.phase_boundary_record=state.get('phase')
    model.phase_boundary_state=state.get('batch_and_monitor_state')
    model.expansion=state.get('expansion',{})
    return model,state


class RealCallbacks:
    def __init__(self, output, args, genes, types, manifest):
        self.output, self.args = Path(output), args
        self.genes, self.types, self.manifest = genes, types, manifest
        self.score_calls = 0
        self.template = None

    def positions(self, model, count, seed):
        return torch.as_tensor(np.random.default_rng(seed).choice(
            model.N, min(count, model.N), replace=False), device=model.device)

    def score(self,model):
        idx=self.positions(model,min(self.args.score_positions,model.N),92025)
        row=fixed_training_score(model,idx,seed=91025,particles=self.args.score_particles,batches=self.args.score_batches)
        row.update(call=self.score_calls,ranks=list(model.Kc))
        write(self.output/'scores'/f'{self.score_calls:05d}.json',row)
        self.score_calls+=1
        return row['mean']

    def probe(self,model):
        idx=self.positions(model,256,41046)
        with model._scope(),torch.no_grad():
            elbo=fixed_training_score(model,idx,seed=51046,particles=self.args.score_particles,batches=1)['mean']
            rate=model._log_expected_counts(idx).exp().cpu().numpy()
            pi=model._indexed_log_pi(idx).exp().cpu().numpy()
            variance=torch.stack([model.v_alpha,model.v_gamma]+[model.v[p] for p in model.programs]).cpu().numpy()
            effects=torch.zeros((model.C,len(idx),model.G),device=model.device,dtype=model.dtype)
            for c,k in model.programs:
                m,_=model._moments(c,k,idx);effects[c]+=m[:,None]*model._w(c,k)
            result=dict(elbo_per_entry=elbo/(model.N*model.G),rate=rate,pi=pi,variance=variance,effect=effects.cpu().numpy())
            if hasattr(model,'v_eps'):
                la=model.structural_log_expected_counts(idx)
                diagnostics=gaussian_epsilon_diagnostics(la,model.Y[idx],model.v_eps)
                result['epsilon_stationarity']=diagnostics['maximum_log_rate_newton_correction']
                result['epsilon_log_rate_reconstruction_error']=diagnostics['maximum_log_rate_reconstruction_error']
                result['epsilon_variance']=model.v_eps.detach().cpu().numpy().copy()
        if not np.isfinite(result['elbo_per_entry']) or not all(np.isfinite(result[k]).all() for k in ['rate','pi','variance','effect']):
            raise FloatingPointError('Nonfinite full-model training diagnostic')
        return result

    def _run_segment(self,model,phase,prefix,cycles,conditional_program=None,*,resume=False):
        prefix=Path(prefix);prefix.mkdir(parents=True,exist_ok=True)
        if cycles<1:raise ValueError('Positive continuation segment length required')
        prior=model.phase_boundary_state if resume else None
        prior_record=model.phase_boundary_record if resume else None
        if resume:
            if phase not in ['k0','terminal'] or conditional_program is not None:
                raise ValueError('External/internal same-phase continuation supports strict K0/terminal only')
            if not isinstance(prior,dict) or prior.get('phase')!=phase or prior.get('conditional_program') is not None:
                raise ValueError('Checkpoint does not contain this exact strict-phase cursor')
            if prior_record.get('batch')!=self.args.batch or prior_record.get('check_interval')!=self.args.check_interval:
                raise ValueError('Batch/check interval differs from the phase checkpoint')
            history=copy.deepcopy(prior['history']);seen=int(prior['training_rows_seen']);cursor=int(prior['cursor'])
            permutation=prior['permutation'].to(model.device).clone()
            if permutation.shape!=(model.N,) or cursor<0 or cursor>model.N or torch.unique(permutation).numel()!=model.N:
                raise ValueError('Invalid saved batch permutation/cursor')
            if int(permutation.min())!=0 or int(permutation.max())!=model.N-1:raise ValueError('Invalid saved permutation support')
            previous=copy.deepcopy(prior['previous_probe']);start_step=int(prior['completed_steps'])
            epsilon_eb_events=copy.deepcopy(prior.get('epsilon_eb_events',prior_record.get('epsilon_EB_events',[])))
            prior_seconds=float(prior_record['seconds'])
        else:
            history=[];epsilon_eb_events=[];seen=0;cursor=model.N;permutation=None
            previous=None if conditional_program is not None else self.probe(model)
            start_step=0;prior_seconds=0.
        stop_step=start_step+cycles;started=time.time()-prior_seconds
        strict=phase in ['k0','terminal'];status='phase_budget_completed'
        free=[] if conditional_program is None else [f'{n}_{conditional_program[0]}_{conditional_program[1]}' for n in ['mu','dvar','w']]
        immutable={n:v.detach().clone() for n,v in model._param_state['params'].items() if conditional_program is not None and n not in free}
        eps_before=None if not hasattr(model,'v_eps') else model.v_eps.detach().clone()
        with model._scope(),model._rng_scope():
            for step in range(start_step+1,stop_step+1):
                if time.time()-self.manifest['started_unix'] > self.args.budget_seconds:
                    self.checkpoint(model,prefix/'budget_checkpoint')
                    raise TimeoutError('Wall-time limit; numerical convergence not established')
                lr=(.02 if step<=150 else .005) if conditional_program is not None else (.001 if step<=500 else .0002)
                model.set_optimization(lr=lr,num_particles=self.args.particles)
                if cursor>=model.N:
                    permutation=torch.randperm(model.N,device=model.device);cursor=0
                idx=permutation[cursor:cursor+self.args.batch];cursor+=len(idx);seen+=len(idx)
                for p in model.programs if conditional_program is None else [conditional_program]:
                    model._step_group((*p,'q'),idx);model._step_group((*p,'w'),idx);model._update_variance(*p)
                if conditional_program is None:
                    for n in ['alpha','gamma']:model._step_group(n,idx)
                    model._update_shared_variances();model._step_group('pi',idx)
                    if hasattr(model,'v_eps') and cursor>=model.N:
                        old_variance=model.v_eps.detach().cpu().numpy().copy()
                        model.update_epsilon_variance()
                        epsilon_eb_events.append(dict(step=step,effective_epochs=seen/model.N,
                            relative_change=epsilon_relative_change(model.v_eps.detach().cpu().numpy(),old_variance,model.epsilon_variance_floor),
                            kind='completed_shuffled_epoch'))
                if step%self.args.check_interval and step!=stop_step:continue
                row=dict(step=step,seconds=time.time()-started,effective_epochs=seen/model.N)
                if conditional_program is None:
                    current=self.probe(model);row['elbo_per_entry']=current['elbo_per_entry']
                    for k in ['rate','pi','variance','effect']:row[k+'_change']=relative_change(current[k],previous[k])
                    if hasattr(model,'v_eps'):
                        row['epsilon_variance_change']=epsilon_relative_change(current['epsilon_variance'],previous['epsilon_variance'],model.epsilon_variance_floor)
                        row['epsilon_stationarity']=current['epsilon_stationarity']
                        row['epsilon_variance_min']=float(model.v_eps.min());row['epsilon_variance_max']=float(model.v_eps.max())
                        row['epsilon_EB_updates']=len(epsilon_eb_events)
                        row['last_epsilon_EB_change']=epsilon_eb_events[-1]['relative_change'] if epsilon_eb_events else None
                    previous=current
                history.append(row)
                epsilon_stable=not hasattr(model,'v_eps') or (len(epsilon_eb_events)>=3 and len(history)>=3 and all(
                    e['relative_change']<.01 for e in epsilon_eb_events[-3:]) and all(
                    h.get('epsilon_variance_change',float('inf'))<.01 for h in history[-3:]) and current['epsilon_stationarity']<1e-7)
                if conditional_program is None and seen>=3*model.N and stable_probe(history) and epsilon_stable:
                    if hasattr(model,'v_eps'):
                        before_final=previous
                        old_variance=model.v_eps.detach().cpu().numpy().copy()
                        model.update_epsilon_variance()
                        final=self.probe(model)
                        change=epsilon_relative_change(final['epsilon_variance'],old_variance,model.epsilon_variance_floor)
                        epsilon_eb_events.append(dict(step=step,effective_epochs=seen/model.N,relative_change=change,kind='strict_boundary_full_data_verification'))
                        final_changes={k:relative_change(final[k],before_final[k]) for k in ['rate','pi','variance','effect']}
                        final_elbo_change=abs(final['elbo_per_entry']-before_final['elbo_per_entry'])/max(1.,abs(before_final['elbo_per_entry']))
                        row['final_full_data_EB_change']=change
                        row['final_full_data_ELBO_change']=final_elbo_change
                        row['final_full_data_state_changes']=final_changes
                        previous=final
                        if change<.01 and max(final_changes.values())<.01 and final_elbo_change<.005 and final['epsilon_stationarity']<1e-7:
                            status='training_probe_stable'
                    else:
                        status='training_probe_stable'
                self.manifest.update(phase='fitting_'+phase,active_prefix=str(prefix),phase_step=step,
                    elapsed_seconds=time.time()-self.manifest['started_unix']+float(self.manifest.get('continuation_lineage',{}).get('prior_elapsed_seconds',0)),active_ranks=list(model.Kc))
                write(self.output/'experiment_status.json',self.manifest)
                write(prefix/'phase.json',dict(phase=phase,status=status,steps=step,history=history,ranks=list(model.Kc)))
                print(json.dumps(dict(phase=phase,prefix=str(prefix),ranks=list(model.Kc),**row)),flush=True)
                if status=='training_probe_stable':break
        for n,v in immutable.items():
            if not torch.equal(v,model._param_state['params'][n]):raise AssertionError('Conditional proposal changed frozen parameter')
        if conditional_program is not None and eps_before is not None and not torch.equal(eps_before,model.v_eps):
            raise AssertionError('Conditional proposal changed residual prior variance')
        record=dict(phase=phase,status=status,steps=step,seconds=time.time()-started,history=history,ranks=list(model.Kc),
            effective_epochs=seen/model.N,conditional_program=None if conditional_program is None else list(conditional_program),
            frozen_old_and_shared_parameters_unchanged=True if conditional_program is not None else None,
            epsilon_update_schedule='full/chunked scan after each complete shuffled epoch and final strict-boundary verification; fixed prior variance within conditional proposal',
            epsilon_EB_events=epsilon_eb_events,
            monitor='>=10 checks and >=3 epochs; legacy rate/pi/structural-variance/type-effect <.01; last3 actual epsilon EB events and last3 epsilon probe changes <.01 truly relative; final full-data EB/probe and local stationarity <1e-7',
            batch=self.args.batch,check_interval=self.args.check_interval)
        model.phase_boundary_record=record
        model.phase_boundary_state=dict(completed_steps=step,training_rows_seen=seen,permutation=permutation.cpu(),cursor=cursor,previous_probe=previous,history=history,conditional_program=conditional_program,phase=phase,epsilon_eb_events=copy.deepcopy(epsilon_eb_events),continuation_schema='real_phase_continuation_v1')
        write(prefix/'phase.json',record)
        segment=prefix/'segments'/f'{start_step+1:09d}_{step:09d}'
        record['continuation_segment']=dict(start_step=start_step+1,end_step=step,resumed=resume,prior_phase_seconds=prior_seconds,
            segment_steps=step-start_step,segment_seconds=record['seconds']-prior_seconds,
            same_learning_rate_progress=True,monitor_and_batch_cursor_preserved=True)
        model.phase_boundary_record=record
        write(segment/'phase.json',record);write(prefix/'phase.json',record)
        self.checkpoint(model,segment/'checkpoint',record)
        return record

    def fit(self,model,phase,prefix,cycles,conditional_program=None):
        strict=phase in ['k0','terminal']
        armed=getattr(model,'_real_continuation_armed_phase',None)
        resume=armed==phase
        if armed is not None and not resume:raise ValueError('Requested phase differs from armed checkpoint continuation')
        if armed is not None:delattr(model,'_real_continuation_armed_phase')
        while True:
            record=self._run_segment(model,phase,prefix,cycles,conditional_program,resume=resume)
            if not strict or record['status']=='training_probe_stable':return record
            # Segment exhaustion is a checkpoint boundary, never convergence.
            # Continue the same phase without resetting Adam, RNG, cursor,
            # history, EB events or the absolute-step LR schedule.
            resume=True
            self.manifest['continuation_segments_completed']=self.manifest.get('continuation_segments_completed',0)+1
            self.manifest['continuation_next_phase_step']=record['steps']+1
            write(self.output/'experiment_status.json',self.manifest)

    def checkpoint(self,model,prefix,phase=None):
        prefix=Path(prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
        basis_path=self.output/'spatial_basis.npy'
        if model.basis is not None and not basis_path.exists():np.save(basis_path,model.basis.cpu().numpy())
        if not hasattr(self,'_basis_sha256'):
            self._basis_sha256=sha(basis_path) if model.basis is not None else None
        model._immutable_input_sha256=self.manifest['input_sha256']
        model._immutable_basis_sha256=self._basis_sha256
        state=dict(config=asdict(model.config),ranks=list(model.Kc),
            comparison_backend='pyro',
            pyro_optimizer_states={k:v.get_state() for k,v in model.optimizers.items()},
            parameters={n:v.detach().cpu().clone() for n,v in model._param_state['params'].items()},
            constraints=model._param_state['constraints'],group_names=model.group_names,
            program_variances={p:v.detach().cpu().clone() for p,v in model.v.items()},
            v_alpha=model.v_alpha.detach().cpu().clone(),v_gamma=model.v_gamma.detach().cpu().clone(),
            initial_pi=model._initial_pi.detach().cpu().clone(),program_uids=getattr(model,'program_uids',{}),
            cpu_rng=model._rng_state.clone(),cuda_rng=None if model._cuda_rng_state is None else model._cuda_rng_state.clone(),
            block_optimizer_states={k:v.state_dict() for k,v in model.block_optimizers.items()},
            basis_path=str(basis_path),basis_metadata=model.basis_metadata,basis_sha256=self._basis_sha256,
            immutable_input=str(self.args.input),input_sha256=self.manifest['input_sha256'],
            phase=phase if phase is not None else getattr(model,'phase_boundary_record',None),
            batch_and_monitor_state=getattr(model,'phase_boundary_state',None),
            expansion=getattr(model,'expansion',{}),inference_backend=model.inference_backend,
            epsilon_variance_mode='shared',epsilon_mean_correction=False,
            continuation='Same strict phase preserves absolute-step LR, batch cursor, monitor, EB events, Adam and RNG across bounded segments; different phases deliberately start their own schedules')
        if hasattr(model,'v_eps'):state.update(epsilon_checkpoint_fields(model))
        target=Path(str(prefix)+'.state.pt');temp=Path(str(target)+'.tmp');torch.save(state,temp);temp.replace(target)
        write(Path(str(prefix)+'.state.json'),dict(state=str(target),ranks=list(model.Kc),saved_unix=time.time(),historical_snapshot_retained=True))
        write(self.output/'latest_state.json',dict(state=str(target),ranks=list(model.Kc)))
        return str(target)

    def export(self,model,prefix):
        prefix=Path(prefix);self.checkpoint(model,prefix)
        write(self.output/'retained_state.json',dict(state=str(prefix)+'.state.pt',ranks=list(model.Kc)))
        write(Path(str(prefix)+'.json'),dict(ranks=list(model.Kc),norms=raw_program_norms(model),
            program_uids=[dict(cell_type=c,program=k,uid=getattr(model,'program_uids',{}).get((c,k))) for c,k in model.programs]))
        return str(prefix)

    def restore(self,path):
        return restore_checkpoint(path,self.args.device,self.template)[0]

    def final_arrays(self,model):
        means = np.empty((len(model.programs), model.N), dtype=np.float32); sd = np.empty_like(means)
        with model._scope(), torch.no_grad():
            for idx in model._chunks():
                ids = idx.cpu().numpy()
                for j, p in enumerate(model.programs):
                    mean, variance = model._moments(*p, idx)
                    means[j, ids] = mean.cpu().numpy(); sd[j, ids] = variance.sqrt().cpu().numpy()
            arrays = dict(h_mean=means, h_sd=sd,
                loadings=np.asarray([model._w(*p).cpu().numpy() for p in model.programs], dtype=np.float32).reshape(len(model.programs), model.G),
                program_cell_type=np.asarray([p[0] for p in model.programs], dtype=int),
                program_index=np.asarray([p[1] for p in model.programs], dtype=int),
                pi=model._pi().cpu().numpy(), eta0=model.eta0.cpu().numpy(),
                library_size=model.library_size.cpu().numpy(), genes=self.genes, cell_types=self.types,
                alpha_mean=model._shared('alpha')[0].cpu().numpy(), alpha_sd=model._shared('alpha')[1].sqrt().cpu().numpy(),
                gamma_mean=model._shared('gamma')[0].cpu().numpy(), gamma_sd=model._shared('gamma')[1].sqrt().cpu().numpy(),
                alpha_prior_variance=model.v_alpha.cpu().numpy(), gamma_prior_variance=model.v_gamma.cpu().numpy(),
                prior_variance=np.asarray([float(model.v[p]) for p in model.programs]),
                epsilon_prior_variance=model.v_eps.detach().cpu().numpy(),
                epsilon_variance_mode=np.asarray('shared'), epsilon_mean_correction=np.asarray(False),
                model_version=np.asarray('MMF_eps'))
        np.savez_compressed(self.output/'final.npz', **arrays)


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--budget-seconds',type=float,default=21600)
    p.add_argument('--resume-checkpoint',type=Path);p.add_argument('--resume-phase',choices=['k0','fixed_terminal'])
    p.add_argument('--prior-status',type=Path)
    for n in ['input','output','execution-root']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--epsilon-variance',type=float,default=.09);p.add_argument('--epsilon-floor',type=float,default=1e-8)
    p.add_argument('--device',default='cuda');p.add_argument('--batch',type=int,default=1024);p.add_argument('--particles',type=int,default=4)
    p.add_argument('--check-interval',type=int,default=25);p.add_argument('--lengthscale',type=float,default=.2)
    p.add_argument('--landmarks',type=int,default=128);p.add_argument('--seed',type=int,default=23)
    p.add_argument('--threshold',type=float);p.add_argument('--dtype',choices=['float32','float64'],default='float32')
    p.add_argument('--k0-cycles',type=int,default=6000);p.add_argument('--forward-cycles',type=int,default=150);p.add_argument('--backfit-cycles',type=int,default=1000)
    p.add_argument('--score-positions',type=int,default=1024);p.add_argument('--score-particles',type=int,default=16);p.add_argument('--score-batches',type=int,default=4)
    p.add_argument('--stop-after-k0',action='store_true');p.add_argument('--fixed-ranks',type=str)
    p.add_argument('--gpu-memory-fraction',type=float,default=.75)
    return p


def run(args):
    args.input=Path(args.input).resolve();args.output=validate_output(args.output,args.execution_root)
    if bool(args.resume_checkpoint)!=bool(args.prior_status):raise ValueError('Resume checkpoint and prior-status must be supplied together')
    if args.resume_phase and not args.resume_checkpoint:raise ValueError('Resume phase requires checkpoint')
    prior_status={};prior_status_path=None
    if args.resume_checkpoint:
        args.resume_checkpoint=args.resume_checkpoint.resolve()
        prior_status_path=args.prior_status.resolve()
        prior_status=json.loads(prior_status_path.read_text())
        if not args.resume_checkpoint.is_relative_to(prior_status_path.parent):raise ValueError('Checkpoint is outside its original attempt status directory')
        if not prior_status.get('source_sha256'):raise ValueError('Original source manifest is missing')
        phase_meta=torch.load(args.resume_checkpoint,map_location='cpu',weights_only=False)
        saved_phase=phase_meta.get('phase',{}).get('phase')
        inferred='k0' if saved_phase=='k0' else 'fixed_terminal' if saved_phase=='terminal' and prior_status.get('run_kind')=='fixed_K_diagnostic' else None
        if inferred is None:raise ValueError('Rank-search/non-strict checkpoint resume is not supported')
        if args.resume_phase and args.resume_phase!=inferred:raise ValueError('Explicit resume phase differs from checkpoint')
        args.resume_phase=inferred
        if prior_status.get('active_ranks') is not None and prior_status['active_ranks']!=phase_meta['ranks']:raise ValueError('Original status/checkpoint ranks differ')
        del phase_meta
        if prior_status.get('source_unchanged') is not True:raise ValueError('Original source integrity not certified')
        for source,digest in prior_status.get('source_sha256',{}).items():
            if sha(source)!=digest:raise ValueError('Original frozen source changed: '+source)
        if prior_status.get('input_sha256')!=sha(args.input):raise ValueError('Continuation input differs from original attempt')
        if args.resume_phase=='fixed_terminal' and (not args.fixed_ranks or prior_status.get('run_kind')!='fixed_K_diagnostic'):
            raise ValueError('Fixed terminal continuation requires a fixed-K original attempt; rank-search resume is unsupported')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,torch.device(args.device))
    began=time.time();sources=sorted(Path(__file__).resolve().parent.rglob('*.py'))
    manifest=dict(backend='pyro',phase='initializing',pid=os.getpid(),started_unix=began,input=str(args.input),input_sha256=sha(args.input),
        initialization='fresh model and K0 from frozen prepared composition only; no old Gaussian state/programs reused',
        variance_mode='shared',epsilon_mean_correction=False,epsilon_initial_variance=args.epsilon_variance,
        source_sha256={str(f):sha(f) for f in sources},heldout_or_truth_read=False,selection_complete=False,stable_final=False,
        imported_modules={n:str(Path(sys.modules[n].__file__).resolve()) for n in ['MMF.mmf','MMF.fit_mmf','MMF.backfit']},
        run_kind='K0_diagnostic' if args.stop_after_k0 else 'fixed_K_diagnostic' if args.fixed_ranks else 'automatic_K_primary')
    if args.resume_checkpoint:
        manifest.update(initialization='Exact numerical checkpoint continuation; no optimization reinitialization',
            continuation_lineage=dict(checkpoint=str(args.resume_checkpoint),checkpoint_sha256=sha(args.resume_checkpoint),
                prior_status=str(prior_status_path),prior_status_sha256=sha(prior_status_path),
                prior_elapsed_seconds=prior_status.get('elapsed_seconds',0),resume_phase=args.resume_phase,
                unsupported='Mid-search rank-selection restart is not implemented'))
    write(args.output/'experiment_status.json',manifest)
    with np.load(args.input,allow_pickle=False) as z:
        data={k:z[k] for k in ['Y','eta0','pi','coords','library_size']};genes=z['genes'];types=z['cell_types']
    if not np.array_equal(data['library_size'],data['Y'].sum(1)) or np.any(data['library_size']<=0):
        raise ValueError('library_size must equal the positive observed row sum of the fitted gene panel')
    threshold=.065*np.sqrt(250/len(genes)) if args.threshold is None else args.threshold
    if args.resume_checkpoint and prior_status.get('threshold')!=threshold:raise ValueError('Continuation threshold differs from original attempt')
    config=FitConfig(kernel='nystrom',device=args.device,dtype=args.dtype,lengthscale=args.lengthscale,n_landmarks=args.landmarks,
        batch_size=args.batch,chunk_size=args.batch,num_particles=args.particles,
        evaluation_particles=args.score_particles,seed=args.seed,lr=.001,q_steps=1,w_steps=1,nuisance_steps=1,pi_steps=1)
    kw=dict(epsilon_variance_mode='shared',epsilon_variance=args.epsilon_variance,epsilon_variance_floor=args.epsilon_floor)
    if args.resume_checkpoint:
        model,restored=restore_checkpoint(args.resume_checkpoint,args.device)
        expected_phase='k0' if args.resume_phase=='k0' else 'terminal'
        if restored.get('phase',{}).get('phase')!=expected_phase:raise ValueError('Wrong checkpoint phase')
        if restored.get('epsilon_variance_mode','shared')!='shared':raise ValueError('Variance mode differs from saved checkpoint')
        checks={'batch_size':args.batch,'num_particles':args.particles,'lengthscale':args.lengthscale,
            'n_landmarks':args.landmarks,'seed':args.seed,'dtype':args.dtype}
        for name,value in checks.items():
            if getattr(model.config,name)!=value:raise ValueError('Continuation changed frozen setting '+name)
        if args.resume_phase=='fixed_terminal' and args.stop_after_k0:raise ValueError('Contradictory terminal/K0-only request')
        if model.epsilon_variance_floor!=args.epsilon_floor:raise ValueError('Continuation epsilon floor differs')
        if args.resume_phase=='k0' and any(model.Kc):raise ValueError('K0 checkpoint already contains programs')
        if args.resume_phase=='fixed_terminal' and model.Kc!=json.loads(args.fixed_ranks):raise ValueError('Fixed ranks differ from checkpoint')
        model._real_continuation_armed_phase=expected_phase
    else:
        model=PyroBackfitMMF(**data,programs_per_type=[0]*len(types),config=config,**kw)
    del data
    callbacks=RealCallbacks(args.output,args,genes,types,manifest);callbacks.template=model
    manifest.update(N=model.N,G=model.G,C=model.C,config=asdict(config),threshold=threshold);write(args.output/'experiment_status.json',manifest)
    try:
        if args.resume_phase!='fixed_terminal':
            callbacks.fit(model,'k0',args.output/'k0',args.k0_cycles)
            callbacks.export(model,args.output/'k0/retained')
        if args.stop_after_k0:
            callbacks.final_arrays(model);manifest.update(phase='finished_k0_diagnostic',stable_final=True,selected_ranks=list(model.Kc))
        elif args.fixed_ranks:
            wanted=json.loads(args.fixed_ranks)
            if len(wanted)!=model.C:raise ValueError('fixed ranks dimension differs from C')
            if args.resume_phase!='fixed_terminal':
                for c,count in enumerate(wanted):
                    for k in range(count):
                        model=add_program(model,c,callbacks.positions(model,4096,93025),args.seed+1009*c+7*k)
                        callbacks.fit(model,'forward',args.output/f'fixed_K/type{c}_program{k}',args.forward_cycles,conditional_program=(c,k))
            callbacks.fit(model,'terminal',args.output/'fixed_K/terminal',args.k0_cycles)
            callbacks.final_arrays(model);callbacks.export(model,args.output/'fixed_K/final')
            manifest.update(phase='finished_fixed_K_diagnostic',stable_final=True,selected_ranks=list(model.Kc))
        else:
            model,selection=select_programs(model,seed_idx=callbacks.positions(model,4096,93025),seed=args.seed,
                output=args.output/'discovery',fit=callbacks.fit,score=callbacks.score,export=callbacks.export,write=write,
                tau=float(threshold),k_max=3,forward_cycles=args.forward_cycles,backfit_cycles=args.backfit_cycles,max_rounds=3*model.C+5,
                restore=callbacks.restore)
            if not selection['selection_complete']:raise RuntimeError('Program selection incomplete: '+selection['stop_reason'])
            callbacks.final_arrays(model)
            manifest.update(phase='finished',selection_complete=True,stable_final=True,selected_ranks=list(model.Kc),stop_reason=selection['stop_reason'])
        manifest['final_npz_sha256']=sha(args.output/'final.npz')
    except Exception as error:
        manifest.update(phase='needs_attention',error=repr(error),selection_complete=False,stable_final=False);raise
    finally:
        manifest.update(elapsed_seconds=time.time()-began+float(prior_status.get('elapsed_seconds',0)),current_attempt_seconds=time.time()-began,source_unchanged=all(sha(f)==manifest['source_sha256'][str(f)] for f in sources))
        if args.device.startswith('cuda'):
            manifest['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated(torch.device(args.device))
        write(args.output/'experiment_status.json',manifest)
    print(json.dumps(manifest),flush=True)
    return manifest


def main():
    run(parser().parse_args())


if __name__=='__main__':main()
