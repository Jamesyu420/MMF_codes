"""Original fixed-pi schedule; additionally plug in K0 alpha/gamma means.

This ablation deliberately does not inherit the new fast inference mixin.
After K0 there are no alpha/gamma sample sites, variational parameters, KLs,
optimizers or EB updates. Epsilon and program inference retain the old policy.
"""
from dataclasses import replace
import pyro
import torch
from MMF._fixed_legacy import FixedAfterK0MMF as LegacyFixedMMF
from MMF.MMF_ctback import select_programs_ct as _select


class FixedPiEffectsMMF(LegacyFixedMMF):
    def __init__(self,*args,**kwargs):
        self.shared_effects_frozen=False
        super().__init__(*args,**kwargs)

    def freeze_proportions(self):
        super().freeze_proportions()
        if self.shared_effects_frozen:
            self.assert_shared_effects_frozen();return self
        with self._scope(),torch.no_grad():
            self._fixed_shared={n:super(FixedPiEffectsMMF,self)._shared(n)[0].detach().clone() for n in ('alpha','gamma')}
        self._fixed_shared_reference={n:v.clone() for n,v in self._fixed_shared.items()}
        for n in ('alpha','gamma'):
            for registry in [self.group_names,self.optimizers,self.svis,self.block_optimizers]:registry.pop(n,None)
            for suffix in ('mean','var'):
                self._param_state['params'].pop(f'{n}_{suffix}',None)
                self._param_state['constraints'].pop(f'{n}_{suffix}',None)
        self.shared_effects_frozen=True
        self.assert_shared_effects_frozen();return self

    def assert_shared_effects_frozen(self):
        if not self.shared_effects_frozen:return
        for n,value in self._fixed_shared.items():
            assert torch.equal(value,self._fixed_shared_reference[n]) and not value.requires_grad
            assert all(f'{n}_{s}' not in self._param_state['params'] for s in ('mean','var'))
            assert all(n not in r for r in [self.group_names,self.optimizers,self.svis])

    def _shared(self,name):
        if self.shared_effects_frozen:return self._fixed_shared[name],torch.zeros_like(self._fixed_shared[name])
        return super()._shared(name)

    def _guide(self,idx,reference_only=False):
        if not self.shared_effects_frozen:return super()._guide(idx,reference_only)
        if not reference_only:
            for c,k in self.programs:pyro.sample(f'z_{c}_{k}',self._q_distribution(c,k))

    def _model(self,idx,reference_only=False):
        if not self.shared_effects_frozen:return super()._model(idx,reference_only)
        from MMF.backfit import _gaussian_epsilon_solution
        z={} if reference_only else {p:pyro.sample(f'z_{p[0]}_{p[1]}',self._program_prior(*p)) for p in self.programs}
        mix=self._log_expected_mixture(idx,True) if reference_only else self._log_mixture(idx,z)
        a=self.structural_log_expected_counts(idx,reference_only)
        m,s,kl,x=_gaussian_epsilon_solution(a,self.Y[idx],self.v_eps)
        alpha,_=self._shared('alpha');gamma,_=self._shared('gamma')
        linear=self.log_N[idx,None].double()+alpha[idx,None].double()+gamma.double()
        y=self.Y[idx].double();rate=(a+x-a.detach()).exp()
        with pyro.poutine.scale(scale=self.N/len(idx)):
            pyro.factor('fixed_effect_likelihood_and_epsilon_KL',(y*(linear+mix.double()+m)-rate-torch.lgamma(y+1)-kl).sum((-2,-1)))

    def _structural_objective_parts(self,idx,particles,reference_only,eps):
        if not self.shared_effects_frozen:return super()._structural_objective_parts(idx,particles,reference_only,eps)
        programs=[] if reference_only else self.programs
        kl=torch.zeros((),device=self.device,dtype=torch.float64)
        for p in programs:
            mu,var=self._q(*p);prior=self.v[p].double()
            kl=kl+.5*((mu.double().square()+var.double())/prior-1+prior.log()-var.double().log()).sum()
        if programs:
            if eps is None:
                count=max(2,int(particles or self.config.num_particles));count+=count%2
                half=torch.randn((count//2,self.C,len(idx),self.G),device=self.device,dtype=self.dtype)
                eps=torch.cat([half,-half],0)
            mix=self._marginal_log_mixture(idx,programs,eps)
        else:mix=(self._indexed_log_pi(idx).T[:,:,None]+self.log_eta0[:,None,:]).logsumexp(0)
        alpha,_=self._shared('alpha');gamma,_=self._shared('gamma')
        linear=self.log_N[idx,None].double()+alpha[idx,None].double()+gamma.double()
        return kl,linear+mix.double(),self.structural_log_expected_counts(idx,reference_only)

    def _step_group(self,block,idx,reference_only=False):
        if self.shared_effects_frozen and block in ('alpha','gamma','pi'):
            raise RuntimeError('K0 plug-in effects cannot be updated')
        return super()._step_group(block,idx,reference_only)

    def backfit_cycle(self,batches,conditional_program=None):
        if not self.shared_effects_frozen:return super().backfit_cycle(batches,conditional_program)
        self.assert_shared_effects_frozen()
        for p in self.programs if conditional_program is None else [conditional_program]:
            for idx in batches:self._step_group((*p,'qw'),idx)
            self._update_variance(*p)

    def _update_shared_variances(self):
        if not self.shared_effects_frozen:super()._update_shared_variances()

    def posterior_summary(self,*,include_epsilon_arrays=False):
        result=super().posterior_summary(include_epsilon_arrays=include_epsilon_arrays)
        if self.shared_effects_frozen:
            result.pop('alpha_prior_variance',None);result.pop('gamma_prior_variance',None)
        return result


def make_model(data,config):
    return FixedPiEffectsMMF(**data,programs_per_type=0,config=replace(config,program_steps=10,fit_pi=True),
                            epsilon_variance=.1,epsilon_variance_floor=1e-8)
def after_k0(model):return model.freeze_proportions()
def phase_plan(phase,requested_steps):return 10,requested_steps
def select_programs_ct(parent,**kwargs):
    parent.freeze_proportions();return _select(parent,**kwargs)
