"""Install external RCTD plug-in values into the unchanged MMF_fixfix model."""
from dataclasses import replace
import torch
from MMF.MMF_fixfix import FixedPiEffectsMMF


def make_fixed_model(data,config,alpha,gamma,programs_per_type=0):
    model=FixedPiEffectsMMF(**data,programs_per_type=programs_per_type,
        config=replace(config,fit_pi=False),epsilon_variance=.1,epsilon_variance_floor=1e-8)
    with torch.no_grad():
        model._fixed_log_pi=model._initial_pi.log().detach().clone()
        model._fixed_pi_reference=model._initial_pi.detach().clone()
        model._fixed_log_pi_reference=model._fixed_log_pi.detach().clone()
        model.pi_frozen_after_k0=True
        model._fixed_shared={'alpha':model._tensor(alpha).detach().clone(),
                             'gamma':model._tensor(gamma).detach().clone()}
        if model._fixed_shared['alpha'].shape!=(model.N,) or model._fixed_shared['gamma'].shape!=(model.G,):
            raise ValueError('Fixed effect dimensions do not match data')
        for value in model._fixed_shared.values():
            if not torch.isfinite(value).all():raise ValueError('Nonfinite fixed effect')
        model._fixed_shared_reference={n:v.clone() for n,v in model._fixed_shared.items()}
        for name in ('alpha','gamma'):
            for registry in (model.group_names,model.optimizers,model.svis,model.block_optimizers):
                registry.pop(name,None)
            for suffix in ('mean','var'):
                model._param_state['params'].pop(f'{name}_{suffix}',None)
                model._param_state['constraints'].pop(f'{name}_{suffix}',None)
        model.shared_effects_frozen=True
    model.fixed_effect_provenance='External RCTD: no MMF pi/alpha/gamma fitting or K0 plug-in estimation'
    assert_fixed(model)
    return model


def assert_fixed(model):
    model.assert_proportions_frozen();model.assert_shared_effects_frozen()
    assert model.shared_effects_frozen and not model.config.fit_pi
    assert not any(n in model.group_names for n in ('pi','alpha','gamma'))
    assert not any(n in model._param_state['params'] for n in
                   ('pi_logits','alpha_mean','alpha_var','gamma_mean','gamma_var'))
    for name in ('alpha','gamma'):
        value,variance=model._shared(name)
        assert torch.equal(value,model._fixed_shared_reference[name])
        assert torch.count_nonzero(variance)==0
