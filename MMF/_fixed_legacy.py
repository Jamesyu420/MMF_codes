"""Fit proportions during K0, then hold those estimates exactly fixed.

All other parameters, priors, conditional forward fits, rank selection,
backfitting, deletion refits and polish follow MMF_ctback. Fixed proportions
come from the fitted K0 model, never from truth or the uniform initialization.
Both the original softmax and log-softmax values are saved to avoid introducing
a floating-point perturbation when the representation changes.
"""
from dataclasses import replace
import torch
from MMF.backfit import PyroBackfitMMF
from MMF.MMF_ctback import select_programs_ct as _select_programs_ct


class FixedAfterK0MMF(PyroBackfitMMF):
    def __init__(self, *args, **kwargs):
        self.pi_frozen_after_k0 = False
        super().__init__(*args, **kwargs)

    def freeze_proportions(self):
        if self.pi_frozen_after_k0:
            self.assert_proportions_frozen()
            return self
        if any(self.Kc) or not self.config.fit_pi:
            raise ValueError('Freeze immediately after a fitted K0 model with learned pi')
        if self.optimizer_calls.get('pi', 0) == 0:
            raise ValueError('K0 must actually update pi before freezing it')
        with self._scope(), torch.no_grad():
            fixed_pi = self._pi().detach().clone()
            fixed_log_pi = self._log_pi().detach().clone()
        self._initial_pi = fixed_pi
        self._fixed_log_pi = fixed_log_pi
        self._fixed_pi_reference = fixed_pi.detach().clone()
        self._fixed_log_pi_reference = fixed_log_pi.detach().clone()
        self.config = replace(self.config, fit_pi=False)
        for registry in [self.group_names, self.optimizers, self.svis, self.block_optimizers]:
            registry.pop('pi', None)
        self._param_state['params'].pop('pi_logits', None)
        self._param_state['constraints'].pop('pi_logits', None)
        self.pi_frozen_after_k0 = True
        self.assert_proportions_frozen()
        return self

    def _log_pi(self):
        if self.pi_frozen_after_k0:
            return self._fixed_log_pi
        return super()._log_pi()

    def _step_group(self, block, idx, reference_only=False):
        if self.pi_frozen_after_k0 and block == 'pi':
            raise RuntimeError('MMF_fix forbids pi updates after K0')
        return super()._step_group(block, idx, reference_only)

    def assert_proportions_frozen(self):
        if not self.pi_frozen_after_k0:
            return
        assert not self.config.fit_pi
        assert 'pi_logits' not in self._param_state['params']
        assert all('pi' not in r for r in [self.group_names,self.optimizers,self.svis])
        assert not self._initial_pi.requires_grad and not self._fixed_log_pi.requires_grad
        assert torch.equal(self._initial_pi,self._fixed_pi_reference)
        assert torch.equal(self._fixed_log_pi,self._fixed_log_pi_reference)


def make_model(data, config):
    return FixedAfterK0MMF(**data, programs_per_type=0,
        config=replace(config, program_steps=10, fit_pi=True),
        epsilon_variance=.1, epsilon_variance_floor=1e-8)


def after_k0(model):
    return model.freeze_proportions()


def phase_plan(phase, requested_steps):
    return 10, requested_steps


def select_programs_ct(parent, **kwargs):
    """Drop-in selection entry after K0 has fitted pi, using make_model above."""
    parent.freeze_proportions()
    return _select_programs_ct(parent, **kwargs)
