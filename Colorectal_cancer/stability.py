"""Unchanged K0 diagnostic functions used by the selected HD fit."""
import numpy as np
import torch

def relative_change(current,previous):
    return float(np.linalg.norm(current-previous)/max(float(np.linalg.norm(previous)),1e-12))

def probe(model,diagnostics):
    """K0 has no program Monte Carlo term: direct_loss is the exact profiled ELBO."""
    rate=np.empty((model.N,model.G),dtype=np.float64);elbo=0.;stationarity=0.
    with model._scope(),torch.no_grad():
        for idx in model._chunks():
            elbo-=float(model.direct_loss(idx))*len(idx)/model.N
            rows=idx.cpu().numpy()
            rate[rows]=model._log_expected_counts(idx).exp().cpu().numpy()
            diagnostic=diagnostics(model.structural_log_expected_counts(idx),model.Y[idx],model.v_eps)
            stationarity=max(stationarity,diagnostic['maximum_log_rate_newton_correction'],
                             diagnostic['maximum_log_rate_reconstruction_error'])
        pi=model._pi().detach().cpu().numpy().copy()
        variance=torch.stack([model.v_alpha,model.v_gamma]).detach().cpu().numpy().copy()
        epsilon=model.v_eps.detach().cpu().numpy().reshape(-1).copy()
    if not all(np.isfinite(v).all() for v in [rate,pi,variance,epsilon]) or not np.isfinite(elbo):
        raise FloatingPointError('Nonfinite complete-data diagnostic')
    updates=model.epsilon_updates[-3:]
    return dict(elbo=elbo,elbo_per_entry=elbo/(model.N*model.G),rate=rate,pi=pi,variance=variance,
                epsilon=epsilon,stationarity=stationarity,
                epsilon_last3=max([u['true_relative_change'] for u in updates]) if len(updates)==3 else None)

def changes(current,previous):
    return {key+'_change':relative_change(current[key],previous[key]) for key in ['rate','pi','variance','epsilon']}

def window_change(history):
    if len(history)<10:return None
    recent=history[-10:]
    before=float(np.mean([x['elbo_per_entry'] for x in recent[:5]]))
    after=float(np.mean([x['elbo_per_entry'] for x in recent[5:]]))
    return abs(after-before)/max(1.,abs(before))

def stable(history,step,elbo_rtol=1e-2):
    if step<500 or len(history)<10:return False
    return bool(window_change(history)<elbo_rtol and all(
        max(row[key+'_change'] for key in ['rate','pi','variance','epsilon'])<.01
        and row['epsilon_last3'] is not None and row['epsilon_last3']<.01
        and row['stationarity']<1e-6 for row in history[-3:]))

def update_epsilon(model):
    old=float(model.v_eps)
    model.update_epsilon_variance()
    model.epsilon_updates[-1]['true_relative_change']=abs(float(model.v_eps)-old)/max(abs(old),1e-8)
