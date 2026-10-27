"""Post-fit paired metrics on frozen outputs; truth never feeds fitting.

Loading W fixes the Hungarian assignment before H is evaluated. Activities
are never sign-flipped, and unmatched true programs score zero in recovery
averages. Native posterior h intervals remain on native L1(W)=1 scale.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import gammaln


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return dict(z)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corr(a,b):
    a=np.asarray(a,dtype=float)-np.mean(a);b=np.asarray(b,dtype=float)-np.mean(b)
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denominator) if denominator>0 else 0.


def run(a):
    a.output.mkdir(parents=True,exist_ok=True)
    truth=arrays(a.truth);fit_input=arrays(a.fit_input)
    heldout=None if a.heldout is None else arrays(a.heldout)
    pi_true=truth['pi_true'];w_true=truth['loadings_true'];h_true=truth['h_true']
    n,c=pi_true.shape;g=w_true.shape[1];true_k=np.bincount(truth['program_cell_type'],minlength=c)
    rows=[];pimatch=[];programmatch=[];effectrows=[];prediction=[];sources={}
    for assignment in a.fit:
        method,path=assignment.split('=',1);path=Path(path)
        if not path.exists():raise FileNotFoundError(path)
        z=arrays(path);sources[str(path.resolve())]=sha(path)
        native='program_cell_type' in z
        rctd=method=='RCTD'
        raw=z['pi'] if native or rctd else z['activity']
        if native or rctd:pi_assignment={j:j for j in range(c)}
        else:
            matrix=np.asarray([[corr(pi_true[:,j],raw[:,k]) for k in range(raw.shape[1])] for j in range(c)])
            rr,cc=linear_sum_assignment(-matrix);pi_assignment=dict(zip(map(int,rr),map(int,cc)))
        composition=[]
        for j in range(c):
            score=corr(pi_true[:,j],raw[:,pi_assignment[j]]);composition.append(score)
            pimatch.append(dict(method=method,true_type_index0=j,native_factor_index0=pi_assignment[j],
                correlation=score,assignment='intrinsic reference label' if native or rctd else 'post-fit composition-proxy Hungarian'))
        row=dict(method=method,composition_correlation=float(np.mean(composition)))
        if rctd:
            row.update(program_recovery_applicable=False);rows.append(row);continue
        w=z['loadings'];h=z['h_mean'] if native else z['activity'].T
        matches={}
        if native:
            for j in range(c):
                ti=np.flatnonzero(truth['program_cell_type']==j);fi=np.flatnonzero(z['program_cell_type']==j)
                if not len(ti) or not len(fi):continue
                matrix=np.asarray([[corr(w_true[t],w[f]) for f in fi] for t in ti])
                rr,cc=linear_sum_assignment(-matrix)
                matches.update({int(ti[t]):int(fi[f]) for t,f in zip(rr,cc)})
        else:
            matrix=np.asarray([[corr(wt,wf) for wf in w] for wt in w_true])
            rr,cc=linear_sum_assignment(-matrix);matches=dict(zip(map(int,rr),map(int,cc)))
        loading_scores=[];activity_scores=[];coverages=[];widths=[]
        for t in range(len(w_true)):
            f=matches.get(t)
            loading=0. if f is None else corr(w_true[t],w[f])
            activity=0. if f is None else corr(h_true[t],h[f])
            recall=0. if f is None else len(set(np.argsort(w_true[t])[-8:])&set(np.argsort(w[f])[-8:]))/8
            loading_scores.append(loading);activity_scores.append(activity)
            record=dict(method=method,true_program_index0=t,true_type_index0=int(truth['program_cell_type'][t]),
                native_factor_index0=f,matched=f is not None,W_centered_cosine=loading,H_correlation=activity,
                top8_recall=recall,matching_uses_H_truth=False,H_sign_flip=False)
            if native and f is not None:
                sd=z['h_sd'][f];lower=h[f]-1.6448536269514722*sd;upper=h[f]+1.6448536269514722*sd
                coverage=float(np.mean((h_true[t]>=lower)&(h_true[t]<=upper)))
                width=float(np.mean(upper-lower))
                coverages.append(coverage);widths.append(width)
                record.update(native_H90_coverage=coverage,native_H90_mean_width=width,
                    H_RMSE=float(np.sqrt(np.mean((h_true[t]-h[f])**2))))
            programmatch.append(record)
        row.update(program_recovery_applicable=True,W_centered_cosine=float(np.mean(loading_scores)),
            H_correlation=float(np.mean(activity_scores)),true_programs_matched=len(matches))
        if native:
            selected=np.bincount(z['program_cell_type'],minlength=c)
            row.update(selected_total=int(selected.sum()),exact_rank_vector=bool(np.array_equal(selected,true_k)),
                native_H90_coverage_matched=float(np.mean(coverages)) if coverages else None,
                native_H90_width_matched=float(np.mean(widths)) if widths else None,
                epsilon_variance_mode=str(z.get('epsilon_variance_mode',np.asarray('none'))),
                epsilon_prior_variance_mean=float(np.mean(z['epsilon_prior_variance'])) if 'epsilon_prior_variance' in z else 0.)
            for j in range(c):row[f'K{j+1}']=int(selected[j])
            for j in range(c):
                true_effect=np.einsum('pn,pg->ng',h_true[truth['program_cell_type']==j],w_true[truth['program_cell_type']==j])
                chosen=z['program_cell_type']==j
                fitted_effect=np.einsum('pn,pg->ng',h[chosen],w[chosen])
                effectrows.append(dict(method=method,type_index0=j,selected_K=int(selected[j]),true_K=int(true_k[j]),
                    total_type_log_effect_RMSE=float(np.sqrt(np.mean((true_effect-fitted_effect)**2))),
                    total_type_log_effect_correlation=corr(true_effect.ravel(),fitted_effect.ravel()),
                    matching_required=False))
            if heldout is not None:
                yy=heldout['Y_test']
                if yy.shape!=(n,g):raise ValueError('Heldout shape differs from fitted panel')
                for key in ['structural_expected_counts','same_entry_expected_counts','fresh_epsilon_expected_counts']:
                    if key not in z:continue
                    # Training exposure .6, test .2. This is explicitly a
                    # Poisson-at-mean summary, not an integrated PLN score.
                    test_mean=np.asarray(z[key],dtype=float)*(a.test_fraction/a.train_fraction)
                    safe=np.maximum(test_mean,np.finfo(float).tiny)
                    nll=float(np.mean(safe-yy*np.log(safe)+gammaln(yy+1)))
                    prediction.append(dict(method=method,mean_kind=key,test_Poisson_at_mean_NLL=nll,
                        integrated_Poisson_lognormal_predictive_score=False,
                        epsilon_posterior_fitted_using_training_only=True))
                target=truth.get('structural_rate_true',truth['rate_true'])
                if 'structural_expected_counts' in z:
                    row['structural_full_rate_RMSE']=float(np.sqrt(np.mean((z['structural_expected_counts']/a.train_fraction-target)**2)))
        rows.append(row)
    for name,data in [('case_metrics',rows),('composition_matching',pimatch),
                      ('program_matching',programmatch),('total_type_effect_metrics',effectrows),
                      ('heldout_mean_prediction',prediction)]:
        pd.DataFrame(data).to_csv(a.output/f'{name}.csv',index=False)
    sources.update({str(p.resolve()):sha(p) for p in [a.truth,a.fit_input]+([] if a.heldout is None else [a.heldout])})
    (a.output/'evaluation_receipt.json').write_text(json.dumps(dict(
        phase='finished',source_sha256=sources,methods=[r['method'] for r in rows],
        truth_opened_only_postfit=True,program_matching='W-only within type for native MMF; global W matching for global baselines',
        unmatched_true_programs_score_zero=True,activity_sign_flip=False,
        native_H_intervals='Marginal Gaussian 90 percent intervals, no post-fit unit-variance rescaling',
        mean_prediction_scope='Separate structural / same-entry / fresh-epsilon mean predictions; Poisson-at-mean NLL is not a proper integrated PLN predictive score',
        integrated_prediction_scope='Full posterior predictive scores are evaluated by the separate proper_predictive.py adapter and linked, not approximated here by a Poisson density at the mean',
        proper_predictive_receipt=None if getattr(a,'proper_predictive_receipt',None) is None else dict(path=str(a.proper_predictive_receipt.resolve()),sha256=sha(a.proper_predictive_receipt)),
        source_code_sha256=sha(__file__)),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit-input',type=Path,required=True)
    p.add_argument('--truth',type=Path,required=True)
    p.add_argument('--heldout',type=Path)
    p.add_argument('--fit',action='append',required=True,help='METHOD=/absolute/path/to/final-or-factors.npz')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-fraction',type=float,default=.6)
    p.add_argument('--test-fraction',type=float,default=.2)
    p.add_argument('--proper-predictive-receipt',type=Path,help='Result receipt from proper_predictive.py; links the independently integrated score')
    run(p.parse_args())


if __name__=='__main__':main()
