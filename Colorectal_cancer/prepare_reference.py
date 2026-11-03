"""Prepare the CRC reference 400-DE + 400-within-type-HVG panel, with DC labels.

This prepares inputs only. No model fitting, external deconvolution or existing
program results enter selection. Every invocation requires a new directory.
"""
import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from Colorectal_cancer.reference_panel import nominate, sha, write_json

DC_FINE = ('mregDC', 'cDC2', 'pDC')


def broad_labels(labels):
    labels = np.asarray(labels, dtype=str)
    return np.where(np.isin(labels, DC_FINE), 'DC', labels)


def check_counts(matrix):
    x = sparse.csr_matrix(matrix)
    x.sum_duplicates(); x.eliminate_zeros(); x.sort_indices()
    for start in range(0, len(x.data), 1000000):
        values = x.data[start:start+1000000]
        if not np.isfinite(values).all() or (values<0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError('Counts must be finite, nonnegative integer UMIs')
    return x


def spatial(path, assay, old_input):
    a = ad.read_h5ad(path)
    counts = check_counts(a.X)
    total = np.asarray(counts.sum(1)).ravel(); detected = np.diff(counts.indptr)
    if assay=='HD':
        keep = (total>=50) & (detected>=100)
    else:
        keep = (a.obs['high_quality'].astype(str).to_numpy()=='1') & (total>=20) & (detected>=20)
    indices = np.flatnonzero(keep)
    names = a.obs_names.to_numpy(dtype=str)[keep]
    with np.load(old_input, allow_pickle=False) as old:
        np.testing.assert_array_equal(indices, old['original_indices'])
        np.testing.assert_array_equal(names, old['spot_names'])
    result = dict(counts=counts[keep], genes=a.var_names.to_numpy(dtype=str),
                  original_indices=indices, spot_names=names,
                  original_coords=np.asarray(a.obsm['spatial'], dtype=np.float64)[keep],
                  total_counts=total[keep], source=str(path), source_sha256=sha(path), raw_shape=list(a.shape))
    if assay=='Xenium':
        result['annotation'] = broad_labels(a.obs['annotation'].astype(str).to_numpy()[keep])
    del a,counts; gc.collect()
    print(assay, 'QC retained', len(indices), 'observations; identical to previous QC', flush=True)
    return result


def statistics(counts, genes, candidates):
    cols = pd.Index(genes).get_indexer(candidates)
    assert (cols>=0).all()
    x = counts[:, cols].tocsr()
    detected = np.asarray((x>0).sum(0)).ravel()
    means = np.zeros(len(candidates), dtype=np.float64)
    for start in range(0, x.shape[0], 4096):
        block = x[start:start+4096].astype(np.float64)
        libs = np.asarray(block.sum(1)).ravel()
        means += np.asarray(block.multiply(1/np.maximum(libs[:,None],1)).sum(0)).ravel()
    return detected, means/x.shape[0]


def initial_pi(counts, eta0):
    weights = np.empty((len(counts),len(eta0)),dtype=np.float32)
    profiles = eta0.astype(np.float64)
    inverse = np.linalg.pinv(profiles@profiles.T,hermitian=True)
    for start in range(0,len(counts),4096):
        y = counts[start:start+4096].astype(np.float64)
        w = ((y@profiles.T)/y.sum(1,keepdims=True))@inverse
        w = np.maximum(w,1e-3); w /= w.sum(1,keepdims=True)
        weights[start:start+len(y)] = w
    return weights


def save_input(assay, data, panel, eta0, types, out):
    cols = pd.Index(data['genes']).get_indexer(panel)
    observed = data['counts'][:,cols].toarray().astype(np.float32)
    positive = observed.sum(1)>0
    if not positive.all():
        raise ValueError(f'{assay}: zero-panel observations; inspect before changing the retained cohort')
    raw = data['original_coords']; origin = raw.min(0); scale = float(np.ptp(raw,axis=0).max())
    payload = dict(Y=observed, eta0=eta0, pi=initial_pi(observed,eta0),
                   coords=((raw-origin)/scale).astype(np.float32), library_size=observed.sum(1,dtype=np.float64),
                   genes=panel, cell_types=types, original_indices=data['original_indices'],
                   original_coords=raw, spot_names=data['spot_names'],total_counts=data['total_counts'])
    if assay=='Xenium':payload['annotation']=data['annotation']
    dest = out/f'{assay}_data.npz'
    np.savez_compressed(dest,**payload)
    print(assay,'prepared',observed.shape,flush=True)
    return dict(assay=assay,file=str(dest),N=len(observed),G=len(panel),sha256=sha(dest)),payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    raw=Path.home()/'Xenium_HD/data/SPATCH/COAD'
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--hd',type=Path,required=True)
    p.add_argument('--xenium',type=Path,required=True)
    p.add_argument('--legacy-data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--per-type',type=int,default=400)
    p.add_argument('--min-spots',type=int,default=10)
    p.add_argument('--min-relative-expression',type=float,default=5e-5)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    write_json(out/'preprocessing_status.json',dict(status='running',started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()))
    sections={'HD':spatial(args.hd,'HD',args.legacy_data/'HD_data.npz'),
              'Xenium':spatial(args.xenium,'Xenium',args.legacy_data/'Xenium_data.npz')}
    ref=ad.read_h5ad(args.reference);fine=ref.obs['major_annotation'].to_numpy(dtype=str)
    labels=broad_labels(fine);types=np.array(sorted(set(labels)),dtype=str)
    assert len(set(fine))==16 and len(types)==14 and 'DC' in types and 'Other' not in types
    assert int((labels=='DC').sum())==78
    for names in [ref.var_names]+[s['genes'] for s in sections.values()]:assert pd.Index(names).is_unique
    genes=np.array(sorted(set(ref.var_names)&set(sections['HD']['genes'])&set(sections['Xenium']['genes'])),dtype=str)
    counts=check_counts(ref.X)[:,ref.var_names.get_indexer(genes)].astype(np.float64).tocsr()
    nominees=nominate(counts,genes,labels,types,args.per_type)
    candidates=np.array(sorted(set(nominees.gene)),dtype=str)
    supported=np.asarray(counts[:,pd.Index(genes).get_indexer(candidates)].sum(0)).ravel()>0
    filtering=pd.DataFrame(dict(gene=candidates,reference_supported=supported))
    for assay,data in sections.items():
        det,expr=statistics(data['counts'],data['genes'],candidates)
        filtering[f'detected_{assay}']=det;filtering[f'mean_relative_expression_{assay}']=expr
    filtering['passes_detection_both']=filtering[['detected_HD','detected_Xenium']].min(axis=1)>=args.min_spots
    filtering['passes_mean_expression_both']=filtering[['mean_relative_expression_HD','mean_relative_expression_Xenium']].min(axis=1)>args.min_relative_expression
    filtering['in_panel']=supported&filtering.passes_detection_both&filtering.passes_mean_expression_both
    panel=filtering.loc[filtering.in_panel,'gene'].to_numpy(dtype=str)
    if not len(panel):raise ValueError('Empty final panel')
    nominees['in_panel']=nominees.gene.isin(set(panel))
    counts_panel=counts[:,pd.Index(genes).get_indexer(panel)].tocsr()
    libs=np.asarray(counts_panel.sum(1)).ravel();positive=libs>0
    normalized=counts_panel[positive].multiply(1/libs[positive,None]).tocsr()
    eta0=np.stack([np.asarray(normalized[labels[positive]==t].mean(0)).ravel() for t in types]).astype(np.float32)
    np.testing.assert_allclose(eta0.sum(1),1,atol=1e-6)
    assert np.isfinite(eta0).all()
    mapping=pd.DataFrame(dict(original_type=fine,cell_type=labels)).value_counts().rename('reference_cells').reset_index()
    nominees.to_csv(out/'gene_nominations.csv',index=False);filtering.to_csv(out/'gene_filtering.csv',index=False)
    mapping.to_csv(out/'cell_type_mapping.csv',index=False);pd.DataFrame(dict(gene=panel)).to_csv(out/'genes.csv',index=False)
    nominees.groupby(['cell_type','method']).agg(nominated=('gene','size'),retained=('in_panel','sum')).to_csv(out/'selection_counts.csv')
    np.savez_compressed(out/'reference_panel.npz',genes=panel,cell_types=types,eta0=eta0)
    summary=dict(common_universe_genes=len(genes),per_type=args.per_type,types=len(types),nomination_entries=len(nominees),
                 nomination_union=len(candidates),passing_detection=int(filtering.passes_detection_both.sum()),final_panel=len(panel))
    write_json(out/'selection_summary.json',summary)
    print('SELECTION',json.dumps(summary),flush=True)
    outputs=[];by_type=[]
    for assay,data in sections.items():
        record,payload=save_input(assay,data,panel,eta0,types,out);outputs.append(record)
        if assay=='Xenium':
            if not set(payload['annotation'])<=set(types):raise ValueError('Unknown Xenium annotation')
            for c,t in enumerate(types):
                support=eta0[c]>0;keep=(payload['annotation']==t)&(payload['Y'][:,support].sum(1)>0)
                subset={key:payload[key][keep] for key in ['Y','coords','original_indices','original_coords','spot_names','total_counts']}
                subset['Y']=subset['Y'][:,support]
                subset.update(eta0=eta0[c:c+1,support],pi=np.ones((int(keep.sum()),1),dtype=np.float32),
                              genes=panel[support],cell_types=np.array([t]),library_size=subset['Y'].sum(1,dtype=np.float64))
                dest=out/'Xenium_by_type'/str(t);dest.mkdir(parents=True,exist_ok=False)
                np.savez_compressed(dest/'data.npz',**subset)
                by_type.append(dict(cell_type=str(t),N=int(keep.sum()),G=int(support.sum()),sha256=sha(dest/'data.npz')))
            write_json(out/'Xenium_by_type/manifest.json',by_type)
        del payload;gc.collect()
    for filename in ['hd32_markers.npz','hd32_original_annotation_fractions.npz']:
        src=args.legacy_data/filename
        if src.exists():
            with np.load(src,allow_pickle=False) as aux:np.testing.assert_array_equal(aux['spot_names'],sections['HD']['spot_names'])
            shutil.copy2(src,out/filename)
    manifest=dict(schema='MMF_CRC_reference_panel_v1',created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        sources={s:{k:d[k] for k in ['source','source_sha256','raw_shape']} for s,d in sections.items()},
        source_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('reference_panel.py')]},
        reference=str(args.reference),reference_sha256=sha(args.reference),reference_shape=list(ref.shape),
        cell_types=types.tolist(),reference_cells_by_type={str(t):int((labels==t).sum()) for t in types},
        type_mapping='mregDC, cDC2, pDC -> DC; all other original labels unchanged; all 14 types nominate genes and can have programs',
        selection=dict(**summary,DE='Positive one-v-rest Welch t ranking on reference log1p(CP10k); no FDR significance claim',
                       HVG='Within-type CP10k log variance/mean, standardized in 20 equal-width log1p(mean) bins',
                       eligibility='At least max(3, ceil(5% of type cells)) reference detections; no mitochondrial genes',
                       normalization='All reference-HD-Xenium common genes before nomination'),
        qc=dict(cohort_unchanged=True,HD='Published high-quality source bins aggregated to 32um, then UMI >=50 and genes >=100',
                Xenium='Published high_quality=1, UMI >=20 and genes >=20',min_observations_per_assay=args.min_spots,
                min_mean_relative_expression_per_assay=args.min_relative_expression,
                relative_expression_denominator='Per-observation UMI over nomination union before final filtering; both assays separately'),
        count_policy='All observed UMI on final panel; no molecular split',
        reference_policy='Mean per-cell final-panel fractions by type; zero-panel reference cells excluded',
        zero_panel_reference_cells=int((~positive).sum()),mixture_initialization='Positive reference projection only; fit estimates mixture weights jointly',
        outputs=outputs,fit_started=False)
    write_json(out/'preprocess_manifest.json',manifest)
    write_json(out/'preprocessing_status.json',dict(status='completed',finished_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),**summary))
    print('COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':main()
