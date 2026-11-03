"""Postfit interpretation inputs from original SPATCH obs and prespecified markers.

No MMF fitting/selection file is read. Original data are never changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import h5py
import numpy as np
import pandas as pd

MARKERS=['MUC2','TFF3','SPINK4','FCGBP','CLCA1','ZG16','AGR2','REG4','EPCAM',
    'KRT8','KRT18','KRT19','CEACAM5','CEACAM6','MKI67','LGR5','OLFM4',
    'CDX2','ASCL2','CA1','CA2','FABP1','NDRG1','TOP2A','PCNA','MCM2','MCM5','TYMS']


def string_values(ds):
    return ds.asstr()[:] if ds.dtype.kind in ['O','S','U'] else ds[:]


def obs_table(f):
    group=f['obs']; key=group.attrs.get('_index','_index')
    if isinstance(key,bytes):key=key.decode()
    index=string_values(group[key]).astype(str)
    values={}; categories={}
    for name,obj in group.items():
        if name==key:continue
        if isinstance(obj,h5py.Dataset) and obj.ndim==1 and len(obj)==len(index):
            values[name]=string_values(obj)
        elif isinstance(obj,h5py.Group) and 'categories' in obj and 'codes' in obj:
            cats=string_values(obj['categories']).astype(str); codes=obj['codes'][:]
            out=np.full(len(codes),'',dtype=object);valid=codes>=0;out[valid]=cats[codes[valid]]
            values[name]=out;categories[name]=cats.tolist()
    result=pd.DataFrame(values,index=index);result.index.name='original_barcode'
    return result,categories


def selected_counts(group,feature_indices,nrows,nfeatures):
    result=np.zeros((nrows,len(feature_indices)),dtype=np.float64)
    feature_to_marker=np.full(nfeatures,-1,dtype=np.int64)
    feature_to_marker[feature_indices]=np.arange(len(feature_indices))
    if isinstance(group,h5py.Dataset):
        order=np.argsort(feature_indices)
        return group[:,feature_indices[order]][:,np.argsort(order)]
    encoding=group.attrs['encoding-type'];encoding=encoding.decode() if isinstance(encoding,bytes) else encoding
    ptr=group['indptr'][:]
    if encoding=='csc_matrix':
        for j,g in enumerate(feature_indices):
            indices=group['indices'][ptr[g]:ptr[g+1]];counts=group['data'][ptr[g]:ptr[g+1]]
            np.add.at(result[:,j],indices,counts)
    elif encoding=='csr_matrix':
        for start in range(0,nrows,10000):
            stop=min(start+10000,nrows);left,right=int(ptr[start]),int(ptr[stop])
            indices=group['indices'][left:right];marker=feature_to_marker[indices];selected=marker>=0
            if not selected.any():continue
            rows=np.repeat(np.arange(start,stop),np.diff(ptr[start:stop+1]))
            counts=group['data'][left:right]
            np.add.at(result,(rows[selected],marker[selected]),counts[selected])
    else:
        raise ValueError(f'Unsupported original X encoding: {encoding}')
    return result


def full_library_totals(group,nrows):
    totals=np.zeros(nrows,dtype=np.int64)
    if isinstance(group,h5py.Dataset):
        for start in range(0,nrows,256):
            totals[start:start+256]=group[start:start+256,:].sum(1,dtype=np.int64)
        return totals
    encoding=group.attrs['encoding-type'];encoding=encoding.decode() if isinstance(encoding,bytes) else encoding
    ptr=group['indptr'][:]
    if encoding=='csr_matrix':
        for start in range(0,nrows,10000):
            stop=min(start+10000,nrows);left,right=int(ptr[start]),int(ptr[stop])
            cumulative=np.r_[0,np.cumsum(group['data'][left:right],dtype=np.int64)]
            totals[start:stop]=np.diff(cumulative[ptr[start:stop+1]-left])
    elif encoding=='csc_matrix':
        for start in range(0,len(ptr)-1,128):
            stop=min(start+128,len(ptr)-1);left,right=int(ptr[start]),int(ptr[stop])
            rows=group['indices'][left:right];counts=group['data'][left:right]
            totals+=np.bincount(rows,weights=counts,minlength=nrows).astype(np.int64)
    else:
        raise ValueError(f'Unsupported original X encoding: {encoding}')
    return totals


def extract(source,output,label):
    started=time.time();output.mkdir(parents=True,exist_ok=True)
    with h5py.File(source,'r') as f:
        obs,categories=obs_table(f)
        feature_index=f['var'].attrs.get('_index','_index')
        if isinstance(feature_index,bytes):feature_index=feature_index.decode()
        genes=string_values(f['var'][feature_index]).astype(str)
        present=[g for g in MARKERS if g in set(genes)]
        assert len(genes)==len(set(genes)), 'Duplicate symbols require an explicit sum policy.'
        feature_indices=np.array([int(np.flatnonzero(genes==g)[0]) for g in present])
        counts=selected_counts(f['X'],feature_indices,len(obs),len(genes))
        library=full_library_totals(f['X'],len(obs))
        assert np.isfinite(counts).all() and (counts>=0).all() and np.all(counts==np.rint(counts))
        coords=f['obsm/spatial'][:] if 'obsm/spatial' in f else np.zeros((0,2))
    obs.to_csv(output/f'{label}_original_obs.csv.gz')
    np.savez_compressed(output/f'{label}_original_marker_counts.npz',counts=counts.astype(np.int64),
        genes=np.array(present),coords=coords,barcodes=obs.index.to_numpy(dtype=str),
        library_size_all_genes=library)
    receipt=dict(dataset=label,original_source=str(source.resolve()),shape=[len(obs),len(genes)],
        original_source_size=source.stat().st_size,obs_columns=list(obs.columns),
        categorical_levels=categories,prespecified_markers=MARKERS,present_markers=present,
        absent_markers=[g for g in MARKERS if g not in present],spatial_coords_shape=list(coords.shape),
        purpose='Postfit biological interpretation only; never rank/threshold selection or training gene selection.',
        no_fit_or_selection_file_read=True,elapsed_seconds=time.time()-started,
        normalization_library='Exact original X row sum across all assayed genes; marker counts remain raw integers.',
        marker_file_sha256=hashlib.sha256((output/f'{label}_original_marker_counts.npz').read_bytes()).hexdigest(),
        obs_file_sha256=hashlib.sha256((output/f'{label}_original_obs.csv.gz').read_bytes()).hexdigest())
    (output/f'{label}_extraction_audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--label',required=True)
    a=ap.parse_args();extract(a.source,a.output,a.label)
