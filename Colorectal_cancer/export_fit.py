"""Export finite, metadata-aligned plotting arrays from a completed selected fit."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from Colorectal_cancer.io_utils import sha, write

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['HD','Xenium'],required=True)
    p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--effects',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();m=json.loads((a.fit/'run_manifest.json').read_text())
    assert m['status']=='completed' and m['selection_complete'] and m['source_unchanged']
    assert sha(a.fit/'final.npz')==m['final_npz_sha256'] and sha(a.input)==m['input_sha256']
    keys=['h_mean','h_sd','loadings','pi','program_cell_type','program_index','genes','cell_types','coords','spot_names','original_indices']
    with np.load(a.fit/'final.npz',allow_pickle=False) as z,np.load(a.input,allow_pickle=False) as inp:
        for k in ['genes','cell_types','coords','spot_names','original_indices']:
            np.testing.assert_array_equal(z[k],inp[k])
        data={k:z[k] for k in keys}
        assert all(np.isfinite(v).all() for v in data.values() if v.dtype.kind in 'fciub')
        np.testing.assert_allclose(data['pi'].sum(1),1,atol=1e-6)
        if a.kind=='HD':
            if a.effects is None:p.error('HD export requires --effects')
            with np.load(a.effects,allow_pickle=False) as e:
                for name in ['alpha','gamma']:
                    np.testing.assert_array_equal(z[name+'_mean'],e[name])
                    assert not np.count_nonzero(z[name+'_sd'])
                np.testing.assert_allclose(data['pi'],e['pi'],rtol=2e-7,atol=2e-8)
            assert m['configuration']['lengthscale']==.2 and m['G']==1343
            assert m['pi_max_abs_drift']==m['shared_effects_max_abs_drift']==0
        else:
            assert data['cell_types'].tolist()==['Epithelial'] and m['G']==1341
            assert m['configuration']['lengthscale']==.06 and m['no_pi_optimizer']
            np.testing.assert_array_equal(data['pi'],np.ones_like(data['pi']))
    a.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(a.output/'plot_source.npz',**data)
    shutil.copy2(a.fit/'run_manifest.json',a.output/'fit_run_manifest.json')
    receipt=dict(status='completed',hashes_verified=True,finite_arrays=True,metadata_aligned=True,
        plot_source_sha256=sha(a.output/'plot_source.npz'),final_npz_sha256=m['final_npz_sha256'],
        input_sha256=m['input_sha256'],pi_max_abs_drift=m['pi_max_abs_drift'],
        convergence_verified=m.get('convergence_verified',False),fit_rerun=False)
    if a.kind=='HD':receipt['shared_effects_max_abs_drift']=m['shared_effects_max_abs_drift']
    else:receipt['no_pi_optimizer']=m['no_pi_optimizer']
    write(a.output/('plot_export_receipt.json' if a.kind=='HD' else 'summary.json'),receipt)

if __name__=='__main__':main()
