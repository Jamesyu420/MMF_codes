"""Run native py-rctd full mode and export its baseline in MMF coordinates."""
import argparse
from datetime import datetime, timezone
import gc
import inspect
import os
from pathlib import Path
import socket
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch
import rctd
import rctd._rctd as pipeline
from rctd._full import run_full_mode
from rctd import Reference, RCTDConfig
from Colorectal_cancer.io_utils import sha, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    args=p.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=False)
    t0=time.monotonic(); torch.set_num_threads(4); torch.manual_seed(23); np.random.seed(23)
    cfg=RCTDConfig(device='cuda',dtype='float64',UMI_min=1,counts_MIN=0)
    receipt=dict(status='loading',host=socket.gethostname(),pid=os.getpid(),
        visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),start_utc=datetime.now(timezone.utc).isoformat(),
        input=str(args.input),input_sha256=sha(args.input),runner_sha256=sha(__file__),
        rctd_py_version=rctd.__version__,mode='full',configuration=cfg._asdict(),batch_size=5000,
        package_sha256={f.name:sha(f) for f in Path(inspect.getfile(rctd)).parent.glob('*.py')})
    write(out/'run_manifest.json',receipt)
    try:
        with np.load(args.input,allow_pickle=False) as z: data={k:z[k] for k in z.files}
        refpath=args.reference
        ref=ad.read_h5ad(refpath)
        labels=ref.obs['major_annotation'].astype(str).to_numpy()
        ref.obs['MMF_type']=np.where(np.isin(labels,['mregDC','cDC2','pDC']),'DC',labels)
        ref=ref[:,list(data['genes'])].copy()
        reference=Reference(ref,cell_type_col='MMF_type',min_UMI=1)
        np.testing.assert_array_equal(reference.cell_type_names,data['cell_types'])
        np.testing.assert_allclose(reference.profiles.T,data['eta0'],rtol=2e-6,atol=1e-9)
        spatial=ad.AnnData(X=sparse.csr_matrix(data['Y']),obs=pd.DataFrame(index=data['spot_names']),
                         var=pd.DataFrame(index=data['genes']))
        spatial.obsm['spatial']=data['coords'].copy()
        del ref; gc.collect()
        # Capture the native bulk result without changing its input or algorithm.
        captured={}; native_fit_bulk=pipeline.fit_bulk
        def capture_bulk(*args,**kwargs):
            weights,profiles=native_fit_bulk(*args,**kwargs)
            captured['weights']=weights.detach().cpu().numpy().copy()
            captured['profiles']=profiles.detach().cpu().numpy().copy()
            captured['base_profiles']=kwargs['cell_type_profiles'].detach().cpu().numpy().copy()
            return weights,profiles
        pipeline.fit_bulk=capture_bulk
        model=pipeline.RCTD(spatial,reference,cfg)
        receipt.update(status='platform_running',N=len(data['Y']),G=len(data['genes']),C=len(data['cell_types']),
                       reference_sha256=sha(refpath))
        write(out/'run_manifest.json',receipt); print('PLATFORM_START',flush=True)
        t=time.monotonic()
        try: model.fit_platform_effects()
        finally: pipeline.fit_bulk=native_fit_bulk
        receipt['platform_and_sigma_seconds']=time.monotonic()-t
        receipt.update(status='full_mode_running',bulk_genes=len(model._gene_list_bulk),reg_genes=len(model._gene_list_reg))
        write(out/'run_manifest.json',receipt)
        t=time.monotonic()
        result=run_full_mode(spatial_counts=model.counts,spatial_numi=model.nUMI,norm_profiles=model.norm_profiles,
            cell_type_names=reference.cell_type_names,q_mat=model.q_mat,sq_mat=model.sq_mat,x_vals=model.x_vals,
            batch_size=5000,device=cfg.device,pixel_mask=model._pixel_mask)
        receipt['decomposition_seconds']=time.monotonic()-t
        receipt['rctd_seconds']=receipt['platform_and_sigma_seconds']+receipt['decomposition_seconds']
        assert result.pixel_mask.all(), 'RCTD dropped HD rows'
        np.testing.assert_array_equal(result.cell_type_names,data['cell_types'])
        beta=np.asarray(result.weights,dtype=np.float64)
        assert beta.shape==data['pi'].shape and np.isfinite(beta).all() and (beta>=0).all()
        scale=beta.sum(1); assert (scale>0).all()
        pi=beta/scale[:,None]
        alpha=np.log(scale)+np.log(model.nUMI.astype(np.float64)/data['library_size'].astype(np.float64))
        # Native RCTD estimates bulk composition on its DE bulk genes. Extend its
        # SAME get_norm_ref equation to the rest of the predeclared MMF panel.
        prop=captured['weights'].astype(np.float64); prop/=prop.sum()
        base=reference.get_profiles_for_genes(list(data['genes'])).astype(np.float64)
        target=data['Y'].sum(0,dtype=np.float64)/model.nUMI.sum(dtype=np.float64)
        gene_factor=(base@prop)/np.maximum(target,1e-10)
        factor=1/np.maximum(gene_factor,1e-10)
        lookup={g:i for i,g in enumerate(data['genes'])}
        bulk_idx=np.array([lookup[g] for g in model._gene_list_bulk])
        reg_idx=np.array([lookup[g] for g in model._gene_list_reg])
        # Use native factors on native genes, retaining the extrapolation only
        # for panel genes excluded from RCTD's bulk list.
        bp=captured['base_profiles']; npf=captured['profiles']
        native_factor=np.array([npf[j,np.argmax(bp[j])]/bp[j,np.argmax(bp[j])] for j in range(len(bp))])
        native_err=float(np.max(np.abs(factor[bulk_idx]/native_factor-1)))
        np.testing.assert_allclose(factor[bulk_idx],native_factor,rtol=5e-6,atol=1e-10)
        factor[bulk_idx]=native_factor
        gamma=np.log(factor); assert np.isfinite(alpha).all() and np.isfinite(gamma).all()
        # Verify full native per-pixel predictions against MMF's fixed baseline.
        max_abs=0.; max_rel=0.
        for first in range(0,len(beta),512):
            sl=slice(first,first+512)
            expected=model.nUMI[sl,None]*(beta[sl]@model.norm_profiles.T)
            actual=data['library_size'][sl,None]*np.exp(alpha[sl,None]+gamma[reg_idx][None,:])*(pi[sl]@data['eta0'][:,reg_idx])
            max_abs=max(max_abs,float(np.max(np.abs(actual-expected))))
            max_rel=max(max_rel,float(np.max(np.abs(actual-expected)/np.maximum(expected,1e-10))))
            np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-7)
        payload=dict(pi=pi.astype(np.float32),alpha=alpha.astype(np.float32),gamma=gamma.astype(np.float32),
            weights_raw=beta,weight_row_sum=scale,rctd_numi=model.nUMI,platform_factor=factor,
            bulk_weights=captured['weights'],bulk_genes=np.array(model._gene_list_bulk),
            regression_genes=np.array(model._gene_list_reg),converged=result.converged,
            rctd_sigma=np.array(model.sigma),genes=data['genes'],cell_types=data['cell_types'],spot_names=data['spot_names'])
        np.savez_compressed(out/'fixed_effects.npz',**payload)
        data['pi']=payload['pi']; np.savez_compressed(out/'HD_rctd_init.npz',**data)
        receipt.update(status='completed',output_sha256=sha(out/'fixed_effects.npz'),
            mmf_input_sha256=sha(out/'HD_rctd_init.npz'),converged_fraction=float(np.mean(result.converged)),
            all_rows_retained=True,native_factor_extension_relative_error=native_err,
            baseline_native_max_abs_error=max_abs,baseline_native_max_relative_error=max_rel,
            extrapolated_platform_genes=len(data['genes'])-len(bulk_idx),
            alpha_formula='log(sum_c raw_weight_ic) + log(RCTD_nUMI_i / MMF_library_size_i)',
            gamma_formula='log(native normalized reference / original reference); non-bulk genes use same bulk-weight get_norm_ref equation',
            pi_formula='raw_weight_ic / sum_c raw_weight_ic',
            rctd_gene_selection_unchanged=True,mmf_panel_unchanged=True,
            alpha_range=[float(alpha.min()),float(alpha.max())],gamma_range=[float(gamma.min()),float(gamma.max())])
    except Exception as exc:
        receipt.update(status='failed',error=repr(exc)); raise
    finally:
        receipt.update(total_seconds=time.monotonic()-t0,finish_utc=datetime.now(timezone.utc).isoformat(),
                       peak_gpu_bytes=torch.cuda.max_memory_allocated())
        write(out/'run_manifest.json',receipt)
    print('FIXED_EFFECTS_COMPLETE',receipt,flush=True)


if __name__=='__main__': main()
