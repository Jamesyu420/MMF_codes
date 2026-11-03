"""RCTD selects cells; MMF then treats the selected cohort as one epithelial type."""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import socket
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from rctd import Reference, RCTDConfig, run_rctd, __version__
from Colorectal_cancer.io_utils import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel', choices=['dehvg100'], default='dehvg100')
    p.add_argument('--prepared-data',type=Path,required=True)
    p.add_argument('--panel-input',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--xenium',type=Path,required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); out = args.output
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); home = Path.home()
    torch.set_num_threads(4); np.random.seed(20250502); torch.manual_seed(20250502)
    config = RCTDConfig(device='cuda', dtype='float64')
    base = args.prepared_data
    panel_input = args.panel_input
    ref_path = args.reference
    spatial_path = args.xenium
    receipt = dict(status='loading', start_utc=datetime.now(timezone.utc).isoformat(),
        host=socket.gethostname(), pid=os.getpid(), visible_gpu=os.getenv('CUDA_VISIBLE_DEVICES'),
        panel=args.panel, target_type='Epithelial', runner_sha256=sha(__file__),
        rctd_py_version=__version__, mode='doublet', config=config._asdict(), batch_size=10000,
        seed=20250502, panel_input=str(panel_input), panel_input_sha256=sha(panel_input),
        RCTD_gene_policy='Full raw measured genes and default internal RCTD DE selection, as in the prior Xenium workflow; independent RCTD run for each experiment.',
        MMF_policy='Single Epithelial reference, exact pi=1, no pi fitting; remove zero epithelial-reference genes as explicitly requested; final-panel counts and normalization.',
        reference_path=str(ref_path), spatial_path=str(spatial_path))
    write(out / 'run_manifest.json', receipt)
    try:
        with np.load(base / 'Xenium_data.npz', allow_pickle=False) as z:
            meta = {k: z[k] for k in ['spot_names', 'original_indices', 'coords', 'original_coords', 'cell_types']}
        with np.load(panel_input, allow_pickle=False) as z:
            genes = z['genes']; eta0 = z['eta0']; types = z['cell_types']
        np.testing.assert_array_equal(types, meta['cell_types'])
        expected_g = 1343
        assert len(genes) == expected_g
        ref = ad.read_h5ad(ref_path)
        labels = ref.obs['major_annotation'].astype(str).to_numpy()
        ref.obs['MMF_type'] = np.where(np.isin(labels, ['mregDC', 'cDC2', 'pDC']), 'DC', labels)
        reference = Reference(ref, cell_type_col='MMF_type')
        np.testing.assert_array_equal(reference.cell_type_names, types)
        del ref; gc.collect()
        spatial = ad.read_h5ad(spatial_path)
        np.testing.assert_array_equal(spatial.obs_names.to_numpy(dtype=str)[meta['original_indices']], meta['spot_names'])
        spatial = spatial[meta['original_indices']].copy()
        assert spatial.var_names.is_unique
        assert np.all(pd.Index(spatial.var_names).get_indexer(genes) >= 0)
        receipt.update(status='rctd_running', N=len(spatial), G=expected_g, C=len(types),
            reference_genes=reference.n_genes, spatial_genes=spatial.n_vars)
        write(out / 'run_manifest.json', receipt)
        print('RCTD_START', json.dumps(receipt), flush=True)
        began = time.monotonic()
        result = run_rctd(spatial, reference, mode='doublet', config=config, batch_size=10000)
        receipt['rctd_seconds'] = time.monotonic() - began
        np.savez_compressed(out / 'rctd_doublet.npz', **{k: np.asarray(v) for k, v in result._asdict().items() if v is not None})
        rows = np.flatnonzero(result.pixel_mask)
        weights = np.asarray(result.weights, dtype=np.float64)
        assert len(rows) == len(weights) and np.isfinite(weights).all() and (weights >= 0).all()
        assert (weights.sum(1) > 0).all()
        weights /= weights.sum(1, keepdims=True)
        np.testing.assert_array_equal(result.cell_type_names, types)
        # Consume the random draws in the original type order to preserve the previous sampling rule.
        rng = np.random.default_rng(20250502); selection = None
        for c, ct in enumerate(types):
            eligible = ((np.isin(result.spot_class, [1, 3]) & (result.first_type == c)) |
                        ((result.spot_class == 2) & ((result.first_type == c) | (result.second_type == c))))
            ii = np.flatnonzero(eligible); mass = float(weights[ii, c].sum())
            if not len(ii) or mass <= 100:
                continue
            selected = np.sort(rng.choice(ii, 10000, replace=False)) if len(ii) > 10000 else ii
            if str(ct) == 'Epithelial':
                selection = selected
                receipt.update(eligible_epithelial_cells=len(ii), target_weight_sum=mass)
        assert selection is not None, 'Insufficient RCTD epithelial support'
        selected_rows = rows[selection]
        nominal_genes = genes.copy()
        epithelial = list(types).index('Epithelial')
        supported = eta0[epithelial] > 0
        removed = genes[~supported]
        counts = sparse.csr_matrix(spatial[selected_rows, list(genes)].X).toarray().astype(np.float32)
        removed_umis = counts[:, ~supported].sum(0).tolist()
        removed_positive_cells = (counts[:, ~supported] > 0).sum(0).tolist()
        counts = counts[:, supported]
        genes = genes[supported]
        eta0 = eta0[epithelial:epithelial+1, supported].copy()
        eta0 /= eta0.sum(1, keepdims=True)
        assert len(genes) == 1341
        pd.DataFrame(dict(gene=removed, observed_UMIs=removed_umis,
                         observed_positive_cells=removed_positive_cells)).to_csv(out / 'excluded_zero_reference_genes.csv', index=False)
        libraries = counts.sum(1, dtype=np.float64)
        positive = libraries > 0
        receipt['zero_panel_cells_removed'] = int((~positive).sum())
        counts = counts[positive]; libraries = libraries[positive]
        selection = selection[positive]; selected_rows = selected_rows[positive]
        assert len(counts) and np.isfinite(counts).all() and (counts >= 0).all()
        payload = {k: (v if k == 'cell_types' else v[selected_rows]) for k, v in meta.items()}
        payload.update(Y=counts, genes=genes, eta0=eta0, library_size=libraries,
            cell_types=np.asarray(['Epithelial']), pi=np.ones((len(counts),1), dtype=np.float32),
            rctd_initial_pi=weights[selection].astype(np.float32), rctd_cell_types=types,
            nominal_genes=nominal_genes, removed_zero_reference_genes=removed, target_type=np.asarray('Epithelial'),
            xenium_input_rows=selected_rows, rctd_spot_class=result.spot_class[selection])
        np.testing.assert_allclose(payload['pi'].sum(1), 1, atol=1e-6)
        np.testing.assert_allclose(eta0.sum(1), 1, atol=1e-6)
        np.savez_compressed(out / 'data.npz', **payload)
        pd.DataFrame(dict(xenium_input_row=selected_rows, original_row=payload['original_indices'],
            spot_name=payload['spot_names'], spot_class=payload['rctd_spot_class'])).to_csv(out / 'selected_cells.csv', index=False)
        receipt.update(status='completed', N_selected=len(counts), nominal_G=len(nominal_genes), G=len(genes), C=1,
            pi_policy='Exact ones, not RCTD weights; RCTD weights retained only as separate audit arrays',
            reference_zero_genes_removed=removed.tolist(), reference_zero_gene_count=len(removed),
            RCTD_C=len(types),
            input_path=str(out / 'data.npz'), input_sha256=sha(out / 'data.npz'),
            rctd_retained=len(rows), spot_class_counts={str(int(k)): int(v) for k, v in zip(*np.unique(result.spot_class, return_counts=True))})
    except Exception as error:
        receipt.update(status='failed', error=repr(error)); raise
    finally:
        receipt.update(total_seconds=time.monotonic() - started, finish_utc=datetime.now(timezone.utc).isoformat(),
            peak_gpu_bytes=torch.cuda.max_memory_allocated())
        write(out / 'run_manifest.json', receipt)
    print('RCTD_COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
