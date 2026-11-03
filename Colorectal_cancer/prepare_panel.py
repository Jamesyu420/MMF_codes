"""Prepare the selected 1343-gene HD input; keep existing experiments and inputs intact."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from Colorectal_cancer.io_utils import sha, write


def initial_pi(y, profiles):
    profiles = profiles.astype(np.float64)
    inverse = np.linalg.pinv(profiles @ profiles.T, hermitian=True)
    pi = np.empty((len(y), len(profiles)), dtype=np.float32)
    for first in range(0, len(y), 4096):
        block = y[first:first + 4096].astype(np.float64)
        weights = ((block @ profiles.T) / block.sum(1, keepdims=True)) @ inverse
        weights = np.maximum(weights, 1e-3)
        pi[first:first + len(block)] = weights / weights.sum(1, keepdims=True)
    return pi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepared-data',type=Path,required=True)
    parser.add_argument('--hd',type=Path)
    parser.add_argument('--xenium',type=Path)
    parser.add_argument('--reference',type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status = dict(status='preparing', started_utc=datetime.now(timezone.utc).isoformat())
    write(root / 'preparation_status.json', status)
    try:
        home = Path.home()
        current = args.prepared_data
        import json
        source = json.loads((current / 'preprocess_manifest.json').read_text())
        if args.hd: source['sources']['HD']['source']=str(args.hd)
        if args.xenium: source['sources']['Xenium']['source']=str(args.xenium)
        if args.reference: source['reference']=str(args.reference)
        nominations = pd.read_csv(current / 'gene_nominations.csv')
        saved_filter = pd.read_csv(current / 'gene_filtering.csv').set_index('gene')
        candidates = {}
        for quota in (100, 400):
            rows = nominations[nominations['rank'] <= quota]
            assert (rows.groupby(['cell_type', 'method']).size() == quota).all()
            candidates[quota] = np.array(sorted(rows.gene.unique()), dtype=str)
        with np.load(current / 'HD_data.npz', allow_pickle=False) as z:
            metadata = {k: z[k] for k in ('coords', 'original_coords', 'original_indices',
                         'spot_names', 'total_counts', 'cell_types')}
        assert len(metadata['spot_names']) == 36503
        input_universe = np.array(sorted(set(candidates[100])), dtype=str)
        hd_counts = np.empty((36503, len(input_universe)), dtype=np.float32)
        stats = {k: {} for k in candidates}
        cohorts = {}
        for assay in ('HD', 'Xenium'):
            with np.load(current / (assay + '_data.npz'), allow_pickle=False) as z:
                retained = z['original_indices'].copy()
            with h5py.File(source['sources'][assay]['source'], 'r') as f:
                names = f['var/_index'].asstr()[:]
                x = f['X']; shape = tuple(x.attrs['shape']); indptr = x['indptr'][:]
                keep = np.zeros(shape[0], dtype=bool); keep[retained] = True
                columns = pd.Index(names).get_indexer(candidates[400])
                subcolumns = {k: pd.Index(candidates[400]).get_indexer(v) for k, v in candidates.items()}
                input_columns = pd.Index(names).get_indexer(input_universe)
                assert (columns >= 0).all()
                if assay == 'HD':
                    assert (input_columns >= 0).all()
                accum = {k: dict(detected=np.zeros(len(v), dtype=np.int64),
                                mean=np.zeros(len(v), dtype=np.float64)) for k, v in candidates.items()}
                seen = 0
                for first in range(0, shape[0], 4096):
                    last = min(first + 4096, shape[0]); mask = keep[first:last]
                    if not mask.any():
                        continue
                    lo, hi = int(indptr[first]), int(indptr[last])
                    raw = sparse.csr_matrix((x['data'][lo:hi], x['indices'][lo:hi],
                          indptr[first:last + 1] - lo), shape=(last - first, shape[1]))
                    raw.sum_duplicates(); raw.eliminate_zeros(); raw = raw[mask]
                    if assay == 'HD':
                        hd_counts[seen:seen + len(raw.indptr) - 1] = raw[:, input_columns].toarray()
                    block = raw[:, columns].astype(np.float64).tocsr()
                    seen += int(mask.sum())
                    for quota in candidates:
                        counts = block if quota == 400 else block[:, subcolumns[quota]]
                        libraries = np.asarray(counts.sum(1)).ravel()
                        accum[quota]['detected'] += np.asarray((counts > 0).sum(0)).ravel()
                        accum[quota]['mean'] += np.asarray(counts.multiply(
                            1 / np.maximum(libraries[:, None], 1)).sum(0)).ravel()
                assert seen == len(retained)
                for quota in candidates:
                    accum[quota]['mean'] /= seen
                    stats[quota][assay] = accum[quota]
                expected = saved_filter.loc[candidates[400]]
                np.testing.assert_array_equal(accum[400]['detected'], expected['detected_' + assay])
                np.testing.assert_allclose(accum[400]['mean'], expected['mean_relative_expression_' + assay],
                                           rtol=1e-10, atol=1e-14)
                cohorts[assay] = seen
            print('EXPRESSION_FILTER_REPRODUCED', assay, seen, flush=True)
        panels = {}
        summaries = {}
        for quota in (100, 400):
            frame = pd.DataFrame(dict(gene=candidates[quota]))
            for assay in ('HD', 'Xenium'):
                frame['detected_' + assay] = stats[quota][assay]['detected']
                frame['mean_relative_expression_' + assay] = stats[quota][assay]['mean']
            detected = frame[['detected_HD', 'detected_Xenium']].min(axis=1) >= 10
            expressed = frame[['mean_relative_expression_HD', 'mean_relative_expression_Xenium']].min(axis=1) > 5e-5
            frame['in_panel'] = detected & expressed & saved_filter.loc[frame.gene, 'reference_supported'].to_numpy()
            selected = frame.loc[frame.in_panel, 'gene'].to_numpy(dtype=str)
            if quota == 400:
                np.testing.assert_array_equal(selected, saved_filter.index[saved_filter.in_panel].to_numpy())
                continue
            label = 'dehvg' + str(quota)
            panels[label] = selected
            dest = root / 'data' / label; dest.mkdir(parents=True, exist_ok=False)
            frame.to_csv(dest / 'gene_filtering.csv', index=False)
            nominations[nominations['rank'] <= quota].to_csv(dest / 'gene_nominations.csv', index=False)
            summaries[label] = dict(per_type=quota, nomination_entries=28 * quota,
                                   candidate_union=len(frame), passing_detection=int(detected.sum()), G=len(selected))
        assert len(panels['dehvg100']) == 1343
        assert __import__('hashlib').sha256('\n'.join(panels['dehvg100']).encode()).hexdigest() == '136899a9a7fb7924dcad47c62b7c466bb6d0a2ac3e705e3e830103e36b77e743'
        reference = ad.read_h5ad(source['reference'])
        fine = reference.obs['major_annotation'].astype(str).to_numpy()
        labels = np.where(np.isin(fine, ['mregDC', 'cDC2', 'pDC']), 'DC', fine)
        types = metadata['cell_types']; np.testing.assert_array_equal(types, sorted(set(labels)))
        for label, genes in panels.items():
            dest = root / 'data' / label; dest.mkdir(parents=True, exist_ok=True)
            counts = sparse.csr_matrix(reference[:, list(genes)].X).astype(np.float64)
            libs = np.asarray(counts.sum(1)).ravel()
            assert (libs > 0).all(), 'Zero-panel reference cell: ' + label
            normalized = counts.multiply(1 / libs[:, None]).tocsr()
            eta0 = np.stack([np.asarray(normalized[labels == t].mean(0)).ravel() for t in types]).astype(np.float32)
            np.testing.assert_allclose(eta0.sum(1), 1, atol=1e-6)
            y = hd_counts[:, pd.Index(input_universe).get_indexer(genes)]
            library = y.sum(1, dtype=np.float64)
            assert (library > 0).all() and np.isfinite(eta0).all()
            payload = dict(metadata, Y=y, genes=genes, eta0=eta0, library_size=library,
                           pi=initial_pi(y, eta0))
            np.savez_compressed(dest / 'HD_data.npz', **payload)
            pd.DataFrame(dict(gene=genes)).to_csv(dest / 'genes.csv', index=False)
            summaries[label].update(N=len(y), C=len(types), input=str(dest / 'HD_data.npz'),
                input_sha256=sha(dest / 'HD_data.npz'), gene_order_sha256=__import__('hashlib').sha256(
                    '\n'.join(genes).encode()).hexdigest(), zero_count_bins=0, zero_reference_cells=0)
            write(dest / 'preprocess_manifest.json', summaries[label])
            print('PANEL_READY', label, summaries[label], flush=True)
        status.update(status='completed', panels=summaries, cohorts=cohorts,
            raw_sources=source['sources'], reference=source['reference'], reference_sha256=source['reference_sha256'],
            source_nomination_sha256=sha(current / 'gene_nominations.csv'),
            count_policy='All observed UMI; no molecule splitting; library sizes and reference fractions normalized to each final panel',
            shared_cohort=True, shared_coordinates=True, source_sha256=sha(__file__),
            elapsed_seconds=time.monotonic() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        write(root / 'preparation_status.json', status)
    except Exception as error:
        status.update(status='failed', error=repr(error), elapsed_seconds=time.monotonic() - started)
        write(root / 'preparation_status.json', status)
        raise


if __name__ == '__main__':
    main()
