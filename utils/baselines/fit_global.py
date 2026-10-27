"""Run official global baselines on training counts only; never read truth."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import signal
import sys
import time
import types
import warnings

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

# Process-local aliases restore the exact old NumPy API used by unchanged official sources.
np.Inf = np.inf
np.float = float
np.int = int


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def graph(coords):
    nn = NearestNeighbors(n_neighbors=11).fit(coords).kneighbors(coords,return_distance=False)
    edges = {(int(i), int(j)) for i, ids in enumerate(nn) for j in ids if i != j}
    edges |= {(j, i) for i, j in edges}
    edges = np.asarray(sorted(edges), dtype=np.int64)
    adjacency = sparse.csr_matrix((np.ones(len(edges), dtype=np.float32), edges.T),
        shape=(len(coords), len(coords)))
    return edges, adjacency


def fit(args):
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('6900-second fit budget')))
    signal.alarm(6900)
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.input, allow_pickle=False) as z:
        counts = z['Y'].astype(np.float64)
        coords = z['coords'].astype(np.float64)
    assert np.isfinite(counts).all() and (counts >= 0).all() and (counts == np.floor(counts)).all()
    assert (counts.sum(1) > 0).all()
    random.seed(args.seed)
    np.random.seed(args.seed)
    edges, adjacency = graph(coords)
    normalized = np.log1p(10000 * counts / counts.sum(1, keepdims=True))
    report = dict(method=args.method, rank=args.rank, seed=args.seed, input=str(args.input.resolve()),
        input_sha256=digest(args.input), truth_accessed=False,
        reference_profiles_accessed=False, known_cell_types_accessed=False,
        n_positions=len(counts), n_genes=counts.shape[1],
        rank_status='Fixed oracle-assisted global capacity; no per-type rank inference',
        graph='Symmetrized 10-nearest-neighbor graph from supplied coordinates, no expression edges',
        graph_directed_edges=len(edges))
    report['compatibility'] = 'Process-local np.Inf=np.inf, np.float=float, np.int=int; official source and numerical algorithms unchanged; no package upgrades'
    if args.method == 'NMF':
        import sklearn
        from sklearn.decomposition import NMF
        model = NMF(n_components=args.rank, init='nndsvda', solver='cd',
            beta_loss='frobenius', tol=1e-4, max_iter=2000, random_state=args.seed)
        with warnings.catch_warnings(record=True) as captured:
            activity = model.fit_transform(normalized)
        loading = model.components_
        report.update(implementation='scikit-learn.decomposition.NMF', sklearn_version=sklearn.__version__,
            preprocessing='log1p(10000 * training counts / observed selected-gene training total)',
            iterations=int(model.n_iter_), max_iterations=2000, tolerance=1e-4,
            stopping_status='native_tolerance' if model.n_iter_ < 2000 else 'iteration_limit',
            warnings=[str(w.message) for w in captured],
            reconstruction_error=float(model.reconstruction_err_), loading_semantics='Nonnegative additive gene weights')
    else:
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        report.update(torch_version=torch.__version__, device=args.device)
        if args.method == 'STAMP':
            import anndata as ad
            import pyro
            # Skip eager, unrelated plotting/enrichment imports; every model file is unchanged official source.
            package = types.ModuleType('sctm')
            package.__path__ = [str(args.vendor/'scTM/sctm')]
            sys.modules['sctm'] = package
            from sctm.stamp import STAMP
            adata = ad.AnnData(sparse.csr_matrix(counts.astype(np.float32)))
            adata.obsm['spatial'] = coords
            adata.obsp['spatial_connectivities'] = adjacency
            model = STAMP(adata, n_topics=args.rank, n_layers=1, hidden_size=128,
                train_size=1, gene_likelihood='nb', mode='sign')
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                model.train(max_epochs=800, min_epochs=100, learning_rate=.01,
                    device=args.device, batch_size=256, early_stop=True, patience=20,
                    num_particles=1)
            print(stdout.getvalue())
            activity = model.get_cell_by_topic(device=args.device).to_numpy()
            loading = model.get_feature_by_topic(device=args.device, pseudocount=0).to_numpy().T
            model.save(str(args.output/'native_parameter_store.pt'))
            torch.save(model.model.state_dict(), args.output/'native_model_state.pt')
            report.update(implementation='Official JinmiaoChenLab/scTM STAMP',
                source_commit='0099ab268c712c91f398a2862db59b1dc793f69b',
                preprocessing='Raw integer training counts; native NB likelihood and graph encoder',
                stopping_status='native_training_early_stop' if 'Early Stopping' in stdout.getvalue() else 'published_fixed_epochs_completed',
                max_epochs=800, early_stop_patience=20, minimum_epochs=100, learning_rate=.01,
                pyro_version=pyro.__version__, numpy_version=np.__version__,
                loading_semantics='Native signed, shrinkage-adjusted log topic gene module weights, pseudocount=0',
                activity_semantics='Native cell-topic proportions, summing to one at each position',
                source_files={str(p.relative_to(args.vendor)):digest(p) for p in (args.vendor/'scTM/sctm').glob('*.py')})
        elif args.method == 'SpiceMix':
            sys.path.insert(0,str(args.vendor/'SpiceMix/SpiceMix'))
            from model import SpiceMix
            files = args.output/'input/files'
            files.mkdir(parents=True,exist_ok=True)
            np.savetxt(files/'expression_0.txt',normalized,delimiter='\t')
            np.savetxt(files/'neighborhood_0.txt',edges,fmt='%d',delimiter='\t')
            (files/'genes_0.txt').write_text('\n'.join(f'gene{g}' for g in range(counts.shape[1]))+'\n')
            np.savetxt(files/'meta_0.csv',coords,delimiter=',',header='coor X,coor Y',comments='')
            context=dict(device=args.device,dtype=torch.float64)
            model = SpiceMix(K=args.rank,lambda_Sigma_x_inv=1e-6,power_Sigma_x_inv=2,
                repli_list=['0'],betas=np.ones(1),context=context,context_Y=context,
                path2result=args.output/'native_output.h5')
            model.load_dataset(args.output/'input')
            model.initialize(method='kmeans',random_state=args.seed,kwargs={})
            for iteration in range(10):
                index=0 if iteration==9 else -1
                model.estimate_weights(iiter=index,use_spatial=[False])
                model.estimate_parameters(iiter=index,use_spatial=[False])
            model.initialize_Sigma_x_inv()
            for iteration in range(1,201):
                model.estimate_parameters(iiter=iteration,use_spatial=[True])
                model.estimate_weights(iiter=iteration,use_spatial=[True])
                if iteration%10==0:print(f'SpiceMix iteration {iteration}/200',flush=True)
            activity=model.Xs[0].detach().cpu().numpy()
            loading=model.M.detach().cpu().numpy().T
            report.update(implementation='Official ma-compbio/SpiceMix GPU revision',
                source_commit='aea69f8081f33e8ca64b734c680804db907ed4eb',
                preprocessing='log1p(10000 * training counts / training total), then native K/mean expression-total scaling',
                initialization='Official supported kmeans initialization, then 10 nonspatial NMF initialization steps',
                stopping_status='published_fixed_iterations_completed', iterations=200,
                loading_semantics='Native nonnegative simplex metagenes',
                activity_semantics='Native additive metagene state matrix X',
                source_files={str(p.relative_to(args.vendor)):digest(p) for p in (args.vendor/'SpiceMix/SpiceMix').glob('*.py')})
    assert activity.shape==(len(counts),args.rank) and loading.shape==(args.rank,counts.shape[1])
    assert np.isfinite(activity).all() and np.isfinite(loading).all()
    np.savez_compressed(args.output/'factors.npz',activity=activity,loadings=loading,
        factor_ids=np.asarray([f'factor{i}' for i in range(args.rank)],dtype=str))
    report.update(elapsed_preparation_fit_export_seconds=time.monotonic()-started,
        all_factors_retained=True, factor_selection=False, status='completed')
    (args.output/'fit.json').write_text(json.dumps(report,indent=2)+'\n')
    signal.alarm(0)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--vendor',type=Path,required=True)
    parser.add_argument('--method',choices=['NMF','STAMP','SpiceMix'],required=True)
    parser.add_argument('--rank',type=int,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--device',default='cuda:0')
    fit(parser.parse_args())
