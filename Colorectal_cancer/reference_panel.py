"""Reference-only gene nominations, using the verified Visium 400+400 definitions."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t as student_t

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

def moments(matrix):
    n = matrix.shape[0]
    mean = np.asarray(matrix.mean(0)).ravel()
    variance = np.maximum(np.asarray(matrix.power(2).mean(0)).ravel() - mean**2, 0)
    if n > 1:
        variance *= n / (n - 1)
    return mean, variance

def hvg_scores(mean, variance, eligible, bins=20):
    """Seurat-style log dispersion standardized within equal-width mean bins.

    Uses CP10k linear-scale moments, log(mean+1), log(variance/mean), and
    sample-SD scaling. A singleton mean bin receives normalized score 1.
    """
    scores = np.full(len(mean), -np.inf)
    idx = np.flatnonzero(eligible & (mean > 0) & (variance > 0))
    logged_mean = np.log1p(mean[idx])
    log_dispersion = np.log(variance[idx] / mean[idx])
    groups = pd.cut(logged_mean, bins=bins, labels=False, duplicates="drop")
    for group in np.unique(groups):
        take = np.flatnonzero(groups == group)
        values = log_dispersion[take]
        if len(take) == 1:
            scores[idx[take]] = 1.
        else:
            sd = values.std(ddof=1)
            scores[idx[take]] = (values - values.mean()) / sd if sd > 0 else 0.
    return scores


def nominate(counts, genes, labels, types, per_type=400):
    libraries = np.asarray(counts.sum(1)).ravel()
    if np.any(libraries <= 0):
        raise ValueError('Reference cell has no common-universe counts')
    normalized = counts.multiply(1e4 / libraries[:, None]).tocsr()
    lognorm = normalized.copy(); lognorm.data = np.log1p(lognorm.data)
    n = counts.shape[0]
    total_log = np.asarray(lognorm.sum(0)).ravel()
    total_log2 = np.asarray(lognorm.power(2).sum(0)).ravel()
    non_mt = np.array([not g.lower().startswith('mt-') for g in genes])
    records = []
    for celltype in types:
        take = labels == celltype; nc = int(take.sum()); nr = n-nc
        detected = np.asarray((counts[take] > 0).sum(0)).ravel()
        eligible = non_mt & (detected >= max(3, int(np.ceil(.05*nc))))
        mean, var = moments(lognorm[take])
        rest_mean = (total_log-mean*nc)/nr
        rest_var = np.maximum((total_log2-(var*(nc-1)+nc*mean**2))/nr-rest_mean**2, 0)*nr/(nr-1)
        denominator = var/nc+rest_var/nr
        tscore = np.divide(mean-rest_mean, np.sqrt(denominator), out=np.zeros_like(mean), where=denominator>0)
        dfden = (var/nc)**2/(nc-1)+(rest_var/nr)**2/(nr-1)
        df = np.divide(denominator**2, dfden, out=np.ones_like(mean), where=dfden>0)
        pvalues = 2*student_t.sf(np.abs(tscore), df)
        linear_mean, linear_var = moments(normalized[take])
        hvgs = hvg_scores(linear_mean, linear_var, eligible)
        for method, score, valid in [('DE', tscore, eligible & (tscore>0)), ('HVG', hvgs, eligible & np.isfinite(hvgs))]:
            available = np.flatnonzero(valid)
            chosen = available[np.argsort(-score[available], kind='stable')[:per_type]]
            if len(chosen) != per_type:
                raise ValueError(f'{celltype}: only {len(chosen)} eligible {method} genes')
            for rank, idx in enumerate(chosen, 1):
                records.append(dict(cell_type=str(celltype), method=method, rank=rank, gene=str(genes[idx]),
                                    score=float(score[idx]), reference_detected_cells=int(detected[idx]),
                                    de_pvalue=float(pvalues[idx]) if method=='DE' else None))
        print(celltype, nc, 'reference cells; 400 DE + 400 HVG nominated', flush=True)
    return pd.DataFrame(records)
