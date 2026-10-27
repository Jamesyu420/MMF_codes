"""Post-fit comparisons. These functions do not open truth during model fitting."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

TRUE_K = np.array([1, 2, 0, 1, 1])
SIM_METHODS = ["MMF", "RCTD", "STAMP", "SpiceMix", "NMF"]


def corr(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else 0.


def match(a, b):
    if not len(a) or not len(b):
        return []
    sim = np.array([[corr(x, y) for y in b] for x in a])
    ii, jj = linear_sum_assignment(-sim)
    return [(int(i), int(j), float(sim[i, j])) for i, j in zip(ii, jj)]


def assignment(wt, w, ct=None, c=None):
    out = {}
    groups = [(np.arange(len(wt)), np.arange(len(w)))] if ct is None else [
        (np.flatnonzero(ct == t), np.flatnonzero(c == t)) for t in sorted(set(ct))]
    for ti, fi in groups:
        if not len(ti) or not len(fi):
            continue
        score = np.array([[corr(wt[i], w[j]) for j in fi] for i in ti])
        rr, cc = linear_sum_assignment(-score)
        out.update({int(ti[i]): int(fi[j]) for i, j in zip(rr, cc)})
    return out


def RNA_share(pi, eta, h, w, types, gamma, hvar=None, gvar=None):
    with np.errstate(divide="ignore"):
        value = np.log(pi.T[:, :, None]) + np.log(eta[:, None, :]) + gamma[None, None, :]
    if gvar is not None:
        value = value + .5 * gvar[None, None, :]
    for p, c in enumerate(types):
        value[int(c)] += h[p, :, None] * w[p, None, :]
        if hvar is not None:
            value[int(c)] += .5 * hvar[p, :, None] * w[p, None, :] ** 2
    totals = logsumexp(value, axis=2).T
    return np.exp(totals - logsumexp(totals, axis=1, keepdims=True))


def one_case(seed, method, truth, fit, seconds, host, device):
    raw = fit["pi"] if method in ["MMF", "RCTD"] else fit["activity"]
    pa = {c: c for c in range(5)}
    if method not in ["MMF", "RCTD"]:
        rr, cc = linear_sum_assignment(-np.array([
            [corr(truth["pi_true"][:, c], raw[:, j]) for j in range(raw.shape[1])] for c in range(5)]))
        pa = dict(zip(map(int, rr), map(int, cc)))
    prows = [dict(seed=seed, method=method, type=c + 1, factor=pa[c],
                  correlation=corr(truth["pi_true"][:, c], raw[:, pa[c]])) for c in range(5)]
    row = dict(seed=seed, method=method,
               composition_correlation=float(np.mean([x["correlation"] for x in prows])),
               seconds=seconds, host=host, device=device)
    if method in ["MMF", "RCTD"]:
        row["composition_RMSE"] = float(np.sqrt(np.mean((fit["pi"] - truth["pi_true"]) ** 2)))
    if method == "RCTD":
        return row, prows, []
    pairs = assignment(truth["loadings_true"], fit["loadings"],
                       truth["program_cell_type"] if method == "MMF" else None,
                       fit.get("program_cell_type"))
    h = fit["h_mean"] if method == "MMF" else fit["activity"].T
    rows = []
    for p in range(len(truth["loadings_true"])):
        q = pairs.get(p)
        rec = dict(seed=seed, method=method, true_program=p,
                   type=int(truth["program_cell_type"][p]) + 1,
                   within_type=int(truth["program_index"][p]) + 1, factor=q,
                   matched=q is not None,
                   loading_cosine=0. if q is None else corr(truth["loadings_true"][p], fit["loadings"][q]),
                   activity_correlation=0. if q is None else corr(truth["h_true"][p], h[q]))
        if method == "MMF" and q is not None:
            error = h[q] - truth["h_true"][p]
            rec.update(native_H_RMSE=float(np.sqrt(np.mean(error ** 2))),
                       native_H_relative_error=float(np.linalg.norm(error) / np.linalg.norm(truth["h_true"][p])),
                       native_H90_coverage=float(np.mean(np.abs(error) <= 1.6448536269514722 * fit["h_sd"][q])))
        rows.append(rec)
    row.update(loading_cosine=float(np.mean([x["loading_cosine"] for x in rows])),
               activity_correlation=float(np.mean([x["activity_correlation"] for x in rows])),
               matched_programs=len(pairs))
    if method == "MMF":
        for key in ["native_H_RMSE", "native_H_relative_error", "native_H90_coverage"]:
            values = [x[key] for x in rows if key in x]
            row[key + "_matched"] = float(np.mean(values)) if values else None
        row.update(epsilon_variance=float(fit["epsilon_prior_variance"]),
                   epsilon_correlation=corr(fit["epsilon_mean"], truth["epsilon_true"]),
                   epsilon90_coverage=float(np.mean(
                       np.abs(fit["epsilon_mean"] - truth["epsilon_true"]) <= 1.6448536269514722 * fit["epsilon_sd"])))
    return row, prows, rows


def partial_rank_residual(x, pi, library):
    ranked = pd.Series(x).rank(method="average").to_numpy()
    design = np.column_stack([
        np.ones(len(ranked)),
        pd.Series(pi).rank(method="average"),
        pd.Series(library).rank(method="average"),
    ])
    return ranked - design @ np.linalg.lstsq(design, ranked, rcond=None)[0]
