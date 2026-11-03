"""Build a common reference DE + within-type HVG panel for two Visium sections.

No spatial coordinates or fitted proportions enter gene ranking. UnknownLowQ
remains a mixture component but, as before, does not nominate program genes.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t as student_t

SECTIONS = ("Visium_1", "Visium_2")
MIN_COUNTS, MIN_GENES, MAX_MT = 1000, 500, .40


def analysis_type(name):
    if name in {"Ext_L23", "Ext_L25", "Ext_L56"}:
        return name
    if name in {"Ext_L5_1", "Ext_L5_2", "Ext_L5_3"}:
        return "Ext_L5"
    if name in {"Ext_L6", "Ext_L6B"}:
        return "Ext_L6"
    for prefix, group in [("Astro_", "Astrocyte"), ("Oligo_", "Oligodendrocyte"),
                          ("OPC_", "OPC"), ("Inh_", "Inhibitory"),
                          ("Nb_", "Neuroblast"), ("Ext_Hpc_", "HippocampalExc"),
                          ("Ext_Thal_", "ThalamicExc"), ("Ext_Amy_", "AmygdalarExc")]:
        if name.startswith(prefix):
            return group
    if name == "Micro":
        return "Microglia"
    if name == "Endo":
        return "Endothelial"
    if name.startswith("Ext_L"):
        raise ValueError(f"Unmapped cortical type: {name}")
    if name.startswith("Ext_"):
        return "OtherExc"
    if name.startswith(("Unk_", "LowQ_")):
        return "UnknownLowQ"
    raise ValueError(f"Unmapped reference type: {name}")


def read_strings(group, name):
    column = group[name]
    if isinstance(column, h5py.Group):
        return column["categories"].asstr()[:][column["codes"][:]]
    if column.dtype.kind in "iu" and "__categories" in group:
        return group["__categories"][name].asstr()[:][column[:]]
    return column.asstr()[:]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def load_section(raw, name):
    folder = raw / name.lower()
    path = folder / f"mouse_brain_{name.lower()}_counts.csv"
    frame = pd.read_csv(path, index_col=0)
    genes = frame.columns.to_numpy(dtype=str)
    if not frame.columns.is_unique or not frame.index.is_unique:
        raise ValueError("Visium genes and barcodes must be unique")
    counts = frame.to_numpy(dtype=np.int32)
    if (counts < 0).any() or not np.array_equal(counts, frame.to_numpy()):
        raise ValueError("Visium counts must be nonnegative integer UMIs")
    spots = frame.index.to_numpy(dtype=str)
    del frame
    coords = pd.read_csv(folder / f"mouse_brain_{name.lower()}_coordinates.csv", index_col=0)
    coords = coords.loc[spots, ["s_1", "s_2"]].to_numpy(dtype=np.float64)
    total = counts.sum(1, dtype=np.float64)
    detected = (counts > 0).sum(1)
    mt = np.array([g.lower().startswith("mt-") for g in genes])
    mt_fraction = counts[:, mt].sum(1) / np.maximum(total, 1)
    keep = (total >= MIN_COUNTS) & (detected >= MIN_GENES) & (mt_fraction <= MAX_MT)
    print(name, "raw", counts.shape, "QC spots", int(keep.sum()), flush=True)
    return dict(counts=counts[keep], genes=genes, spots=spots[keep], coords=coords[keep],
                original_indices=np.flatnonzero(keep), total_counts=total[keep],
                summary=dict(spots_before=len(keep), spots_after=int(keep.sum()),
                             raw_genes=len(genes), removed_spots=int((~keep).sum())),
                source=str(path), source_sha256=sha(path))


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


def initial_pi(counts, profiles):
    projected = (counts.astype(np.float64) @ profiles.T) / counts.sum(1, keepdims=True)
    weights = projected @ np.linalg.pinv(profiles @ profiles.T, hermitian=True)
    weights = np.maximum(weights, 1e-3)
    return (weights / weights.sum(1, keepdims=True)).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New directory; existing outputs are never replaced")
    p.add_argument("--per-type", type=int, default=400)
    p.add_argument("--min-spots", type=int, default=10)
    p.add_argument("--min-relative-expression", type=float, default=5e-5)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sections = {name: load_section(args.raw, name) for name in SECTIONS}
    ref_path = args.reference / "all_cells_20200625.h5ad"
    annotation_path = args.reference / "snRNA_annotation_astro_subtypes_refined59_20200823.csv"
    with h5py.File(ref_path) as f:
        x = f["X"]
        raw_shape = tuple(int(n) for n in x.attrs["shape"])
        counts = sparse.csr_matrix((x["data"][:], x["indices"][:], x["indptr"][:]), shape=raw_shape)
        symbols = read_strings(f["var"], "SYMBOL").astype(str)
        barcodes = read_strings(f["obs"], "_index").astype(str)
    annotation = pd.read_csv(annotation_path, index_col=0)["annotation_1"].reindex(barcodes)
    labelled = annotation.notna().to_numpy()
    fine = annotation[labelled].to_numpy(dtype=str)
    labels = np.array([analysis_type(t) for t in fine])
    types = np.array(sorted(set(labels)), dtype=str)
    if len(types) != 17 or "CorticalExc" in types:
        raise ValueError(f"Expected the approved 17 groups, got {types}")
    common = sorted(set(symbols) & set(sections["Visium_1"]["genes"]) & set(sections["Visium_2"]["genes"]))
    genes = np.array(common, dtype=str)
    lookup = {g: i for i, g in enumerate(genes)}
    columns = np.array([i for i, g in enumerate(symbols) if g in lookup])
    merger = sparse.csr_matrix((np.ones(len(columns), dtype=np.float64),
                               (columns, [lookup[symbols[i]] for i in columns])),
                              shape=(len(symbols), len(genes)))
    counts = (counts[labelled] @ merger).tocsr()
    libraries = np.asarray(counts.sum(1)).ravel()
    if (libraries <= 0).any():
        raise ValueError("An annotated reference cell has zero common-universe counts")
    normalized = counts.multiply(1e4 / libraries[:, None]).tocsr()
    lognorm = normalized.copy()
    lognorm.data = np.log1p(lognorm.data)
    n = counts.shape[0]
    total_log = np.asarray(lognorm.sum(0)).ravel()
    total_log2 = np.asarray(lognorm.power(2).sum(0)).ravel()
    non_mt = np.array([not g.lower().startswith("mt-") for g in genes])
    nominations, type_records = [], []
    for celltype in types:
        take = labels == celltype
        nc = int(take.sum())
        type_records.append(dict(cell_type=celltype, reference_cells=nc,
                                 fine_types=sorted(set(fine[take]))))
        if celltype == "UnknownLowQ":
            continue
        detected = np.asarray((counts[take] > 0).sum(0)).ravel()
        eligible = non_mt & (detected >= max(3, int(np.ceil(.05 * nc))))
        mean, var = moments(lognorm[take])
        nr = n - nc
        rest_mean = (total_log - mean * nc) / nr
        rest_var = np.maximum((total_log2 - (var * (nc - 1) + nc * mean**2)) / nr - rest_mean**2, 0) * nr / (nr - 1)
        denominator = var / nc + rest_var / nr
        tscore = np.divide(mean - rest_mean, np.sqrt(denominator),
                           out=np.zeros_like(mean), where=denominator > 0)
        dfden = (var / nc)**2 / (nc - 1) + (rest_var / nr)**2 / (nr - 1)
        df = np.divide(denominator**2, dfden, out=np.ones_like(mean), where=dfden > 0)
        pvalue = 2 * student_t.sf(np.abs(tscore), df)
        linear_mean, linear_var = moments(normalized[take])
        hvgs = hvg_scores(linear_mean, linear_var, eligible)
        for method, score, valid in [("DE", tscore, eligible & (tscore > 0)),
                                     ("HVG", hvgs, eligible & np.isfinite(hvgs))]:
            available = np.flatnonzero(valid)
            order = available[np.argsort(-score[available], kind="stable")[:args.per_type]]
            if len(order) != args.per_type:
                raise ValueError(f"{celltype} has only {len(order)} eligible {method} genes")
            for rank, idx in enumerate(order, 1):
                nominations.append(dict(cell_type=celltype, method=method, rank=rank,
                                        gene=genes[idx], score=float(score[idx]),
                                        reference_detected_cells=int(detected[idx]),
                                        de_pvalue=float(pvalue[idx]) if method == "DE" else None))
        print(celltype, nc, "cells; nominated", args.per_type, "DE +", args.per_type, "HVG", flush=True)
    nominated = sorted({row["gene"] for row in nominations})
    index = np.array([lookup[g] for g in nominated])
    supported = np.asarray(counts[:, index].sum(0)).ravel() > 0
    gene_qc = pd.DataFrame(dict(gene=nominated, reference_supported=supported))
    for name, data in sections.items():
        st_index = pd.Index(data["genes"]).get_indexer(nominated)
        selected_counts = data["counts"][:, st_index]
        gene_qc[f"detected_{name}"] = (selected_counts > 0).sum(0)
        candidate_library = selected_counts.sum(1, dtype=np.float64)
        gene_qc[f"mean_relative_expression_{name}"] = (
            selected_counts / np.maximum(candidate_library[:, None], 1)).mean(0)
    gene_qc["passes_detection_both"] = gene_qc[[f"detected_{name}" for name in SECTIONS]].min(axis=1) >= args.min_spots
    gene_qc["passes_mean_expression_both"] = gene_qc[[f"mean_relative_expression_{name}" for name in SECTIONS]].min(axis=1) > args.min_relative_expression
    gene_qc["in_panel"] = supported & gene_qc.passes_detection_both & gene_qc.passes_mean_expression_both
    panel = gene_qc.loc[gene_qc.in_panel, "gene"].to_numpy(dtype=str)
    panel_index = np.array([lookup[g] for g in panel])
    panel_counts = counts[:, panel_index]
    panel_lib = np.asarray(panel_counts.sum(1)).ravel()
    positive_ref = panel_lib > 0
    norm_panel = panel_counts[positive_ref].multiply(1 / panel_lib[positive_ref, None]).tocsr()
    eta0 = np.stack([np.asarray(norm_panel[labels[positive_ref] == c].mean(0)).ravel() for c in types]).astype(np.float32)
    if not np.isfinite(eta0).all() or not np.allclose(eta0.sum(1), 1, atol=1e-6):
        raise ValueError("Reference panel means must be finite and sum to one")
    selected = pd.DataFrame(nominations)
    selected["in_panel"] = selected.gene.isin(set(panel))
    selected.to_csv(args.output / "gene_nominations.csv", index=False)
    gene_qc.to_csv(args.output / "gene_filtering.csv", index=False)
    mapping = pd.DataFrame(dict(fine_type=fine, cell_type=labels)).value_counts().rename("reference_cells").reset_index()
    mapping.to_csv(args.output / "cell_type_mapping.csv", index=False)
    pd.DataFrame(dict(gene=panel)).to_csv(args.output / "genes.csv", index=False)
    selected.groupby(["cell_type", "method"]).agg(nominated=("gene", "size"), retained=("in_panel", "sum")).to_csv(args.output / "selection_counts.csv")
    outputs = []
    for name, data in sections.items():
        st_index = pd.Index(data["genes"]).get_indexer(panel)
        observed = data["counts"][:, st_index]
        positive = observed.sum(1) > 0
        observed = observed[positive]
        original = data["coords"][positive]
        origin, scale = original.min(0), float(np.ptp(original, axis=0).max())
        payload = dict(Y=observed.astype(np.float32), eta0=eta0, pi=initial_pi(observed, eta0),
                       coords=((original - origin) / scale).astype(np.float32),
                       library_size=observed.sum(1, dtype=np.float64), genes=panel, cell_types=types,
                       original_indices=data["original_indices"][positive].astype(np.int64),
                       original_coords=original, spot_names=data["spots"][positive],
                       K_list=np.array([0 if c == "UnknownLowQ" else 1 for c in types]),
                       total_counts=data["total_counts"][positive])
        assert np.array_equal(payload["library_size"], payload["Y"].sum(1, dtype=np.float64))
        destination = args.output / f"{name}.npz"
        np.savez_compressed(destination, **payload)
        outputs.append(dict(section=name, spots=len(observed), genes=len(panel),
                            dropped_zero_panel=int((~positive).sum()), file=str(destination),
                            sha256=sha(destination)))
        print(name, "prepared", observed.shape, flush=True)
    manifest = dict(schema="MMF_mouse_brain_reference_panel_v1",
                    created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                    code_sha256=sha(__file__), reference=str(ref_path), reference_sha256=sha(ref_path),
                    annotation_sha256=sha(annotation_path), raw_reference_shape=raw_shape,
                    reference_cells=len(labels), zero_panel_reference_cells=int((~positive_ref).sum()),
                    common_universe_genes=len(genes), cell_types=types.tolist(), type_mapping=type_records,
                    cortex_policy="Keep Ext_L23, Ext_L25, Ext_L56; merge L5_1/2/3 as Ext_L5 and L6/L6B as Ext_L6; no CorticalExc.",
                    selection=dict(per_type=args.per_type, discovery_types=16,
                                   DE="Positive Welch t statistic on reference log1p(CP10k), one type versus all other annotated cells; rank selection, no significance claim.",
                                   HVG="Within-type CP10k log dispersion, standardized in 20 equal-width log1p(mean) bins; singleton bin score 1.",
                                   candidate_detection="At least max(3, ceil(5% of type cells)); no mitochondrial genes.",
                                   normalization="Per-cell UMI over all common genes before marker/HVG ranking.",
                                   unknown_policy="UnknownLowQ remains in mixture and one-v-rest background, does not nominate genes or programs.",
                                   nomination_union=len(nominated), final_panel=len(panel)),
                    qc=dict(min_counts=MIN_COUNTS, min_genes=MIN_GENES, max_mitochondrial_fraction=MAX_MT,
                            gene_min_spots=args.min_spots, gene_spot_rule="Both sections separately after spot QC",
                            gene_mean_relative_expression_threshold=args.min_relative_expression,
                            gene_mean_expression_rule="Both sections separately must exceed threshold; per-spot denominator is UMI over the nomination union, before final gene filtering.",
                            reference_rule="Positive reference support", spatial_coordinates_used_for_selection=False),
                    count_policy="Full observed raw UMI; no molecule split or heldout partition.",
                    eta0="Mean per-cell fractions on final panel within each approved type; zero-panel cells excluded; rows sum to one.",
                    pi="Positive normalized reference projection for initialization only; MMF jointly estimates mixing weights.",
                    sections={name:dict(**data["summary"], source=data["source"], source_sha256=data["source_sha256"]) for name,data in sections.items()},
                    outputs=outputs)
    write_json(args.output / "preprocess_manifest.json", manifest)
    print("COMPLETE", json.dumps(dict(types=len(types), nominated=len(nominated), panel=len(panel))), flush=True)


if __name__ == "__main__":
    main()
