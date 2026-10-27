"""Match programs across the two Visium sections and write the section tables."""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils.io import load_npz, load_spatial, repo_root
from utils.metrics import match

ROOT = repo_root()
RESULTS = ROOT / "Mouse_brain/results"
OUT = ROOT / "Mouse_brain/analysis"
SECTIONS = ("Visium_1", "Visium_2")


def program_rows(section, spatial, fit):
    genes, types = spatial["genes"], spatial["cell_types"]
    rows, genes_out = [], []
    for p, (c, k) in enumerate(zip(fit["program_cell_type"], fit["program_index"])):
        weights = fit["loadings"][p]
        order = np.argsort(-weights, kind="stable")
        visible = fit["pi"][:, c] >= .05
        rho = float(spearmanr(fit["pi"][visible, c], fit["h_mean"][p, visible]).statistic) if visible.sum() > 2 else None
        rows.append(dict(section=section, cell_type=str(types[c]), program_index=int(k) + 1, flat_index=p,
                         n_visible=int(visible.sum()), composition_activity_spearman=rho,
                         top_genes=";".join(genes[order[:15]])))
        for rank, g in enumerate(order):
            genes_out.append(dict(section=section, cell_type=str(types[c]), program_index=int(k) + 1,
                                  flat_index=p, gene=str(genes[g]), weight=float(weights[g]), gene_rank=rank + 1))
    return rows, genes_out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spatial, fits, programs, genes = {}, {}, [], []
    for section in SECTIONS:
        spatial[section] = load_spatial(ROOT / "data/mouse_brain" / f"{section}.npz")
        fits[section] = load_npz(RESULTS / section / "final.npz")
        rows, gene_rows = program_rows(section, spatial[section], fits[section])
        programs.extend(rows)
        genes.extend(gene_rows)
    cross = []
    types = spatial["Visium_1"]["cell_types"]
    for c, name in enumerate(types):
        left = np.flatnonzero(fits["Visium_1"]["program_cell_type"] == c)
        right = np.flatnonzero(fits["Visium_2"]["program_cell_type"] == c)
        for i, j, score in match(fits["Visium_1"]["loadings"][left], fits["Visium_2"]["loadings"][right]):
            p, q = int(left[i]), int(right[j])
            cross.append(dict(
                cell_type=str(name), brain1_flat_index=p, brain2_flat_index=q,
                brain1_program=int(fits["Visium_1"]["program_index"][p]) + 1,
                brain2_program=int(fits["Visium_2"]["program_index"][q]) + 1,
                loading_correlation=score))
    pd.DataFrame(programs).to_csv(OUT / "programs.csv", index=False)
    pd.DataFrame(genes).to_csv(OUT / "gene_weights.csv", index=False)
    pd.DataFrame(cross).to_csv(OUT / "cross_section_matching.csv", index=False)
    chosen = {}
    for name in types:
        candidates = [row for row in cross if row["cell_type"] == name]
        if candidates:
            chosen[str(name)] = max(candidates, key=lambda row: row["loading_correlation"])
    (OUT / "summary.json").write_text(json.dumps({"main_program_pairs": chosen}, indent=2))
    print(json.dumps({"programs": len(programs), "matched_types": sorted(chosen)}, indent=2))


if __name__ == "__main__":
    main()
