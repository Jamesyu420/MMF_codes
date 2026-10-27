"""Program table for the Visium HD fit."""
from pathlib import Path

import numpy as np
import pandas as pd

from utils.io import load_npz, load_spatial, repo_root

ROOT = repo_root()
OUT = ROOT / "Colorectal_cancer/analysis"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spatial = load_spatial(ROOT / "data/colorectal_cancer/HD_data.npz")
    fit = load_npz(ROOT / "Colorectal_cancer/results/HD/final.npz")
    genes, types = spatial["genes"], spatial["cell_types"]
    ranks, weights = [], []
    counts = np.bincount(fit["program_cell_type"], minlength=len(types))
    for c, name in enumerate(types):
        ranks.append(dict(dataset="HD", cell_type=str(name), rank=int(counts[c])))
    for p, (c, k) in enumerate(zip(fit["program_cell_type"], fit["program_index"])):
        order = np.argsort(-fit["loadings"][p], kind="stable")
        for rank, g in enumerate(order):
            weights.append(dict(dataset="HD", cell_type=str(types[c]), program=int(k) + 1,
                                gene=str(genes[g]), weight=float(fit["loadings"][p, g]), gene_rank=rank + 1))
    pd.DataFrame(ranks).to_csv(OUT / "ranks.csv", index=False)
    pd.DataFrame(weights).to_csv(OUT / "gene_weights.csv", index=False)
    print(pd.DataFrame(ranks).to_string(index=False))


if __name__ == "__main__":
    main()
