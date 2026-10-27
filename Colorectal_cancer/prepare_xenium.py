"""One Xenium count matrix per supplied annotation, with that type's reference support."""
import json
from pathlib import Path

import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/colorectal_cancer/Xenium_data.npz"
RAW = Path("/home/byual/Xenium_HD/data/SPATCH/COAD/Xenium/section_unresolved/original/extracted/transcriptome/adata.h5ad")
OUT = ROOT / "data/colorectal_cancer/Xenium_by_type"


def main():
    with np.load(SRC, allow_pickle=False) as stored:
        data = {key: stored[key] for key in stored.files}
    adata = ad.read_h5ad(RAW, backed="r")
    obs = adata.obs.iloc[data["original_indices"]]
    assert np.array_equal(obs.index.astype(str).to_numpy(), data["spot_names"].astype(str))
    assert (obs.high_quality.astype(str) == "1").all()
    labels = obs.annotation.astype(str).to_numpy()
    labels = np.where(np.isin(labels, ["mregDC", "cDC2", "pDC"]), "Other", labels)
    assert set(labels) <= set(data["cell_types"].tolist())
    records = []
    for c, name in enumerate(data["cell_types"]):
        support = data["eta0"][c] > 0
        keep = (labels == str(name)) & (data["Y"][:, support].sum(1) > 0)
        n = int(keep.sum())
        assert n > 100
        subset = {key: data[key][keep] for key in ["Y", "coords", "original_indices", "original_coords", "spot_names"]}
        subset["Y"] = subset["Y"][:, support]
        subset["library_size"] = subset["Y"].sum(1).astype(float)
        subset["eta0"] = data["eta0"][c:c + 1, support]
        subset["pi"] = np.ones((n, 1), dtype=data["pi"].dtype)
        subset["genes"] = data["genes"][support]
        subset["cell_types"] = np.asarray([name])
        dest = OUT / str(name)
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / "data.npz", **subset)
        records.append(dict(cell_type=str(name), N=n, G=int(support.sum())))
    adata.file.close()
    (OUT / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
