"""Load prepared count matrices."""
from pathlib import Path

import numpy as np

SPATIAL_FIELDS = ("Y", "eta0", "pi", "coords", "library_size", "genes", "cell_types")


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_spatial(path):
    """Visium / Visium HD / Xenium matrix with a reference profile and initial composition."""
    data = load_npz(path)
    return {key: data[key] for key in SPATIAL_FIELDS}


def load_fit_input(path):
    """Simulation fitting arrays. Truth lives in a separate file and is not read here."""
    data = load_npz(path)
    return {key: data[key] for key in ("Y", "eta0", "pi", "coords", "library_size")}


def repo_root():
    return Path(__file__).resolve().parents[1]
