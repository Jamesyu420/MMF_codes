"""Reference-guided spatial factorization.

``fit`` takes a prepared count matrix (Visium or Visium HD, plus a
single-cell reference profile and an initial composition) and writes the
posterior programs, loadings, and mixing weights.
"""
from MMF.mmf import MMF, FitConfig, compute_reference_mean
from MMF.fit import fit

__all__ = ["MMF", "FitConfig", "compute_reference_mean", "fit"]
