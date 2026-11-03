"""Reference-guided spatial factorization.

``fit`` loads a prepared count matrix, fits the K=0 reference model, then
runs ``select_programs_ct``.
"""
from MMF.mmf import MMF, FitConfig, compute_reference_mean
from MMF.fit import fit
from MMF.MMF_ctback import select_programs_ct

__all__ = ["MMF", "FitConfig", "compute_reference_mean", "fit", "select_programs_ct"]
