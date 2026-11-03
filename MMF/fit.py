"""One prepared spatial matrix to one MMF fit."""
import argparse
from pathlib import Path

from MMF.real import run


def fit(input_npz, output, *, device="cuda", seed=23, particles=8,
        lengthscale=0.2, landmarks=128, dtype="float32",
        epsilon_variance=0.1, epsilon_floor=1e-8, threshold=0.065, k_max=3,
        k0_cycles=6000, forward_cycles=100, backfit_cycles=500):
    """Fit MMF. ``input_npz`` holds Y, eta0, pi, coords, library_size, genes, cell_types."""
    args = argparse.Namespace(
        input=Path(input_npz), output=Path(output), device=device, seed=seed,
        particles=particles, lengthscale=lengthscale, landmarks=landmarks, dtype=dtype,
        epsilon_variance=epsilon_variance, epsilon_floor=epsilon_floor,
        threshold=threshold, k_max=k_max, k0_cycles=k0_cycles,
        forward_cycles=forward_cycles, backfit_cycles=backfit_cycles)
    return run(args)
