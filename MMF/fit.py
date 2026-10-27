"""One prepared spatial matrix to one automatic-rank MMF fit."""
import argparse
from pathlib import Path

from MMF.real import run


def fit(input_npz, output, *, execution_root, device="cuda", seed=23,
        batch=1024, particles=4, lengthscale=0.2, landmarks=128, dtype="float32",
        budget_seconds=1e12, epsilon_variance=0.09, epsilon_floor=1e-8):
    """Fit MMF. ``input_npz`` holds Y, eta0, pi, coords, library_size, genes, cell_types."""
    args = argparse.Namespace(
        budget_seconds=budget_seconds,
        resume_checkpoint=None, resume_phase=None, prior_status=None,
        input=Path(input_npz), output=Path(output), execution_root=Path(execution_root),
        epsilon_variance=epsilon_variance, epsilon_floor=epsilon_floor,
        device=device, batch=batch, particles=particles, check_interval=25,
        lengthscale=lengthscale, landmarks=landmarks,
        seed=seed, threshold=None, dtype=dtype, k0_cycles=6000,
        forward_cycles=150, backfit_cycles=1000, score_positions=1024,
        score_particles=16, score_batches=4, stop_after_k0=False,
        fixed_ranks=None, gpu_memory_fraction=0.75)
    return run(args)
