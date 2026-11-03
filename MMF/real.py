"""K=0 backfitting, then select_programs_ct."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from MMF.backfit import PyroBackfitMMF, fixed_training_score
from MMF.mmf import FitConfig
from MMF.MMF_ctback import select_programs_ct


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


class Callbacks:
    def __init__(self, particles, genes, types):
        self.particles = particles
        self.genes = genes
        self.types = types

    def fit(self, model, phase, prefix, cycles, conditional_program=None):
        prefix = Path(prefix)
        prefix.mkdir(parents=True, exist_ok=True)
        idx = torch.arange(model.N, device=model.device)
        step = 0
        with model._scope(), model._rng_scope():
            for first in range(1, cycles + 1, model.config.program_steps):
                step = min(first + model.config.program_steps - 1, cycles)
                lr = .02 if first <= 150 else .005 if first <= 1000 else .001 if first <= 3000 else .0002
                model.set_optimization(lr=lr, num_particles=self.particles)
                model.backfit_cycle([idx] * (step - first + 1), conditional_program)
                if conditional_program is None:
                    model.update_epsilon_variance()
        record = dict(phase=phase, steps=step, ranks=list(model.Kc))
        write(prefix / "phase.json", record)
        return record

    def score(self, model):
        idx = torch.arange(model.N, device=model.device)
        return fixed_training_score(model, idx, seed=91025, particles=self.particles, batches=4)["mean"]

    def export(self, model, prefix):
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        summary = model.posterior_summary()
        summary.update(genes=self.genes, cell_types=self.types)
        np.savez_compressed(prefix.with_suffix(".npz"), **summary)
        return str(prefix)


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.input, allow_pickle=False) as z:
        data = {k: z[k] for k in ["Y", "eta0", "pi", "coords", "library_size"]}
        genes, types = z["genes"], z["cell_types"]
    n = data["Y"].shape[0]
    config = FitConfig(
        kernel="nystrom", lengthscale=args.lengthscale, n_landmarks=min(args.landmarks, n),
        dtype=args.dtype, device=args.device, batch_size=n, chunk_size=n,
        num_particles=args.particles, seed=args.seed, lr=.02)
    model = PyroBackfitMMF(
        **data, programs_per_type=[0] * len(types), config=config,
        epsilon_variance=args.epsilon_variance, epsilon_variance_floor=args.epsilon_floor)
    callbacks = Callbacks(args.particles, genes, types)
    callbacks.fit(model, "k0", output / "k0", args.k0_cycles)
    model, report = select_programs_ct(
        model, seed_idx=torch.arange(model.N, device=model.device), seed=args.seed,
        output=output, fit=callbacks.fit, score=callbacks.score, export=callbacks.export,
        write=write, restore=None, tau=args.threshold, k_max=args.k_max,
        forward_cycles=args.forward_cycles, backfit_cycles=args.backfit_cycles)
    print(json.dumps(dict(stop_reason=report["stop_reason"], ranks=report["final_ranks"])), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--particles", type=int, default=8)
    p.add_argument("--lengthscale", type=float, default=.2)
    p.add_argument("--landmarks", type=int, default=128)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--epsilon-variance", type=float, default=.1)
    p.add_argument("--epsilon-floor", type=float, default=1e-8)
    p.add_argument("--threshold", type=float, default=.065)
    p.add_argument("--k-max", type=int, default=3)
    p.add_argument("--k0-cycles", type=int, default=6000)
    p.add_argument("--forward-cycles", type=int, default=100)
    p.add_argument("--backfit-cycles", type=int, default=500)
    return p


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
