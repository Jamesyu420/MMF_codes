"""Fit every gradient replicate. Each replicate is a separate process."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(202610201, 202610251)
RUNNER = Path(__file__).with_name("run_replicate.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    args = parser.parse_args()
    for seed in (args.seeds or SEEDS):
        source = ROOT / f"data/simu_data/simulation/gradient_{seed}_depth4x/fit_input.npz"
        output = ROOT / f"Simulation/results/sim_{seed}_MMF"
        subprocess.run([
            sys.executable, str(RUNNER),
            "--input", str(source), "--output", str(output),
            "--seed", str(seed), "--device", args.device,
            "--backend", "pyro", "--guide-rank", "0",
        ], check=True)


if __name__ == "__main__":
    main()
