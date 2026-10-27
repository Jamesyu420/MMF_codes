"""Fit Visium HD, the full Xenium section, and each annotation-conditioned Xenium type."""
import argparse
import subprocess
import sys
from pathlib import Path

from utils.io import load_spatial

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/colorectal_cancer"
HERE = Path(__file__).resolve().parent


def jobs():
    spatial = load_spatial(DATA / "HD_data.npz")
    listed = [
        ("HD", DATA / "HD_data.npz"),
        ("Xenium", DATA / "Xenium_data.npz"),
    ]
    for name in spatial["cell_types"]:
        listed.append((f"Xenium_{name}", DATA / "Xenium_by_type" / str(name) / "data.npz"))
    return listed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for name, path in jobs():
        subprocess.run([
            sys.executable, str(HERE / "run.py"),
            "--name", name, "--input", str(path), "--device", args.device,
        ], check=True)


if __name__ == "__main__":
    main()
