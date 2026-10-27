"""Fit one colorectal count matrix (Visium HD, Xenium, or one annotated Xenium type)."""
import argparse
from pathlib import Path

from MMF import fit

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    fit(args.input, ROOT / "Colorectal_cancer/results" / args.name,
        execution_root=ROOT / "Colorectal_cancer", device=args.device, seed=23)


if __name__ == "__main__":
    main()
