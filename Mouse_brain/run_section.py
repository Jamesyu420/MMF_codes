"""Fit one Visium mouse-brain section."""
import argparse
from pathlib import Path

from MMF import fit

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = {
    "Visium_1": ROOT / "data/mouse_brain/Visium_1.npz",
    "Visium_2": ROOT / "data/mouse_brain/Visium_2.npz",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", required=True, choices=tuple(SECTIONS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    fit(SECTIONS[args.section], ROOT / "Mouse_brain/results" / args.section,
        execution_root=ROOT / "Mouse_brain", device=args.device, seed=23)


if __name__ == "__main__":
    main()
