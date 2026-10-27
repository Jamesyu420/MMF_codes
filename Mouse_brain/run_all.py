"""Fit both Visium sections, one process each."""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for section in ("Visium_1", "Visium_2"):
        subprocess.run([sys.executable, str(HERE / "run_section.py"),
                        "--section", section, "--device", args.device], check=True)


if __name__ == "__main__":
    main()
