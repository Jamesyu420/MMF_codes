"""Evaluate a complete primary study and export the Fig 1–3 source-data layout."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True, help='primary50 directory')
    p.add_argument('--output', type=Path, required=True, help='New figure-source directory')
    a = p.parse_args()
    settings = json.loads((a.study / 'settings.json').read_text())
    if settings['seeds'] != list(range(2026102901, 2026102951)):
        raise ValueError('Paper figures require all 50 prespecified seeds')
    a.output.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, '-m', 'Simulation.evaluate', '--results', str(a.study),
                    '--data', str(a.study / 'inputs'), '--output', str(a.output),
                    '--seeds', ','.join(map(str, settings['seeds']))], check=True)
    summary = json.loads((a.output / 'summary.json').read_text())
    if not summary['all_requested_complete']:
        raise RuntimeError('Incomplete fits retained in availability.csv; paper plot sources withheld')
    for name in ['case_metrics', 'composition_matching', 'program_matching', 'rank_selection']:
        frame = pd.read_csv(a.output / f'{name}.csv').assign(study='primary50', condition='main')
        frame.to_csv(a.output / f'{name}.csv', index=False)
    seed = settings['representative_seed']
    rep = a.output / 'representative'; rep.mkdir()
    for method in ['MMF', 'RCTD', 'STAMP', 'SpiceMix', 'NMF']:
        source = a.study / f'sim_{seed}_{method}' / ('final.npz' if method == 'MMF' else 'factors.npz')
        shutil.copy2(source, rep / f'{method}.npz')
    for name in ['fit_input.npz', 'evaluation_only.npz', 'design.json']:
        shutil.copy2(a.study / 'inputs' / f'input_{seed}' / name, rep / name)
    (a.output / 'validation_summary.json').write_text(json.dumps(
        dict(representative_seed=seed, validated_primary=250,
             rank_search_stationarity_verified=summary['rank_search_stationary_verified']), indent=2) + '\n')


if __name__ == '__main__':
    main()
