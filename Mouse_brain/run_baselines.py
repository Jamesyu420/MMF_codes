"""Reproduce the six matched 2,359-gene comparator fits used in Fig 5."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New comparator output and metadata directory')
    p.add_argument('--vendor', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'input_metadata').mkdir()
    baseline_code = Path(__file__).resolve().parent / 'baselines'
    cases = []
    state = dict(completed_cases=0, all_outputs_finite=False, cases=cases)
    validation = a.output / 'validation.json'
    validation.write_text(json.dumps(state, indent=2) + '\n')
    for section, rank, n in [('Visium_1', 30, 4097), ('Visium_2', 31, 3892)]:
        inp = a.inputs / f'{section}.npz'
        with np.load(inp, allow_pickle=False) as z:
            if z['Y'].shape != (n, 2359) or len(z['cell_types']) != 17:
                raise ValueError('Expected the frozen 17-group, 2359-gene reference panel')
            np.savez_compressed(a.output / f'input_metadata/{section}_metadata.npz',
                                genes=z['genes'], coords=z['coords'])
        for method in ['NMF', 'STAMP', 'SpiceMix']:
            case = f'{section}_{method}'; out = a.output / 'results' / case
            cmd = [sys.executable, '-m', 'Simulation.run_baselines', '--input', str(inp),
                   '--output', str(out), '--seed', '23', '--method', method, '--rank', str(rank),
                   '--baseline-code', str(baseline_code), '--vendor', str(a.vendor),
                   '--device', 'cpu' if method == 'NMF' else a.device]
            subprocess.run(cmd, check=True)
            with np.load(out / 'factors.npz', allow_pickle=False) as z:
                assert z['activity'].shape == (n, rank) and z['loadings'].shape == (rank, 2359)
                assert np.isfinite(z['activity']).all() and np.isfinite(z['loadings']).all()
            cases.append(dict(case=case, input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),
                              factors_sha256=hashlib.sha256((out / 'factors.npz').read_bytes()).hexdigest(),
                              N=n, G=2359, rank=rank))
            state['completed_cases'] = len(cases)
            state['all_outputs_finite'] = len(cases) == 6
            validation.write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    main()
