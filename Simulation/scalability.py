"""Single-seed Fig 1d scaling protocol, with explicit serial execution.

Use --execute to launch; otherwise print the plan only. Fresh timing values
depend on hardware. The submitted curve uses its archived source CSV.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--vendor', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--baseline-python', default=sys.executable)
    p.add_argument('--sizes', nargs='+', type=int, choices=[400, 2500, 10000, 40000],
                   default=[400, 2500, 10000, 40000])
    p.add_argument('--methods', nargs='+', choices=['MMF', 'STAMP', 'SpiceMix', 'NMF'],
                   default=['MMF', 'STAMP', 'SpiceMix', 'NMF'])
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan = [dict(N=n, method=m, rho=.7 if n == 40000 and m == 'MMF' else .25 if m == 'MMF' else None)
            for n in a.sizes for m in a.methods]
    if not a.execute:
        print(json.dumps(dict(executed=False, plan=plan), indent=2)); return
    if any(m in a.methods for m in ['STAMP', 'SpiceMix']) and a.vendor is None:
        p.error('STAMP and SpiceMix require --vendor')
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(root))
    rows = []
    for n in a.sizes:
        inp = a.output / 'inputs' / f'N{n}'
        subprocess.run([sys.executable, '-m', 'Simulation.generate', '--output', str(inp),
                        '--seed', '2026102901', '--scenario', 'gradient', '--side', str(int(n**.5))],
                       check=True, env=env, cwd=root)
        for method in a.methods:
            out = a.output / 'results' / f'N{n}_{method}'
            rho = .7 if n == 40000 else .25
            if method == 'MMF':
                command = [sys.executable, '-m', 'Simulation.fit_scalability', '--input', str(inp/'fit_input.npz'),
                           '--output', str(out), '--seed', '2026102901', '--device', a.device,
                           '--threshold', '.065', '--landmarks', '48', '--particles', '8',
                           '--rho', str(rho), '--budget', '86400' if n == 40000 else '21600']
            else:
                command = [a.baseline_python, '-m', 'Simulation.run_baselines', '--input', str(inp/'fit_input.npz'),
                           '--output', str(out), '--seed', '2026102901', '--method', method, '--rank', '10',
                           '--baseline-code', str(root/'MMF/_utils/baselines'), '--vendor', str(a.vendor or root),
                           '--device', 'cpu' if method == 'NMF' else a.device]
            started = time.perf_counter()
            # A failed or budget-limited attempt is recorded and stops this run.
            proc = subprocess.run(command, env=env, cwd=root)
            elapsed = time.perf_counter() - started
            receipt = out / ('experiment_status.json' if method == 'MMF' else 'execution_receipt.json')
            child = json.loads(receipt.read_text()) if receipt.exists() else {}
            row = dict(method=method, N=n, seed=2026102901,
                       status='completed' if proc.returncode == 0 and child.get('phase') == 'finished' else 'incomplete',
                       wall_seconds=elapsed, minutes=elapsed/60, host=socket.gethostname(),
                       rho=rho if method == 'MMF' else None,
                       continuous_fit_stable=child.get('continuous_fit_stable'),
                       rank_search_stationary_verified=child.get('rank_search_stationary_verified'),
                       input_sha256=hashlib.sha256((inp/'fit_input.npz').read_bytes()).hexdigest())
            rows.append(row)
            with (a.output/'runtime_source.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(row)); w.writeheader(); w.writerows(rows)
            if row['status'] != 'completed':
                raise RuntimeError(f'{method}, N={n}: incomplete attempt retained in {out}')


if __name__ == '__main__':
    main()
