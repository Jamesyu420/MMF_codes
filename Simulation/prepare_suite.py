"""Generate the paper study inputs and a portable serial execution catalog."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['primary50', 'robustness'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', help='Comma-separated seeds; omitted means the full prespecified study')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--baseline-python', default=sys.executable)
    p.add_argument('--vendor', type=Path, help='Pinned scTM and SpiceMix checkouts')
    p.add_argument('--rscript', type=Path)
    p.add_argument('--rlib', type=Path)
    p.add_argument('--methods', nargs='+', choices=['MMF', 'RCTD', 'STAMP', 'SpiceMix', 'NMF'])
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    a.output = a.output.resolve()
    primary = a.stage == 'primary50'
    seeds = list(range(2026102901, 2026102951) if primary else range(2026103001, 2026103011))
    if a.seeds:
        seeds = list(map(int, a.seeds.split(',')))
    methods = a.methods or (['MMF', 'RCTD', 'STAMP', 'SpiceMix', 'NMF'] if primary else ['MMF'])
    if not primary and methods != ['MMF']:
        p.error('The paper sensitivity study runs MMF only')
    if any(m in methods for m in ['STAMP', 'SpiceMix']) and a.vendor is None:
        p.error('STAMP/SpiceMix require --vendor; see README for pinned checkouts')
    if 'RCTD' in methods and (a.rscript is None or a.rlib is None):
        p.error('RCTD requires --rscript and --rlib')
    a.output.mkdir(parents=True, exist_ok=False)
    conditions = [dict(name='main')]
    if not primary:
        for key, values in dict(threshold=[.09, .12], depth=[12000, 48000],
                                epsilon_sd=[0., .3], lengthscale=[.1, .3], landmarks=[32, 96]).items():
            conditions.extend(dict(name=f'{key}_{v:g}', **{key: v}) for v in values)
    generator = root / 'Simulation/generate.py'
    settings = dict(stage=a.stage, threshold=.065, seeds=seeds, N=400, G=250,
                    truth_K=[1, 2, 0, 1, 1], depth=24000, signal=2.4, epsilon_sd=.15,
                    conditions=conditions, representative_seed=seeds[0],
                    generator_sha256=hashlib.sha256(generator.read_bytes()).hexdigest(),
                    methods=methods, input_policy='All counts; truth used only after fitting.')
    (a.output / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    commands = []
    env = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(root))
    for condition in conditions:
        group = a.output if primary else a.output / condition['name']
        for seed in seeds:
            data = group / 'inputs' / f'input_{seed}'
            cmd = [sys.executable, '-m', 'Simulation.generate', '--output', str(data),
                   '--seed', str(seed), '--scenario', 'gradient']
            for key in ['depth', 'epsilon_sd']:
                if key in condition:
                    cmd += ['--' + key.replace('_', '-'), str(condition[key])]
            subprocess.run(cmd, check=True, cwd=root)
            for method in methods:
                dest = group / f'sim_{seed}_{method}'
                command = [sys.executable if method == 'MMF' else a.baseline_python, '-u', '-m',
                           'Simulation.fit_paper' if method == 'MMF' else 'Simulation.run_baselines',
                           '--input', str(data / 'fit_input.npz'), '--output', str(dest), '--seed', str(seed)]
                if method == 'MMF':
                    command += ['--threshold', str(condition.get('threshold', .065)),
                                '--budget', '86400', '--device', a.device]
                    for key in ['lengthscale', 'landmarks']:
                        if key in condition:
                            command += ['--' + key, str(condition[key])]
                else:
                    command += ['--method', method, '--rank', '10', '--baseline-code',
                                str(root / 'MMF/_utils/baselines'), '--vendor',
                                str(a.vendor.resolve() if a.vendor else root),
                                '--device', 'cpu' if method in ['NMF', 'RCTD'] else a.device]
                    if method == 'RCTD':
                        command += ['--reference', str(data / 'reference_input.npz'),
                                    '--rscript', str(a.rscript.resolve()), '--rlib', str(a.rlib.resolve())]
                commands.append(dict(run_id=f"{condition['name']}_{seed}_{method}", method=method,
                                     experiment=condition['name'], seed=seed, output=str(dest),
                                     command=command, environment=env))
    (a.output / 'catalog.json').write_text(json.dumps(dict(commands=commands), indent=2) + '\n')
    print(json.dumps(dict(datasets=len(conditions)*len(seeds), fits=len(commands), executed=False)))


if __name__ == '__main__':
    main()
