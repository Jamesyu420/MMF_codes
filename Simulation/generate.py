"""Five-type, heterogeneous-rank gradient/hotspot experiments.

Each seed independently regenerates reference cells, mixtures, factors and counts.
The generator writes fitting inputs separately from final-evaluation truth.
"""
import argparse
import json
from pathlib import Path

import numpy as np


PROGRAMS = [1, 2, 0, 1, 1]


def softmax(logits, axis):
    values = np.exp(logits - logits.max(axis=axis, keepdims=True))
    return values / values.sum(axis=axis, keepdims=True)


def simulate(seed, scenario, side=20, genes=250, depth=24000, signal=2.4,
             epsilon_sd=.15):
    rng = np.random.default_rng(seed)
    axis = np.linspace(0, 1, side)
    coords = np.array(np.meshgrid(axis, axis)).reshape(2, -1).T
    n, cell_types = len(coords), 5
    block = genes // cell_types
    reference_rates = rng.gamma(2, 1, (cell_types, genes))
    for c in range(cell_types):
        reference_rates[c, c * block:(c + 1) * block] *= 10
    reference_rates /= reference_rates.sum(1, keepdims=True)
    reference_cells = rng.poisson(5000 * reference_rates[:, None, :],
                                 size=(cell_types, 200, genes))
    eta = (reference_cells / reference_cells.sum(2, keepdims=True)).mean(1)
    angles = np.linspace(0, 2 * np.pi, cell_types, endpoint=False)
    x, y = coords.T
    logits = np.array([.7 * np.sin(2 * np.pi * (np.cos(a) * x + np.sin(a) * y))
                       for a in angles]).T
    pi = softmax(logits + rng.normal(0, .08, (n, cell_types)), axis=1)
    pi_init = np.full((n, cell_types), 1 / cell_types)
    weights, activities, types, indices = [], [], [], []
    for c, count in enumerate(PROGRAMS):
        chosen = rng.permutation(np.arange(c * block, (c + 1) * block))
        for k in range(count):
            w = np.zeros(genes)
            w[chosen[8 * k:8 * (k + 1)]] = rng.dirichlet(np.full(8, 12.))
            angle = angles[c] + np.pi / 2 * k + rng.uniform(-.1, .1)
            if scenario == "gradient":
                raw = np.tanh(2.5 * (np.cos(angle) * (x - .5) + np.sin(angle) * (y - .5)))
            else:
                center = .5 + .28 * np.array([np.cos(angle), np.sin(angle)])
                width = .13 + .02 * k + rng.uniform(-.01, .01)
                raw = np.exp(-np.square(coords - center).sum(1) / (2 * width ** 2))
            raw -= raw.mean()
            h = raw / abs(raw).max() * signal / w.max()
            weights.append(w)
            activities.append(h)
            types.append(c)
            indices.append(k)
    weights, activities = np.array(weights), np.array(activities)
    effect = np.zeros((cell_types, n, genes))
    for c, h, w in zip(types, activities, weights):
        effect[c] += h[:, None] * w
    library = rng.lognormal(np.log(depth), .2, n)
    alpha, gamma = rng.normal(0, .05, n), rng.normal(0, .06, genes)
    rate = library[:, None] * np.exp(alpha[:, None] + gamma) * (
        pi.T[:, :, None] * eta[:, None, :] * np.exp(effect)).sum(0)
    epsilon = rng.normal(0, epsilon_sd, rate.shape)
    structural = rate.copy()
    rate *= np.exp(epsilon)
    counts = rng.poisson(rate)
    observed_library = counts.sum(1)
    return dict(Y=counts,
                eta0=eta, pi=pi_init, coords=coords,
                library_size=observed_library), dict(
                    pi_true=pi, rate_true=rate, loadings_true=weights,
                    h_true=activities, program_cell_type=np.array(types),
                    program_index=np.array(indices), reference_cells=reference_cells,
                    alpha_true=alpha + np.log(library / observed_library),
                    alpha_generated=alpha, exposure_generated=library,
                    gamma_true=gamma, epsilon_true=epsilon,
                    epsilon_sigma_true=np.asarray(epsilon_sd),
                    structural_rate_true=structural)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--scenario", choices=["gradient", "hotspot"], required=True)
    parser.add_argument("--side", type=int, default=20)
    parser.add_argument("--genes", type=int, default=250)
    parser.add_argument("--depth", type=float, default=24000)
    parser.add_argument("--signal", type=float, default=2.4)
    parser.add_argument("--epsilon-sd", type=float, default=.15)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fit, truth = simulate(args.seed, args.scenario, args.side, args.genes,
                          args.depth, args.signal, args.epsilon_sd)
    np.savez_compressed(args.output / "fit_input.npz", **fit)
    reference_cells = truth.pop('reference_cells')
    np.savez_compressed(args.output / "reference_input.npz",
        X=reference_cells.reshape(-1,args.genes),
        cell_type=np.repeat(np.arange(5), reference_cells.shape[1]),
        genes=np.array([f'g{j}' for j in range(args.genes)]))
    np.savez_compressed(args.output / "evaluation_only.npz", **truth)
    (args.output / "design.json").write_text(json.dumps(dict(
        scenario=args.scenario, seed=args.seed, spots=args.side ** 2, genes=args.genes,
        cell_types=5, programs_per_type=PROGRAMS, depth=args.depth,
        signal=args.signal, epsilon_sd=args.epsilon_sd, global_capacity=10,
        rank="Unknown to fitter; automatic per-type selection evaluated after freezing output",
        reference="200 independently simulated cells per type; fixed normalized mean",
        mixture_initialization="uniform; refined using all observed counts",
        reference_setting="matched reference: simulated generation uses the fixed reference mean",
        counts="Poisson counts with Gaussian log residual; all counts used for fitting",
        offset="Observed row sum; alpha_true absorbs log(generated exposure / observed sum). This changes the generating alpha distribution after reexpression.",
        truth_access="evaluation_only.npz is never read by fitter"), indent=2) + "\n")


if __name__ == "__main__":
    main()
