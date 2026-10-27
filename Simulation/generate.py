"""Five-type, heterogeneous-rank gradient/hotspot experiments.

Each seed independently regenerates reference cells, mixtures, factors and counts.
The generator writes fitting inputs separately from final-evaluation truth.
"""
import argparse
import json
from pathlib import Path

import numpy as np


PROGRAMS = [1, 2, 0, 1, 2]
FRACTIONS = [.6, .2, .2]


def softmax(logits, axis):
    values = np.exp(logits - logits.max(axis=axis, keepdims=True))
    return values / values.sum(axis=axis, keepdims=True)


def simulate(seed, scenario, side=20, genes=250):
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
            angle = angles[c] + .75 * k + rng.uniform(-.1, .1)
            if scenario == "gradient":
                raw = np.tanh(2.5 * (np.cos(angle) * (x - .5) + np.sin(angle) * (y - .5)))
            else:
                center = .5 + .28 * np.array([np.cos(angle), np.sin(angle)])
                width = .13 + .02 * k + rng.uniform(-.01, .01)
                raw = np.exp(-np.square(coords - center).sum(1) / (2 * width ** 2))
            raw -= raw.mean()
            h = raw / abs(raw).max() * 1.8 / w.max()
            weights.append(w)
            activities.append(h)
            types.append(c)
            indices.append(k)
    weights, activities = np.array(weights), np.array(activities)
    effect = np.zeros((cell_types, n, genes))
    for c, h, w in zip(types, activities, weights):
        effect[c] += h[:, None] * w
    library = rng.lognormal(np.log(6000), .2, n)
    alpha, gamma = rng.normal(0, .05, n), rng.normal(0, .06, genes)
    rate = library[:, None] * np.exp(alpha[:, None] + gamma) * (
        pi.T[:, :, None] * eta[:, None, :] * np.exp(effect)).sum(0)
    counts = [rng.poisson(fraction * rate) for fraction in FRACTIONS]
    return dict(Y=counts[0], Y_validation=counts[1], Y_test=counts[2],
                eta0=eta, pi=pi_init, coords=coords,
                library_size=FRACTIONS[0] * library), dict(
                    pi_true=pi, rate_true=rate, loadings_true=weights,
                    h_true=activities, program_cell_type=np.array(types),
                    program_index=np.array(indices), reference_cells=reference_cells,
                    alpha_true=alpha, gamma_true=gamma)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--scenario", choices=["gradient", "hotspot"], required=True)
    parser.add_argument("--side", type=int, default=20)
    parser.add_argument("--genes", type=int, default=250)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fit, truth = simulate(args.seed, args.scenario, args.side, args.genes)
    np.savez_compressed(args.output / "fit_input.npz", **{k: v for k, v in fit.items()
                        if k not in {"Y_validation", "Y_test"}})
    np.savez_compressed(args.output / "heldout.npz", **{k: fit[k]
                        for k in ["Y_validation", "Y_test"]})
    reference_cells = truth.pop('reference_cells')
    np.savez_compressed(args.output / "reference_input.npz",
        X=reference_cells.reshape(-1,args.genes),
        cell_type=np.repeat(np.arange(5), reference_cells.shape[1]),
        genes=np.array([f'g{j}' for j in range(args.genes)]))
    np.savez_compressed(args.output / "evaluation_only.npz", **truth)
    (args.output / "design.json").write_text(json.dumps(dict(
        scenario=args.scenario, seed=args.seed, spots=args.side ** 2, genes=args.genes,
        cell_types=5, programs_per_type=PROGRAMS, split=FRACTIONS,
        rank="Unknown to fitter; automatic per-type selection evaluated after freezing output",
        reference="200 independently simulated cells per type; fixed normalized mean",
        mixture_initialization="uniform; refined using training counts only",
        reference_setting="matched reference: simulated generation uses the fixed reference mean",
        counts="Independent Poisson components equivalent in law to Poisson thinning",
        truth_access="evaluation_only.npz is never read by fitter"), indent=2) + "\n")


if __name__ == "__main__":
    main()
