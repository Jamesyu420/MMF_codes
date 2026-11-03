"""Score completed fits against simulation truth; retain every requested status."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from MMF._utils.io import load_npz, repo_root
from MMF._utils.metrics import RNA_share, SIM_METHODS, TRUE_K, corr, one_case

ROOT = repo_root()
DATA = ROOT / "results/20261028/primary50/inputs"
RESULTS = ROOT / "results/20261028/primary50"
OUT = ROOT / "results/20261028/analysis"


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_simulation_selection(directory, status):
    selection = read_json(directory / "selection.json")
    assert selection["phase"] == "finished" and selection["selection_complete"] and not selection["heldout_or_truth_read"]
    assert selection["final_ranks"] == status["selected_ranks"]
    phase = read_json(directory / "polish" / "phase.json")
    assert phase["ranks"] == status["selected_ranks"] and status["continuous_fit_stable"]
    assert phase["status"] == "training_probe_stable" and phase["steps"] >= 500 and phase["strict_exit_validation"]["passed"]
    history = phase["history"][-10:]
    assert len(history) == 10
    first = np.mean([x["elbo_per_entry"] for x in history[:5]])
    last_score = np.mean([x["elbo_per_entry"] for x in history[5:]])
    assert abs(first - last_score) / max(1., abs(first)) < .005
    for x in history[-3:]:
        assert max(x[k + "_change"] for k in ["rate", "pi", "variance", "effect"]) < .01
        assert x["epsilon_variance_change"] < .01 and x["epsilon_EB_last3_relative_change"] < .01 and x["epsilon_stationarity_relative"] < 1e-6
    meta = read_json(directory / "final.metadata.json")
    assert meta["input_sha256"] == status["input_sha256"] and meta["ranks"] == status["selected_ranks"]


def finished_fit(directory, method):
    receipt = directory / ("experiment_status.json" if method == "MMF" else "execution_receipt.json")
    if not receipt.exists():
        return None
    status = read_json(receipt)
    if status.get("phase") != "finished":
        return None
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default=",".join(map(str, range(2026102901, 2026102951))))
    parser.add_argument("--methods", nargs="+", choices=SIM_METHODS, default=SIM_METHODS)
    args = parser.parse_args()
    seeds = list(map(int, args.seeds.split(",")))
    args.output.mkdir(parents=True, exist_ok=True)
    rows, pirows, programs, ranks, availability, rna = [], [], [], [], [], []
    for seed in seeds:
        case = args.data / f"input_{seed}"
        truth = load_npz(case / "evaluation_only.npz")
        training = load_npz(case / "fit_input.npz")
        assert np.array_equal(np.bincount(truth["program_cell_type"], minlength=5), TRUE_K)
        target = RNA_share(truth["pi_true"], training["eta0"], truth["h_true"], truth["loadings_true"],
                           truth["program_cell_type"], truth["gamma_true"])
        for method in args.methods:
            directory = args.results / f"sim_{seed}_{method}"
            status = finished_fit(directory, method)
            receipt = directory / ("experiment_status.json" if method == "MMF" else "execution_receipt.json")
            observed = read_json(receipt).get("phase", "pending") if receipt.exists() else "pending"
            availability.append(dict(seed=seed, method=method, status="completed" if status else observed))
            if status is None:
                continue
            assert status["input_sha256"] == sha(case / "fit_input.npz")
            assert status["source_unchanged"]
            if method == "RCTD":
                table = pd.read_csv(directory / "proportions.csv", index_col=0)
                fit = {"pi": table[[f"T{c + 1}" for c in range(5)]].to_numpy()}
            elif method == "MMF":
                validate_simulation_selection(directory, status)
                fit = load_npz(directory / "final.npz")
                assert all(np.isfinite(v).all() for v in fit.values() if v.dtype.kind in "biufc")
                assert (fit["pi"] >= 0).all() and (fit["loadings"] >= 0).all() and (fit["h_sd"] >= 0).all()
                np.testing.assert_allclose(fit["pi"].sum(1), 1, atol=1e-5)
                np.testing.assert_allclose(fit["loadings"].sum(1), 1, atol=1e-5)
                selected = np.bincount(fit["program_cell_type"], minlength=5)
                assert selected.tolist() == status["selected_ranks"]
                ranks.append(dict(
                    seed=seed, exact_vector=bool(np.array_equal(selected, TRUE_K)),
                    rank_search_stationary_verified=status["rank_search_stationary_verified"],
                    total_rank=int(selected.sum()), null_type_false_positive=bool(selected[2] > 0),
                    **{f"K{c + 1}": int(selected[c]) for c in range(5)}))
            else:
                fit = load_npz(directory / "factors.npz")
            seconds = status["elapsed_seconds"]
            row, composition, matched = one_case(seed, method, truth, fit, seconds, status["host"], status["device"])
            rows.append(row)
            pirows.extend(composition)
            programs.extend(matched)
            if method in ["MMF", "RCTD"]:
                estimate = fit["pi"] if method == "RCTD" else RNA_share(
                    fit["pi"], fit["eta0"], fit["h_mean"], fit["loadings"], fit["program_cell_type"],
                    fit["gamma_mean"], fit["h_sd"] ** 2, fit["gamma_sd"] ** 2)
                rna.append(dict(seed=seed, method=method, structural_RNA_share_correlation=float(
                    np.mean([corr(target[:, c], estimate[:, c]) for c in range(5)]))))
    tables = {
        "case_metrics": rows, "composition_matching": pirows, "program_matching": programs,
        "rank_selection": ranks, "availability": availability, "RNA_share_metrics": rna,
    }
    for name, values in tables.items():
        pd.DataFrame(values).to_csv(args.output / f"{name}.csv", index=False)
    summary = dict(all_requested_complete=len(rows) == len(args.methods) * len(seeds),
                   methods=args.methods, requested_fits=len(args.methods) * len(seeds), completed_fits=len(rows), n_MMF=len(ranks),
                   exact_rank_successes=sum(x["exact_vector"] for x in ranks),
                   rank_denominator_requested=len(seeds),
                   rank_search_stationary_verified=all(x["rank_search_stationary_verified"] for x in ranks) if ranks else False)
    summary["representative"] = dict(seed=seeds[0], rule="First prespecified seed, independent of outcomes")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
