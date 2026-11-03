"""Run one section with the current MMF API in an explicit output directory."""
import argparse
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--section", choices=["Visium_1", "Visium_2"])
    p.add_argument("--code", type=Path)
    p.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--budget", type=float, default=86400)
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--preflight", action="store_true", help="Bounded real-input optimizer check; no formal rank search")
    p.add_argument("--preflight-wide", action="store_true", help="Memory check with one program in every eligible type")
    p.add_argument("--serial-particles", action="store_true", help="Evaluate the same eight Pyro particles sequentially to bound training memory")
    args = p.parse_args()
    # All input/output paths are explicit; no private server configuration.
    if args.input is None or args.output is None:
        p.error("Supply both --input and --output; --section is an optional label")
    args.code = args.code or root
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(args.code.resolve()))
    from MMF.backfit import PyroBackfitMMF, gaussian_epsilon_diagnostics, add_program
    from MMF.mmf import FitConfig
    from MMF.MMF_ctback import select_programs_ct
    from Simulation.fit_paper import SimulationCallbacks, write

    with np.load(args.input, allow_pickle=False) as data:
        caps = np.where(data["cell_types"] == "UnknownLowQ", 0, 3).tolist()
        metadata = {k: data[k] for k in ("genes", "cell_types", "coords")}
        arrays = {k: data[k] for k in ("Y", "eta0", "pi", "coords", "library_size")}
    args.seed, args.particles, args.variance_mode = 23, 8, "shared"

    class VisiumCallbacks(SimulationCallbacks):
        def full_score(self, model, seed, particles, batches=4):
            # Weighted chunk losses sum to the complete-data likelihood and
            # exactly one global KL. Parameters never change between chunks.
            # This bounds the C x N x G x particles Monte Carlo allocation.
            values = []
            count = max(2, particles + particles % 2)
            with model._scope(), torch.no_grad():
                for repeat in range(batches):
                    generators = [torch.Generator(device=model.device).manual_seed(
                        (seed + repeat*1000003 + c*15485863) % (2**63-1)) for c in range(model.C)]
                    total = 0.
                    for idx in model._chunks():
                        eps = None
                        if model.programs:
                            noises = []
                            for generator in generators:
                                half = torch.randn((count//2, len(idx), model.G), generator=generator,
                                                   device=model.device, dtype=model.dtype)
                                noises.append(torch.cat([half, -half], 0))
                            eps = torch.stack(noises, 1)
                            del noises, half
                        total -= float(model.direct_loss(idx, particles=count, eps=eps)) * len(idx) / model.N
                        del eps
                    values.append(total)
            return dict(mean=float(np.mean(values)), elbo_batches=values,
                        mc_standard_error=float(np.std(values, ddof=1)/np.sqrt(batches)) if batches > 1 else None,
                        seed=seed, particles=particles, batches=batches,
                        rows=model.N, chunk_size=model.config.chunk_size)

        def score(self, model):
            row = self.full_score(model, seed=91025, particles=args.particles)
            row.update(call=self.score_calls, ranks=list(model.Kc))
            write(self.output / "scores" / f"{self.score_calls:05d}.json", row)
            self.score_calls += 1
            return row["mean"]

        def probe(self, model):
            elbo = self.full_score(model, seed=51023, particles=16)["mean"]
            rate = np.empty((model.N, model.G), dtype=np.float64)
            pi = np.empty((model.N, model.C), dtype=np.float32)
            effect = np.empty((model.C, model.N, model.G), dtype=np.float32)
            stationarity = 0.
            with model._scope(), torch.no_grad():
                for idx in model._chunks():
                    rows = idx.cpu().numpy()
                    rate[rows] = model._log_expected_counts(idx).exp().cpu().numpy()
                    pi[rows] = model._indexed_log_pi(idx).exp().cpu().numpy()
                    diagnostic = gaussian_epsilon_diagnostics(
                        model.structural_log_expected_counts(idx), model.Y[idx], model.v_eps)
                    stationarity = max(stationarity, diagnostic["maximum_log_rate_newton_correction"],
                                       diagnostic["maximum_log_rate_reconstruction_error"])
                    for c in range(model.C):
                        value = torch.zeros((len(idx), model.G), device=model.device, dtype=model.dtype)
                        for k in range(model.Kc[c]):
                            mean, _ = model._moments(c, k, idx)
                            value += mean[:, None] * model._w(c, k)
                        effect[c, rows] = value.cpu().numpy()
                variance = torch.stack([model.v_alpha, model.v_gamma] + [model.v[p] for p in model.programs]).reshape(-1).cpu().numpy()
                epsilon_variance = model.v_eps.reshape(-1).cpu().numpy().copy()
                updates = model.epsilon_updates[-3:]
                epsilon_EB_last3 = max([r.get("true_relative_change", float("inf")) for r in updates]) if len(updates) == 3 else float("inf")
            if not np.isfinite(elbo) or not all(np.isfinite(a).all() for a in [rate, pi, variance, effect]):
                raise FloatingPointError("Nonfinite full-data Visium diagnostic")
            return dict(elbo_per_entry=elbo/(model.N*model.G), rate=rate, pi=pi,
                        variance=variance, effect=effect, epsilon_variance=epsilon_variance,
                        epsilon_EB_last3_relative_change=epsilon_EB_last3,
                        epsilon_stationarity_relative=stationarity)

        def export(self, model, prefix, summary=True):
            super().export(model, prefix, summary=False)
            if summary:
                payload = model.posterior_summary()
                payload.update(metadata)
                np.savez_compressed(str(prefix) + ".npz", **payload)
            return str(prefix)

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    sources = {str(path.relative_to(args.code)): digest(path)
               for path in sorted((args.code / "MMF").glob("*.py")) if not path.name.startswith(".")}
    sources["Simulation/fit_paper.py"] = digest(args.code / "Simulation/fit_paper.py")
    sources["Mouse_brain/run_section.py"] = digest(Path(__file__))
    args.output.mkdir(parents=True, exist_ok=False)
    n, g = arrays["Y"].shape
    config = FitConfig(kernel="nystrom", lengthscale=.2, n_landmarks=128,
                       dtype="float32", device=args.device, batch_size=n, chunk_size=args.chunk_size,
                       num_particles=8, seed=23, lr=.02)
    receipt = dict(status="running", host=socket.gethostname(), device=args.device,
                   gpu=torch.cuda.get_device_name(torch.device(args.device)), seed=23, N=n, G=g,
                   input_sha256=digest(args.input), input_path=str(args.input.resolve()),
                   cell_types=metadata["cell_types"].tolist(), source_sha256=sources, K_max=caps,
                   scoring="Complete data, weighted exact-KL chunks; 8/16 particles, four fixed-seed batches.",
                   vectorized_training_particles=not args.serial_particles,
                   configuration=dict(kernel="nystrom", landmarks=128, lengthscale=.2,
                                      particles=8, score_particles=8, probe_particles=16,
                                      dtype="float32", epsilon_variance=.1, epsilon_floor=1e-8,
                                      threshold=.065, k0_steps=6000, forward_steps=100,
                                      backfit_steps=500, polish_max_steps=6000,
                                      diagnostic_chunk_size=args.chunk_size, budget_seconds=args.budget))
    (args.output / "run_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    started = time.monotonic()
    initial_ranks = [int(k > 0) for k in caps] if args.preflight_wide else [0] * len(caps)
    if args.preflight_wide and not args.preflight:
        p.error("--preflight-wide requires --preflight")
    model = PyroBackfitMMF(**arrays, programs_per_type=initial_ranks, config=config,
                          epsilon_variance=.1, epsilon_variance_floor=1e-8)
    if args.serial_particles:
        model._training_elbo.vectorize_particles = False
    callbacks = VisiumCallbacks(args)
    try:
        if args.preflight:
            if args.preflight_wide:
                with model._scope(), model._rng_scope():
                    model._step_group("pi", torch.arange(model.N, device=model.device))
            else:
                # Exercise full-size K0 and a seeded program allocation.
                callbacks.cycle(model, steps=1)
                model = add_program(model, int(np.argmax(arrays["pi"].sum(0))),
                                    torch.arange(model.N, device=model.device), seed=23)
                callbacks.cycle(model, steps=1)
            check = callbacks.full_score(model, seed=91025, particles=16, batches=1)
            if not np.isfinite(check["mean"]):
                raise FloatingPointError("Nonfinite preflight score")
            receipt.update(status="preflight_passed", score=check,
                           tested_ranks=list(model.Kc),
                           peak_gpu_bytes=torch.cuda.max_memory_allocated(model.device))
            return
        callbacks.fit(model, "k0", args.output / "k0", 6000)
        model, selection = select_programs_ct(
            model, seed_idx=torch.arange(n, device=model.device), seed=23,
            output=args.output, fit=callbacks.fit, score=callbacks.score,
            export=callbacks.export, write=write, restore=None,
            tau=.065, k_max=caps, forward_cycles=100, backfit_cycles=500,
            polish_cycles=6000)
        receipt.update(status="completed", selection_complete=selection["selection_complete"],
                       continuous_fit_stable=True, rank_search_stationary_verified=False,
                       final_ranks=selection["final_ranks"],
                       final_npz_sha256=digest(args.output / "final.npz"))
    except Exception as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["source_unchanged"] = all(digest(args.code / path) == value for path, value in sources.items())
        (args.output / "run_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
