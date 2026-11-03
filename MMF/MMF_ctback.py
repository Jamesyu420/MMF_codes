"""Sequential adds within a round, then one joint backfit.

A round walks cell types in order. Each type gets a 100-step conditional
forward fit. If that fit raises the training ELBO, the new program is
committed to the same model before the next type is considered. After the
walk, every program kept so far is backfitted together for 500 steps.
Programs whose RMS is at or below the threshold are then removed when
refitting without them raises the ELBO. The next round starts from that model.
"""
from pathlib import Path

import numpy as np

from MMF.backfit import add_program, copy_subset, ranks, raw_program_norms


def program_rms(model, program):
    for row in raw_program_norms(model, programs=[program]):
        if (row["cell_type"], row["program"]) == program:
            return row["rms"]
    raise KeyError(program)


def select_programs_ct(parent, *, seed_idx, seed, output, fit, score, export, write, restore,
                       tau=.065, k_max=3, forward_cycles=100, backfit_cycles=500,
                       max_passes=None, elbo_keep=.25, polish_cycles=1000):
    """Add several programs in type order, then backfit them together.

    ``fit`` with ``conditional_program`` set updates only that program.
    The round's single backfit updates every program in the model, plus the
    shared effects, proportions, and epsilon variance. ``restore`` is accepted
    so the notebook hook stays stable; this procedure keeps one model.
    """
    del restore
    if forward_cycles <= 0 or backfit_cycles <= 0:
        raise ValueError("forward_cycles and backfit_cycles must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    caps = np.broadcast_to(k_max, (parent.C,)).astype(int)
    if max_passes is None:
        max_passes = int(caps.max())
    report = dict(
        phase="selecting", threshold=tau, K_max=caps.tolist(), max_passes=max_passes,
        forward_cycles=forward_cycles, backfit_cycles=backfit_cycles, elbo_keep=elbo_keep,
        rank_selection="sequential adds, then one joint backfit",
        heldout_or_truth_read=False,
        rule=("Each round walks types in order and commits a program only when "
              "its forward ELBO is positive. After the walk, programs whose "
              "forward ELBO is below elbo_keep times the round maximum are "
              "removed before the joint backfit. Programs with RMS at or below "
              "tau are deleted when refitting without them raises the ELBO. "
              "After selection stops, the kept ranks are polished for "
              "polish_cycles steps with no new programs."),
        rounds=[], current_ranks=ranks(parent), selection_complete=False)

    def save():
        report["current_ranks"] = ranks(parent)
        write(output / "selection.json", report)

    def one_round(round_index):
        nonlocal parent
        start_ranks = ranks(parent)
        room = [c for c in range(parent.C) if parent.Kc[c] < caps[c]]
        entry = dict(round=round_index, parent_ranks=start_ranks, trials=[],
                     deletions=[], accepted_types=[])
        report["rounds"].append(entry)
        print(f"round {round_index}: ranks {start_ranks}", flush=True)
        save()

        def delete_weak(model, folder):
            checked = set()
            while True:
                weak = [row for row in raw_program_norms(model)
                        if row["rms"] <= tau and (row["cell_type"], row["program"]) not in checked]
                if not weak:
                    return model
                row = min(weak, key=lambda item: (item["rms"], item["cell_type"], item["program"]))
                program = (row["cell_type"], row["program"])
                before = score(model)
                reduced, _ = copy_subset(model, [q for q in model.programs if q != program])
                prefix = folder / f"delete_{len(entry['deletions']):02d}_T{program[0]}P{program[1]}"
                fit(reduced, "backfit", prefix, backfit_cycles)
                after = score(reduced)
                removed = bool(after > before)
                entry["deletions"].append(dict(
                    cell_type=program[0], program=program[1], rms=row["rms"],
                    delta_elbo=after - before, removed=removed,
                    before_ranks=ranks(model), after_ranks=ranks(reduced)))
                decision = "removed" if removed else "kept"
                print(f"  delete type {program[0] + 1} program {program[1] + 1}: {decision}"
                      f"  dELBO={after - before:+.4g}  rms={row['rms']:.4f}", flush=True)
                if removed:
                    model = reduced
                    checked = set()
                    export(model, prefix / "accepted")
                else:
                    del reduced
                    checked.add(program)
                save()

        for c in room:
            baseline = score(parent)
            trial_seed = int(seed + 100003 * round_index + 1009 * c)
            candidate = add_program(parent, c, seed_idx, trial_seed)
            program = (c, parent.Kc[c])
            prefix = output / f"round{round_index:02d}" / f"type{c}"
            fit(candidate, "forward", prefix / "forward", forward_cycles, conditional_program=program)
            delta = score(candidate) - baseline
            norm = program_rms(candidate, program)
            accepted = bool(delta > 0)
            entry["trials"].append(dict(
                cell_type=c, program=program[1], seed=trial_seed,
                forward_delta_elbo=delta, forward_rms=norm, accepted=accepted))
            print(f"  forward type {c + 1}: {'ADDED' if accepted else 'ineligible'}"
                  f"  dELBO={delta:+.4g}  rms={norm:.4f}", flush=True)
            if accepted:
                parent = candidate
                entry["accepted_types"].append(c)
            else:
                del candidate
            save()

        added = [trial for trial in entry["trials"] if trial["accepted"]]
        if not added:
            print("  no program added; skip joint backfit", flush=True)
            return False
        best = max(trial["forward_delta_elbo"] for trial in added)
        cutoff = elbo_keep * best
        entry["forward_elbo_max"] = best
        entry["forward_elbo_cutoff"] = cutoff
        drop = {(trial["cell_type"], trial["program"]) for trial in added
                if trial["forward_delta_elbo"] < cutoff}
        print(f"  forward max dELBO={best:+.4g}  cutoff={cutoff:+.4g}", flush=True)
        for trial in added:
            if (trial["cell_type"], trial["program"]) not in drop:
                continue
            trial["dropped_small_elbo"] = True
            print(f"  drop type {trial['cell_type'] + 1} program {trial['program'] + 1}: "
                  f"dELBO={trial['forward_delta_elbo']:+.4g}", flush=True)
        if drop:
            parent, _ = copy_subset(parent, [p for p in parent.programs if tuple(p) not in drop])
            entry["accepted_types"] = [c for c in entry["accepted_types"] if c not in {item[0] for item in drop}]
            print(f"  ranks after drop {ranks(parent)}", flush=True)
            save()
        if ranks(parent) == start_ranks:
            print("  no program left above the cutoff; skip joint backfit", flush=True)
            return False
        print(f"  joint backfit ranks {ranks(parent)}", flush=True)
        prefix = output / f"round{round_index:02d}"
        fit(parent, "backfit", prefix / "backfit", backfit_cycles)
        parent = delete_weak(parent, prefix / "cleanup")
        entry["retained_ranks"] = ranks(parent)
        export(parent, prefix / "accepted")
        print(f"  ranks now {entry['retained_ranks']}", flush=True)
        save()
        return entry["retained_ranks"] != start_ranks

    save()
    stop_reason = "reached the round limit"
    for round_index in range(1, max_passes + 1):
        if not any(parent.Kc[c] < caps[c] for c in range(parent.C)):
            stop_reason = "every cell type is at K_max"
            break
        if not one_round(round_index):
            stop_reason = "a full round kept no new program"
            break
    print(f"  selection stopped; polish {polish_cycles} steps at the current ranks", flush=True)
    fit(parent, "polish", output / "polish", polish_cycles)
    export(parent, output / "final")
    report.update(phase="finished", selection_complete=True, stop_reason=stop_reason,
                  final_ranks=ranks(parent), final_training_elbo=score(parent),
                  final_program_norms=raw_program_norms(parent))
    save()
    return parent, report
