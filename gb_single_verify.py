from __future__ import annotations

"""Timeout-bounded deep gate for one saved generalized-bicycle finalist."""

import argparse
import json
from pathlib import Path

import cpp_fast
from generalized_bicycle import build_matrices
from gb_aggressive_campaign import M, TARGET_D, _candidate_doc, _json_default
from hyper_validator import structural_validate


def run(args):
    source = Path(args.source)
    data = json.loads((source / "refinement.json").read_text())
    rows = sorted(data.get("refined", []),
                  key=lambda r: ((r.get("d_proxy") or -1), r.get("semantic_hash", "")),
                  reverse=True)
    if not rows:
        raise RuntimeError("refinement.json has no rows")
    row = rows[int(args.index)]
    hx, hz = build_matrices(M, tuple(row["A"][0]), tuple(row["B"][0]))
    gate = cpp_fast.css_ris_parallel(
        hx, hz, trials=int(args.gate_trials), seed=int(args.seed), pair_depth=24,
        max_seconds_per_side=float(args.gate_seconds), target=TARGET_D,
        stop_on_target=True, threads=0)
    result = {"row": row, "gate": gate, "submission_sent": False,
              "git_commit_performed": False}
    if (gate.get("d_upper") is not None and gate["d_upper"] >= TARGET_D
            and int(args.full_trials) > 0):
        full = cpp_fast.css_ris_parallel(
            hx, hz, trials=int(args.full_trials), seed=int(args.seed) + 0x100000,
            pair_depth=24, max_seconds_per_side=(None if args.full_seconds <= 0
                                                 else float(args.full_seconds)),
            target=TARGET_D, stop_on_target=True, threads=0)
        result["full"] = full
        out = full
    else:
        out = gate
    dx, dz = out["x"].get("best_weight"), out["z"].get("best_weight")
    if dx is not None and dz is not None:
        candidate = _candidate_doc(row, dx, dz, out["x"]["witness"],
                                   out["z"]["witness"], out.get("mode"))
        path = source / f"single_verify_{min(dx, dz)}.json"
        path.write_text(json.dumps(candidate, indent=2) + "\n")
        result["candidate_path"] = str(path.resolve())
        result["candidate"] = candidate
        result["structural"] = structural_validate(candidate)
    result["gate_seconds_per_side"] = float(args.gate_seconds)
    result["full_trials_per_side"] = int(args.full_trials) if "full" in result else 0
    (source / "single_verify.json").write_text(
        json.dumps(result, indent=2, default=_json_default) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--gate-trials", type=int, default=200_000)
    ap.add_argument("--gate-seconds", type=float, default=180.0)
    ap.add_argument("--full-trials", type=int, default=2_000_000)
    ap.add_argument("--full-seconds", type=float, default=0.0,
                    help="optional full-pass timeout per side; 0 means unlimited")
    return run(ap.parse_args())


if __name__ == "__main__":
    result = main()
    print(json.dumps({"candidate_path": result.get("candidate_path"),
                      "gate": result.get("gate"), "full": result.get("full")},
                     indent=2, default=_json_default))
