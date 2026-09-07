from __future__ import annotations

"""Large native RIS validation for one saved candidate document."""

import argparse
import json
from pathlib import Path

import cpp_fast
import distance_sketch as ds
from hyper_validator import structural_validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--trials", type=int, default=20_000_000)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--pair-depth", type=int, default=24)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    candidate = json.loads(args.candidate.read_text())
    hx = ds.supports_to_binary(candidate["checks"]["X"], candidate["n"])
    hz = ds.supports_to_binary(candidate["checks"]["Z"], candidate["n"])
    result = cpp_fast.css_ris_parallel(
        hx, hz, trials=int(args.trials), seed=int(args.seed),
        pair_depth=int(args.pair_depth), target=77,
        stop_on_target=True, threads=int(args.threads))
    out = {"kind": "candidate_deep_validation",
           "candidate_path": str(args.candidate.resolve()),
           "candidate": candidate, "run": result,
           "structural": structural_validate(candidate),
           "configured_trials_per_sector": int(args.trials),
           "early_stop_target": 76,
           "submission_sent": False, "git_commit_performed": False}
    out_path = args.out or args.candidate.with_name(
        args.candidate.stem + "_20m.json")
    out_path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps({
        "out": str(out_path.resolve()),
        "dx": result.get("dx_upper"), "dz": result.get("dz_upper"),
        "d": result.get("d_upper"),
        "x_trials": result["x"].get("trials_run"),
        "z_trials": result["z"].get("trials_run"),
        "stopped": bool(result["x"].get("stopped_early") or
                         result["z"].get("stopped_early")),
        "screen_refuted": result.get("screen_refuted"),
        "seconds": result.get("total_seconds")}, indent=2))


if __name__ == "__main__":
    main()
