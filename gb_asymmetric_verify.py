from __future__ import annotations

"""Bounded verifier for the Z-ranked generalized-bicycle finalists."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
from pathlib import Path

import cpp_fast
from generalized_bicycle import build_matrices
from gb_aggressive_campaign import M, TARGET_D, _candidate_doc, _json_default
from hyper_validator import structural_validate


def _task(task):
    row, seed, trials, threads = task
    a = tuple(row["A"][0]); b = tuple(row["B"][0])
    hx, hz = build_matrices(M, a, b)
    gate = cpp_fast.css_ris_parallel(
        hx, hz, trials=int(trials), seed=int(seed), pair_depth=24,
        target=TARGET_D, stop_on_target=True, threads=int(threads))
    dx = gate["x"].get("best_weight")
    dz = gate["z"].get("best_weight")
    result = {"row": row, "gate": gate, "submission_sent": False,
              "git_commit_performed": False}
    if dx is not None and dz is not None:
        candidate = _candidate_doc(row, dx, dz, gate["x"]["witness"],
                                   gate["z"]["witness"], gate.get("mode"))
        result["candidate"] = candidate
        result["structural"] = structural_validate(candidate)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path,
                    default=Path("results/gb_asymmetric_campaign_01"))
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--trials", type=int, default=200_000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260902)
    args = ap.parse_args()
    data = json.loads((args.source / "z_deep.json").read_text())
    rows = data["rows"][:max(1, int(args.count))]
    tasks = [(row, int(args.seed) + i * 0x9E3779B9,
              int(args.trials), int(args.threads)) for i, row in enumerate(rows)]
    results = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = [pool.submit(_task, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda r: (
        -1 if r["gate"].get("dz_upper") is None else int(r["gate"]["dz_upper"]),
        -1 if r["gate"].get("dx_upper") is None else int(r["gate"]["dx_upper"])),
                  reverse=True)
    out = {"kind": "gb_asymmetric_gate_campaign",
           "submission_sent": False, "git_commit_performed": False,
           "trials_per_side": int(args.trials), "results": results}
    (args.source / "z_gate.json").write_text(
        json.dumps(out, indent=2, default=_json_default) + "\n")
    print(json.dumps({"count": len(results), "top": [
        {"hash": r["row"]["semantic_hash"][:8],
         "dx": r["gate"].get("dx_upper"),
         "dz": r["gate"].get("dz_upper"),
         "d": r["gate"].get("d_upper"),
         "stopped": bool(r["gate"]["x"].get("stopped_early") or
                          r["gate"]["z"].get("stopped_early"))}
        for r in results]}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
