from __future__ import annotations

"""Full early-stop validation for the strongest asymmetric-gate survivors."""

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
    result = cpp_fast.css_ris_parallel(
        hx, hz, trials=int(trials), seed=int(seed), pair_depth=24,
        target=TARGET_D, stop_on_target=True, threads=int(threads))
    dx = result["x"].get("best_weight")
    dz = result["z"].get("best_weight")
    out = {"row": row, "full": result, "submission_sent": False,
           "git_commit_performed": False}
    if dx is not None and dz is not None:
        candidate = _candidate_doc(row, dx, dz, result["x"]["witness"],
                                   result["z"]["witness"], result.get("mode"))
        out["candidate"] = candidate
        out["structural"] = structural_validate(candidate)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path,
                    default=Path("results/gb_asymmetric_campaign_01"))
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--trials", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()
    gate = json.loads((args.source / "z_gate.json").read_text())
    eligible = [r["row"] for r in gate["results"]
                if not (r["gate"]["x"].get("stopped_early") or
                         r["gate"]["z"].get("stopped_early"))]
    eligible.sort(key=lambda r: r.get("z_rank", 10**9))
    rows = eligible[:max(1, int(args.count))]
    tasks = [(row, int(args.seed) + i * 0x9E3779B9,
              int(args.trials), int(args.threads)) for i, row in enumerate(rows)]
    results = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = [pool.submit(_task, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda r: (
        -1 if r["full"].get("dz_upper") is None else int(r["full"]["dz_upper"]),
        -1 if r["full"].get("dx_upper") is None else int(r["full"]["dx_upper"])),
                  reverse=True)
    out = {"kind": "gb_asymmetric_full_campaign",
           "submission_sent": False, "git_commit_performed": False,
           "configured_trials_per_side": int(args.trials),
           "results": results}
    (args.source / "z_full.json").write_text(
        json.dumps(out, indent=2, default=_json_default) + "\n")
    for i, result in enumerate(results):
        candidate = result.get("candidate")
        if candidate is None:
            continue
        d = min(candidate["distance"]["X"]["value"],
                candidate["distance"]["Z"]["value"])
        path = args.source / f"candidate_zfull_{i}_{d}.json"
        path.write_text(json.dumps(candidate, indent=2) + "\n")
        result["candidate_path"] = str(path.resolve())
    (args.source / "z_full.json").write_text(
        json.dumps(out, indent=2, default=_json_default) + "\n")
    print(json.dumps({"count": len(results), "top": [
        {"hash": r["row"]["semantic_hash"][:8],
         "dx": r["full"].get("dx_upper"),
         "dz": r["full"].get("dz_upper"),
         "d": r["full"].get("d_upper"),
         "x_trials": r["full"]["x"].get("trials_run"),
         "z_trials": r["full"]["z"].get("trials_run"),
         "stopped": bool(r["full"]["x"].get("stopped_early") or
                          r["full"]["z"].get("stopped_early"))}
        for r in results]}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
