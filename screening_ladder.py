from __future__ import annotations

"""Independent RIS screening ladder for large CSS candidates.

Runs fresh native parallel trials at explicit budgets, validates every returned
witness in Python GF(2), transfers improvements through an audited code
symmetry, then runs the trusted challenge gate on the refined artifact.
"""

import argparse
import copy
import json
from pathlib import Path
import time

import cpp_fast
from hyper_validator import (DEFAULT_STAGES, _logical_check, block_swap_reverse_support,
                             run_official_gate, structural_validate, symmetry_audit)


DEFAULT_LADDER = (200_000, 2_000_000, 20_000_000)
SEED_STEP = 0x9E3779B9


def _install_if_better(candidate: dict, side: str, weight: int | None,
                       witness: list[int] | None, best: dict, checks: dict) -> bool:
    if weight is None or witness is None:
        return False
    if not checks.get("ok") or int(weight) >= int(best[side]):
        return False
    best[side] = int(weight)
    candidate["distance"][side]["value"] = int(weight)
    candidate["distance"][side]["witness"] = [int(x) for x in witness]
    return True


def run_ladder(path: Path, *, budgets=DEFAULT_LADDER, seed: int = 20260829,
               threads: int = 0, pair_depth: int = 8,
               out: Path = Path("results/screening_ladder/run"),
               challenge_root: Path = Path("challenge_data")) -> dict:
    source = json.loads(path.read_text())
    structural = structural_validate(source)
    if not structural["ok"]:
        report = {"status": "invalid", "structural": structural}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        return report
    if not cpp_fast.available():
        raise RuntimeError("qldpc_fast native library unavailable")

    candidate = copy.deepcopy(source)
    hx, hz = structural["hx"], structural["hz"]
    symmetry = symmetry_audit(candidate, structural)
    best = {"X": int(candidate["distance"]["X"]["value"]),
            "Z": int(candidate["distance"]["Z"]["value"])}
    stages = []
    total_trials = 0
    t0 = time.perf_counter()

    def transfer_symmetry() -> dict:
        transfers = {}
        if not symmetry.get("present"):
            return transfers
        m = int(candidate["n"]) // 2
        for src, dst, kernel, row in (("X", "Z", hx, hz), ("Z", "X", hz, hx)):
            mapped = block_swap_reverse_support(candidate["distance"][src]["witness"], m)
            check = _logical_check(mapped, kernel, row)
            transfers[f"{src}_to_{dst}"] = {k: v for k, v in check.items()
                                              if k != "row_rank"}
            _install_if_better(candidate, dst, len(mapped), mapped, best, check)
        candidate["distance"]["d"] = min(best.values())
        return transfers

    initial_transfer = transfer_symmetry()
    for index, budget in enumerate(int(x) for x in budgets if int(x) > 0):
        stage_seed = int(seed) + index * SEED_STEP
        result = cpp_fast.css_ris_parallel(
            hx, hz, trials=budget, seed=stage_seed, pair_depth=int(pair_depth),
            target=None, stop_on_target=False, threads=int(threads))
        stage = {"stage": index + 1, "trials_per_side": budget,
                 "seed": stage_seed, "mode": result.get("mode"),
                 "seconds": result.get("total_seconds"),
                 "throughput_trials_per_second": (
                     2 * budget / result["total_seconds"]
                     if result.get("total_seconds") else None),
                 "sectors": {}, "symmetry_transfers": {}}
        for side, kernel, row in (("X", hz, hx), ("Z", hx, hz)):
            native = result.get(side.lower(), {})
            weight, witness = native.get("best_weight"), native.get("witness")
            check = ({"ok": True, "present": False} if weight is None or witness is None
                     else _logical_check(witness, kernel, row))
            stage["sectors"][side] = {
                "native_best": weight, "native_trials": native.get("trials_run"),
                "witness_check": {k: v for k, v in check.items() if k != "row_rank"},
            }
            _install_if_better(candidate, side, weight, witness, best, check)
        stage["symmetry_transfers"] = transfer_symmetry()
        candidate["distance"]["d"] = min(best.values())
        stage["d_after"] = candidate["distance"]["d"]
        stage["score_upper"] = (int(candidate["k"]) * stage["d_after"] ** 2 /
                                 int(candidate["n"]))
        total_trials += 2 * budget
        stages.append(stage)
        # Crash-safe progress artifact; final report written after trusted gate.
        checkpoint = {
            "status": "running", "source": str(path.resolve()),
            "completed_stages": stages, "best": {"dx": best["X"], "dz": best["Z"],
                                                   "d": min(best.values())},
            "seconds": time.perf_counter() - t0,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_name(out.stem + "_checkpoint.json").write_text(
            json.dumps(checkpoint, indent=2, default=str) + "\n")

    d = min(best.values())
    candidate["distance"]["d"] = d
    out.parent.mkdir(parents=True, exist_ok=True)
    refined_path = out.with_name(f"{out.stem}_{candidate['n']}_{candidate['k']}_{d}.json")
    gate = run_official_gate(candidate, refined_path, challenge_root)
    symmetry_final = symmetry_audit(candidate, structural)
    report = {
        "status": "ok", "source": str(path.resolve()),
        "candidate_path": str(refined_path.resolve()),
        "ladder": {"budgets_per_side": [int(x) for x in budgets if int(x) > 0],
                    "total_trials_both_sectors": total_trials, "seed": int(seed),
                    "threads": int(threads), "pair_depth": int(pair_depth),
                    "seconds": time.perf_counter() - t0},
        "structural": {k: v for k, v in structural.items() if k not in ("hx", "hz")},
        "symmetry_initial": symmetry,
        "initial_symmetry_transfers": initial_transfer,
        "stages": stages,
        "final": {"n": candidate["n"], "k": candidate["k"], "dx": best["X"],
                  "dz": best["Z"], "d": d,
                  "score_upper": int(candidate["k"]) * d * d / int(candidate["n"])},
        "symmetry_final": symmetry_final,
        "official_gate": gate,
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/screening_ladder/run"))
    ap.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_LADDER))
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--pair-depth", type=int, default=8)
    ap.add_argument("--challenge-root", type=Path,
                    default=Path("challenge_data"))
    args = ap.parse_args()
    report = run_ladder(args.candidate, budgets=args.budgets, seed=args.seed,
                        threads=args.threads, pair_depth=args.pair_depth,
                        out=args.out, challenge_root=args.challenge_root)
    print(json.dumps({k: report.get(k) for k in ("status", "final", "ladder",
                                                  "candidate_path", "official_gate")
                      if k in report}, indent=2, default=str))
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
