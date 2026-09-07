from __future__ import annotations

"""Measure GCD-ideal witness distance against randomized trial count."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

from ideal_x_logical_search import _ab_from_candidate, _poly, scan_candidate
from gf2_factor import degree, gcd


def _ladder_task(task: tuple[int, str, tuple[int, ...]]) -> tuple[int, dict]:
    candidate_index, candidate_name, budgets = task
    candidate = Path(candidate_name)
    doc = json.loads(candidate.read_text(encoding="utf-8"))
    series = []
    best = None
    cumulative_trials = 0
    for budget_index, budget in enumerate(budgets):
        report = scan_candidate(candidate, trials=int(budget), seed=20260905 + candidate_index * 100003 + budget_index * 1009,
                                pair_depth=24, max_seconds=None)
        found = report.get("found_X_logical") or {}
        raw_weight = found.get("weight")
        css_ok = bool((found.get("css_logical_check") or {}).get("ok"))
        weight = raw_weight if css_ok else None
        actual_trials = int((report.get("run") or {}).get("trials_run") or 0)
        cumulative_trials += actual_trials
        if weight is not None:
            best = int(weight) if best is None else min(best, int(weight))
        score = (int(doc["k"]) * best ** 2 / int(doc["n"])) if best is not None else None
        series.append({"trials_per_side_requested": int(budget),
                       "trials_per_side_actual": actual_trials,
                       "cumulative_trials_per_side": cumulative_trials,
                       "trial_accounting": "cumulative_compute_independent_seeds",
                       "best_ideal_weight": best,
                       "best_ideal_score_upper": score, "new_weight": weight,
                       "new_weight_raw": raw_weight, "witness_verified": css_ok,
                       "new_witness": found.get("block_support"),
                       "g_degree": report.get("algebra", {}).get("g_degree_K"),
                       "css_ok": css_ok})
    return candidate_index, {"candidate_path": str(candidate), "n": doc["n"], "k": doc["k"], "series": series,
                             "final_best_ideal_weight": best,
                             "final_best_ideal_score_upper": (int(doc["k"]) * best ** 2 / int(doc["n"])) if best is not None else None}


def _update_family_cache(cache_path: Path | None, rows: list[dict]) -> dict:
    """Promote independently CSS-verified GCD witnesses into the family cache."""
    if cache_path is None:
        return {"path": None, "updated": 0}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        payload = {}
    families = payload.get("families", payload) if isinstance(payload, dict) else {}
    if not isinstance(families, dict):
        families = {}
    updated = 0
    for row in rows:
        candidate = Path(row["candidate_path"])
        doc = json.loads(candidate.read_text(encoding="utf-8"))
        ell, a, b = _ab_from_candidate(doc)
        g = gcd(gcd(_poly(a, ell), _poly(b, ell)), (1 << ell) | 1)
        key = f"{ell}:{hex(int(g))}"
        key_value = [int(ell), hex(int(g))]
        # Only cache a witness that passed the full CSS logical check and belongs
        # to the expected ideal dimension.  Raw RIS hits are never promoted.
        for point in row.get("series", []):
            witness = point.get("new_witness")
            if not point.get("witness_verified") or not witness:
                continue
            if int(point.get("g_degree") or -1) != int(doc["k"]) // 2:
                continue
            weight = int(point.get("new_weight") or len(witness))
            old = families.get(key)
            if old is None or weight < int(old.get("weight", 10**9)):
                families[key] = {
                    "family_key": key_value,
                    "weight": weight,
                    "witness": [int(x) for x in witness],
                    "source_candidate_path": str(candidate.resolve()),
                    "validation": "gcd_ladder_css_verified",
                    "trials_per_side": int(point.get("trials_per_side_actual") or 0),
                    "cache_schema": "family_witness_v1",
                }
                updated += 1
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"schema_version": "1.0", "families": families},
                                     indent=2) + "\n", encoding="utf-8")
    return {"path": str(cache_path.resolve()), "updated": updated}


def run(candidates: list[Path], budgets: list[int], out: Path,
        workers: int | None = None, witness_cache: Path | None = None) -> dict:
    rows: list[dict | None] = [None] * len(candidates)
    max_workers = max(1, min(len(candidates), int(workers or min(8, os.cpu_count() or 1))))
    tasks = [(candidate_index, str(candidate.resolve()), tuple(int(x) for x in budgets))
             for candidate_index, candidate in enumerate(candidates)]
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_ladder_task, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            candidate_index, row = future.result()
            rows[candidate_index] = row
            print(json.dumps({"done": done, "total": len(candidates), "candidate": row["candidate_path"],
                              "best_weight": row["final_best_ideal_weight"]}), flush=True)
    rows = [row for row in rows if row is not None]
    payload = {"schema_version": "1.0", "kind": "gcd_ideal_trial_curve", "target_score": 1542.0,
               "budgets": budgets, "workers": max_workers, "candidates": rows,
               "regulation": {"distance_is_not_proven": True, "submission_sent": False, "git_commit_performed": False}}
    payload["witness_cache_update"] = _update_family_cache(witness_cache, rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/gcd_screen_ml_queue/gcd_ladder.json"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[1024, 4096, 16384, 65536, 262144])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--witness-cache", type=Path, default=None,
                        help="promote CSS-verified GCD witnesses into this family cache")
    args = parser.parse_args()
    run(args.candidates, args.budgets, args.out, args.workers, args.witness_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
