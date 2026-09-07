from __future__ import annotations

"""Parallel common-gcd ideal sieve for existing GB candidate artifacts.

The common ideal depends only on g=gcd(a,b,x^ell-1), not on the rest of the
CSS pair.  This tool therefore deduplicates candidates by g and screens one
representative per gcd family before spending a full CSS RIS budget.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from ideal_x_logical_search import scan_candidate
from generalized_bicycle import build_matrices
from gf2_factor import gcd
from hyper_validator import structural_validate


def _poly(support: list[int] | tuple[int, ...], ell: int) -> int:
    value = 0
    for exponent in support:
        value ^= 1 << (int(exponent) % int(ell))
    return value


def _g_key(candidate: dict) -> str:
    ell = int(candidate["n"]) // 2
    first = candidate["checks"]["X"][0]
    a = tuple(int(value) for value in first if int(value) < ell)
    b = tuple(int(value) - ell for value in first if int(value) >= ell)
    modulus = (1 << ell) | 1
    g = gcd(gcd(_poly(a, ell), _poly(b, ell)), modulus)
    return hex(int(g))


def _task(row: tuple[str, int, int, int]) -> dict:
    path, trials, seed, pair_depth = row
    candidate_path = Path(path)
    try:
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        structural = structural_validate(candidate)
        if not structural["ok"]:
            return {"candidate": str(candidate_path.resolve()),
                    "status": "rejected_structural",
                    "error": structural.get("distance_errors") or
                             structural.get("commutation_violations")}
        report = scan_candidate(candidate_path, trials=int(trials), seed=int(seed),
                                pair_depth=int(pair_depth), max_seconds=None)
        found = report.get("found_X_logical") or {}
        return {
            "candidate": str(candidate_path.resolve()),
            "claimed_d": int(candidate.get("distance", {}).get("d", 0)),
            "g_hex": report.get("algebra", {}).get("g_polynomial_hex"),
            "g_degree": report.get("algebra", {}).get("g_degree_K"),
            "status": "scanned",
            "trials": int(trials),
            "seed": int(seed),
            "ideal_weight": found.get("weight"),
            "ideal_witness": found.get("block_support"),
            "ideal_css_ok": (found.get("css_logical_check") or {}).get("ok"),
            "h_divides": found.get("h_divides_word"),
            "ideal_search": report.get("run", {}).get("mode"),
            "regulation": {"submission_sent": False,
                            "git_commit_performed": False,
                            "distance_is_not_proven": True},
        }
    except (OSError, ValueError, ArithmeticError, KeyError, json.JSONDecodeError) as exc:
        return {"candidate": str(candidate_path.resolve()),
                "status": "rejected_input", "error_type": type(exc).__name__,
                "error": str(exc),
                "regulation": {"submission_sent": False,
                                "git_commit_performed": False,
                                "admissible_candidate": False}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--pattern", default="candidate_*_682_132_*.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-claimed-d", type=int, default=90)
    parser.add_argument("--trials", type=int, default=1024)
    parser.add_argument("--pair-depth", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    selected: dict[str, tuple[str, int]] = {}
    rejected = []
    for path in sorted(args.root.rglob(args.pattern)):
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
            claimed = int(candidate.get("distance", {}).get("d", 0))
            if claimed < int(args.min_claimed_d):
                continue
            structural = structural_validate(candidate)
            if not structural["ok"]:
                rejected.append({"candidate": str(path.resolve()),
                                 "status": "rejected_structural"})
                continue
            key = _g_key(candidate)
            previous = selected.get(key)
            if previous is None or claimed > previous[1]:
                selected[key] = (str(path), claimed)
        except (OSError, ValueError, ArithmeticError, KeyError,
                json.JSONDecodeError) as exc:
            rejected.append({"candidate": str(path.resolve()),
                             "status": "rejected_input",
                             "error_type": type(exc).__name__,
                             "error": str(exc)})

    tasks = [(path, int(args.trials), int(args.seed) + index * 10007,
              int(args.pair_depth))
             for index, (path, _claimed) in enumerate(selected.values())]
    results = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(_task, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            row = future.result()
            results.append(row)
            print(json.dumps({"done": done, "total": len(futures),
                              "candidate": row.get("candidate"),
                              "g_degree": row.get("g_degree"),
                              "ideal_weight": row.get("ideal_weight"),
                              "status": row.get("status")}), flush=True)

    results.sort(key=lambda row: (
        -int(row.get("ideal_weight") or 0),
        -int(row.get("claimed_d") or 0),
        str(row.get("candidate"))))
    payload = {
        "schema_version": "1.0",
        "kind": "parallel_common_gcd_ideal_frontier_sieve",
        "target": {"n": 682, "k": 132, "target_d": int(args.min_claimed_d)},
        "deduplication": {"key": "g=gcd(a,b,x^ell-1)",
                           "candidate_families": len(selected),
                           "candidate_artifacts_seen": len(list(args.root.rglob(args.pattern)))},
        "search": {"trials_per_family": int(args.trials),
                   "pair_depth": int(args.pair_depth),
                   "workers": int(args.workers), "seed": int(args.seed)},
        "results": results,
        "rejected": rejected,
        "regulation": {"submission_sent": False,
                        "git_commit_performed": False,
                        "distance_is_not_proven": True},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()),
                      "families": len(selected),
                      "scanned": len(results),
                      "rejected": len(rejected)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
