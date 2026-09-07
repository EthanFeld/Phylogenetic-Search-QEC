from __future__ import annotations

"""Targeted same-family search for improving the weak Z sector.

The generalized-bicycle gcd constraint makes ordinary one-coordinate mutations
invalid. This campaign therefore keeps the Z_341, 16+16, k=182 construction and
enumerates valid cross-orbit pairings from the mined cyclic ideal words. It ranks
Z and X independently so a strong Z witness is not hidden by a weaker X side.
"""

import argparse
import json
import time
from pathlib import Path

from gb_aggressive_campaign import (
    M, ROOTS, _json_default, _make_candidates, _parallel_map,
    _pool_from_ideal, _task_screen,
)


def _score(row, field):
    value = row.get(field)
    return -1 if value is None else int(value)


def run(args):
    started = time.perf_counter()
    pool = _pool_from_ideal(args.miner_iterations, args.max_words, args.seed)
    candidates = _make_candidates(pool, 0, 1, args.seed + 17)
    print(json.dumps({"stage": "population", "ideal_words": len(pool),
                      "candidates": len(candidates)}), flush=True)
    screened = _parallel_map(
        _task_screen, candidates, seed=args.seed + 101,
        trials=args.proxy_trials, workers=args.workers, threads=args.threads,
        label="z_proxy", repeats=args.proxy_repeats)
    # Primary objective: maximize the observed Z-side upper bound. Secondary
    # objective: retain X-side strength and the global proxy. These are ranking
    # signals only; final distance remains verifier-owned.
    screened.sort(key=lambda r: (
        _score(r, "dz_proxy"), _score(r, "dx_proxy"), _score(r, "d_proxy"),
        r.get("semantic_hash", "")), reverse=True)
    for rank, row in enumerate(screened, 1):
        row["z_rank"] = rank
        row["z_target_clear"] = bool(_score(row, "dz_proxy") >= 77)
    report = {
        "schema_version": "1.0",
        "kind": "gb_asymmetric_z_target_campaign",
        "submission_sent": False,
        "git_commit_performed": False,
        "target": {"m": M, "n": 682, "k": 182,
                    "weak_sector": "Z", "desired_z": ">=77",
                    "check_weight": 32},
        "predictor": {
            "method": "sector-specific native CSS-RIS proxy",
            "proxy_trials_per_side": int(args.proxy_trials),
            "proxy_repeats": int(args.proxy_repeats),
            "ranking": "dz_proxy, then dx_proxy, then d_proxy",
            "proof_status": "ranking signal only",
        },
        "population": {"ideal_words": len(pool),
                       "candidate_count": len(candidates),
                       "root_lineages": sorted(ROOTS)},
        "z_clear_count": sum(bool(r["z_target_clear"]) for r in screened),
        "top_z": screened[:int(args.keep)],
        "seconds_wall": time.perf_counter() - started,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "asymmetric_campaign.json").write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n")
    (out / "z_ranked.json").write_text(
        json.dumps({"rows": screened, "submission_sent": False,
                    "git_commit_performed": False}, indent=2,
                   default=_json_default) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("results/gb_asymmetric_campaign_01"))
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--miner-iterations", type=int, default=50_000_000)
    ap.add_argument("--max-words", type=int, default=20_000)
    ap.add_argument("--proxy-trials", type=int, default=64)
    ap.add_argument("--proxy-repeats", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--keep", type=int, default=64)
    return run(ap.parse_args())


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=_json_default))
