from __future__ import annotations

"""Resume a mined Z_350 campaign with signature-diverse large pairing."""

import argparse
import json
from pathlib import Path
import time

import gb_m350_k188_repeated_campaign as campaign


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path,
                        default=Path("results/gb_m350_k188_adaptive_01.json"))
    parser.add_argument("--pairs-per-branch", type=int, default=256)
    parser.add_argument("--pair-attempts", type=int, default=250000)
    parser.add_argument("--min-check-weight", type=int, default=28)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--target-d", type=int, default=77)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--top", type=int, default=256)
    args = parser.parse_args()

    started = time.perf_counter()
    payload = json.loads(args.source.read_text())
    candidates = []
    branch_counts = {}
    for index, branch in enumerate(payload["mined"]):
        rows = campaign._pair_population(
            branch, count=args.pairs_per_branch, attempts=args.pair_attempts,
            min_check_weight=args.min_check_weight,
            seed=args.seed + index * 10007)
        candidates.extend(rows)
        branch_counts[branch["branch_id"]] = len(rows)
    print(json.dumps({"stage": "pair", "candidates": len(candidates),
                      "branches": sum(value > 0 for value in branch_counts.values())}),
          flush=True)

    screened = campaign.core._replicated_screen(
        candidates, replicas=args.replicas, trials=args.trials,
        seed=args.seed + 1_000_003, workers=args.workers, threads=args.threads,
        target_d=args.target_d, label="adaptive_proxy")
    screened.sort(key=lambda row: campaign.core._screen_quality(
        row, args.target_d), reverse=True)
    survivors = [row for row in screened
                 if campaign.core._survives(row, args.target_d)]
    mode_histogram = {}
    distance_histogram = {}
    signature_histogram = {}
    for row in screened:
        mode = row["screen"].get("mode", "css_ris")
        mode_histogram[mode] = mode_histogram.get(mode, 0) + 1
        distance = str(row["screen"].get("d_upper"))
        distance_histogram[distance] = distance_histogram.get(distance, 0) + 1
        signatures = row.get("individual_factor_exponents", [])
        signature = json.dumps(sorted(signatures), separators=(",", ":"))
        current = signature_histogram.setdefault(signature, {
            "count": 0, "best_d_upper": None, "survivors": 0})
        current["count"] += 1
        d_upper = row["screen"].get("d_upper")
        if d_upper is not None:
            current["best_d_upper"] = max(int(d_upper),
                                           int(current["best_d_upper"] or -1))
        current["survivors"] += int(campaign.core._survives(row, args.target_d))

    report = {
        "kind": "gb_m350_k188_signature_adaptive_continuation",
        "source": str(args.source.resolve()),
        "target": {"n": campaign.N, "k": campaign.K, "d": args.target_d,
                   "score": campaign.K * args.target_d ** 2 / campaign.N},
        "search": {"candidates": len(candidates), "screened": len(screened),
                   "survivors": len(survivors),
                   "pairs_per_branch": args.pairs_per_branch,
                   "pair_attempts": args.pair_attempts,
                   "replicas": args.replicas, "trials_per_replica": args.trials,
                   "branch_candidate_counts": branch_counts,
                   "mode_histogram": mode_histogram,
                   "distance_histogram": distance_histogram,
                   "signature_histogram": signature_histogram,
                   "seconds_wall": time.perf_counter() - started},
        "survivors": survivors,
        "top": screened[:args.top],
        "regulation": {"stage_only": True, "submission_sent": False,
                       "git_commit_performed": False,
                       "distance_is_randomized_upper_bound": True,
                       "board_claim_allowed": False},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "candidates": len(candidates),
                      "survivors": len(survivors),
                      "best": [(row["screen"].get("d_upper"), row["branch_id"],
                                row["individual_gcd_degrees"])
                               for row in screened[:10]]}, indent=2))


if __name__ == "__main__":
    main()
