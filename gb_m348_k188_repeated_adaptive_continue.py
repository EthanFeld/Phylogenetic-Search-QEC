from __future__ import annotations

"""Re-pair an existing m348 mining report after a pairing-policy fix.

No new word mining is performed.  This is deliberately stage-only: candidates
are screened with randomized, witness-backed upper bounds and are not
submitted or committed.
"""

import argparse
import json
from pathlib import Path
import time

import gb_m348_k188_repeated_campaign as wrapper


ENGINE = wrapper.campaign
# The source report came from the mixed campaign; importing the base m348
# wrapper defaults to exact-only, so explicitly restore complementary pairing.
ENGINE.PAIR_EXACT_ONLY = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path,
                        default=Path("results/gb_m348_k188_adaptive_01.json"))
    parser.add_argument("--pairs-per-branch", type=int, default=256)
    parser.add_argument("--pair-attempts", type=int, default=250_000)
    parser.add_argument("--min-check-weight", type=int, default=28)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--target-d", type=int, default=77)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--top", type=int, default=256)
    args = parser.parse_args()

    started = time.perf_counter()
    payload = json.loads(args.source.read_text())
    candidates = []
    branch_counts = {}
    for index, branch in enumerate(payload["mined"]):
        rows = ENGINE._pair_population(
            branch, count=args.pairs_per_branch, attempts=args.pair_attempts,
            min_check_weight=args.min_check_weight,
            seed=args.seed + index * 10007)
        candidates.extend(rows)
        branch_counts[branch["branch_id"]] = len(rows)
    raw_count = len(candidates)

    # Match the campaign's affine deduplication so a repaired pairing policy
    # does not reintroduce equivalent cyclic/unit-multiplier codes.
    affine_unique = {}
    for row in candidates:
        key = ENGINE.core._affine_pair_key(
            ENGINE.M, tuple(row["A"][0]), tuple(row["B"][0]))
        row["translation_semantic_hash"] = row["semantic_hash"]
        row["semantic_hash"] = ENGINE.core._semantic_hash(*key)
        row["affine_unit_canonical"] = True
        previous = affine_unique.get(row["semantic_hash"])
        if previous is None or ENGINE._candidate_prior(row) > ENGINE._candidate_prior(previous):
            affine_unique[row["semantic_hash"]] = row
    candidates = list(affine_unique.values())
    print(json.dumps({"stage": "pair", "raw": raw_count,
                      "affine_unique": len(candidates),
                      "branches": sum(value > 0 for value in branch_counts.values())}),
          flush=True)

    screened = ENGINE.core._replicated_screen(
        candidates, replicas=args.replicas, trials=args.trials,
        seed=args.seed + 1_000_003, workers=args.workers, threads=args.threads,
        target_d=args.target_d, label="m348_adaptive_proxy")
    screened.sort(key=lambda row: ENGINE.core._screen_quality(row, args.target_d),
                  reverse=True)
    survivors = [row for row in screened
                 if ENGINE.core._survives(row, args.target_d)]

    distance_histogram = {}
    mode_histogram = {}
    signature_histogram = {}
    for row in screened:
        screen = row.get("screen", {})
        distance = str(screen.get("d_upper"))
        distance_histogram[distance] = distance_histogram.get(distance, 0) + 1
        mode = screen.get("mode", "css_ris")
        mode_histogram[mode] = mode_histogram.get(mode, 0) + 1
        signature = json.dumps(sorted(row.get("individual_factor_exponents", [])),
                               separators=(",", ":"))
        current = signature_histogram.setdefault(signature, {
            "count": 0, "best_d_upper": None, "survivors": 0})
        current["count"] += 1
        if screen.get("d_upper") is not None:
            current["best_d_upper"] = max(
                int(screen["d_upper"]), int(current["best_d_upper"] or -1))
        current["survivors"] += int(ENGINE.core._survives(row, args.target_d))

    report = {
        "kind": "gb_m348_k188_repaired_pairing_screen",
        "source": str(args.source.resolve()),
        "target": {"m": ENGINE.M, "n": ENGINE.N, "k": ENGINE.K,
                   "d": args.target_d,
                   "score": ENGINE.K * args.target_d ** 2 / ENGINE.N},
        "search": {
            "raw_candidates": raw_count, "candidates": len(candidates),
            "screened": len(screened), "survivors": len(survivors),
            "pairs_per_branch": args.pairs_per_branch,
            "pair_attempts": args.pair_attempts,
            "branch_candidate_counts": branch_counts,
            "replicas": args.replicas, "trials_per_replica": args.trials,
            "distance_histogram": distance_histogram,
            "mode_histogram": mode_histogram,
            "signature_histogram": signature_histogram,
            "seconds_wall": time.perf_counter() - started,
        },
        "survivors": survivors,
        "top": screened[:args.top],
        "regulation": {"stage_only": True, "submission_sent": False,
                       "git_commit_performed": False,
                       "distance_is_randomized_upper_bound": True,
                       "official_gate_run": False, "board_claim_allowed": False},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "candidates": len(candidates),
                      "survivors": len(survivors),
                      "best": [(row["screen"].get("d_upper"), row["branch_id"],
                                row["word_weights"], row["individual_gcd_degrees"])
                               for row in screened[:10]]}, indent=2))


if __name__ == "__main__":
    main()
