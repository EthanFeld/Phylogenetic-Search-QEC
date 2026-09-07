from __future__ import annotations

"""Sample the full repeated-root factor-signature lattice for one-block caps."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import random
import time

import gb_m350_k188_repeated_campaign as campaign
from generalized_bicycle import common_gcd_degree


def _screen_task(task):
    row, seed, target_d, targeted_trials, replica = task
    campaign.core.TARGETED_IDEAL_TRIALS = int(targeted_trials)
    tagged = {**row, "_replica": int(replica)}
    return campaign.core._screen_task((tagged, 1, int(seed), 1, int(target_d)))


def _signature_pairs(parent, count: int, seed: int):
    rng = random.Random(int(seed))
    options = {
        0: ((0, 0), (0, 1), (1, 0), (0, 2), (2, 0)),
        1: ((1, 1), (1, 2), (2, 1)),
        2: ((2, 2),),
    }
    seen = set()
    attempts = max(1000, int(count) * 50)
    for _ in range(attempts):
        coordinates = [rng.choice(options[int(value)]) for value in parent]
        left = tuple(value[0] for value in coordinates)
        right = tuple(value[1] for value in coordinates)
        # Signature (2,...,2) is the zero polynomial in the quotient ring and
        # cannot define a nonempty circulant check block.
        if all(value == 2 for value in left) or all(value == 2 for value in right):
            continue
        key = tuple(sorted((left, right)))
        seen.add(key)
        if len(seen) >= int(count):
            break
    return sorted(seen)


def _row(parent_branch, left, right):
    a = campaign.core._support(campaign._divisor(left))
    b = campaign.core._support(campaign._divisor(right))
    if common_gcd_degree(campaign.M, a, b) != campaign.Q:
        raise ArithmeticError("signature pair lost q94 common gcd")
    key = campaign.core._pair_key(a, b)
    actual_signatures = (campaign._word_signature(key[0]),
                         campaign._word_signature(key[1]))
    return {
        "A": [list(key[0])], "B": [list(key[1])],
        "m": campaign.M, "n": campaign.N, "k": campaign.K,
        "gcd_degree": campaign.Q,
        "branch_id": parent_branch["branch_id"],
        "branch_index": parent_branch["branch_index"],
        "factor_indices": parent_branch["factor_indices"],
        "factor_exponents": parent_branch["factor_exponents"],
        "individual_factor_exponents": [list(x) for x in actual_signatures],
        "individual_gcd_degrees": [campaign.core._word_gcd_degree(key[0]),
                                   campaign.core._word_gcd_degree(key[1])],
        "word_weights": [len(key[0]), len(key[1])],
        "semantic_hash": campaign.core._semantic_hash(*key),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("results/gb_m350_signature_ceiling01.json"))
    parser.add_argument("--per-branch", type=int, default=128)
    parser.add_argument("--target-d", type=int, default=77)
    parser.add_argument("--targeted-trials", type=int, default=32)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    campaign.core.TARGETED_IDEAL_TRIALS = int(args.targeted_trials)
    started = time.perf_counter()

    rows = []
    sampled_by_branch = {}
    for index, branch in enumerate(campaign.enumerate_branches()):
        pairs = _signature_pairs(branch["factor_exponents"], args.per_branch,
                                 args.seed + index * 10007)
        sampled_by_branch[branch["branch_id"]] = len(pairs)
        rows.extend(_row(branch, left, right) for left, right in pairs)
    # Different signatures can have identical canonical divisor pairs only if
    # factor accounting is broken; hash dedupe is a defensive audit.
    rows = list({row["semantic_hash"]: row for row in rows}.values())
    print(json.dumps({"stage": "signature_sample", "candidates": len(rows)}),
          flush=True)

    tasks = [(row, args.seed + 1_000_003 + index * 0x9E3779B9 +
              replica * 0xD1B54A35, args.target_d, args.targeted_trials, replica)
             for index, row in enumerate(rows) for replica in range(args.replicas)]
    raw = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(_screen_task, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            raw.append(future.result())
            if done == 1 or done % 256 == 0 or done == len(futures):
                print(json.dumps({"stage": "signature_ceiling", "done": done,
                                  "total": len(futures)}), flush=True)
    grouped = {}
    for row in raw:
        grouped.setdefault(row["semantic_hash"], []).append(row)
    screened = []
    annihilator_escapes = []
    for items in grouped.values():
        winner = min(items, key=lambda row: int(row["screen"].get("d_upper") or
                                                campaign.N + 1))
        winner.pop("_replica", None)
        winner["screen"]["replicas"] = len(items)
        winner["screen"]["replica_modes"] = [
            item["screen"].get("mode", "css_ris") for item in items]
        screened.append(winner)
        if all(item["screen"].get("mode") !=
               "dynamic_one_block_annihilator_logical_ris" for item in items):
            annihilator_escapes.append(winner)
    histogram = {}
    for row in screened:
        key = str(row["screen"].get("d_upper"))
        histogram[key] = histogram.get(key, 0) + 1
    screened.sort(key=lambda row: int(row["screen"].get("d_upper") or -1),
                  reverse=True)
    report = {
        "kind": "gb_m350_q94_repeated_root_signature_ceiling",
        "hypothesis": (
            "one-block logical quotient depends only on individual repeated-root "
            "factor signatures; canonical divisors sample that quotient directly"),
        "target": {"n": campaign.N, "k": campaign.K, "d": args.target_d},
        "search": {"branches": 66, "per_branch": args.per_branch,
                   "sampled": len(rows), "replicas": args.replicas,
                   "targeted_trials": args.targeted_trials,
                   "annihilator_escapes": len(annihilator_escapes),
                   "distance_histogram": histogram,
                   "sampled_by_branch": sampled_by_branch,
                   "seconds_wall": time.perf_counter() - started},
        "annihilator_escapes": annihilator_escapes,
        "top": screened[:256],
        "regulation": {"stage_only": True, "submission_sent": False,
                       "git_commit_performed": False,
                       "distance_is_randomized_upper_bound": True,
                       "board_claim_allowed": False},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "sampled": len(rows),
                      "annihilator_escapes": len(annihilator_escapes),
                      "top_d_upper": [row["screen"].get("d_upper")
                                      for row in screened[:20]]}, indent=2))


if __name__ == "__main__":
    main()
