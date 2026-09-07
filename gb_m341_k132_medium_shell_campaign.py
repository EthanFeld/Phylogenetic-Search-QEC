from __future__ import annotations

"""Medium-weight q=66 divisor-shell campaign for [[682,132,d]].

The sparse-divisor campaign covered only the low-weight tail.  This driver
enumerates the finite q=66 factor lattice without retaining its millions of
members, reservoir-samples divisor supports in medium shells, then delegates
scouting/mining/pairing/RIS screening to the trusted accelerated pipeline.

Stage-only.  No commit, PR, submission, or board claim.
"""

import argparse
import itertools
import json
import random
import time
from pathlib import Path

import gb_m341_k132_target as target
import gb_m345_k188_phylo_campaign as core
from gf2_factor import degree, factor_xm_plus_one, mul


M, N, Q, K = 341, 682, 66, 132
FACTORS = tuple(factor_xm_plus_one(M))
DEGREES = tuple(degree(factor) for factor in FACTORS)
SHAPES = ((5, 4), (3, 5), (1, 6))


def _branch(subset, divisor, weight, shape_index):
    subset = tuple(sorted(int(value) for value in subset))
    return {
        "branch_index": 0,
        "branch_id": (f"m341_q66_medium_w{int(weight)}_s{int(shape_index)}_"
                       + "-".join(map(str, subset))),
        "factor_indices": list(subset),
        "factor_degrees": [DEGREES[index] for index in subset],
        "divisor": int(divisor),
        "divisor_weight": int(weight),
        "inherited_q55_ancestors": [],
        "phylogenetic_arm": "q66_medium_divisor_shell",
        "shape_index": int(shape_index),
    }


def _medium_branches(seed: int, count: int, min_weight: int,
                     max_weight: int) -> list[dict]:
    """Reservoir-sample all exact-q=66 medium shells with bounded memory."""
    if min_weight > max_weight:
        raise ValueError("min divisor weight must not exceed max divisor weight")
    degree5 = tuple(index for index, value in enumerate(DEGREES) if value == 5)
    degree10 = tuple(index for index, value in enumerate(DEGREES) if value == 10)
    degree1 = tuple(index for index, value in enumerate(DEGREES) if value == 1)
    if (len(degree1), len(degree5), len(degree10)) != (1, 6, 31):
        raise ArithmeticError(f"unexpected Z341 factor spectrum: {DEGREES}")

    # Per-shell reservoirs keep exploration broad while avoiding an 8m-row
    # Python list.  Capacity scales with requested campaign size.
    shell_count = (max_weight - min_weight + 1) * len(SHAPES)
    reservoir_cap = max(96, (max(1, int(count)) * 2 + shell_count - 1)
                        // shell_count)
    reservoirs: dict[tuple[int, int], list[tuple[tuple[int, ...], int, int]]] = {}
    seen: dict[tuple[int, int], int] = {}
    rng = random.Random(int(seed))
    enumerated = 0
    matched = 0

    for shape_index, (take5, take10) in enumerate(SHAPES):
        for chosen5 in itertools.combinations(degree5, take5):
            partial = 1
            for index in chosen5:
                partial = mul(partial, FACTORS[index])
            for chosen10 in itertools.combinations(degree10, take10):
                enumerated += 1
                divisor = partial
                for index in chosen10:
                    divisor = mul(divisor, FACTORS[index])
                weight = int(divisor).bit_count()
                if not (int(min_weight) <= weight <= int(max_weight)):
                    continue
                matched += 1
                key = (int(weight), int(shape_index))
                seen[key] = seen.get(key, 0) + 1
                bucket = reservoirs.setdefault(key, [])
                item = (tuple(sorted((*degree1, *chosen5, *chosen10))),
                        int(divisor), int(weight))
                if len(bucket) < reservoir_cap:
                    bucket.append(item)
                else:
                    slot = rng.randrange(seen[key])
                    if slot < reservoir_cap:
                        bucket[slot] = item

    # Shuffle each shell, then round-robin sorted (weight, shape) buckets.
    # This prevents abundant high-weight shells from consuming campaign.
    for bucket in reservoirs.values():
        rng.shuffle(bucket)
    selected: list[tuple[tuple[int, ...], int, int, int]] = []
    buckets = {key: list(value) for key, value in sorted(reservoirs.items())}
    keys = list(buckets)
    while len(selected) < int(count) and any(buckets.values()):
        for key in keys:
            if len(selected) >= int(count):
                break
            if buckets[key]:
                subset, divisor, weight = buckets[key].pop()
                selected.append((subset, divisor, weight, key[1]))

    branches = [_branch(subset, divisor, weight, shape)
                for subset, divisor, weight, shape in selected]
    for index, branch in enumerate(branches):
        branch["branch_index"] = index
    print(json.dumps({
        "stage": "medium_shell_enumeration",
        "enumerated": enumerated,
        "matched": matched,
        "reservoir_cap": reservoir_cap,
        "shells": {f"w{weight}_s{shape}": len(rows)
                    for (weight, shape), rows in sorted(reservoirs.items())},
        "selected": len(branches),
    }), flush=True)
    return branches


def run(args):
    # target.run installs target.enumerate_branches into core by name. Replace
    # that module-global before call so core receives this bounded sampler.
    target.enumerate_branches = lambda _parent_report=None: _medium_branches(
        int(args.seed), int(args.branches), int(args.divisor_weight_min),
        int(args.divisor_weight_max))
    args.branch_mode = "sparse_divisor"
    report = target.run(args)
    report["kind"] = "gb_m341_k132_medium_shell_campaign"
    report["algebra"] = {
        "factor_spectrum": {"1": 1, "5": 6, "10": 31},
        "degree66_shapes": [[1, 5, 4], [1, 3, 5], [1, 1, 6]],
        "sampled_branches": int(args.branches),
        "exact_shape_count": "6*C(31,4) + C(6,3)*C(31,5) + 6*C(31,6)",
        "square_free": True,
        "branch_mode": "bounded_reservoir_medium_shell",
        "divisor_weight_range": [int(args.divisor_weight_min),
                                 int(args.divisor_weight_max)],
        "enumeration_memory": "O(shells * reservoir_cap)",
    }
    report["phylogenetic_hypothesis"] = {
        "representation": "square-free Z341 q66 factor-shape phylogeny",
        "branch_sampling": "reservoir-balanced medium divisor shells, all shapes",
        "nested_subideal_escape": "detectors outside every immediate factor child",
        "mutation": "exact-q66 parent XOR shifted nested descendants",
        "selection": "cycle-aware, branch-capped, diversity-preserving RIS beam",
        "reason": "low-weight q66 roots saturated at d85-88; medium shells probe new arms",
    }
    Path(args.out, "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("results/gb_m341_k132_medium_shell_01"))
    parser.add_argument("--parent-report", type=Path, default=Path("unused.json"))
    parser.add_argument("--factor-subset")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--leader-score", type=float,
                        default=target.LEADER_SCORE)
    parser.add_argument("--target-d", type=int, default=90)
    parser.add_argument("--plain-scout-runs", type=int, default=1)
    parser.add_argument("--exact-scout-runs", type=int, default=2)
    parser.add_argument("--quotient-scout-runs", type=int, default=1)
    parser.add_argument("--scout-trials", type=int, default=256)
    parser.add_argument("--branches", type=int, default=900)
    parser.add_argument("--branch-mode", default="sparse_divisor")
    parser.add_argument("--divisor-weight-min", type=int, default=23)
    parser.add_argument("--divisor-weight-max", type=int, default=31)
    parser.add_argument("--divisor-weight-max-unused", type=int, default=31)
    parser.add_argument("--mine-branches", type=int, default=300)
    parser.add_argument("--detector-runs", type=int, default=6)
    parser.add_argument("--detector-trials", type=int, default=768)
    parser.add_argument("--lattice-detector-runs", type=int, default=1)
    parser.add_argument("--child-runs", type=int, default=2)
    parser.add_argument("--child-trials", type=int, default=768)
    parser.add_argument("--max-companion-gcd", type=int, default=110)
    parser.add_argument("--ideal-iterations", type=int, default=75_000)
    parser.add_argument("--shell-iterations", type=int, default=75_000)
    parser.add_argument("--max-words", type=int, default=768)
    parser.add_argument("--hill-restarts", type=int, default=24)
    parser.add_argument("--hill-steps", type=int, default=1024)
    parser.add_argument("--hill-bases", type=int, default=8)
    parser.add_argument("--mutation-generations", type=int, default=2)
    parser.add_argument("--max-exact-weight", type=int, default=28)
    parser.add_argument("--min-check-weight", type=int, default=24)
    parser.add_argument("--pairs-per-branch", type=int, default=12)
    parser.add_argument("--pair-attempts", type=int, default=6_000)
    parser.add_argument("--proxy-trials", type=int, default=128)
    parser.add_argument("--proxy-replicas", type=int, default=1)
    parser.add_argument("--deep-per-branch", type=int, default=3)
    parser.add_argument("--deep-count", type=int, default=48)
    parser.add_argument("--deep-trials", type=int, default=10_000)
    parser.add_argument("--deep-replicas", type=int, default=2)
    parser.add_argument("--confirm-count", type=int, default=4)
    parser.add_argument("--confirm-trials", type=int, default=200_000)
    parser.add_argument("--confirm-replicas", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    started = time.perf_counter()
    report = run(args)
    print(json.dumps({"out": str(args.out), "target": report["target"],
                      "population": report["search"], "final": report["final"],
                      "seconds_wall": round(time.perf_counter() - started, 1)},
                     indent=2))


if __name__ == "__main__":
    main()
