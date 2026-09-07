from __future__ import annotations

"""Targeted Z_341 generalized-bicycle search for [[682,132,d]].

q=66 has three factor-shape branches in x^341+1:
  x+1 + 5 degree-5 factors + 4 degree-10 factors
  x+1 + 3 degree-5 factors + 5 degree-10 factors
  x+1 + 1 degree-5 factor  + 6 degree-10 factors

This wrapper reuses the trusted phylogenetic/factor-lattice pipeline while
sampling each shape evenly. It is stage-only: no submission, commit, or PR.
Distance remains a randomized upper bound until the challenge verifier passes.
"""

import argparse
import itertools
import json
import random
from pathlib import Path

import gb_m345_k188_phylo_campaign as core
from gf2_factor import degree, factor_xm_plus_one, mul
from generalized_bicycle import build_supports, normalize_code


M, N, Q, K = 341, 682, 66, 132
CHECK_CAP = 32
LEADER_SCORE = 182 * 76 * 76 / 682
FACTORS = tuple(factor_xm_plus_one(M))
DEGREES = tuple(degree(factor) for factor in FACTORS)
_BRANCH_MODE = "random_shape"
_DIVISOR_WEIGHT_MAX = 21
_REQUESTED_SUBSET = None

core.M, core.N, core.Q, core.K = M, N, Q, K
core.CHECK_CAP = CHECK_CAP
core.LEADER_SCORE = LEADER_SCORE
core.MODULUS = (1 << M) | 1
core.KILLER_IDEAL_FACTOR_SETS = ()
core.ONE_BLOCK_KILLER_IDEAL_FACTOR_SETS = ()
core._two_block_ideal_span.cache_clear()
core._one_block_ideal_span.cache_clear()
core._one_block_divisor_span.cache_clear()


def _sample_branches(seed: int, count: int) -> list[dict]:
    degree5 = tuple(index for index, value in enumerate(DEGREES) if value == 5)
    degree10 = tuple(index for index, value in enumerate(DEGREES) if value == 10)
    degree1 = tuple(index for index, value in enumerate(DEGREES) if value == 1)
    if (len(degree1), len(degree5), len(degree10)) != (1, 6, 31):
        raise ArithmeticError(f"unexpected Z341 factor spectrum: {DEGREES}")

    shapes = ((5, 4), (3, 5), (1, 6))
    rng = random.Random(int(seed))
    quota = max(1, (int(count) + len(shapes) - 1) // len(shapes))
    branches = []
    seen = set()
    for shape_index, (take5, take10) in enumerate(shapes):
        while sum(item["shape_index"] == shape_index for item in branches) < quota:
            subset = tuple(sorted((*degree1,
                                   *rng.sample(degree5, take5),
                                   *rng.sample(degree10, take10))))
            if subset in seen:
                continue
            seen.add(subset)
            divisor = 1
            for index in subset:
                divisor = mul(divisor, FACTORS[index])
            branches.append({
                "branch_index": len(branches),
                "branch_id": f"m341_q66_s{shape_index}_" +
                             "-".join(map(str, subset)),
                "factor_indices": list(subset),
                "factor_degrees": [DEGREES[index] for index in subset],
                "divisor": int(divisor),
                "divisor_weight": int(divisor).bit_count(),
                "inherited_q55_ancestors": [],
                "phylogenetic_arm": f"q66_shape_{shape_index}_out_of_sample",
                "shape_index": shape_index,
            })
    return branches[:max(0, int(count))]


def _sparse_branches(seed: int, count: int) -> list[dict]:
    """Sample exact-q branches from the sparse-divisor tail.

    The ordinary sampler almost never reaches this tail: only 1,448 of the
    8,004,696 q=66 divisors have support weight <=21.  Those divisors are
    useful roots because their ideal contains a broad, non-tensor sparse
    shell.  Enumerating the finite lattice is an algebraic gate; distance
    still comes only from the later randomized RIS screens.
    """
    degree5 = tuple(index for index, value in enumerate(DEGREES) if value == 5)
    degree10 = tuple(index for index, value in enumerate(DEGREES) if value == 10)
    shapes = ((5, 4), (3, 5), (1, 6))
    pool = []
    for shape_index, (take5, take10) in enumerate(shapes):
        for chosen5 in itertools.combinations(degree5, take5):
            partial = 1
            for index in chosen5:
                partial = mul(partial, FACTORS[index])
            for chosen10 in itertools.combinations(degree10, take10):
                subset = tuple(sorted((0, *chosen5, *chosen10)))
                divisor = partial
                for index in chosen10:
                    divisor = mul(divisor, FACTORS[index])
                weight = int(divisor).bit_count()
                if weight <= int(_DIVISOR_WEIGHT_MAX):
                    pool.append({
                        "subset": subset, "divisor": int(divisor),
                        "divisor_weight": weight, "shape_index": shape_index,
                    })
    rng = random.Random(int(seed))
    rng.shuffle(pool)
    # Stratify by shape first, then by divisor weight.  This avoids spending
    # the whole sparse budget on the 1+5+4 branch.
    pool.sort(key=lambda row: (row["shape_index"], row["divisor_weight"]))
    selected = []
    for shape_index in range(len(shapes)):
        selected.extend(row for row in pool if row["shape_index"] == shape_index)
    if len(selected) > int(count):
        # Round-robin gives every shape a chance; within a shape retain
        # weight diversity by cycling through the shuffled equal-weight pool.
        buckets = {shape: [row for row in selected if row["shape_index"] == shape]
                   for shape in range(len(shapes))}
        out = []
        while len(out) < int(count) and any(buckets.values()):
            for shape in range(len(shapes)):
                if buckets[shape] and len(out) < int(count):
                    out.append(buckets[shape].pop(0))
        selected = out
    branches = []
    for item in selected:
        subset = item["subset"]
        branches.append({
            "branch_index": len(branches),
            "branch_id": f"m341_q66_sparse_w{item['divisor_weight']}_s"
                          f"{item['shape_index']}_" + "-".join(map(str, subset)),
            "factor_indices": list(subset),
            "factor_degrees": [DEGREES[index] for index in subset],
            "divisor": item["divisor"],
            "divisor_weight": item["divisor_weight"],
            "inherited_q55_ancestors": [],
            "phylogenetic_arm": "q66_sparse_divisor_tail",
            "shape_index": item["shape_index"],
        })
    return branches


def _explicit_branch(subset) -> list[dict]:
    subset = tuple(sorted(int(x) for x in subset))
    if 0 not in subset or sum(DEGREES[index] for index in subset) != Q:
        raise ValueError(f"factor subset {subset} is not a degree-{Q} branch")
    divisor = 1
    for index in subset:
        divisor = mul(divisor, FACTORS[index])
    return [{
        "branch_index": 0,
        "branch_id": "m341_q66_explicit_" + "-".join(map(str, subset)),
        "factor_indices": list(subset),
        "factor_degrees": [DEGREES[index] for index in subset],
        "divisor": int(divisor), "divisor_weight": int(divisor).bit_count(),
        "inherited_q55_ancestors": [],
        "phylogenetic_arm": "q66_explicit_branch",
        "shape_index": -1,
    }]


def enumerate_branches(_parent_report=None):
    # Parent report intentionally ignored: target q=66 has no trusted parent
    # lineage. Equal shape quotas prevent factor-family saturation.
    if _REQUESTED_SUBSET is not None:
        return _explicit_branch(_REQUESTED_SUBSET)
    if _BRANCH_MODE == "sparse_divisor":
        return _sparse_branches(core._TARGET_BRANCH_SEED,
                                core._TARGET_BRANCH_COUNT)
    return _sample_branches(core._TARGET_BRANCH_SEED,
                            core._TARGET_BRANCH_COUNT)


def _candidate_doc(row: dict, stage: str) -> dict | None:
    screen = row.get("screen", {})
    dx, dz = screen.get("dx_upper"), screen.get("dz_upper")
    wx = row.get("screen_witnesses", {}).get("X")
    wz = row.get("screen_witnesses", {}).get("Z")
    if dx is None or dz is None or not wx or not wz:
        return None
    code = normalize_code(M, row["A"][0], row["B"][0])
    hx, hz = build_supports(M, code.a, code.b)
    distance = min(int(dx), int(dz))
    return {
        "schema_version": "0.1",
        "name": f"[[{N},{K},d<={distance}]] Z_341 q66 phylogenetic GB",
        "code_type": "CSS", "n": N, "k": K,
        "checks": {"X": hx, "Z": hz},
        "distance": {
            "d": distance,
            "X": {"value": int(dx), "confidence": "upper_bound",
                  "witness": list(wx)},
            "Z": {"value": int(dz), "confidence": "upper_bound",
                  "witness": list(wz)},
        },
        "family": "generalized-bicycle",
        "provenance": {
            "authors": ["stage-only autonomous research"],
            "origin": "gb_m341_k132_target",
            "novelty": "unknown",
            "model": "GPT-5 Codex + phylogenetic factor-lattice search",
            "construction": (f"Z_341 GB q66; factor indices={row['factor_indices']}; "
                             f"A={tuple(code.a)}; B={tuple(code.b)}"),
            "references": [],
            "notes": f"stage={stage}; randomized witness-backed upper bound",
        },
        "search": {
            "stage": stage,
            "semantic_hash": row.get("semantic_hash"),
            "branch_id": row.get("branch_id"),
            "factor_indices": row.get("factor_indices"),
            "factor_degrees": row.get("factor_degrees"),
            "phylogenetic_arm": row.get("phylogenetic_arm"),
            "cycle_profile": row.get("cycle_profile"),
        },
        "regulation": {
            "stage_only": True, "submission_sent": False,
            "git_commit_performed": False, "official_gate_status": "not_run",
            "board_claim_allowed": False, "distance_is_not_proven": True,
            "literature_novelty": "unverified",
        },
    }


def run(args):
    core._TARGET_BRANCH_SEED = int(args.seed) ^ 0x66A11CE
    core._TARGET_BRANCH_COUNT = int(args.branches)
    global _BRANCH_MODE, _DIVISOR_WEIGHT_MAX, _REQUESTED_SUBSET
    _BRANCH_MODE = str(args.branch_mode)
    _DIVISOR_WEIGHT_MAX = int(args.divisor_weight_max)
    _REQUESTED_SUBSET = (tuple(sorted(int(value) for value in
                                      str(args.factor_subset).split(",")
                                      if value.strip()))
                         if args.factor_subset else None)
    core.enumerate_branches = enumerate_branches
    core._candidate_doc = _candidate_doc
    report = core.run(args)
    report["kind"] = "gb_m341_k132_target_phylogeny_campaign"
    report["target"]["reference_bar"] = {
        "code": "[[682,182,76]]",
        "score": LEADER_SCORE,
        "required_d_at_k132": 90,
        "target_91_score": K * 91 * 91 / N,
    }
    report["algebra"] = {
        "factor_spectrum": {"1": 1, "5": 6, "10": 31},
        "degree66_shapes": [[1, 5, 4], [1, 3, 5], [1, 1, 6]],
        "sampled_branches": int(args.branches),
        "exact_shape_count": "6*C(31,4) + C(6,3)*C(31,5) + 6*C(31,6)",
        "square_free": True,
        "branch_mode": _BRANCH_MODE,
        "divisor_weight_max": _DIVISOR_WEIGHT_MAX,
    }
    report["phylogenetic_hypothesis"] = {
        "representation": "square-free Z341 q66 factor-shape phylogeny",
        "branch_sampling": "equal quota across all three exact-degree shapes",
        "nested_subideal_escape": "detectors outside every immediate factor child",
        "mutation": "exact-q66 parent XOR shifted nested descendants",
        "selection": "cycle-aware, branch-capped, diversity-preserving RIS beam",
    }
    Path(args.out, "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("results/gb_m341_k132_target_01"))
    parser.add_argument("--parent-report", type=Path, default=Path("unused.json"))
    parser.add_argument("--factor-subset")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--leader-score", type=float, default=LEADER_SCORE)
    parser.add_argument("--target-d", type=int, default=90)
    parser.add_argument("--plain-scout-runs", type=int, default=2)
    parser.add_argument("--exact-scout-runs", type=int, default=4)
    parser.add_argument("--quotient-scout-runs", type=int, default=2)
    parser.add_argument("--scout-trials", type=int, default=512)
    parser.add_argument("--branches", type=int, default=24)
    parser.add_argument("--branch-mode", choices=("random_shape", "sparse_divisor"),
                        default="random_shape")
    parser.add_argument("--divisor-weight-max", type=int, default=21)
    parser.add_argument("--mine-branches", type=int, default=24)
    parser.add_argument("--detector-runs", type=int, default=16)
    parser.add_argument("--detector-trials", type=int, default=2048)
    parser.add_argument("--lattice-detector-runs", type=int, default=1)
    parser.add_argument("--child-runs", type=int, default=4)
    parser.add_argument("--child-trials", type=int, default=2048)
    parser.add_argument("--max-companion-gcd", type=int, default=110)
    parser.add_argument("--ideal-iterations", type=int, default=250_000)
    parser.add_argument("--shell-iterations", type=int, default=250_000)
    parser.add_argument("--max-words", type=int, default=2048)
    parser.add_argument("--hill-restarts", type=int, default=24)
    parser.add_argument("--hill-steps", type=int, default=1024)
    parser.add_argument("--hill-bases", type=int, default=8)
    parser.add_argument("--mutation-generations", type=int, default=3)
    parser.add_argument("--max-exact-weight", type=int, default=28)
    parser.add_argument("--min-check-weight", type=int, default=24)
    parser.add_argument("--pairs-per-branch", type=int, default=32)
    parser.add_argument("--pair-attempts", type=int, default=20_000)
    parser.add_argument("--proxy-trials", type=int, default=256)
    parser.add_argument("--proxy-replicas", type=int, default=2)
    parser.add_argument("--deep-per-branch", type=int, default=2)
    parser.add_argument("--deep-count", type=int, default=32)
    parser.add_argument("--deep-trials", type=int, default=20_000)
    parser.add_argument("--deep-replicas", type=int, default=4)
    parser.add_argument("--confirm-count", type=int, default=4)
    parser.add_argument("--confirm-trials", type=int, default=2_000_000)
    parser.add_argument("--confirm-replicas", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"out": str(args.out), "target": report["target"],
                      "population": report["search"], "final": report["final"],
                      "seconds_wall": round(report["seconds_wall"], 1)}, indent=2))


if __name__ == "__main__":
    main()
