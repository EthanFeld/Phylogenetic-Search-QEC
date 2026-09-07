from __future__ import annotations

"""Sampled square-free Z_341 [[682,190,*]] campaign."""

import json
import random
from pathlib import Path

import gb_m345_k188_phylo_campaign as core
from gf2_factor import degree, factor_xm_plus_one, mul
from generalized_bicycle import build_supports, normalize_code


M, N, Q, K = 341, 682, 95, 190
CHECK_CAP = 32
BRANCH_LIMIT = 96
FACTORS = tuple(factor_xm_plus_one(M))
BASE_DEGREES = [degree(factor) for factor in FACTORS]

core.M, core.N, core.Q, core.K = M, N, Q, K
core.CHECK_CAP = CHECK_CAP
core.MODULUS = (1 << M) | 1
core.EXACT_EXACT_ONLY = False
core.KILLER_IDEAL_FACTOR_SETS = ()
core.ONE_BLOCK_KILLER_IDEAL_FACTOR_SETS = ()
core._two_block_ideal_span.cache_clear()
core._one_block_ideal_span.cache_clear()
core._one_block_divisor_span.cache_clear()
_base_run = core.run


def _sample_subsets(seed: int = 20261015):
    degree5 = tuple(index for index, value in enumerate(BASE_DEGREES)
                    if value == 5)
    degree10 = tuple(index for index, value in enumerate(BASE_DEGREES)
                     if value == 10)
    if len(degree5) != 6 or len(degree10) != 31:
        raise ArithmeticError(f"unexpected Z341 factor spectrum: {BASE_DEGREES}")
    rng = random.Random(seed)
    rows = []
    for shape_index, (u, v) in enumerate(((9, 1), (8, 3), (7, 5))):
        seen = set()
        while len(seen) < BRANCH_LIMIT // 3:
            subset = tuple(sorted((*rng.sample(degree10, u),
                                   *rng.sample(degree5, v))))
            if subset not in seen:
                seen.add(subset)
                rows.append((shape_index, subset))
    return rows


def enumerate_branches(_parent_report=None):
    branches = []
    for branch_index, (shape_index, subset) in enumerate(_sample_subsets()):
        divisor = 1
        for index in subset:
            divisor = mul(divisor, FACTORS[index])
        branches.append({
            "branch_index": branch_index,
            "branch_id": "m341_q95_s" + str(shape_index) + "_" +
                         "-".join(map(str, subset)),
            "factor_indices": list(subset),
            "factor_degrees": [BASE_DEGREES[index] for index in subset],
            "divisor": int(divisor),
            "divisor_weight": int(divisor).bit_count(),
            "inherited_q55_ancestors": [],
            "phylogenetic_arm": f"q95_shape_{shape_index}_out_of_sample",
        })
    return branches


def _candidate_doc(row: dict, stage: str):
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
        "name": f"[[{N},{K},d<={distance}]] Z_341 q95 sampled GB",
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
            "origin": "gb_m341_k190_squarefree_sample",
            "novelty": "unknown",
            "model": "GPT-5 Codex + phylogenetic factor-shape sampling",
            "construction": (f"Z_341 GB; factor indices={row['factor_indices']}; "
                             f"A={tuple(code.a)}; B={tuple(code.b)}"),
            "references": [],
            "notes": f"stage={stage}; randomized witness-backed upper bound",
        },
        "search": {"stage": stage, "semantic_hash": row["semantic_hash"],
                   "branch_id": row["branch_id"],
                   "factor_indices": row["factor_indices"],
                   "cycle_profile": row["cycle_profile"]},
        "regulation": {
            "stage_only": True, "submission_sent": False,
            "git_commit_performed": False, "official_gate_status": "not_run",
            "board_claim_allowed": False, "distance_is_not_proven": True,
            "literature_novelty": "unverified",
        },
    }


core.enumerate_branches = enumerate_branches
core._candidate_doc = _candidate_doc


def run(args):
    report = _base_run(args)
    report["kind"] = "gb_m341_k190_squarefree_sampled_phylogeny_campaign"
    report["algebra"] = {
        "factor_spectrum": {"1": 1, "5": 6, "10": 31},
        "q95_shapes": [[9, 1], [8, 3], [7, 5]],
        "full_shape_count": "31C9*6 + 31C8*20 + 31C7*6",
        "sampled_branches": BRANCH_LIMIT,
        "square_free": True,
        "target_score_at_d76": K * 76 * 76 / N,
    }
    report["phylogenetic_hypothesis"] = {
        "representation": "square-free Z341 q95 odd-parity branch",
        "branch_sampling": "equal deterministic quota across all three factor shapes",
        "nested_subideal_escape": "x+1 and immediate-factor quotient detectors",
        "mutation": "exact-parent XOR shifted shallow descendants",
        "selection": "affine deduplication, dynamic annihilator gate, replicated RIS",
    }
    Path(args.out, "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    core.main.__globals__["run"] = run
    core.main()


if __name__ == "__main__":
    main()
