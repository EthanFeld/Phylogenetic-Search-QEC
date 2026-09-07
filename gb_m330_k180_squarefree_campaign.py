from __future__ import annotations

"""Square-free Z_330 [[660,180,*]] high-rate phylogeny campaign.

Z_330 has factor degrees 1,1,2,2,4^6,10^6,20^12.  q=90 admits several
factor geometries; this wrapper exposes the 20/10 mixed lattice while the
shared engine performs sparse mining, affine deduplication, RIS ranking, and
regulation-safe receipts.
"""

import argparse
import itertools
import json
from pathlib import Path

import gb_m345_k188_phylo_campaign as core
from gf2_factor import degree, factor_xm_plus_one, mul
from generalized_bicycle import build_supports, normalize_code


M, N, Q, K = 330, 660, 90, 180
CHECK_CAP = 32
FACTORS = tuple(factor_xm_plus_one(M))
DEGREES = [degree(factor) for factor in FACTORS]
_base_run = core.run

core.M, core.N, core.Q, core.K = M, N, Q, K
core.CHECK_CAP = CHECK_CAP
core.MODULUS = (1 << M) | 1
core.EXACT_EXACT_ONLY = False
core.KILLER_IDEAL_FACTOR_SETS = ()
core.ONE_BLOCK_KILLER_IDEAL_FACTOR_SETS = ()
core._two_block_ideal_span.cache_clear()
core._one_block_ideal_span.cache_clear()
core._one_block_divisor_span.cache_clear()


def enumerate_branches(_parent_report=None):
    groups = {value: [i for i, degree_value in enumerate(DEGREES)
                      if degree_value == value]
              for value in sorted(set(DEGREES))}
    if len(groups.get(20, [])) != 12 or len(groups.get(10, [])) != 6:
        raise ArithmeticError(f"unexpected Z330 spectrum: {DEGREES}")
    branches = []

    # q=90 sparse mixed families: 3*20+3*10, 4*20+10, 2*20+5*10.
    shapes = ((3, 3), (4, 1), (2, 5))
    for twenty_count, ten_count in shapes:
        for twenties in itertools.combinations(groups[20], twenty_count):
            for tens in itertools.combinations(groups[10], ten_count):
                subset = tuple(sorted((*twenties, *tens)))
                divisor = 1
                for factor_index in subset:
                    divisor = mul(divisor, FACTORS[factor_index])
                branches.append({
                    "branch_index": len(branches),
                    "branch_id": "m330_q90_" + "-".join(map(str, subset)),
                    "factor_indices": list(subset),
                    "factor_degrees": [DEGREES[i] for i in subset],
                    "divisor": int(divisor),
                    "divisor_weight": int(divisor).bit_count(),
                    "inherited_q55_ancestors": [],
                    "phylogenetic_arm": "q90_mixed_20_10",
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
        "name": f"[[{N},{K},d<={distance}]] Z_330 q90 square-free GB",
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
            "origin": "gb_m330_k180_squarefree_campaign",
            "novelty": "unknown",
            "model": "GPT-5 Codex + phylogenetic factor-lattice search",
            "construction": (f"Z_{M} GB; factor indices={row['factor_indices']}; "
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
    report["kind"] = "gb_m330_k180_squarefree_phylogeny_campaign"
    report["algebra"] = {
        "factor_spectrum": {"1": 2, "2": 2, "4": 6, "10": 6, "20": 12},
        "degree90_factor_branches": 7766,
        "square_free": True,
        "q90_shapes": [[3, 3], [4, 1], [2, 5]],
        "target_score_at_d90": K * 90 * 90 / N,
    }
    report["phylogenetic_hypothesis"] = {
        "representation": "Z330 q90 mixed 20/10-factor sibling",
        "branching": "degree-90 mixed factor subsets",
        "sparsity_prior": "rank branches by divisor support weight",
        "mutation": "exact-parent XOR shifted shallow descendants",
        "selection": "affine deduplication, annihilator gate, replicated RIS",
        "score_threshold": "d90 gives kd^2/n = " +
                           f"{K * 90 * 90 / N:.3f}",
    }
    Path(args.out, "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    core.main.__globals__["run"] = run
    core.main()


if __name__ == "__main__":
    main()
