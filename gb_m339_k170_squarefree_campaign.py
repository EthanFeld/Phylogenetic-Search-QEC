from __future__ import annotations

"""Square-free Z_339 [[678,170,*]] q=85 sibling campaign."""

import argparse
import itertools
import json
from pathlib import Path

import gb_m345_k188_phylo_campaign as core
from gf2_factor import degree, factor_xm_plus_one, mul
from generalized_bicycle import build_supports, normalize_code


M, N, Q, K = 339, 678, 85, 170
CHECK_CAP = 32
FACTORS = tuple(factor_xm_plus_one(M))
DEGREES = [degree(factor) for factor in FACTORS]

core.M, core.N, core.Q, core.K = M, N, Q, K
core.CHECK_CAP = CHECK_CAP
core.MODULUS = (1 << M) | 1
core.EXACT_EXACT_ONLY = False
core.KILLER_IDEAL_FACTOR_SETS = ()
core.ONE_BLOCK_KILLER_IDEAL_FACTOR_SETS = ()
core._two_block_ideal_span.cache_clear()
core._one_block_ideal_span.cache_clear()
_base_run = core.run


def enumerate_branches(_parent_report=None):
    degree28 = [i for i, value in enumerate(DEGREES) if value == 28]
    if DEGREES[0] != 1 or len(degree28) != 12:
        raise ArithmeticError(f"unexpected Z339 spectrum: {DEGREES}")
    branches = []
    for index, triple in enumerate(itertools.combinations(degree28, 3)):
        subset = tuple(sorted((0, *triple)))
        divisor = 1
        for factor_index in subset:
            divisor = mul(divisor, FACTORS[factor_index])
        branches.append({
            "branch_index": index,
            "branch_id": "m339_q85_" + "-".join(map(str, subset)),
            "factor_indices": list(subset),
            "factor_degrees": [DEGREES[i] for i in subset],
            "divisor": int(divisor),
            "divisor_weight": int(divisor).bit_count(),
            "inherited_q55_ancestors": [],
            "phylogenetic_arm": "q85_xplus1_plus_28_triple",
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
        "name": f"[[{N},{K},d<={distance}]] Z_339 q85 square-free GB",
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
            "origin": "gb_m339_k170_squarefree_campaign",
            "novelty": "unknown",
            "model": "GPT-5 Codex + phylogenetic factor-lattice search",
            "construction": (f"Z_339 GB; factor indices={row['factor_indices']}; "
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
    report["kind"] = "gb_m339_k170_squarefree_phylogeny_campaign"
    report["algebra"] = {
        "factor_spectrum": {"1": 1, "2": 1, "28": 12},
        "degree85_factor_branches": 220,
        "square_free": True,
        "includes_x_plus_1": True,
        "omits_degree2": True,
        "target_score_at_d80": K * 80 * 80 / N,
    }
    report["phylogenetic_hypothesis"] = {
        "representation": "q85 sibling of Z339 q87 and q86 families",
        "branching": "all C(12,3) degree-28 triples plus x+1",
        "nested_subideal_escape": "every immediate-factor quotient detector",
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
