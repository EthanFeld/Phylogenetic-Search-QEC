from __future__ import annotations

"""Repeated-root Z_342 [[684,184,*]] phylogenetic campaign.

This wrapper reuses the tested repeated-root engine while moving to the
2*171 ring, whose q=92 lattice has many more factor-multiplicity branches than
the exhausted m348 pocket.  It samples that lattice deterministically so the
scout remains bounded, then applies the full cap-32 pairing/RIS pipeline.
Stage-only: no submission and no commit.
"""

import math

import gb_m350_k188_repeated_campaign as campaign
from gf2_factor import factor_xm_plus_one, degree


M = 342
N = 684
Q = 92
K = 184
CHECK_CAP = 32
ROOT_MULTIPLICITY = 2
BASE_FACTORS = tuple(factor_xm_plus_one(171))
BRANCH_LIMIT = 192
ALL_BRANCH_COUNT = None


campaign.M = M
campaign.N = N
campaign.Q = Q
campaign.K = K
campaign.CHECK_CAP = CHECK_CAP
campaign.MODULUS = (1 << M) | 1
campaign.ROOT_MULTIPLICITY = ROOT_MULTIPLICITY
campaign.BASE_FACTORS = BASE_FACTORS
campaign.PAIR_EXACT_ONLY = False
campaign._gcd_exponents.cache_clear()
campaign._configure_core()

_base_enumerate_branches = campaign.enumerate_branches
_base_rank_preflight = campaign._rank_preflight
_base_run = campaign.run


def enumerate_branches(_parent_report=None):
    """Return an evenly spaced branch beam over the full q=92 lattice."""
    global ALL_BRANCH_COUNT
    branches = _base_enumerate_branches(_parent_report)
    ALL_BRANCH_COUNT = len(branches)
    if len(branches) <= BRANCH_LIMIT:
        return branches
    # Include both endpoints and evenly spaced multiplicity signatures.  The
    # original branch index is retained for reproducibility and audit trails.
    indices = sorted({
        round(index * (len(branches) - 1) / (BRANCH_LIMIT - 1))
        for index in range(BRANCH_LIMIT)
    })
    return [branches[index] for index in indices]


def _rank_preflight():
    result = _base_rank_preflight()
    result["full_lattice_branches"] = int(ALL_BRANCH_COUNT or len(
        campaign.degree94_exponents()))
    result["sampled_lattice_branches"] = len(enumerate_branches())
    return result


campaign.enumerate_branches = enumerate_branches
campaign._rank_preflight = _rank_preflight
campaign._install_overrides()


def run(args):
    report = _base_run(args)
    report["kind"] = "gb_m342_k184_repeated_root_phylogeny_campaign"
    report["algebra"] = {
        "odd_core": M // ROOT_MULTIPLICITY,
        "root_multiplicity": ROOT_MULTIPLICITY,
        "identity": f"x^{M}+1=(x^{M // ROOT_MULTIPLICITY}+1)^{ROOT_MULTIPLICITY} over GF(2)",
        "base_factor_degrees": [degree(factor) for factor in BASE_FACTORS],
        "full_lattice_degree92_exponent_vectors": int(ALL_BRANCH_COUNT or 0),
        "sampled_lattice_branches": len(enumerate_branches()),
        "rank_preflight": report["algebra"].get("rank_preflight", {}),
        "literal_girth_tradeoff": (
            "balanced total check weight >=28 forces 4-cycles; exact cycle energy used"),
    }
    report["phylogenetic_hypothesis"] = {
        "representation": "repeated-root Z_342 over the odd core Z_171",
        "branch_beam": "evenly spaced multiplicity signatures from the full q=92 lattice",
        "nested_subideal_escape": "immediate multiplicity-child quotient detectors",
        "mutation": "exact-parent XOR shifted shallow repeated-root descendants",
        "selection": "dynamic annihilator kill-gate, affine deduplication, replicated RIS",
    }
    from pathlib import Path
    import json
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    campaign.main.__globals__["run"] = run
    campaign.main()


if __name__ == "__main__":
    main()
