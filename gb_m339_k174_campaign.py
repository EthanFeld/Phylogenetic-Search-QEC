from __future__ import annotations

"""Square-free Z_339 [[678,174,*]] exact/exact factor campaign.

The degree-87 parent contains x+1, the unique degree-2 factor, and three of
twelve degree-28 factors.  Unlike the Z_345 q94 frontier, two sparse words can
both have the exact parent gcd while respecting the challenge's weight-32 cap.
"""

import argparse
import json
from pathlib import Path

import gb_m345_k188_phylo_campaign as core
from generalized_bicycle import build_supports, normalize_code


M, N, Q, K = 339, 678, 87, 174


def _configure() -> None:
    core.M, core.N, core.Q, core.K = M, N, Q, K
    core.MODULUS = (1 << M) | 1
    core.CHECK_CAP = 32
    core.EXACT_EXACT_ONLY = True
    core.KILLER_IDEAL_FACTOR_SETS = ()
    core.ONE_BLOCK_KILLER_IDEAL_FACTOR_SETS = ()
    core._two_block_ideal_span.cache_clear()
    core._one_block_ideal_span.cache_clear()
    core._one_block_divisor_span.cache_clear()


_configure()


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
        "name": f"[[{N},{K},d<={distance}]] Z_{M} q{Q} exact-parent GB",
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
            "origin": "gb_m339_k174_campaign",
            "novelty": "unknown",
            "model": "GPT-5 Codex + phylogenetic factor-lattice search",
            "construction": (f"Z_{M} GB; degree-{Q} factors="
                             f"{row['factor_indices']}; A={tuple(code.a)}; "
                             f"B={tuple(code.b)}"),
            "references": [],
            "notes": f"stage={stage}; randomized witness-backed upper bound",
        },
        "search": {
            "stage": stage,
            "semantic_hash": row["semantic_hash"],
            "branch_id": row["branch_id"],
            "pairing_mode": row.get("pairing_mode"),
            "individual_gcd_degrees": row.get("individual_gcd_degrees"),
            "cycle_profile": row.get("cycle_profile"),
        },
        "regulation": {
            "stage_only": True, "submission_sent": False,
            "git_commit_performed": False, "official_gate_status": "not_run",
            "board_claim_allowed": False, "distance_is_not_proven": True,
        },
    }


core._candidate_doc = _candidate_doc


def run(args):
    report = core.run(args)
    report["kind"] = "gb_m339_k174_squarefree_exact_parent_campaign"
    report["algebra"] = {
        "factor_spectrum": {"1": 1, "2": 1, "28": 12},
        "degree87_factor_subsets": 220,
        "square_free": True,
        "exact_parent_parity": "even; q87 includes x+1",
        "legal_escape": "exact-parent plus exact-parent, normally 16+16",
    }
    report["phylogenetic_hypothesis"] = {
        "failed_lineage": "Z345 required a deeper child under weight 32",
        "representation_jump": "Z339 q87 permits legal exact/exact checks",
        "exploration": "all C(12,3)=220 degree-28 factor branches",
        "selection": "annihilator gate, replicated RIS, branch diversity",
        "score_threshold": "d79 gives kd^2/n = 1601.673",
    }
    Path(args.out, "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("results/gb_m339_k174_01"))
    parser.add_argument("--parent-report", type=Path, default=Path("unused.json"))
    parser.add_argument("--factor-subset")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--leader-score", type=float, default=core.LEADER_SCORE)
    parser.add_argument("--target-d", type=int)
    parser.add_argument("--plain-scout-runs", type=int, default=1)
    parser.add_argument("--exact-scout-runs", type=int, default=2)
    parser.add_argument("--quotient-scout-runs", type=int, default=1)
    parser.add_argument("--scout-trials", type=int, default=256)
    parser.add_argument("--mine-branches", type=int, default=64)
    parser.add_argument("--detector-runs", type=int, default=8)
    parser.add_argument("--detector-trials", type=int, default=1024)
    parser.add_argument("--lattice-detector-runs", type=int, default=1)
    parser.add_argument("--child-runs", type=int, default=1)
    parser.add_argument("--child-trials", type=int, default=512)
    parser.add_argument("--max-companion-gcd", type=int, default=115)
    parser.add_argument("--ideal-iterations", type=int, default=500000)
    parser.add_argument("--shell-iterations", type=int, default=100000)
    parser.add_argument("--max-words", type=int, default=4096)
    parser.add_argument("--hill-restarts", type=int, default=512)
    parser.add_argument("--hill-steps", type=int, default=4096)
    parser.add_argument("--hill-bases", type=int, default=16)
    parser.add_argument("--mutation-generations", type=int, default=2)
    parser.add_argument("--max-exact-weight", type=int, default=18)
    parser.add_argument("--min-check-weight", type=int, default=30)
    parser.add_argument("--pairs-per-branch", type=int, default=32)
    parser.add_argument("--pair-attempts", type=int, default=30000)
    parser.add_argument("--proxy-trials", type=int, default=512)
    parser.add_argument("--proxy-replicas", type=int, default=2)
    parser.add_argument("--deep-per-branch", type=int, default=2)
    parser.add_argument("--deep-count", type=int, default=64)
    parser.add_argument("--deep-trials", type=int, default=10000)
    parser.add_argument("--deep-replicas", type=int, default=3)
    parser.add_argument("--confirm-count", type=int, default=4)
    parser.add_argument("--confirm-trials", type=int, default=100000)
    parser.add_argument("--confirm-replicas", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"out": str(args.out),
                      "seconds_wall": round(report["seconds_wall"], 1),
                      "final": report["final"]}, indent=2))


if __name__ == "__main__":
    main()
