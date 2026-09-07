from __future__ import annotations

"""Square-free Z_331 [[662,180,*]] whole-lattice campaign wrapper."""

import argparse
import json
from pathlib import Path

import gb_m345_k188_phylo_campaign as core
from generalized_bicycle import build_supports, normalize_code


M, N, Q, K = 331, 662, 90, 180


def _configure():
    core.M, core.N, core.Q, core.K = M, N, Q, K
    core.MODULUS = (1 << M) | 1
    core.EXACT_EXACT_ONLY = False
    core.KILLER_IDEAL_FACTOR_SETS = ()
    core.ONE_BLOCK_KILLER_IDEAL_FACTOR_SETS = ()
    core._two_block_ideal_span.cache_clear()
    core._one_block_ideal_span.cache_clear()
    core._one_block_divisor_span.cache_clear()


_configure()


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
        "name": f"[[{N},{K},d<={distance}]] Z_331 q{Q} phylogenetic GB",
        "code_type": "CSS", "n": N, "k": K,
        "checks": {"X": hx, "Z": hz},
        "distance": {"d": distance,
                     "X": {"value": int(dx), "confidence": "upper_bound",
                           "witness": list(wx)},
                     "Z": {"value": int(dz), "confidence": "upper_bound",
                           "witness": list(wz)}},
        "family": "generalized-bicycle",
        "provenance": {"authors": ["stage-only autonomous research"],
                       "origin": f"gb_m331_k{K}_campaign",
                       "novelty": "unknown",
                       "model": "GPT-5 Codex + phylogenetic factor-lattice search",
                       "construction": (f"Z_331 GB q{Q} factors={row['factor_indices']}; "
                                        f"A={tuple(code.a)}; B={tuple(code.b)}"),
                       "references": [],
                       "notes": f"stage={stage}; randomized upper bound"},
        "regulation": {"stage_only": True, "submission_sent": False,
                       "git_commit_performed": False,
                       "official_gate_status": "not_run",
                       "board_claim_allowed": False,
                       "distance_is_not_proven": True},
    }


core._candidate_doc = _candidate_doc


def run(args):
    report = core.run(args)
    report["kind"] = "gb_m331_k180_squarefree_factor_phylogeny_campaign"
    report["algebra"] = {
        "factor_spectrum": {"1": 1, "30": 11},
        "degree90_factor_subsets": 165,
        "square_free": True,
        "exact_parent_parity": "odd; q90 omits x+1",
    }
    report["phylogenetic_hypothesis"] = {
        "representation_jump": "square-free Z331 after m345/m350 structural collapse",
        "exploration": "all 165 triples of degree-30 irreducible factors",
        "nested_subideal_escape": "x+1 and every immediate-child quotient detector",
        "selection": "dynamic annihilator gate plus replicated RIS and branch diversity",
    }
    Path(args.out, "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("results/gb_m331_k180_01"))
    parser.add_argument("--parent-report", type=Path, default=Path("unused.json"))
    parser.add_argument("--factor-subset")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--leader-score", type=float, default=core.LEADER_SCORE)
    parser.add_argument("--target-d", type=int)
    parser.add_argument("--plain-scout-runs", type=int, default=1)
    parser.add_argument("--exact-scout-runs", type=int, default=2)
    parser.add_argument("--quotient-scout-runs", type=int, default=1)
    parser.add_argument("--scout-trials", type=int, default=256)
    parser.add_argument("--mine-branches", type=int, default=48)
    parser.add_argument("--detector-runs", type=int, default=8)
    parser.add_argument("--detector-trials", type=int, default=1024)
    parser.add_argument("--lattice-detector-runs", type=int, default=1)
    parser.add_argument("--child-runs", type=int, default=2)
    parser.add_argument("--child-trials", type=int, default=1024)
    parser.add_argument("--max-companion-gcd", type=int, default=120)
    parser.add_argument("--ideal-iterations", type=int, default=250000)
    parser.add_argument("--shell-iterations", type=int, default=100000)
    parser.add_argument("--max-words", type=int, default=2048)
    parser.add_argument("--mutation-generations", type=int, default=2)
    parser.add_argument("--max-exact-weight", type=int, default=24)
    parser.add_argument("--min-check-weight", type=int, default=28)
    parser.add_argument("--pairs-per-branch", type=int, default=32)
    parser.add_argument("--pair-attempts", type=int, default=20000)
    parser.add_argument("--proxy-trials", type=int, default=256)
    parser.add_argument("--proxy-replicas", type=int, default=2)
    parser.add_argument("--deep-per-branch", type=int, default=2)
    parser.add_argument("--deep-count", type=int, default=32)
    parser.add_argument("--deep-trials", type=int, default=20000)
    parser.add_argument("--deep-replicas", type=int, default=4)
    parser.add_argument("--confirm-count", type=int, default=4)
    parser.add_argument("--confirm-trials", type=int, default=2000000)
    parser.add_argument("--confirm-replicas", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"out": str(args.out),
                      "seconds_wall": round(report["seconds_wall"], 1),
                      "final": report["final"]}, indent=2))


if __name__ == "__main__":
    main()
