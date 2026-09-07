"""Independently validate a near-target projection-breakout candidate.

This bridge intentionally does not turn a proxy result into a submission
candidate.  It first rechecks CSS structure and both proxy witnesses, then
runs fresh native RIS chunks with a configurable early-refutation threshold.
The default is useful for a score attempt that is near, but below, the raw
projection target: a clean budget is evidence only, never a lower-bound proof.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gb_ris_validation_runner import run as validate
from hyper_validator import structural_validate
from projection_breakout_campaign import _candidate_doc


def stage(campaign_path: Path, out: Path, rank: int) -> Path:
    campaign = json.loads(campaign_path.read_text())
    rows = campaign.get("proxy_top", [])
    if not 0 <= int(rank) < len(rows):
        raise ValueError(f"rank must select one of {len(rows)} proxy rows")
    row = rows[int(rank)]
    result = {
        "x": {"best_weight": row["screen"]["dx_upper"],
              "witness": row["screen_witnesses"]["X"]},
        "z": {"best_weight": row["screen"]["dz_upper"],
              "witness": row["screen_witnesses"]["Z"]},
    }
    doc = _candidate_doc(row, result, stage="proxy_near_target_revalidation")
    if doc is None:
        raise ValueError("selected proxy row has no two-sided witness")
    structural = structural_validate(doc)
    if not structural["ok"]:
        raise ValueError("selected proxy row fails independent structural check")
    doc["regulation"]["official_gate_status"] = "not_run"
    doc["regulation"]["board_claim_allowed"] = False
    doc["provenance"]["notes"] = (
        "Proxy-near-target staging only; fresh-seed RIS validation pending. "
        "Not a distance lower-bound or submission claim."
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--trials", type=int, default=2_000_000,
                        help="maximum fresh RIS trials per side")
    parser.add_argument("--chunk", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2026083001)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--pair-depth", type=int, default=24)
    args = parser.parse_args()
    if args.chunk <= 0 or args.trials <= 0:
        raise ValueError("trials and chunk must be positive")
    candidate = stage(args.campaign, args.out, args.rank)
    chunks = (int(args.trials) + int(args.chunk) - 1) // int(args.chunk)
    checkpoint = args.out.with_name(args.out.stem + "_validation.json")
    state = validate(candidate, out=checkpoint, chunks=chunks,
                     trials=int(args.chunk), target=int(args.target),
                     seed=int(args.seed), threads=int(args.threads),
                     pair_depth=int(args.pair_depth))
    print(json.dumps({"candidate": str(candidate), "validation": str(checkpoint),
                      "status": state["status"], "chunks": len(state["chunks"]),
                      "best": state["best_observed"],
                      "refutation": state["refutation"]}, indent=2))
    return 2 if state["refutation"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
