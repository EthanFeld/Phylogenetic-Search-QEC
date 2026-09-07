from __future__ import annotations

"""Package and gate one campaign row; no submission is sent."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from inverse_design_core import build_product_checks, candidate_groups_of_order, canonical_1x2_logicals


def _supports(matrix):
    return [np.flatnonzero(row).astype(int).tolist() for row in matrix]


def package(row, out: Path):
    group = candidate_groups_of_order(int(row["group_order"]))[int(row["group_index"])]
    A = tuple(tuple(int(v) for v in xs) for xs in row["A"])
    B = tuple(tuple(int(v) for v in xs) for xs in row["B"])
    hx, hz = build_product_checks(A, B, group)
    lx, lz = canonical_1x2_logicals(A, B, group)
    doc = {
        "schema_version": "0.1",
        "name": f"[[{row['n']},{row['k']},{row['d_upper']}]] campaign candidate",
        "code_type": "CSS", "n": int(row["n"]), "k": int(row["k"]),
        "checks": {"X": _supports(hx), "Z": _supports(hz)},
        "distance": {
            "d": int(row["d_upper"]),
            "X": {"value": int(row["dx_upper"]), "confidence": "upper_bound",
                  "witness": row["witness_x"]},
            "Z": {"value": int(row["dz_upper"]), "confidence": "upper_bound",
                  "witness": row["witness_z"]},
        },
        "provenance": {
            "authors": ["stage-only autonomous research"], "origin": "submission",
            "novelty": "unknown",
            "construction": f"generic 1x2 group-algebra chain product over {group.name}; A={row['A']}; B={row['B']}",
            "notes": "Stage only; replace authorship before publication.",
        },
        "family": "lifted-product",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    np.savez_compressed(out.with_suffix(".npz"), hx=hx, hz=hz, lx=lx, lz=lz)
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign", type=Path, default=Path("results/campaign_01/campaign.json"), nargs="?")
    ap.add_argument("--n", type=int, default=220)
    ap.add_argument("--semantic-hash", default=None,
                    help="validate exact campaign row instead of best row")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--challenge-root", type=Path,
                    default=Path("challenge_data"))
    args = ap.parse_args()
    campaign = json.loads(args.campaign.read_text())
    rows = [r for r in campaign["results"] if r.get("status") == "ok" and r.get("n") == args.n]
    if args.semantic_hash:
        rows = [r for r in rows if r.get("semantic_hash") == args.semantic_hash]
    if not rows:
        suffix = f" hash={args.semantic_hash}" if args.semantic_hash else ""
        raise SystemExit(f"no rows for n={args.n}{suffix}")
    row = max(rows, key=lambda r: (r["d_upper"], r["dx_upper"] + r["dz_upper"]))
    out = args.out or args.campaign.parent / f"top_{args.n}.json"
    out = out.resolve()
    package(row, out)
    validator = args.challenge_root / "verify" / "validate_candidate.py"
    print(json.dumps({"selected": {k: row[k] for k in ("n", "k", "d_upper", "dx_upper", "dz_upper", "semantic_hash")},
                      "json": str(out), "validator": str(validator)}, indent=2))
    if validator.exists():
        proc = subprocess.run([sys.executable, str(validator), str(out)],
                              cwd=args.challenge_root, capture_output=True, text=True)
        receipt = out.with_name(out.stem + "_official.json")
        if proc.stdout.strip():
            receipt.write_text(proc.stdout)
        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
