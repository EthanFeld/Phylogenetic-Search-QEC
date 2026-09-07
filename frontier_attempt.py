"""Run one reproducible, stage-only frontier search attempt.

The output is challenge-shaped but is never submitted automatically.  Distance
values are witness-backed upper bounds; exactness needs a separate certificate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from distance_sketch import fast_refute_supports
from inverse_design_core import (
    TargetSpec,
    build_product_checks,
    candidate_groups_of_order,
    canonical_1x2_logicals,
    design_from_specs,
    css_specs,
)


def _supports(matrix: np.ndarray) -> list[list[int]]:
    return [np.flatnonzero(row).astype(int).tolist() for row in matrix]


def run_attempt(outdir: Path, *, seed: int = 4401, search_trials: int = 240,
                deep_trials: int = 160, refute_trials: int = 2_000) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    search = design_from_specs(
        TargetSpec(220, 44, 9, 12),
        seed=seed,
        seed_draws=search_trials,
        seed_screen_trials=8,
        product_screen_trials=12,
        deep_trials=deep_trials,
        max_deep_products=5,
        backend="auto",
    )
    candidate = search.get("candidate")
    if candidate is None:
        raise RuntimeError("search produced no candidate")

    group = candidate_groups_of_order(int(candidate["group_order"]))[
        int(candidate["group_index"])
    ]
    A = tuple(tuple(int(v) for v in row) for row in candidate["A"])
    B = tuple(tuple(int(v) for v in row) for row in candidate["B"])
    hx, hz = build_product_checks(A, B, group)
    lx, lz = canonical_1x2_logicals(A, B, group)
    X, Z = _supports(hx), _supports(hz)

    # Claim d=1 only to disable early refutation.  We want both side witnesses
    # for packaging, while each witness remains independently checkable.
    ris = fast_refute_supports(X, Z, hx.shape[1], 1, seed=seed,
                               trials=refute_trials, max_seconds=None)
    sx, sz = ris["sectors"].get("x"), ris["sectors"].get("z")
    if not sx or not sz or sx["witness"] is None or sz["witness"] is None:
        raise RuntimeError("deep RIS did not produce witnesses on both sectors")
    dx, dz = int(sx["best_weight"]), int(sz["best_weight"])
    d = min(dx, dz)
    doc = {
        "schema_version": "0.1",
        "name": f"[[{hx.shape[1]},{candidate['specs']['k']},{d}]] stage-only search candidate",
        "code_type": "CSS",
        "n": int(hx.shape[1]),
        "k": int(candidate["specs"]["k"]),
        "checks": {"X": X, "Z": Z},
        "distance": {
            "d": d,
            "X": {"value": dx, "confidence": "upper_bound",
                  "witness": sx["witness"]},
            "Z": {"value": dz, "confidence": "upper_bound",
                  "witness": sz["witness"]},
        },
        "provenance": {
            "authors": ["stage-only autonomous research"],
            "origin": "submission",
            "novelty": "unknown",
            "construction": (
                f"generic 1x2 group-algebra chain product over {group.name}; "
                f"A={candidate['A']}; B={candidate['B']}"
            ),
            "notes": "Stage only; replace authorship before publication.",
        },
        "family": "lifted-product",
    }
    np.savez_compressed(outdir / "candidate.npz", hx=hx, hz=hz, lx=lx, lz=lz)
    (outdir / "candidate.json").write_text(json.dumps(doc, indent=2) + "\n")
    (outdir / "search.json").write_text(json.dumps(search, indent=2) + "\n")

    score = float(doc["k"] * d * d / doc["n"])
    result = {
        "candidate": {"n": doc["n"], "k": doc["k"], "d_upper": d,
                      "dx_upper": dx, "dz_upper": dz,
                      "max_check_weight": css_specs(hx, hz)["max_check_weight"],
                      "kd2_over_n_upper": score,
                      "semantic_hash": candidate["semantic_hash"]},
        "construction": {"group": group.name, "group_index": candidate["group_index"],
                         "A": candidate["A"], "B": candidate["B"]},
        "search": {"seed": seed, "draws": search_trials,
                   "deep_trials": deep_trials,
                   "screened_products": search.get("screened_product_count"),
                   "deepened_products": search.get("deepened_product_count")},
        "refuter": {"trials_per_side": refute_trials,
                    "total_trials": ris["trials_run"],
                    "seconds": ris["seconds"],
                    "mode": ris["mode"]},
        "frontier_note": (
            "Same [[220,44]] weight-9 incumbent is d=10 on the live board; "
            "this attempt raises the observed upper-bound witness to d<=11. "
            "Board advancement and exactness remain to be checked by the official gate."
        ),
    }
    (outdir / "attempt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/frontier_attempt"))
    parser.add_argument("--seed", type=int, default=4401)
    parser.add_argument("--search-trials", type=int, default=240)
    parser.add_argument("--deep-trials", type=int, default=160)
    parser.add_argument("--refute-trials", type=int, default=2_000)
    args = parser.parse_args()
    print(json.dumps(run_attempt(
        args.out,
        seed=args.seed,
        search_trials=args.search_trials,
        deep_trials=args.deep_trials,
        refute_trials=args.refute_trials,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
