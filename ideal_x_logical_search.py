from __future__ import annotations

"""Search cyclic ideal <h> for X logicals of a generalized-bicycle code.

For g=gcd(a,b,x^ell-1) and h=(x^ell-1)/g, <h> has dimension deg(g).
Every returned word is checked both as h-divisible polynomial and as the
full CSS witness (w,0).  RIS is randomized discovery, not an exhaustive
minimum-distance proof.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import cpp_fast
from generalized_bicycle import build_matrices
from gf2_factor import degree, gcd, mod, quotient
from hyper_validator import _logical_check, structural_validate


def _poly(support: list[int] | tuple[int, ...], ell: int) -> int:
    value = 0
    for exponent in support:
        exponent = int(exponent)
        if exponent < 0 or exponent >= int(ell):
            raise ValueError("polynomial support outside cyclic block")
        value ^= 1 << exponent
    return value


def _support(poly: int) -> list[int]:
    out = []
    value = int(poly)
    while value:
        bit = value & -value
        out.append(bit.bit_length() - 1)
        value ^= bit
    return out


def _ab_from_candidate(candidate: dict) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    n = int(candidate["n"])
    if n % 2:
        raise ValueError("candidate n must be even for two-block GB")
    ell = n // 2
    first = candidate["checks"]["X"][0]
    a = tuple(sorted(int(value) for value in first if int(value) < ell))
    b = tuple(sorted(int(value) - ell for value in first if int(value) >= ell))
    if not a or not b:
        raise ValueError("first X check does not expose nonempty A,B blocks")
    return ell, a, b


def ideal_basis(h: int, ell: int, k: int) -> np.ndarray:
    """Return k independent non-wrapping shifts of h, shape k×ell."""
    rows = np.zeros((int(k), int(ell)), dtype=np.uint8)
    base = _support(h)
    if not base or (max(base) + int(k) >= int(ell) + 1):
        raise ValueError("<h> basis would wrap; invalid g/h degrees")
    for shift in range(int(k)):
        rows[shift, [shift + value for value in base]] = 1
    return rows


def scan_candidate(candidate_path: Path, *, trials: int, seed: int,
                   pair_depth: int, max_seconds: float | None) -> dict:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    structural = structural_validate(candidate)
    if not structural["ok"]:
        raise ValueError(f"candidate fails structural validation: {candidate_path}")
    ell, a, b = _ab_from_candidate(candidate)
    modulus = (1 << ell) | 1
    g = gcd(gcd(_poly(a, ell), _poly(b, ell)), modulus)
    h = quotient(modulus, g)
    k = degree(g)
    if degree(h) != ell - k or mod(modulus, g) != 0:
        raise ArithmeticError("invalid g/h quotient")
    basis = ideal_basis(h, ell, k)
    rank = cpp_fast.gf2_rank(basis)
    if rank != k:
        raise ArithmeticError(f"<h> basis rank {rank} != deg(g) {k}")

    result = cpp_fast.classical_ris(
        basis, trials=int(trials), seed=int(seed), pair_depth=int(pair_depth),
        max_seconds=max_seconds, target=None, stop_on_target=False)
    word = result.get("witness")
    found = None
    if word is not None:
        word = [int(value) for value in word]
        word_poly = _poly(word, ell)
        full_witness = word
        check = _logical_check(full_witness, structural["hz"], structural["hx"])
        found = {
            "weight": len(word),
            "block_support": word,
            "full_X_witness_(w,0)": full_witness,
            "h_divides_word": mod(word_poly, h) == 0,
            "css_logical_check": {key: value for key, value in check.items()
                                  if key != "row_rank"},
        }

    return {
        "schema_version": "1.0",
        "kind": "cyclic_ideal_x_logical_search",
        "candidate_path": str(candidate_path.resolve()),
        "candidate_signature": {"n": int(candidate["n"]),
                                "k_claim": int(candidate["k"])},
        "algebra": {
            "ell": ell, "g_degree_K": k, "h_degree": degree(h),
            "g_support": _support(g), "h_support": _support(h),
            "modulus_support": _support(modulus),
            "g_polynomial_hex": hex(g), "h_polynomial_hex": hex(h),
            "modulus_hex": hex(modulus),
            "ideal_dimension": int(rank),
        },
        "search": {
            "basis": "non-wrapping shifts h*x^i, 0<=i<K",
            "method": "native classical RIS over <h>",
            "trials": int(trials), "seed": int(seed),
            "pair_depth": int(pair_depth), "max_seconds": max_seconds,
            "minimum_weight_is_heuristic": True,
        },
        "run": result,
        "found_X_logical": found,
        "regulation": {
            "submission_sent": False, "git_commit_performed": False,
            "distance_is_not_proven": True,
            "witness_requires_independent_verifier": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path, nargs="+",
                        help="candidate JSON; repeat for multiple codes")
    parser.add_argument("--out", type=Path,
                        default=Path("results/ideal_x_logical_search.json"))
    parser.add_argument("--trials", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=2026090343)
    parser.add_argument("--pair-depth", type=int, default=24)
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args()
    if not cpp_fast.available():
        raise RuntimeError(cpp_fast.accelerator_info())
    reports = []
    for index, path in enumerate(args.candidate):
        try:
            reports.append(scan_candidate(
                path, trials=args.trials,
                seed=int(args.seed) + index * 0x9E3779B9,
                pair_depth=args.pair_depth,
                max_seconds=args.max_seconds))
        except (OSError, ValueError, ArithmeticError) as exc:
            # Batch campaigns must keep going when an old artifact is malformed
            # or fails the structural gate.  This is a rejection receipt, not a
            # distance result and can never be admitted as a candidate.
            reports.append({
                "schema_version": "1.0",
                "kind": "cyclic_ideal_x_logical_search_rejected_input",
                "candidate_path": str(Path(path).resolve()),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "regulation": {
                    "submission_sent": False,
                    "git_commit_performed": False,
                    "admissible_candidate": False,
                },
            })
    payload = reports[0] if len(reports) == 1 else {
        "schema_version": "1.0", "kind": "cyclic_ideal_x_logical_search_batch",
        "reports": reports, "submission_sent": False,
        "git_commit_performed": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out.resolve()),
        "results": [{
            "candidate": r["candidate_path"],
            "status": "rejected_input" if r["kind"].endswith("rejected_input") else "scanned",
            "K": r.get("algebra", {}).get("g_degree_K"),
            "weight": (r.get("found_X_logical") or {}).get("weight"),
            "h_divides": (r.get("found_X_logical") or {}).get("h_divides_word"),
            "css_ok": ((r.get("found_X_logical") or {})
                       .get("css_logical_check", {}).get("ok")),
            "error": r.get("error"),
        } for r in reports],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
