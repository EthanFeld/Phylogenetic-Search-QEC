from __future__ import annotations

"""Exact certificate for independently shifted cyclic GB presentations."""

import argparse
import json
from pathlib import Path

import numpy as np

from hyper_validator import _logical_check, structural_validate
from inverse_design_core import gf2_rank


def _first_pair(doc: dict) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    n = int(doc["n"])
    if n % 2:
        raise ValueError("GB presentation requires even n")
    m = n // 2
    row = tuple(int(q) for q in doc["checks"]["X"][0])
    a = tuple(sorted(q for q in row if q < m))
    b = tuple(sorted(q - m for q in row if q >= m))
    if not a or not b:
        raise ValueError("first X check is not a two-block GB row")
    return m, a, b


def _shift(word, amount: int, m: int) -> tuple[int, ...]:
    return tuple(sorted((int(q) + int(amount)) % int(m) for q in word))


def _shift_between(source, target, m: int) -> int:
    matches = [amount for amount in range(int(m))
               if _shift(source, amount, m) == tuple(target)]
    if not matches:
        raise ValueError("supports are not cyclic translates")
    return min(matches)


def _map_matrix(matrix: np.ndarray, m: int, shifts: tuple[int, int]) -> np.ndarray:
    mapped = np.zeros_like(matrix)
    for old in range(2 * int(m)):
        block, offset = divmod(old, int(m))
        new = block * int(m) + (offset + int(shifts[block])) % int(m)
        mapped[:, new] = matrix[:, old]
    return mapped


def map_witness(witness, m: int, shifts: tuple[int, int]) -> list[int]:
    out = []
    for q in witness:
        block, offset = divmod(int(q), int(m))
        out.append(block * int(m) + (offset + int(shifts[block])) % int(m))
    return sorted(out)


def equivalence_certificate(source: dict, target: dict, witness,
                            side: str) -> dict:
    side = str(side).upper()
    if side not in ("X", "Z"):
        raise ValueError("side must be X or Z")
    source_struct = structural_validate(source)
    target_struct = structural_validate(target)
    if not source_struct["ok"] or not target_struct["ok"]:
        raise ValueError("source/target failed structural validation")
    m, source_a, source_b = _first_pair(source)
    target_m, target_a, target_b = _first_pair(target)
    if target_m != m:
        raise ValueError("block lengths differ")
    shifts = (_shift_between(source_a, target_a, m),
              _shift_between(source_b, target_b, m))
    mapped_hx = _map_matrix(source_struct["hx"], m, shifts)
    mapped_hz = _map_matrix(source_struct["hz"], m, shifts)
    hx_equal = (gf2_rank(np.vstack([mapped_hx, target_struct["hx"]])) ==
                gf2_rank(target_struct["hx"]))
    hz_equal = (gf2_rank(np.vstack([mapped_hz, target_struct["hz"]])) ==
                gf2_rank(target_struct["hz"]))
    mapped = map_witness(witness, m, shifts)
    if side == "X":
        check = _logical_check(mapped, target_struct["hz"], target_struct["hx"])
    else:
        check = _logical_check(mapped, target_struct["hx"], target_struct["hz"])
    ok = bool(hx_equal and hz_equal and check.get("ok"))
    return {
        "schema_version": "1.0",
        "kind": "gb_phase_equivalence_certificate",
        "ok": ok,
        "m": m,
        "independent_block_shifts": list(shifts),
        "rowspace_equivalence": {"X": bool(hx_equal), "Z": bool(hz_equal)},
        "side": side,
        "source_witness": [int(q) for q in witness],
        "mapped_witness": mapped,
        "mapped_witness_check": check,
        "conclusion": (f"target has a {side}-logical of weight {len(mapped)}"
                       if ok else "equivalence certificate failed"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("validation", type=Path,
                        help="runner JSON containing a refutation witness")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    target = json.loads(args.target.read_text())
    validation = json.loads(args.validation.read_text())
    refutation = validation.get("refutation")
    if not isinstance(refutation, dict):
        raise ValueError("validation has no refutation")
    result = equivalence_certificate(
        source, target, refutation["witness"], refutation["side"])
    result["source_path"] = str(args.source.resolve())
    result["target_path"] = str(args.target.resolve())
    result["validation_path"] = str(args.validation.resolve())
    text = json.dumps(result, indent=2) + "\n"
    if args.out is not None:
        args.out.write_text(text)
    else:
        print(text, end="")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
