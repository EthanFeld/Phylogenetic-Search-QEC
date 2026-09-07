from __future__ import annotations

"""Validated CSS serial-concatenation primitives.

For outer CSS [[n_o,k_o,d_o]] and inner CSS [[n_i,1,d_i]], this module builds
the standard block CSS code with [[n_o*n_i,k_o,d_o*d_i]].  Distance claims remain
claims: challenge validation still must refute-test them.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from inverse_design_core import css_specs, gf2_rank


@dataclass(frozen=True)
class CSSCode:
    name: str
    hx: np.ndarray
    hz: np.ndarray
    lx: np.ndarray
    lz: np.ndarray


def _binary(value, *, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.uint8)
    if out.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if np.any((out != 0) & (out != 1)):
        raise ValueError(f"{name} must be binary")
    return out.copy()


def _block_repeat(matrix: np.ndarray, count: int) -> np.ndarray:
    rows, cols = matrix.shape
    out = np.zeros((rows * count, cols * count), dtype=np.uint8)
    for block in range(count):
        out[block * rows:(block + 1) * rows,
            block * cols:(block + 1) * cols] = matrix
    return out


def _block_diagonal(matrices: list[np.ndarray]) -> np.ndarray:
    """Block diagonal matrix for a non-uniform inner-block sequence."""
    if not matrices:
        return np.zeros((0, 0), dtype=np.uint8)
    rows = sum(int(matrix.shape[0]) for matrix in matrices)
    cols = sum(int(matrix.shape[1]) for matrix in matrices)
    out = np.zeros((rows, cols), dtype=np.uint8)
    row_offset = col_offset = 0
    for matrix in matrices:
        matrix = _binary(matrix, name="block matrix")
        block_rows, block_cols = matrix.shape
        out[row_offset:row_offset + block_rows,
            col_offset:col_offset + block_cols] = matrix
        row_offset += block_rows
        col_offset += block_cols
    return out


def _lift(outer_rows: np.ndarray, inner_logical: np.ndarray) -> np.ndarray:
    """Lift outer bits to inner logical operators, block-major ordering."""
    return np.asarray(np.kron(outer_rows, inner_logical), dtype=np.uint8)


def _lift_multilogical(outer_rows: np.ndarray, inner_logicals: np.ndarray) -> np.ndarray:
    """Lift groups of outer qubits through all logicals of one inner block.

    If inner encodes ``q`` qubits, each block receives ``q`` adjacent outer
    coordinates.  This is stabilizer concatenation through a symplectic
    logical basis, not a direct sum: inner checks protect every block and the
    outer checks act on its logical qubits.
    """
    outer_rows = _binary(outer_rows, name="outer rows")
    inner_logicals = _binary(inner_logicals, name="inner logicals")
    q, inner_n = inner_logicals.shape
    if q < 1 or outer_rows.shape[1] % q:
        raise ValueError("outer block length must be divisible by inner k")
    blocks = outer_rows.shape[1] // q
    lifted = np.zeros((outer_rows.shape[0], blocks, inner_n), dtype=np.uint8)
    for logical_index in range(q):
        selected = outer_rows[:, logical_index::q]
        lifted ^= (selected[:, :, None] *
                   inner_logicals[logical_index][None, None, :]).astype(np.uint8)
    return lifted.reshape(outer_rows.shape[0], blocks * inner_n)


def _lift_heterogeneous(outer_rows: np.ndarray,
                        inner_logicals: list[np.ndarray]) -> np.ndarray:
    """Lift consecutive outer-coordinate groups through unequal inner blocks."""
    outer_rows = _binary(outer_rows, name="outer rows")
    logicals = [_binary(block, name="inner logicals") for block in inner_logicals]
    if sum(int(block.shape[0]) for block in logicals) != outer_rows.shape[1]:
        raise ValueError("inner logical dimensions must cover outer coordinates")
    pieces = []
    outer_offset = 0
    for block in logicals:
        q, _ = block.shape
        pieces.append((outer_rows[:, outer_offset:outer_offset + q] @ block) & 1)
        outer_offset += q
    return np.concatenate(pieces, axis=1) if pieces else np.zeros(
        (outer_rows.shape[0], 0), dtype=np.uint8)


def validate_css_code(code: CSSCode, *, expected_k: int | None = None,
                      check_rank: bool = True) -> dict:
    hx = _binary(code.hx, name="Hx")
    hz = _binary(code.hz, name="Hz")
    lx = _binary(code.lx, name="Lx")
    lz = _binary(code.lz, name="Lz")
    if hx.shape[1] != hz.shape[1] or hx.shape[1] != lx.shape[1] or hx.shape[1] != lz.shape[1]:
        raise ValueError("CSS matrices have inconsistent block length")
    n = hx.shape[1]
    if check_rank:
        specs = css_specs(hx, hz)
        k = int(specs["k"])
    else:
        if expected_k is None:
            raise ValueError("expected_k required when check_rank=False")
        wx = hx.sum(1)
        wz = hz.sum(1)
        weights = np.r_[wx, wz]
        specs = {
            "n": int(n), "k": int(expected_k), "rank_x": None, "rank_z": None,
            "rate": float(expected_k / n),
            "commutation_violations": int(np.count_nonzero((hx @ hz.T) & 1)),
            "min_check_weight": int(min(wx.min() if len(wx) else 0,
                                         wz.min() if len(wz) else 0)),
            "max_check_weight": int(max(wx.max() if len(wx) else 0,
                                         wz.max() if len(wz) else 0)),
            "mean_check_weight": float(np.mean(weights)) if len(weights) else 0.0,
        }
    k = int(specs["k"])
    if lx.shape != (k, n) or lz.shape != (k, n):
        raise ValueError(f"logical bases must have shape ({k}, {n})")
    if specs["commutation_violations"]:
        raise ValueError("CSS stabilizers do not commute")
    if np.count_nonzero((hz @ lx.T) & 1) or np.count_nonzero((hx @ lz.T) & 1):
        raise ValueError("logical basis does not commute with opposite stabilizers")
    pairing = (lx @ lz.T) & 1
    pairing_rank = gf2_rank(pairing)
    if pairing_rank != k:
        raise ValueError("X/Z logical pairing is singular")
    return {
        **specs,
        "logical_pairing_rank": int(pairing_rank),
        "logical_pairing": pairing.astype(int).tolist(),
    }


def css_concatenate(outer: CSSCode, inner: CSSCode, *, name: str | None = None) -> CSSCode:
    """Serial-concatenate CSS codes; inner code must encode exactly one qubit."""
    outer_specs = validate_css_code(outer)
    inner_specs = validate_css_code(inner)
    if inner_specs["k"] != 1:
        raise ValueError("serial CSS concatenation requires inner k=1")
    inner_x = inner.lx[0]
    inner_z = inner.lz[0]
    blocks = int(outer_specs["n"])
    hx = np.vstack((_block_repeat(inner.hx, blocks), _lift(outer.hx, inner_x)))
    hz = np.vstack((_block_repeat(inner.hz, blocks), _lift(outer.hz, inner_z)))
    lx = _lift(outer.lx, inner_x)
    lz = _lift(outer.lz, inner_z)
    result = CSSCode(
        name=name or f"{outer.name} ⊗ {inner.name}",
        hx=hx,
        hz=hz,
        lx=lx,
        lz=lz,
    )
    # Construction proves rank/k algebraically. Avoid dense O(n^3) rank here;
    # large concatenated blocks are validated by commutation + logical pairing.
    result_specs = validate_css_code(result, expected_k=outer_specs["k"], check_rank=False)
    if result_specs["k"] != outer_specs["k"]:
        raise ValueError("concatenation changed encoded-qubit count")
    return result


def css_block_concatenate(outer: CSSCode, inner: CSSCode,
                          *, name: str | None = None) -> CSSCode:
    """Concatenate outer CSS qubits in groups of ``inner.k``.

    For an inner ``[[n_i,q,d_i]]`` and an outer whose n is divisible by q,
    resulting parameters are ``[[n_o*n_i/q, k_o, >= d_o*d_i]]`` under the
    usual concatenation theorem.  Search results still record distance only
    as a witness-backed upper bound.
    """
    outer_specs = validate_css_code(outer)
    inner_specs = validate_css_code(inner)
    q = int(inner_specs["k"])
    if int(outer_specs["n"]) % q:
        raise ValueError("outer n must be divisible by inner k")
    blocks = int(outer_specs["n"]) // q
    hx = np.vstack((_block_repeat(inner.hx, blocks),
                    _lift_multilogical(outer.hx, inner.lx)))
    hz = np.vstack((_block_repeat(inner.hz, blocks),
                    _lift_multilogical(outer.hz, inner.lz)))
    lx = _lift_multilogical(outer.lx, inner.lx)
    lz = _lift_multilogical(outer.lz, inner.lz)
    result = CSSCode(
        name=name or f"{outer.name} block-{q} {inner.name}",
        hx=hx, hz=hz, lx=lx, lz=lz,
    )
    result_specs = validate_css_code(result, expected_k=outer_specs["k"],
                                     check_rank=False)
    if result_specs["k"] != outer_specs["k"]:
        raise ValueError("block concatenation changed encoded-qubit count")
    return result


def css_heterogeneous_block_concatenate(outer: CSSCode, inners: list[CSSCode],
                                        *, name: str | None = None) -> CSSCode:
    """Concatenate consecutive outer-coordinate groups through mixed blocks.

    The inner blocks may encode different positive numbers of qubits.  Their
    logical dimensions must sum to the outer length.  This is useful for
    constrained searches where a high-rate outer code must stay under a hard
    physical-length cap.  It establishes CSS structure and k exactly; all
    distance values still require independent validation.
    """
    outer_specs = validate_css_code(outer)
    if not inners:
        raise ValueError("at least one inner block is required")
    inner_specs = [validate_css_code(inner) for inner in inners]
    if any(int(spec["k"]) < 1 for spec in inner_specs):
        raise ValueError("each inner block must encode at least one qubit")
    if sum(int(spec["k"]) for spec in inner_specs) != int(outer_specs["n"]):
        raise ValueError("inner logical dimensions must sum to outer n")
    hx = np.vstack((_block_diagonal([inner.hx for inner in inners]),
                    _lift_heterogeneous(outer.hx, [inner.lx for inner in inners])))
    hz = np.vstack((_block_diagonal([inner.hz for inner in inners]),
                    _lift_heterogeneous(outer.hz, [inner.lz for inner in inners])))
    lx = _lift_heterogeneous(outer.lx, [inner.lx for inner in inners])
    lz = _lift_heterogeneous(outer.lz, [inner.lz for inner in inners])
    result = CSSCode(name=name or f"{outer.name} heterogeneous block concatenate",
                     hx=hx, hz=hz, lx=lx, lz=lz)
    result_specs = validate_css_code(result, expected_k=outer_specs["k"],
                                     check_rank=False)
    if result_specs["k"] != outer_specs["k"]:
        raise ValueError("heterogeneous concatenation changed encoded-qubit count")
    return result


def steane_code() -> CSSCode:
    """Steane [[7,1,3]] CSS code, using weight-3 logical representatives."""
    h = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ], dtype=np.uint8)
    logical = np.array([[1, 1, 1, 0, 0, 0, 0]], dtype=np.uint8)
    code = CSSCode("Steane-7", h, h.copy(), logical, logical.copy())
    validate_css_code(code)
    return code


def repetition3_code() -> CSSCode:
    """Valid CSS [[3,1,1]] repetition code; useful as a baseline only."""
    hz = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    hx = np.zeros((0, 3), dtype=np.uint8)
    lx = np.array([[1, 1, 1]], dtype=np.uint8)
    lz = np.array([[1, 0, 0]], dtype=np.uint8)
    code = CSSCode("Repetition-3", hx, hz, lx, lz)
    validate_css_code(code)
    return code


def identity_css_code() -> CSSCode:
    """Trivial CSS [[1,1,1]] passthrough block for partial concatenation."""
    empty = np.zeros((0, 1), dtype=np.uint8)
    logical = np.ones((1, 1), dtype=np.uint8)
    code = CSSCode("Identity-1", empty, empty.copy(), logical, logical.copy())
    validate_css_code(code)
    return code


def four_two_two_code() -> CSSCode:
    """CSS [[4,2,2]] inner block with weight-two logical representatives."""
    hx = np.array([[1, 1, 1, 1]], dtype=np.uint8)
    hz = hx.copy()
    lx = np.array([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=np.uint8)
    lz = np.array([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=np.uint8)
    code = CSSCode("Four-Two-Two", hx, hz, lx, lz)
    validate_css_code(code)
    return code


def even_parity_css_code(k: int) -> CSSCode:
    """CSS ``[[k+2,k,2]]`` block for positive even ``k``.

    Both stabilizers are the all-one even-length row.  The displayed logical
    bases use two-qubit representatives with exact X/Z pairing.  These blocks
    are the most length-efficient distance-two CSS inners; heterogeneous use
    lets a campaign trade expansion against block-level logical mixing.
    """
    k = int(k)
    if k < 2 or k % 2:
        raise ValueError("even-parity CSS block requires positive even k")
    n = k + 2
    h = np.ones((1, n), dtype=np.uint8)
    lx = np.zeros((k, n), dtype=np.uint8)
    lz = np.zeros((k, n), dtype=np.uint8)
    for index in range(k):
        lx[index, 0] = lx[index, index + 1] = 1
        lz[index, index + 1] = lz[index, n - 1] = 1
    code = CSSCode(f"Even-Parity-{n}-{k}", h, h.copy(), lx, lz)
    validate_css_code(code)
    return code


def shor9_code() -> CSSCode:
    """Valid CSS form of Shor [[9,1,3]] code; usually over cap for this challenge."""
    hx = np.array([
        [1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1],
    ], dtype=np.uint8)
    hz = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1],
    ], dtype=np.uint8)
    lx = np.ones((1, 9), dtype=np.uint8)
    lz = np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0]], dtype=np.uint8)
    code = CSSCode("Shor-9", hx, hz, lx, lz)
    validate_css_code(code)
    return code


def inner_code_catalog() -> dict[str, CSSCode]:
    return {
        "steane7": steane_code(),
        "repetition3": repetition3_code(),
        "shor9": shor9_code(),
    }


def load_npz_code(path: str | Path, *, name: str | None = None) -> CSSCode:
    path = Path(path)
    with np.load(path) as data:
        code = CSSCode(
            name=name or path.stem,
            hx=data["hx"], hz=data["hz"], lx=data["lx"], lz=data["lz"],
        )
    validate_css_code(code)
    return code
