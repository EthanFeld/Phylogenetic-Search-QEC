from __future__ import annotations

"""ctypes bridge for the optional native GF(2)/RIS accelerator."""

import ctypes as C
from pathlib import Path
import time

import numpy as np


_ROOT = Path(__file__).resolve().parent
_NATIVE_NAMES = ("qldpc_fast.dll", "qldpc_fast.so", "qldpc_fast.dylib")
_LIB = None
_LIB_PATH = None
_LOAD_ERROR = None


def _load():
    global _LIB, _LIB_PATH, _LOAD_ERROR
    if _LIB is not None:
        return _LIB
    for name in _NATIVE_NAMES:
        path = _ROOT / name
        if not path.exists():
            continue
        try:
            lib = C.CDLL(str(path))
            u8p = C.POINTER(C.c_ubyte)
            i32p = C.POINTER(C.c_int)
            lib.qldpc_ris_generator.argtypes = [
                u8p, C.c_int, C.c_int, u8p, C.c_int, C.c_int,
                C.c_uint64, C.c_int, C.c_double, C.c_int, C.c_int,
                i32p, i32p, i32p, u8p,
            ]
            lib.qldpc_ris_generator.restype = C.c_int
            lib.qldpc_gf2_rank.argtypes = [u8p, C.c_int, C.c_int]
            lib.qldpc_gf2_rank.restype = C.c_int
            lib.qldpc_batch_systematic.argtypes = [
                C.POINTER(C.c_int), C.POINTER(C.c_int), C.c_int,
                C.POINTER(C.c_int), C.c_int, C.c_int, C.c_int,
                u8p, i32p, i32p,
            ]
            lib.qldpc_batch_systematic.restype = C.c_int
            lib.qldpc_cyclic_ideal_mine.argtypes = [
                C.POINTER(C.c_int), C.c_int, C.c_int, C.c_int,
                C.c_int, C.c_int, C.c_int, C.c_uint64, C.c_int,
                C.POINTER(C.c_int), C.POINTER(C.c_int), C.POINTER(C.c_int),
            ]
            lib.qldpc_cyclic_ideal_mine.restype = C.c_int
            lib.qldpc_cyclic_ideal_hillclimb.argtypes = [
                C.POINTER(C.c_int), C.c_int, C.c_int, C.c_int, C.c_int,
                C.c_int, C.c_int, C.c_uint64, C.c_int,
                C.POINTER(C.c_int), C.POINTER(C.c_int), C.POINTER(C.c_int),
            ]
            lib.qldpc_cyclic_ideal_hillclimb.restype = C.c_int
            lib.qldpc_css_ris.argtypes = [
                u8p, C.c_int, u8p, C.c_int, C.c_int, C.c_int,
                C.c_uint64, C.c_int, C.c_double, C.c_int, C.c_int,
                i32p, i32p, i32p, i32p, i32p, i32p, u8p, u8p, i32p,
            ]
            lib.qldpc_css_ris.restype = C.c_int
            lib.qldpc_css_ris_parallel.argtypes = [
                u8p, C.c_int, u8p, C.c_int, C.c_int, C.c_int,
                C.c_uint64, C.c_int, C.c_double, C.c_int, C.c_int, C.c_int,
                i32p, i32p, i32p, i32p, i32p, i32p, u8p, u8p, i32p,
            ]
            lib.qldpc_css_ris_parallel.restype = C.c_int
            lib.qldpc_css_ris_spanned.argtypes = [
                u8p, C.c_int, u8p, C.c_int, u8p, C.c_int, u8p, C.c_int,
                C.c_int, C.c_int, C.c_uint64, C.c_int, C.c_double, C.c_int,
                C.c_int, i32p, i32p, i32p, i32p, i32p, i32p, u8p, u8p, i32p,
            ]
            lib.qldpc_css_ris_spanned.restype = C.c_int
            lib.qldpc_partition_anneal.argtypes = [
                u8p, C.c_int, C.c_int, C.POINTER(C.c_int), u8p, u8p,
                u8p, C.c_int, C.c_int, C.c_uint64, C.c_int,
                C.POINTER(C.c_int), i32p, i32p,
            ]
            lib.qldpc_partition_anneal.restype = C.c_int
            lib.qldpc_exact_logical.argtypes = [
                u8p, C.c_int, u8p, C.c_int, C.c_int, C.c_int, C.c_double,
                C.c_int, i32p, C.POINTER(C.c_int64), i32p, u8p, i32p, u8p, i32p,
            ]
            lib.qldpc_exact_logical.restype = C.c_int
            _LIB = lib
            _LIB_PATH = path
            return lib
        except Exception as exc:  # pragma: no cover - platform/load dependent
            _LOAD_ERROR = exc
    return None


def available() -> bool:
    return _load() is not None


def load_error():
    _load()
    return _LOAD_ERROR


def accelerator_info() -> dict:
    """Return auditable native-backend status without running a search."""
    lib = _load()
    return {
        "backend": "qldpc_fast",
        "available": bool(lib is not None),
        "library": _LIB_PATH.name if _LIB_PATH is not None else None,
        "library_path": str(_LIB_PATH) if _LIB_PATH is not None else None,
        "load_error": None if lib is not None else str(_LOAD_ERROR or "unavailable"),
        "kernels": [
            "gf2_rank",
            "batch_systematic",
            "cyclic_ideal_mine",
            "cyclic_ideal_hillclimb",
            "classical_ris",
            "css_ris",
            "css_ris_parallel",
            "css_ris_spanned",
            "partition_anneal",
            "exact_connected_support_tanner"
        ],
    }


def _matrix(A, *, cols: int | None = None) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(A, dtype=np.uint8))
    if a.ndim != 2:
        raise ValueError("expected a 2-D binary matrix")
    if cols is not None and a.shape[1] != int(cols):
        raise ValueError("matrix column count mismatch")
    return a


def _ptr(a: np.ndarray):
    # ctypes still needs a valid address for an empty row matrix.
    if a.size:
        return a.ctypes.data_as(C.POINTER(C.c_ubyte))
    dummy = np.zeros(1, dtype=np.uint8)
    return dummy.ctypes.data_as(C.POINTER(C.c_ubyte))


def _witness(a: np.ndarray):
    return np.flatnonzero(a).astype(int).tolist()


def _require():
    lib = _load()
    if lib is None:
        detail = f": {_LOAD_ERROR}" if _LOAD_ERROR else ""
        raise RuntimeError("qldpc_fast native library unavailable" + detail)
    return lib


def gf2_rank(matrix) -> int:
    """Packed native GF(2) row rank for search-time structural filtering."""
    lib = _require()
    data = _matrix(matrix)
    rank = int(lib.qldpc_gf2_rank(_ptr(data), data.shape[0], data.shape[1]))
    if rank < 0:
        raise RuntimeError("native GF(2) rank failed")
    return rank


def partition_anneal(checks, labels, *, block_count=None, frozen=None,
                     must_active=None, tags=None, steps=100_000, seed=0,
                     target_odd=0):
    """Native fixed-capacity block-partition optimizer.

    Labels are swapped only, so every block keeps its supplied capacity.  The
    objective is the number of odd intersections between every check and the
    blocks, which directly controls check growth for even-parity inner codes.
    """
    lib = _require()
    h = _matrix(checks)
    initial = np.ascontiguousarray(np.asarray(labels, dtype=np.int32))
    if initial.shape != (h.shape[1],):
        raise ValueError("labels must have one entry per check column")
    blocks = int(initial.max()) + 1 if block_count is None else int(block_count)
    if blocks < 1 or np.any(initial < 0) or np.any(initial >= blocks):
        raise ValueError("invalid block labels")
    locked = np.zeros(initial.shape, dtype=np.uint8) if frozen is None else np.ascontiguousarray(
        np.asarray(frozen, dtype=np.uint8))
    if locked.shape != initial.shape:
        raise ValueError("frozen must have one entry per check column")
    active = np.zeros(initial.shape, dtype=np.uint8) if must_active is None else np.ascontiguousarray(
        np.asarray(must_active, dtype=np.uint8))
    tag_data = np.zeros(initial.shape, dtype=np.uint8) if tags is None else np.ascontiguousarray(
        np.asarray(tags, dtype=np.uint8))
    if active.shape != initial.shape or tag_data.shape != initial.shape:
        raise ValueError("must_active and tags must have one entry per check column")
    out = np.empty_like(initial)
    max_odd = C.c_int(0)
    cost = C.c_int(0)
    rc = lib.qldpc_partition_anneal(
        _ptr(h), int(h.shape[0]), int(h.shape[1]),
        initial.ctypes.data_as(C.POINTER(C.c_int)), _ptr(locked), _ptr(active),
        _ptr(tag_data), blocks, int(steps),
        C.c_uint64(int(seed) & ((1 << 64) - 1)), int(target_odd),
        out.ctypes.data_as(C.POINTER(C.c_int)), C.byref(max_odd), C.byref(cost),
    )
    if rc:
        raise RuntimeError(f"qldpc_fast partition call failed ({rc})")
    return {"labels": out, "max_odd": int(max_odd.value),
            "cost": int(cost.value), "steps": int(steps),
            "mode": "cpp_fixed_capacity_partition_anneal"}


def classical_ris(W, *, detectors=None, trials=1000, seed=0, pair_depth=8,
                  max_seconds=None, target=None, stop_on_target=False):
    """Native full-kernel RIS on a generator basis.

    Optional detector rows restrict results to words having nonzero pairing
    with at least one detector.  This efficiently searches a code outside a
    specified subcode instead of repeatedly rediscovering its lighter shell.
    """
    lib = _require()
    w = _matrix(W)
    n = int(w.shape[1])
    dual = (np.zeros((0, n), dtype=np.uint8) if detectors is None else
            _matrix(detectors, cols=n))
    best = C.c_int(0); ran = C.c_int(0); timed = C.c_int(0)
    witness = np.zeros(n, dtype=np.uint8)
    rc = lib.qldpc_ris_generator(
        _ptr(w), int(w.shape[0]), n, _ptr(dual), int(dual.shape[0]), int(trials),
        C.c_uint64(int(seed) & ((1 << 64) - 1)), int(pair_depth),
        -1.0 if max_seconds is None else float(max_seconds),
        0 if target is None else int(target), int(bool(stop_on_target)),
        C.byref(best), C.byref(ran), C.byref(timed), _ptr(witness),
    )
    if rc:
        raise RuntimeError(f"qldpc_fast classical call failed ({rc})")
    return {
        "best_weight": None if best.value == 0 else int(best.value),
        "witness": None if best.value == 0 else _witness(witness),
        "trials_run": int(ran.value),
        "seconds": 0.0,
        "stopped_early": bool(target is not None and best.value > 0 and best.value < int(target)),
        "timed_out": bool(timed.value),
        "mode": "cpp_full_kernel_ris",
    }


def batch_systematic(mult, inv, pairs, *, left=True):
    """Build systematic seed generators/probes for one group-side batch."""
    lib = _require()
    table = np.ascontiguousarray(np.asarray(mult, dtype=np.int32))
    inverse = np.ascontiguousarray(np.asarray(inv, dtype=np.int32))
    if table.ndim != 2 or table.shape[0] != table.shape[1] or inverse.shape != (table.shape[0],):
        raise ValueError("invalid finite-group tables")
    s = int(table.shape[0])
    raw = np.asarray(pairs, dtype=np.int32)
    if raw.ndim != 3 or raw.shape[1] != 2 or raw.shape[2] < 1:
        raise ValueError("pairs must have shape (draws, 2, support_len)")
    draws, _, support_len = raw.shape
    flat = np.ascontiguousarray(raw.reshape(draws, 2 * support_len))
    generators = np.zeros((draws, s, 2 * s), dtype=np.uint8)
    probes = np.zeros(draws, dtype=np.int32)
    valid = np.zeros(draws, dtype=np.int32)
    dummy_support = np.zeros(1, dtype=np.int32) if flat.size == 0 else flat
    rc = lib.qldpc_batch_systematic(
        table.ctypes.data_as(C.POINTER(C.c_int)),
        inverse.ctypes.data_as(C.POINTER(C.c_int)), s,
        dummy_support.ctypes.data_as(C.POINTER(C.c_int)), draws, support_len,
        int(bool(left)), _ptr(generators),
        probes.ctypes.data_as(C.POINTER(C.c_int)),
        valid.ctypes.data_as(C.POINTER(C.c_int)),
    )
    if rc:
        raise RuntimeError(f"qldpc_fast batch systematic call failed ({rc})")
    return generators, probes, valid


def cyclic_ideal_mine(bases, *, m, iterations=100000, min_weight=10,
                      max_weight=18, seed=0, max_words=4096):
    """Mine low-weight words in the cyclic ideal spanned by seed supports.

    This is a discovery heuristic: every emitted word is an XOR of cyclic
    shifts of the supplied bases.  It does not certify a code distance or a
    common gcd.  Fixed-width native masks make this suitable for large
    mutation campaigns without Python bitset/object allocation.
    """
    m = int(m); iterations = max(0, int(iterations)); max_words = max(1, int(max_words))
    min_weight = max(0, int(min_weight)); max_weight = min(m, int(max_weight))
    # Sparse GB seeds frequently have different weights (for example the
    # 16+12 Z_337 frontier).  Pad to a rectangular native buffer; the C++
    # kernel treats -1 as a no-op rather than a cyclic exponent.
    try:
        rows = [tuple(int(value) for value in row) for row in bases]
    except TypeError as exc:
        raise ValueError("bases must be an iterable of nonempty supports") from exc
    if not rows or any(not row for row in rows):
        raise ValueError("bases must contain nonempty supports")
    if any(value < 0 or value >= m for row in rows for value in row):
        raise ValueError("base support exponent out of range")
    width_in = max(len(row) for row in rows)
    raw = np.full((len(rows), width_in), -1, dtype=np.int32)
    for index, row in enumerate(rows):
        raw[index, :len(row)] = row
    raw = np.ascontiguousarray(raw)
    width = max_weight
    supports = np.zeros((max_words, width), dtype=np.int32)
    weights = np.zeros(max_words, dtype=np.int32)
    count = C.c_int(0)
    rc = _require().qldpc_cyclic_ideal_mine(
        raw.ctypes.data_as(C.POINTER(C.c_int)), int(raw.shape[0]), int(raw.shape[1]), m,
        iterations, min_weight, max_weight,
        C.c_uint64(int(seed) & ((1 << 64) - 1)), max_words,
        supports.ctypes.data_as(C.POINTER(C.c_int)),
        weights.ctypes.data_as(C.POINTER(C.c_int)), C.byref(count),
    )
    if rc:
        raise RuntimeError(f"qldpc_fast cyclic ideal miner failed ({rc})")
    return [tuple(int(x) for x in supports[i, :int(weights[i])])
            for i in range(int(count.value))]


def cyclic_ideal_hillclimb(base, *, m, restarts=64, steps=2048,
                           min_weight=3, max_weight=31, seed=0,
                           max_words=4096):
    """Native best-of-random coordinate search for sparse ideal words.

    The ordinary ideal miner is excellent when a sparse seed word already
    exists.  A new divisor branch may have only dense generators, so this
    kernel searches XOR combinations of cyclic shifts before the RIS gate.
    It is still heuristic and never certifies distance.
    """
    raw = np.ascontiguousarray(np.asarray(base, dtype=np.int32).reshape(-1))
    if raw.size < 1:
        raise ValueError("base must be nonempty")
    m = int(m); max_words = max(1, int(max_words))
    min_weight = max(0, int(min_weight)); max_weight = min(m, int(max_weight))
    supports = np.zeros((max_words, max_weight), dtype=np.int32)
    weights = np.zeros(max_words, dtype=np.int32)
    count = C.c_int(0)
    rc = _require().qldpc_cyclic_ideal_hillclimb(
        raw.ctypes.data_as(C.POINTER(C.c_int)), int(raw.size), m,
        max(0, int(restarts)), max(0, int(steps)), min_weight, max_weight,
        C.c_uint64(int(seed) & ((1 << 64) - 1)), max_words,
        supports.ctypes.data_as(C.POINTER(C.c_int)),
        weights.ctypes.data_as(C.POINTER(C.c_int)), C.byref(count))
    if rc:
        raise RuntimeError(f"qldpc_fast cyclic hillclimb failed ({rc})")
    return [tuple(int(x) for x in supports[i, :int(weights[i])])
            for i in range(int(count.value))]


def css_ris(hx, hz, *, lx=None, lz=None, trials=1000, seed=0, pair_depth=8,
            max_seconds_per_side=None, target=None, stop_on_target=False):
    """Native CSS RIS; logical quotient bases derived inside C++."""
    lib = _require()
    xh = _matrix(hx)
    zh = _matrix(hz, cols=xh.shape[1])
    n = int(xh.shape[1])
    dx = C.c_int(0); dz = C.c_int(0)
    xt = C.c_int(0); zt = C.c_int(0); xo = C.c_int(0); zo = C.c_int(0)
    refuted = C.c_int(0)
    xw = np.zeros(n, dtype=np.uint8); zw = np.zeros(n, dtype=np.uint8)
    t0 = time.perf_counter()
    timeout = -1.0 if max_seconds_per_side is None else float(max_seconds_per_side)
    target_value = 0 if target is None else int(target)
    if lx is None or lz is None:
        rc = lib.qldpc_css_ris(
            _ptr(xh), int(xh.shape[0]), _ptr(zh), int(zh.shape[0]), n,
            int(trials), C.c_uint64(int(seed) & ((1 << 64) - 1)), int(pair_depth),
            timeout, target_value, int(bool(stop_on_target)),
            C.byref(dx), C.byref(dz), C.byref(xt), C.byref(zt),
            C.byref(xo), C.byref(zo), _ptr(xw), _ptr(zw), C.byref(refuted),
        )
    else:
        xl = _matrix(lx, cols=n); zl = _matrix(lz, cols=n)
        rc = lib.qldpc_css_ris_spanned(
            _ptr(xh), int(xh.shape[0]), _ptr(zh), int(zh.shape[0]),
            _ptr(xl), int(xl.shape[0]), _ptr(zl), int(zl.shape[0]), n,
            int(trials), C.c_uint64(int(seed) & ((1 << 64) - 1)), int(pair_depth),
            timeout, target_value, int(bool(stop_on_target)),
            C.byref(dx), C.byref(dz), C.byref(xt), C.byref(zt),
            C.byref(xo), C.byref(zo), _ptr(xw), _ptr(zw), C.byref(refuted),
        )
    if rc:
        raise RuntimeError(f"qldpc_fast CSS call failed ({rc})")

    def side(best, ran, timed, witness):
        return {
            "best_weight": None if best.value == 0 else int(best.value),
            "witness": None if best.value == 0 else _witness(witness),
            "trials_run": int(ran.value),
            "seconds": 0.0,
            "stopped_early": bool(target is not None and best.value > 0 and best.value < int(target)),
            "timed_out": bool(timed.value),
            "mode": "cpp_full_kernel_ris",
        }

    x = side(dx, xt, xo, xw); z = side(dz, zt, zo, zw)
    vals = [v for v in (x["best_weight"], z["best_weight"]) if v is not None]
    return {
        "dx_upper": x["best_weight"],
        "dz_upper": z["best_weight"],
        "d_upper": min(vals) if vals else None,
        "x": x,
        "z": z,
        "screen_refuted": bool(refuted.value),
        "total_sector_shots": int(xt.value + zt.value),
        "total_seconds": time.perf_counter() - t0,
        "mode": "cpp_full_kernel_ris",
    }


def css_ris_parallel(hx, hz, *, trials=1000, seed=0, pair_depth=8,
                     combo_depth=2, max_seconds_per_side=None, target=None,
                     stop_on_target=False, threads=0):
    """Parallel native CSS RIS; kernels/logicals built once, sectors concurrent.

    ``combo_depth=3`` checks triples among the light ``pair_depth`` RREF rows;
    it is a bounded stronger screen, not a proof. ``threads=0`` lets C++ use
    hardware concurrency, split across X/Z sectors.
    This is research acceleration only; returned witnesses still require
    independent verifier validation.
    """
    lib = _require()
    xh = _matrix(hx)
    zh = _matrix(hz, cols=xh.shape[1])
    n = int(xh.shape[1])
    dx = C.c_int(0); dz = C.c_int(0)
    xt = C.c_int(0); zt = C.c_int(0); xo = C.c_int(0); zo = C.c_int(0)
    refuted = C.c_int(0)
    xw = np.zeros(n, dtype=np.uint8); zw = np.zeros(n, dtype=np.uint8)
    t0 = time.perf_counter()
    timeout = -1.0 if max_seconds_per_side is None else float(max_seconds_per_side)
    target_value = 0 if target is None else int(target)
    if int(combo_depth) not in (2, 3):
        raise ValueError("combo_depth must be 2 or 3")
    native_depth = abs(int(pair_depth))
    if native_depth < 1:
        raise ValueError("pair_depth must be positive")
    if int(combo_depth) == 3:
        native_depth = -native_depth
    rc = lib.qldpc_css_ris_parallel(
        _ptr(xh), int(xh.shape[0]), _ptr(zh), int(zh.shape[0]), n,
        int(trials), C.c_uint64(int(seed) & ((1 << 64) - 1)), native_depth,
        timeout, target_value, int(bool(stop_on_target)), int(threads),
        C.byref(dx), C.byref(dz), C.byref(xt), C.byref(zt),
        C.byref(xo), C.byref(zo), _ptr(xw), _ptr(zw), C.byref(refuted),
    )
    if rc:
        raise RuntimeError(f"qldpc_fast parallel CSS call failed ({rc})")

    def side(best, ran, timed, witness):
        return {
            "best_weight": None if best.value == 0 else int(best.value),
            "witness": None if best.value == 0 else _witness(witness),
            "trials_run": int(ran.value),
            "seconds": 0.0,
            "stopped_early": bool(target is not None and best.value > 0 and best.value < int(target)),
            "timed_out": bool(timed.value),
            "mode": "cpp_parallel_full_kernel_ris",
        }

    x = side(dx, xt, xo, xw); z = side(dz, zt, zo, zw)
    vals = [v for v in (x["best_weight"], z["best_weight"]) if v is not None]
    return {
        "dx_upper": x["best_weight"],
        "dz_upper": z["best_weight"],
        "d_upper": min(vals) if vals else None,
        "x": x,
        "z": z,
        "screen_refuted": bool(refuted.value),
        "total_sector_shots": int(xt.value + zt.value),
        "total_seconds": time.perf_counter() - t0,
        "mode": "cpp_parallel_full_kernel_ris",
    }


def exact_logical(H, D, cap, *, time_limit=None, variable_order="degree_desc"):
    """Native exact connected-support Tanner search (n <= 384)."""
    lib = _require()
    h = _matrix(H)
    d = _matrix(D, cols=h.shape[1])
    n = int(h.shape[1])
    if n > 384:
        raise ValueError("native exact bitmask supports n <= 384")
    status = C.c_int(0); nodes = C.c_int64(0); starts = C.c_int(0)
    witness = np.zeros(n, dtype=np.uint8); weight = C.c_int(0)
    signature = np.zeros(d.shape[0], dtype=np.uint8); signature_len = C.c_int(0)
    order = 0 if variable_order in ("identity", None) else 1
    if variable_order not in ("identity", None, "degree_desc"):
        raise ValueError(f"unknown variable_order: {variable_order}")
    t0 = time.perf_counter()
    rc = lib.qldpc_exact_logical(
        _ptr(h), int(h.shape[0]), _ptr(d), int(d.shape[0]), n, int(cap),
        -1.0 if time_limit is None else float(time_limit), order,
        C.byref(status), C.byref(nodes), C.byref(starts), _ptr(witness),
        C.byref(weight), _ptr(signature), C.byref(signature_len),
    )
    if rc:
        raise RuntimeError(f"qldpc_fast exact call failed ({rc})")
    names = {0: "certified_none", 1: "found", 2: "timeout"}
    found = status.value == 1
    return {
        "status": names.get(status.value, "error"), "cap": int(cap),
        "nodes": int(nodes.value), "starts_done": int(starts.value),
        "seconds": time.perf_counter() - t0,
        "witness_support": _witness(witness) if found else None,
        "witness_weight": int(weight.value) if found else None,
        "logical_signature": _witness(signature) if found else None,
        "timed_out": status.value == 2,
        "mode": "cpp_exact_connected_support_tanner",
    }
