from __future__ import annotations

"""ctypes bridge for packed CUDA randomized-echelon RIS."""

import ctypes as C
import hashlib
import os
from pathlib import Path

import numpy as np

import distance_sketch as ds


ROOT = Path(__file__).resolve().parent
CUDA_ROOT = Path(os.environ.get(
    "CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"))
_LIB = None
_ERROR = None
_CSS_CACHE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _load():
    global _LIB, _ERROR
    if _LIB is not None:
        return _LIB
    try:
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            for directory in (CUDA_ROOT / "bin", CUDA_ROOT / "bin" / "x64"):
                if directory.exists():
                    os.add_dll_directory(str(directory))
        path = ROOT / ("qldpc_gpu.dll" if os.name == "nt" else "qldpc_gpu.so")
        lib = C.CDLL(str(path))
        u32p = C.POINTER(C.c_uint32)
        lib.qldpc_gpu_classical_ris.argtypes = [
            u32p, C.c_int, C.c_int, C.c_int, u32p, C.c_int, C.c_int,
            C.c_uint64, C.c_int, C.c_int, C.c_int,
            C.c_int, C.c_int, C.c_int, C.c_int,
            C.POINTER(C.c_int), C.POINTER(C.c_int), u32p, C.POINTER(C.c_double),
        ]
        lib.qldpc_gpu_classical_ris.restype = C.c_int
        _LIB = lib
        return lib
    except Exception as exc:  # pragma: no cover - machine dependent
        _ERROR = exc
        return None


def available() -> bool:
    return bool(_load() is not None)


def load_error():
    _load()
    return _ERROR


def _pack_words(matrix) -> np.ndarray:
    data = np.ascontiguousarray(np.asarray(matrix, dtype=np.uint8))
    if data.ndim != 2:
        raise ValueError("expected 2-D binary matrix")
    rows, n = data.shape
    limbs = (int(n) + 31) // 32
    packed = np.packbits(data, axis=1, bitorder="little")
    raw = np.zeros((rows, limbs * 4), dtype=np.uint8)
    raw[:, :packed.shape[1]] = packed
    return np.ascontiguousarray(raw.view(np.uint32).reshape(rows, limbs))


def _ptr(array: np.ndarray | None):
    if array is None or array.size == 0:
        return C.POINTER(C.c_uint32)()
    return array.ctypes.data_as(C.POINTER(C.c_uint32))


def _witness(words: np.ndarray, n: int) -> list[int]:
    out = []
    for limb, value in enumerate(np.asarray(words, dtype=np.uint32).tolist()):
        value = int(value)
        while value:
            bit = value & -value
            index = 32 * limb + bit.bit_length() - 1
            if index < int(n):
                out.append(index)
            value ^= bit
    return out


def classical_ris(kernel, *, detectors=None, trials=1000, seed=0,
                  pair_depth=24, target=None, stop_on_target=False,
                  threads=448, batch_size=8192, block_width=4,
                  rank_cap=None) -> dict:
    """CUDA randomized-column forward-echelon RIS; witnesses remain upper bounds."""
    basis = _pack_words(kernel)
    n = int(np.asarray(kernel).shape[1])
    dual = None if detectors is None else _pack_words(detectors)
    if dual is not None and dual.shape[1] != basis.shape[1]:
        raise ValueError("detector width mismatch")
    best = C.c_int(n + 1)
    completed = C.c_int(0)
    witness = np.zeros(basis.shape[1], dtype=np.uint32)
    milliseconds = C.c_double(0.0)
    lib = _load()
    if lib is None:
        raise RuntimeError(f"CUDA RIS unavailable: {_ERROR}")
    rc = lib.qldpc_gpu_classical_ris(
        _ptr(basis), int(basis.shape[0]), int(basis.shape[1]), n,
        _ptr(dual), 0 if dual is None else int(dual.shape[0]), int(trials),
        C.c_uint64(int(seed) & ((1 << 64) - 1)), int(pair_depth),
        0 if target is None else int(target), int(bool(stop_on_target)),
        int(block_width), 0 if rank_cap is None else int(rank_cap),
        int(threads), int(batch_size),
        C.byref(best), C.byref(completed), _ptr(witness), C.byref(milliseconds))
    if rc:
        raise RuntimeError(f"CUDA RIS failed ({rc})")
    found = None if best.value > n else _witness(witness, n)
    return {
        "backend": "cuda_randomized_echelon_ris",
        "reduction": "forward_echelon",
        "best_weight": None if found is None else len(found),
        "witness": found,
        "trials_run": int(completed.value),
        "seconds": float(milliseconds.value) / 1000.0,
        "stopped_early": bool(stop_on_target and found is not None and len(found) < int(target or 0)),
        "pair_depth": int(pair_depth),
        "threads": int(threads),
        "batch_size": int(batch_size),
        "block_width": int(block_width),
        "rank_cap": None if rank_cap is None else int(rank_cap),
        "device": "CUDA",
    }


def _css_bases(hx: np.ndarray, hz: np.ndarray):
    digest = hashlib.blake2b(digest_size=16)
    digest.update(hx.shape.__repr__().encode())
    digest.update(hx.tobytes())
    digest.update(hz.shape.__repr__().encode())
    digest.update(hz.tobytes())
    key = digest.hexdigest()
    cached = _CSS_CACHE.get(key)
    if cached is not None:
        return cached
    n = int(hx.shape[1])
    value = (
        ds.gf2_nullspace(hz), ds.gf2_nullspace(hx),
        ds.unpack_rows(ds.gf2_logical_basis_packed(hx, hz), n),
        ds.unpack_rows(ds.gf2_logical_basis_packed(hz, hx), n),
    )
    if len(_CSS_CACHE) >= 4:
        _CSS_CACHE.pop(next(iter(_CSS_CACHE)))
    _CSS_CACHE[key] = value
    return value


def css_ris(hx, hz, *, trials=1000, seed=0, pair_depth=24,
            target=None, stop_on_target=False, threads=448,
            batch_size=8192, block_width=4, rank_cap=None) -> dict:
    """CUDA CSS RIS using CPU-built exact kernel/logical bases."""
    hx = np.ascontiguousarray(np.asarray(hx, dtype=np.uint8))
    hz = np.ascontiguousarray(np.asarray(hz, dtype=np.uint8))
    n = int(hx.shape[1])
    x_kernel, z_kernel, x_dual, z_dual = _css_bases(hx, hz)
    x = classical_ris(x_kernel, detectors=x_dual, trials=trials, seed=seed,
                      pair_depth=pair_depth, target=target,
                      stop_on_target=stop_on_target, threads=threads,
                      batch_size=batch_size, block_width=block_width,
                      rank_cap=rank_cap)
    z = classical_ris(z_kernel, detectors=z_dual, trials=trials, seed=int(seed) + 1,
                      pair_depth=pair_depth, target=target,
                      stop_on_target=stop_on_target, threads=threads,
                      batch_size=batch_size, block_width=block_width,
                      rank_cap=rank_cap)
    values = [v["best_weight"] for v in (x, z) if v["best_weight"] is not None]
    return {
        "backend": "cuda_randomized_echelon_css_ris",
        "reduction": "forward_echelon",
        "rank_cap": None if rank_cap is None else int(rank_cap),
        "dx_upper": x["best_weight"], "dz_upper": z["best_weight"],
        "d_upper": min(values) if values else None,
        "x": x, "z": z,
        "total_sector_shots": int(x["trials_run"] + z["trials_run"]),
        "seconds": float(x["seconds"] + z["seconds"]),
    }
