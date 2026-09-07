"""CUDA RIS scout for cheap, independently checkable witness discovery.

This is deliberately a scout, not a distance proof.  It samples exact GF(2)
linear combinations of a kernel basis on CUDA, returns the lightest masks,
and leaves CSS/logical verification to the existing CPU verifier.
"""

from __future__ import annotations

import time

import numpy as np

import distance_sketch as ds

try:  # Optional dependency; CPU/native paths must remain importable.
    import torch
except Exception:  # pragma: no cover - environment dependent
    torch = None


def available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def _limb_tensor(rows: list[int], n: int, device):
    limbs = (int(n) + 63) // 64
    raw = np.zeros((len(rows), limbs), dtype=np.uint64)
    mask = (1 << 64) - 1
    for i, row in enumerate(rows):
        value = int(row)
        for j in range(limbs):
            raw[i, j] = value & mask
            value >>= 64
    # CUDA bitwise ops support signed int64, not uint64, on the installed
    # PyTorch build.  The bit pattern is preserved by this view.
    signed = raw.view(np.int64)
    return torch.from_numpy(signed).to(device=device, non_blocking=True)


def _python_masks(tensor) -> list[int]:
    raw = tensor.detach().cpu().numpy().astype(np.int64, copy=False)
    out = []
    for row in raw:
        value = 0
        for j, limb in enumerate(row):
            part = int(limb)
            if part < 0:
                part += 1 << 64
            value |= part << (64 * j)
        out.append(value)
    return out


def _support(mask: int, n: int) -> list[int]:
    mask = int(mask)
    out = []
    while mask:
        bit = mask & -mask
        index = bit.bit_length() - 1
        if index >= int(n):
            break
        out.append(index)
        mask ^= bit
    return out


def scout(matrix, n: int, *, trials: int = 1_000_000, seed: int = 0,
          batch: int = 32_768, keep: int = 32) -> dict:
    """Return light kernel words sampled with CUDA GF(2) XOR batches."""
    if not available():
        raise RuntimeError("CUDA/PyTorch unavailable")
    n = int(n); trials = max(0, int(trials)); batch = max(256, int(batch))
    keep = max(1, int(keep))
    rows = ds.pack_rows(np.asarray(matrix, dtype=np.uint8))
    basis = ds.gf2_nullspace_packed_basis_rows(rows, n)
    if not basis or not trials:
        return {"backend": "torch_cuda_gf2_combo", "combination_samples": 0,
                "trial_semantics": "CUDA GF(2) combination samples; not RIS trials",
                "rank": len(basis), "candidates": [], "seconds": 0.0}
    device = torch.device("cuda")
    base = _limb_tensor(basis, n, device)
    lut = torch.tensor([int(i).bit_count() for i in range(256)],
                       dtype=torch.int16, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) & ((1 << 63) - 1))
    # Key by mask, not weight.  Equal-weight words are distinct evidence and
    # must all survive the GPU-to-CPU verification handoff.
    found: dict[int, int] = {}
    remaining = trials
    term_counts = (1, 2, 3, 4, 6, 8)
    started = time.perf_counter()
    batch_index = 0
    while remaining > 0:
        draw = min(batch, remaining)
        terms = term_counts[batch_index % len(term_counts)]
        batch_index += 1
        indices = torch.randint(len(basis), (draw, terms), generator=generator,
                                device=device)
        values = base[indices[:, 0]].clone()
        for term in range(1, terms):
            values = torch.bitwise_xor(values, base[indices[:, term]])
        byte_values = values.view(torch.uint8).to(torch.long)
        weights = lut[byte_values].sum(dim=1)
        take = min(keep, draw)
        best_indices = torch.topk(weights, k=take, largest=False).indices
        best_values = values[best_indices]
        best_weights = weights[best_indices].to(torch.int32)
        for weight, value in zip(best_weights.cpu().tolist(), _python_masks(best_values)):
            weight = int(weight)
            if weight > 0:
                found[value] = weight
        remaining -= draw
    torch.cuda.synchronize(device)
    ranked = sorted(((weight, mask) for mask, weight in found.items()),
                    key=lambda item: (item[0], item[1]))[:keep]
    return {
        "backend": "torch_cuda_gf2_combo",
        "combination_samples": trials,
        "trial_semantics": "CUDA GF(2) combination samples; not RIS trials",
        "rank": len(basis), "seconds": time.perf_counter() - started,
        "device": torch.cuda.get_device_name(device),
        "candidates": [{"weight": int(weight), "support": _support(mask, n)}
                       for weight, mask in ranked],
    }
