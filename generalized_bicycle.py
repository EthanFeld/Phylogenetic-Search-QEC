from __future__ import annotations

"""Cyclic generalized-bicycle construction and low-weight word mining.

For m and supports a,b in Z_m, build H_X=[A|B], H_Z=[B^T|A^T].
Commutation is automatic because cyclic circulants commute over GF(2).
The polynomial gcd gives k=2*deg(gcd(a,b,x^m-1)); distance remains an
upper-bound claim until witness and official refutation gates pass.
"""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

REFERENCE_SOURCE = "https://unitaryfoundation.github.io/qldpc-challenge/codes/682-182-75.html"
REFERENCE_M = 341
REFERENCE_A = (37, 55, 71, 102, 159, 169, 172, 182, 190, 244, 246, 265, 268, 272, 290, 301)
REFERENCE_B = (0, 11, 70, 81, 114, 153, 217, 227, 228, 235, 268, 280, 311, 320, 323, 339)
REFERENCE_WITNESS_X = (0, 5, 16, 25, 29, 30, 33, 36, 38, 44, 63, 71, 77, 98, 109, 117,
                      121, 127, 189, 199, 205, 208, 223, 229, 239, 241, 248, 258, 273,
                      288, 295, 296, 314, 349, 350, 362, 378, 379, 383, 388, 389, 397,
                      401, 408, 409, 411, 417, 424, 432, 436, 441, 446, 461, 465, 473,
                      493, 514, 518, 520, 535, 550, 561, 563, 566, 574, 583, 598, 606,
                      609, 617, 624, 658, 661, 668, 681)
REFERENCE_WITNESS_Z = (1, 14, 21, 24, 58, 65, 73, 76, 84, 99, 108, 116, 119, 121, 132,
                      147, 162, 164, 168, 189, 209, 217, 221, 236, 241, 246, 250, 258,
                      265, 271, 273, 274, 281, 285, 293, 294, 299, 303, 304, 320, 332,
                      333, 341, 368, 386, 387, 394, 409, 424, 434, 441, 443, 453, 459,
                      474, 477, 483, 493, 555, 561, 565, 573, 584, 605, 611, 619, 638,
                      644, 646, 649, 652, 653, 657, 666, 677)


@dataclass(frozen=True)
class GeneralizedBicycle:
    m: int
    a: tuple[int, ...]
    b: tuple[int, ...]

    @property
    def n(self) -> int:
        return 2 * self.m

    @property
    def gcd_degree(self) -> int:
        return common_gcd_degree(self.m, self.a, self.b)

    @property
    def k(self) -> int:
        return 2 * self.gcd_degree

    @property
    def check_weight(self) -> int:
        return len(self.a) + len(self.b)


def _poly_mod(a: int, b: int) -> int:
    if not b:
        raise ValueError("polynomial divisor is zero")
    degree = b.bit_length() - 1
    while a and a.bit_length() - 1 >= degree:
        a ^= b << (a.bit_length() - 1 - degree)
    return a


def _poly_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, _poly_mod(a, b)
    return a


def _poly_from_support(support: tuple[int, ...] | list[int], m: int) -> int:
    mask = 0
    for exponent in support:
        exponent = int(exponent)
        if not 0 <= exponent < m:
            raise ValueError("support exponent out of range")
        mask ^= 1 << exponent
    return mask


def common_gcd_degree(m: int, a, b) -> int:
    modulus = (1 << int(m)) | 1  # x^m - 1 == x^m + 1 over GF(2)
    return _poly_gcd(_poly_gcd(_poly_from_support(tuple(a), m),
                               _poly_from_support(tuple(b), m)), modulus).bit_length() - 1


def _check_support(support, m: int, name: str) -> tuple[int, ...]:
    out = tuple(sorted(int(x) for x in support))
    if len(out) != len(set(out)):
        raise ValueError(f"{name} has duplicate exponents")
    if any(x < 0 or x >= m for x in out):
        raise ValueError(f"{name} exponent out of range")
    return out


def normalize_code(m: int, a, b) -> GeneralizedBicycle:
    m = int(m)
    if m < 2:
        raise ValueError("m must be >=2")
    a = _check_support(a, m, "a")
    b = _check_support(b, m, "b")
    if not a or not b:
        raise ValueError("a,b must be nonempty")
    return GeneralizedBicycle(m, a, b)


def build_supports(m: int, a, b) -> tuple[list[list[int]], list[list[int]]]:
    code = normalize_code(m, a, b)
    hx, hz = [], []
    for i in range(code.m):
        hx.append(sorted([(i + e) % code.m for e in code.a] +
                         [code.m + (i + e) % code.m for e in code.b]))
        hz.append(sorted([(i - e) % code.m for e in code.b] +
                         [code.m + (i - e) % code.m for e in code.a]))
    return hx, hz


def build_matrices(m: int, a, b) -> tuple[np.ndarray, np.ndarray]:
    hx, hz = build_supports(m, a, b)
    n = 2 * int(m)
    return _supports_to_binary(hx, n), _supports_to_binary(hz, n)


def _supports_to_binary(rows, n: int) -> np.ndarray:
    """Convert sparse support rows to packed-free uint8 matrices."""
    out = np.zeros((len(rows), int(n)), dtype=np.uint8)
    for row, support in enumerate(rows):
        out[row, list(support)] = 1
    return out


def cyclic_shift(mask: int, shift: int, m: int) -> int:
    shift %= int(m)
    if not shift:
        return int(mask) & ((1 << m) - 1)
    ring = (1 << m) - 1
    return ((int(mask) << shift) | (int(mask) >> (m - shift))) & ring


def mask_support(mask: int) -> tuple[int, ...]:
    out = []
    mask = int(mask)
    while mask:
        bit = mask & -mask
        out.append(bit.bit_length() - 1)
        mask ^= bit
    return tuple(out)


def mine_low_words(m: int, bases, *, iterations: int = 100_000,
                   min_weight: int = 10, max_weight: int = 18, seed: int = 0,
                   max_words: int = 2048) -> list[tuple[int, ...]]:
    """Mine low-weight cyclic-code words from shifted seed-word combinations.

    Every result lies in span of cyclic shifts of supplied bases, hence shares
    their common polynomial divisor. This is a discovery heuristic, not proof.
    """
    m = int(m)
    base_masks = [_poly_from_support(tuple(base), m) for base in bases]
    shifts = [[cyclic_shift(mask, t, m) for t in range(m)] for mask in base_masks]
    found = {mask_support(mask) for mask in base_masks
             if min_weight <= mask.bit_count() <= max_weight}
    # Deterministic pair shell catches structured cancellation cheaply.
    for left in shifts:
        for right in shifts:
            for x in left:
                for y in right:
                    word = x ^ y
                    weight = word.bit_count()
                    if min_weight <= weight <= max_weight:
                        found.add(mask_support(word))
                        if len(found) >= max_words:
                            return sorted(found)
    rng = np.random.default_rng(int(seed))
    for _ in range(int(iterations)):
        word = 0
        terms = int(rng.integers(2, 9))
        for _ in range(terms):
            side = int(rng.integers(0, len(shifts)))
            offset = int(rng.integers(0, m))
            word ^= shifts[side][offset]
        weight = word.bit_count()
        if min_weight <= weight <= max_weight:
            found.add(mask_support(word))
            if len(found) >= max_words:
                break
    return sorted(found)


def reference_code() -> GeneralizedBicycle:
    return normalize_code(REFERENCE_M, REFERENCE_A, REFERENCE_B)
