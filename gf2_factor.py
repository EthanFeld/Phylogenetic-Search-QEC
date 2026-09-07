"""Small, dependency-free GF(2) polynomial factoring helpers.

Polynomials use Python integers: bit ``i`` is the coefficient of ``x**i``.
This is intentionally separate from the distance code.  It is a cheap
algebraic branch generator, not a distance or novelty certificate.
"""

from __future__ import annotations


def degree(poly: int) -> int:
    return int(poly).bit_length() - 1


def mod(poly: int, divisor: int) -> int:
    poly = int(poly)
    divisor = int(divisor)
    if not divisor:
        raise ValueError("polynomial divisor is zero")
    shift = degree(divisor)
    while poly and degree(poly) >= shift:
        poly ^= divisor << (degree(poly) - shift)
    return poly


def quotient(poly: int, divisor: int) -> int:
    """Exact polynomial quotient over GF(2) (remainder is ignored)."""
    poly = int(poly)
    divisor = int(divisor)
    if not divisor:
        raise ValueError("polynomial divisor is zero")
    out = 0
    shift = degree(divisor)
    while poly and degree(poly) >= shift:
        amount = degree(poly) - shift
        out ^= 1 << amount
        poly ^= divisor << amount
    return out


def mul(left: int, right: int) -> int:
    left, right = int(left), int(right)
    out = 0
    while right:
        if right & 1:
            out ^= left
        right >>= 1
        left <<= 1
    return out


def gcd(left: int, right: int) -> int:
    left, right = int(left), int(right)
    while right:
        left, right = right, mod(left, right)
    return left


def _nullspace(rows: list[int], width: int) -> list[int]:
    """Return a GF(2) nullspace basis for a bit-packed matrix."""
    rows = [int(row) for row in rows]
    pivots: list[int] = []
    rank = 0
    for col in range(int(width)):
        pivot = next((i for i in range(rank, len(rows))
                      if (rows[i] >> col) & 1), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for i in range(len(rows)):
            if i != rank and ((rows[i] >> col) & 1):
                rows[i] ^= rows[rank]
        pivots.append(col)
        rank += 1
        if rank == len(rows):
            break
    pivot_set = set(pivots)
    free = [col for col in range(int(width)) if col not in pivot_set]
    basis = []
    for free_col in free:
        vector = 1 << free_col
        for row, pivot_col in reversed(list(zip(rows[:rank], pivots))):
            if (row & vector).bit_count() & 1:
                vector ^= 1 << pivot_col
        basis.append(vector)
    return basis


def berlekamp_basis(poly: int) -> list[int]:
    """Return the Berlekamp subalgebra basis modulo a monic polynomial."""
    poly = int(poly)
    n = degree(poly)
    if n <= 0:
        return [1]

    # Column j of Q is x**(2*j) mod poly.  Multiplication by x**2 generates
    # the columns without exponentiation or dense Python matrices.  (Squaring
    # the previous column would generate x**(2**j), a different matrix.)
    columns = []
    value = 1
    x_squared = 1 << 2
    for _ in range(n):
        columns.append(value)
        value = mod(mul(value, x_squared), poly)
    rows = [0] * n
    for col, column in enumerate(columns):
        bits = column
        while bits:
            bit = bits & -bits
            rows[bit.bit_length() - 1] ^= 1 << col
            bits ^= bit
    for row in range(n):
        rows[row] ^= 1 << row
    return _nullspace(rows, n)


def _split_squarefree(poly: int) -> list[int]:
    poly = int(poly)
    if degree(poly) <= 1:
        return [poly]
    basis = berlekamp_basis(poly)
    if len(basis) <= 1:
        return [poly]
    parts = [poly]
    # Over GF(2), the only Berlekamp shifts are b and b+1.  Each basis vector
    # refines at least one composite part for square-free input.
    for basis_vector in basis[1:]:
        refined = []
        for part in parts:
            if degree(part) <= 1:
                refined.append(part)
                continue
            split = False
            for shift in (0, 1):
                divisor = gcd(part, basis_vector ^ shift)
                if divisor not in (1, part) and divisor:
                    refined.extend((divisor, quotient(part, divisor)))
                    split = True
                    break
            if not split:
                refined.append(part)
        parts = refined
    out = []
    for part in parts:
        if degree(part) <= 1 or len(berlekamp_basis(part)) <= 1:
            out.append(part)
        else:
            out.extend(_split_squarefree(part))
    return out


def factor(poly: int) -> list[int]:
    """Factor a square-free monic GF(2) polynomial into monic irreducibles."""
    poly = int(poly)
    if not poly or not (poly >> degree(poly)) & 1:
        raise ValueError("factor expects a nonzero monic polynomial")
    factors = _split_squarefree(poly)
    factors.sort(key=lambda item: (degree(item), item))
    product = 1
    for item in factors:
        product = mul(product, item)
    if product != poly:
        raise ArithmeticError("GF(2) factorization product mismatch")
    if any(len(berlekamp_basis(item)) != 1 for item in factors):
        raise ArithmeticError("GF(2) factorization left a reducible factor")
    return factors


def factor_xm_plus_one(m: int) -> list[int]:
    """Factor ``x**m + 1`` over GF(2), preserving repeated factors.

    If ``m = odd * 2**s``, Frobenius gives
    ``x**m + 1 = (x**odd + 1)**(2**s)``.  The square-free Berlekamp routine
    therefore runs only on the odd core and each irreducible is returned with
    multiplicity ``2**s``.
    """
    m = int(m)
    if m < 1:
        raise ValueError("x**m + 1 must use a positive m")
    multiplicity = 1
    odd = m
    while odd % 2 == 0:
        odd //= 2
        multiplicity *= 2
    squarefree = factor((1 << odd) | 1)
    return sorted(squarefree * multiplicity,
                  key=lambda item: (degree(item), item))
