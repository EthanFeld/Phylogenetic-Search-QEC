from __future__ import annotations

"""Whole-lattice Z_345, [[690,188,*]] generalized-bicycle campaign.

The campaign combines two phylogenetic arms:

* exploitation: extend the two strongest observed degree-55 Z_345 divisors
  by every compatible degree-39 factor graft;
* exploration: scout all 99 degree-94 divisors of x^345+1.

Plain minimum-word search repeatedly falls into deeper subideals.  Since every
degree-94 divisor omits x+1, an x+1-child detector forces odd words outside
that child.  Exact gcd filtering then keeps genuine degree-94 words.  XORing
those words with shifted nested descendants gives cheap, lineage-preserving
mutations which remain in the parent ideal and often improve check geometry.

All distance values are witness-backed randomized upper bounds.  This file
never submits, commits, or represents a clean randomized run as a proof.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import time

import numpy as np

import cpp_fast
from distance_sketch import gf2_nullspace
from generalized_bicycle import (build_matrices, build_supports,
                                 common_gcd_degree, normalize_code)
from gf2_factor import degree, factor_xm_plus_one, gcd, mod, mul, quotient
from hyper_validator import _logical_check, structural_validate
from phylogenetic_search import select_phylogenetic_elites


M = 345
N = 2 * M
Q = 94
K = 2 * Q
CHECK_CAP = 32
MODULUS = (1 << M) | 1
UNIT_MULTIPLIERS = tuple(unit for unit in range(1, M) if math.gcd(unit, M) == 1)
# Current board line supplied by the user; keep search admission aligned with
# the live challenge instead of the obsolete historical 1582.226 baseline.
LEADER_SCORE = 1541.0
TARGETED_IDEAL_TRIALS = 4
# Common-gcd ideal gate. <h> has dimension deg(g); valid first-block words
# are X logicals. Small budget = refutation sieve; finalists need long gates.
COMMON_GCD_IDEAL_TRIALS = 1024
EXACT_EXACT_ONLY = False
# Learned from independently checked d=16/d=20/d=24 witnesses.  These are
# search subspaces, not assumed witnesses: every emitted vector is still
# checked by exact kernel construction plus outside-stabilizer detectors.
KILLER_IDEAL_FACTOR_SETS = (
    (0, 1, 4, 5, 6, 7, 8, 11, 14),       # degree 161
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 14), # degree 169
)
ONE_BLOCK_KILLER_IDEAL_FACTOR_SETS = (
    (1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13),  # degree285, weight21 generator
)
DEFAULT_PARENT_REPORT = Path(
    "results/projection_breakout_05_m345q55_pair24/campaign.json")


def _support(poly: int) -> tuple[int, ...]:
    return tuple(index for index in range(M) if (int(poly) >> index) & 1)


def _poly(word) -> int:
    out = 0
    for value in word:
        out ^= 1 << (int(value) % M)
    return out


def _shift(word, amount: int) -> tuple[int, ...]:
    return tuple(sorted((int(value) + int(amount)) % M for value in word))


@lru_cache(maxsize=500_000)
def _orbit_cached(values: tuple[int, ...]) -> tuple[int, ...]:
    if not values:
        return ()
    # Lexicographically minimal cyclic translate starts at zero, so only
    # translates anchored by a support element can win.
    return min(tuple(sorted((value - anchor) % M for value in values))
               for anchor in values)


def _orbit(word) -> tuple[int, ...]:
    return _orbit_cached(tuple(sorted(int(value) % M for value in word)))


def _pair_key(a, b) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Canonicalize all distance-preserving GB shifts and block swap.

    The two circulant blocks may be shifted independently.  A relative shift
    of B is absorbed by a cyclic permutation of its qubit block together with
    independent cyclic row permutations of H_X and H_Z.  Keeping that phase
    generated hundreds of WL-equivalent genomes and converted RIS seed noise
    into fake phylogenetic diversity.
    """
    direct = (_orbit(tuple(a)), _orbit(tuple(b)))
    swapped = (direct[1], direct[0])
    return min(direct, swapped)


def _semantic_hash(a, b) -> str:
    return hashlib.sha256(json.dumps(
        [list(a), list(b)], separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=200_000)
def _affine_pair_key(m: int, a: tuple[int, ...], b: tuple[int, ...]):
    """Canonical pair under unit multipliers plus independent translations."""
    m = int(m)
    units = UNIT_MULTIPLIERS if m == M else tuple(
        unit for unit in range(1, m) if math.gcd(unit, m) == 1)
    return min(
        _pair_key(tuple((unit * value) % m for value in a),
                  tuple((unit * value) % m for value in b))
        for unit in units)


@lru_cache(maxsize=500_000)
def _word_gcd_degree_cached(word: tuple[int, ...]) -> int:
    return degree(gcd(_poly(word), MODULUS))


def _word_gcd_degree(word) -> int:
    return _word_gcd_degree_cached(tuple(int(value) for value in word))


def _factor_indices(poly: int, factors: list[int]) -> tuple[int, ...]:
    return tuple(index for index, factor in enumerate(factors)
                 if mod(int(poly), factor) == 0)


def _ideal_generator(divisor: int) -> np.ndarray:
    """Generator for degree-<345 words divisible by ``divisor``."""
    support = _support(int(divisor))
    width = M - degree(int(divisor))
    out = np.zeros((width, M), dtype=np.uint8)
    for shift in range(width):
        out[shift, [shift + value for value in support]] = 1
    return out


@lru_cache(maxsize=None)
def _two_block_ideal_span(factor_indices: tuple[int, ...]) -> np.ndarray:
    factors = factor_xm_plus_one(M)
    divisor = 1
    for factor_index in factor_indices:
        divisor = mul(divisor, factors[int(factor_index)])
    generator = _ideal_generator(divisor)
    width = generator.shape[0]
    span = np.zeros((2 * width, N), dtype=np.uint8)
    span[:width, :M] = generator
    span[width:, M:] = generator
    return span


@lru_cache(maxsize=None)
def _one_block_ideal_span(factor_indices: tuple[int, ...], block: int) -> np.ndarray:
    factors = factor_xm_plus_one(M)
    divisor = 1
    for factor_index in factor_indices:
        divisor = mul(divisor, factors[int(factor_index)])
    generator = _ideal_generator(divisor)
    span = np.zeros((generator.shape[0], N), dtype=np.uint8)
    start = int(block) * M
    span[:, start:start + M] = generator
    return span


@lru_cache(maxsize=None)
def _one_block_divisor_span(divisor: int, block: int) -> np.ndarray:
    generator = _ideal_generator(int(divisor))
    span = np.zeros((generator.shape[0], N), dtype=np.uint8)
    start = int(block) * M
    span[:, start:start + M] = generator
    return span


def _reverse_support(word) -> tuple[int, ...]:
    return tuple(sorted((-int(value)) % M for value in word))


def _targeted_span_logical(h_check: np.ndarray, h_stabilizer: np.ndarray,
                           span: np.ndarray, *, trials: int,
                           seed: int, target: int) -> dict:
    """RIS inside span intersect ker(H), excluding stabilizer rowspace."""
    coefficient_checks = (h_check @ span.T) & 1
    coefficient_kernel = gf2_nullspace(coefficient_checks)
    restricted_kernel = (coefficient_kernel @ span) & 1
    detectors = gf2_nullspace(h_stabilizer)
    return cpp_fast.classical_ris(
        restricted_kernel, detectors=detectors, trials=int(trials), seed=int(seed),
        pair_depth=24, target=int(target), stop_on_target=True)


def _targeted_ideal_logical(h_check: np.ndarray, h_stabilizer: np.ndarray,
                            factor_indices: tuple[int, ...], *, trials: int,
                            seed: int, target: int) -> dict:
    return _targeted_span_logical(
        h_check, h_stabilizer, _two_block_ideal_span(tuple(factor_indices)),
        trials=trials, seed=seed, target=target)


def _common_gcd_ideal_probe(a, b, hx: np.ndarray, hz: np.ndarray, *,
                            trials: int, seed: int, target: int) -> dict:
    """Probe X logical ideal <h> before full CSS RIS."""
    g = gcd(gcd(_poly(a), _poly(b)), MODULUS)
    h = quotient(MODULUS, g)
    generator = _ideal_generator(h)
    result = cpp_fast.classical_ris(
        generator, trials=int(trials), seed=int(seed), pair_depth=24,
        target=int(target), stop_on_target=True)
    weight = result.get("best_weight")
    witness = result.get("witness")
    probe = {
        "g_degree": int(degree(g)), "h_degree": int(degree(h)),
        "ideal_dimension": int(generator.shape[0]), "trials": int(trials),
        "best_weight": weight, "trials_run": result.get("trials_run"),
        "mode": result.get("mode"), "valid_X_logical": False,
    }
    if weight is None or witness is None:
        return probe
    local = [int(value) for value in witness]
    word_poly = _poly(local)
    check = _logical_check(local, hz, hx)
    probe.update({
        "witness": local,
        "h_divides_w": bool(mod(word_poly, h) == 0),
        "css_check": {key: value for key, value in check.items()
                       if key != "row_rank"},
        "valid_X_logical": bool(mod(word_poly, h) == 0 and check.get("ok")),
    })
    return probe


def target_distance(score: float = LEADER_SCORE) -> int:
    value = max(1, math.floor(math.sqrt(float(score) * N / K)) + 1)
    while K * value * value / N <= float(score):
        value += 1
    return value


def degree94_factor_subsets(factors: list[int] | None = None):
    factors = factor_xm_plus_one(M) if factors is None else factors
    out = []
    for size in range(1, len(factors) + 1):
        for subset in itertools.combinations(range(len(factors)), size):
            if sum(degree(factors[index]) for index in subset) == Q:
                out.append(subset)
    return out


def _parent_factor_sets(path: Path, factors: list[int]) -> dict[frozenset, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    out = {}
    for item in payload.get("mined", []):
        divisor = int(item["divisor"])
        indices = frozenset(_factor_indices(divisor, factors))
        out[indices] = str(item.get("branch_id", "degree55_parent"))
    return out


def enumerate_branches(parent_report: Path = DEFAULT_PARENT_REPORT) -> list[dict]:
    factors = factor_xm_plus_one(M)
    parents = _parent_factor_sets(Path(parent_report), factors)
    branches = []
    for branch_index, subset in enumerate(degree94_factor_subsets(factors)):
        divisor = 1
        for factor_index in subset:
            divisor = mul(divisor, factors[factor_index])
        ancestors = [name for indices, name in parents.items()
                     if indices.issubset(subset)]
        branches.append({
            "branch_index": branch_index,
            "branch_id": f"m{M}_q{Q}_" + "-".join(map(str, subset)),
            "factor_indices": list(subset),
            "factor_degrees": [degree(factors[index]) for index in subset],
            "divisor": int(divisor),
            "divisor_weight": int(divisor).bit_count(),
            "inherited_q55_ancestors": ancestors,
            "phylogenetic_arm": "q55_extension" if ancestors else "q94_novel_branch",
        })
    return branches


def _cycle_profile(a, b) -> dict:
    """Exact 4-cycle profile of H_X=[A|B] under cyclic row shifts."""
    mask_a, mask_b = _poly(tuple(a)), _poly(tuple(b))
    ring = (1 << M) - 1
    overlaps = []
    for delta in range(1, M):
        shifted_a = ((mask_a << delta) | (mask_a >> (M - delta))) & ring
        shifted_b = ((mask_b << delta) | (mask_b >> (M - delta))) & ring
        overlap = (mask_a & shifted_a).bit_count() + (mask_b & shifted_b).bit_count()
        overlaps.append(int(overlap))
    four_cycle_energy = sum(value * (value - 1) // 2 for value in overlaps) // 2
    maximum = max(overlaps, default=0)
    check_weight = len(a) + len(b)
    return {
        "girth_at_least_6": maximum <= 1,
        "maximum_row_overlap": maximum,
        "four_cycle_energy": int(four_cycle_energy),
        "row_pair_min_stabilizer_weight": int(2 * check_weight - 2 * maximum),
    }


def _pair_multiplier_symmetry(a, b) -> int:
    """Count exponent units preserving pair up to shift and block swap."""
    base = _pair_key(a, b)
    count = 0
    for multiplier in UNIT_MULTIPLIERS:
        ma = tuple(sorted((multiplier * value) % M for value in a))
        mb = tuple(sorted((multiplier * value) % M for value in b))
        if _pair_key(ma, mb) == base:
            count += 1
    return count


def _divisor_multiplier_symmetry(divisor: int) -> int:
    """Count exponent units which preserve the exact cyclic ideal."""
    support = _support(divisor)
    count = 0
    for multiplier in UNIT_MULTIPLIERS:
        transformed = tuple((multiplier * value) % M for value in support)
        if degree(gcd(_poly(transformed), int(divisor))) == Q:
            count += 1
    return count


def _classical(generator, *, detectors=None, trials: int, seed: int,
               target: int | None = None):
    result = cpp_fast.classical_ris(
        generator, detectors=detectors, trials=int(trials), seed=int(seed),
        pair_depth=24, target=target, stop_on_target=target is not None)
    witness = result.get("witness")
    return None if witness is None else tuple(int(x) for x in witness)


def _scout_task(task):
    branch, plain_runs, exact_runs, quotient_runs, trials, seed = task
    divisor = int(branch["divisor"])
    factors = factor_xm_plus_one(M)
    generator = _ideal_generator(divisor)
    # Escape one immediate child.  For the original q94 lattice this is x+1;
    # wrappers whose parent already contains x+1 select the first unused
    # irreducible instead.
    used = set(int(index) for index in branch["factor_indices"])
    escape_index = next(index for index in range(len(factors)) if index not in used)
    detector = gf2_nullspace(_ideal_generator(mul(divisor, factors[escape_index])))
    words = {}
    source_counts = {"plain": 0, "x_plus_1_detector": 0,
                     "factor_quotient_detector": 0}
    for run in range(int(plain_runs)):
        word = _classical(generator, trials=trials,
                          seed=int(seed) + run * 104729)
        if word is not None:
            words.setdefault(_orbit(word), "plain")
            source_counts["plain"] += 1
    for run in range(int(exact_runs)):
        word = _classical(generator, detectors=detector, trials=trials,
                          seed=(int(seed) ^ 0x9E3779B9) + run * 104729)
        if word is not None:
            words.setdefault(_orbit(word), "x_plus_1_detector")
            source_counts["x_plus_1_detector"] += 1
    # Escape every immediate child, not only x+1.  Different factor shapes
    # have different nested sinks; complementary shallow descendants can be
    # paired even when no sparse exact-parent word exists.
    for factor_index, factor in enumerate(factors):
        if factor_index in used:
            continue
        child_detector = gf2_nullspace(
            _ideal_generator(mul(divisor, factor)))
        for run in range(int(quotient_runs)):
            word = _classical(
                generator, detectors=child_detector, trials=trials,
                seed=(int(seed) ^ ((factor_index + 1) * 0xD1B54A35)) + run * 104729)
            if word is not None:
                words.setdefault(_orbit(word), f"factor_detector_{factor_index}")
                source_counts["factor_quotient_detector"] += 1
    classified = []
    for word, source in words.items():
        classified.append({"support": list(word), "weight": len(word),
                           "gcd_degree": _word_gcd_degree(word), "source": source})
    exact = [row for row in classified if row["gcd_degree"] == Q]
    nested = [row for row in classified if row["gcd_degree"] > Q]
    complementary = []
    for left, right in itertools.combinations(classified, 2):
        if (left["weight"] + right["weight"] > CHECK_CAP or
                max(left["gcd_degree"], right["gcd_degree"]) > Q + 44):
            continue
        if common_gcd_degree(M, left["support"], right["support"]) == Q:
            complementary.append((left["weight"] + right["weight"],
                                  left["gcd_degree"], right["gcd_degree"]))
    return {
        **branch,
        "divisor_multiplier_symmetry": _divisor_multiplier_symmetry(divisor),
        "scout_words": sorted(classified, key=lambda row: (row["weight"], row["gcd_degree"])),
        "source_counts": source_counts,
        "exact_best": min((row["weight"] for row in exact), default=None),
        "nested_best": min((row["weight"] for row in nested), default=None),
        "exact_count": len(exact), "nested_count": len(nested),
        "complementary_pair_count": len(complementary),
        "complementary_pair_best_shell": max(
            (row[0] for row in complementary), default=None),
        "complementary_pair_depths": sorted({tuple(sorted(row[1:]))
                                              for row in complementary}),
    }


def _mutate_exact_words(exact_words, nested_words, *, generations: int,
                        max_words: int, max_weight: int, seed: int):
    """Grow exact-parent words by XORing shifted nested descendants."""
    rng = random.Random(int(seed))
    exact = {_orbit(word) for word in exact_words if _word_gcd_degree(word) == Q}
    nested = sorted({_orbit(word) for word in nested_words}, key=lambda x: (len(x), x))
    # Native ideal mining fills its fixed buffer with low shells first.  Keep
    # mutating even when that buffer is full: frontier checks need heavier
    # exact parents to pair with 4--8 weight nested descendants.
    frontier = sorted(exact, key=lambda x: (-len(x), x))[:256]
    hard_cap = max(int(max_words) * 4, len(exact) + 4096)
    for _ in range(max(0, int(generations))):
        if not frontier or not nested:
            break
        parents = sorted(frontier, key=lambda x: (-len(x), x))[:256]
        frontier = []
        attempts = max(4096, min(int(max_words) * 64,
                                 len(parents) * len(nested) * M))
        for _ in range(attempts):
            parent = rng.choice(parents)
            child = rng.choice(nested[:min(128, len(nested))])
            shifted = _shift(child, rng.randrange(M))
            word = tuple(sorted(set(parent).symmetric_difference(shifted)))
            if not word or len(word) > int(max_weight):
                continue
            canonical = _orbit(word)
            if canonical in exact or _word_gcd_degree(canonical) != Q:
                continue
            exact.add(canonical)
            frontier.append(canonical)
            if len(exact) >= hard_cap:
                break
        if len(exact) >= hard_cap:
            break
    # Preserve every check-cap-relevant parent shell.  Algebraic parity then
    # selects 15/17 when x+1 is omitted and 14/16/18 when it is included.
    frontier_shell = sorted((word for word in exact
                             if 14 <= len(word) <= CHECK_CAP - 14),
                            key=lambda x: (len(x), x))
    selected = set(frontier_shell[:max(1, int(max_words) * 3 // 4)])
    heavy = sorted(exact, key=lambda x: (-len(x), x))
    for word in heavy:
        selected.add(word)
        if len(selected) >= int(max_words):
            break
    return sorted(selected, key=lambda x: (len(x), x))


def _mine_task(task):
    (branch, detector_runs, detector_trials, lattice_detector_runs,
     child_runs, child_trials,
     max_companion_gcd, ideal_iterations, shell_iterations, max_words,
     hill_restarts, hill_steps, hill_bases,
     mutation_generations, max_exact_weight, seed) = task
    divisor = int(branch["divisor"])
    factors = factor_xm_plus_one(M)
    generator = _ideal_generator(divisor)
    used = set(int(index) for index in branch["factor_indices"])
    escape_index = next(index for index in range(len(factors)) if index not in used)
    detector = gf2_nullspace(_ideal_generator(mul(divisor, factors[escape_index])))
    initial = [tuple(row["support"]) for row in branch.get("scout_words", [])]
    pool = {_orbit(word) for word in initial}

    for run in range(int(detector_runs)):
        word = _classical(generator, detectors=detector,
                          trials=int(detector_trials),
                          seed=int(seed) + run * 104729)
        if word is not None:
            pool.add(_orbit(word))

    for factor_index, factor in enumerate(factors):
        if factor_index in used:
            continue
        quotient_detector = gf2_nullspace(
            _ideal_generator(mul(divisor, factor)))
        for run in range(int(lattice_detector_runs)):
            word = _classical(
                generator, detectors=quotient_detector,
                trials=int(detector_trials),
                seed=(int(seed) ^ ((factor_index + 1) * 0xD1B54A35)) + run * 104729)
            if word is not None:
                pool.add(_orbit(word))

    bases = sorted(pool, key=lambda word: (len(word), word))[:32]
    if bases and int(ideal_iterations) > 0:
        mined = cpp_fast.cyclic_ideal_mine(
            bases, m=M, iterations=int(ideal_iterations), min_weight=3,
            max_weight=int(max_exact_weight), seed=int(seed) ^ 0xA5A55A5A,
            max_words=int(max_words))
        pool.update(_orbit(tuple(int(x) for x in word)) for word in mined)

    # Broad ideal mining can fill its fixed native output buffer with dense
    # words before rare 14--18 weight words appear.  Protect low-shell
    # discovery with separate native hill-climbs from exact-parent seeds.
    # Output remains heuristic; exact gcd classification below is authority.
    low_shell_bases = [word for word in bases if _word_gcd_degree(word) == Q]
    low_shell_bases = low_shell_bases[:max(1, int(hill_bases))]
    for base_index, base in enumerate(low_shell_bases):
        climbed = cpp_fast.cyclic_ideal_hillclimb(
            base, m=M, restarts=int(hill_restarts), steps=int(hill_steps),
            min_weight=14, max_weight=min(18, int(max_exact_weight)),
            seed=(int(seed) ^ 0x51ED2705) + base_index * 0x9E3779B9,
            max_words=int(max_words))
        pool.update(_orbit(tuple(int(x) for x in word)) for word in climbed)
    if bases and int(shell_iterations) > 0:
        # Expand each shallow descendant separately.  Mixing q117 and q138
        # bases lets abundant q117 combinations fill native output before any
        # q138 shell appears, erasing complementary-factor diversity.
        representatives = {}
        for word in bases:
            depth = _word_gcd_degree(word)
            if Q < depth <= Q + 44:
                previous = representatives.get(depth)
                if previous is None or (len(word), word) < (len(previous), previous):
                    representatives[depth] = word
        for depth, base in sorted(representatives.items()):
            shell_mined = cpp_fast.cyclic_ideal_mine(
                [base], m=M, iterations=int(shell_iterations), min_weight=14,
                max_weight=18, seed=(int(seed) ^ 0x6A09E667) + depth * 0x9E37,
                max_words=int(max_words))
            pool.update(_orbit(tuple(int(x) for x in word)) for word in shell_mined)

    exact = [word for word in pool if _word_gcd_degree(word) == Q and
             len(word) <= int(max_exact_weight)]
    nested = [word for word in pool if _word_gcd_degree(word) > Q and
              len(word) <= CHECK_CAP - 1]
    exact = _mutate_exact_words(
        exact, nested, generations=int(mutation_generations),
        max_words=int(max_words), max_weight=int(max_exact_weight), seed=int(seed) ^ 0xC0FFEE)

    # Search immediate factor children explicitly.  Unconstrained parent RIS
    # prefers saturated q=200+ descendants; those produced d=2 in smoke runs.
    # Exact q+11 children empirically expose 13--15 weight companions and keep
    # enough quotient dimension to avoid that collapse mechanism.
    companions = {}
    for factor_index, factor in enumerate(factors):
        if factor_index in used or factor_index == escape_index:
            continue
        child_divisor = mul(divisor, factor)
        child_q = degree(child_divisor)
        if child_q > int(max_companion_gcd):
            continue
        child_generator = _ideal_generator(child_divisor)
        child_detector = gf2_nullspace(
            _ideal_generator(mul(child_divisor, factors[escape_index])))
        for run in range(int(child_runs)):
            word = _classical(
                child_generator, detectors=child_detector,
                trials=int(child_trials),
                seed=(int(seed) ^ ((factor_index + 1) * 0x9E3779B9)) + run * 104729)
            if word is None:
                continue
            canonical = _orbit(word)
            actual_q = _word_gcd_degree(canonical)
            if actual_q != child_q or len(canonical) >= CHECK_CAP:
                continue
            companions.setdefault((canonical, actual_q), {
                "support": list(canonical), "weight": len(canonical),
                "gcd_degree": actual_q, "added_factor_index": factor_index,
                "added_factor_degree": degree(factor),
            })
    return {
        **branch,
        "exact_words": [list(word) for word in exact],
        "nested_words": [list(word) for word in sorted(set(nested), key=lambda x: (len(x), x))],
        "exact_word_count": len(exact), "nested_word_count": len(set(nested)),
        "companion_words": sorted(companions.values(),
                                  key=lambda row: (row["weight"], row["gcd_degree"],
                                                   row["support"])),
        "companion_word_count": len(companions),
        "exact_weight_histogram": _histogram(len(word) for word in exact),
        "nested_weight_histogram": _histogram(len(word) for word in nested),
        "companion_weight_histogram": _histogram(
            row["weight"] for row in companions.values()),
        "companion_gcd_histogram": _histogram(
            row["gcd_degree"] for row in companions.values()),
    }


def _histogram(values) -> dict[str, int]:
    out = {}
    for value in values:
        key = str(int(value))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda item: int(item[0])))


def _categorical_histogram(values) -> dict[str, int]:
    out = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _candidate_prior(row: dict):
    cycle = row["cycle_profile"]
    weights = row["word_weights"]
    return (
        {"complementary_descendant_shell": 2,
         "exact_exact_frontier_shell": 1}.get(row.get("pairing_mode"), 0),
        int(cycle["girth_at_least_6"]),
        -int(cycle["four_cycle_energy"]),
        int(min(weights) >= 10),
        sum(weights),
        int(row["individual_gcd_degrees"][0] != row["individual_gcd_degrees"][1]),
        int(row.get("divisor_multiplier_symmetry", 1)),
        row["semantic_hash"],
    )


def _pair_population(branch: dict, *, count: int, attempts: int,
                     min_check_weight: int, seed: int) -> list[dict]:
    exact = [tuple(word) for word in branch.get("exact_words", [])]
    nested = [(tuple(word), _word_gcd_degree(word))
              for word in branch.get("nested_words", [])]
    companion_rows = branch.get("companion_words", [])
    companions = [(tuple(row["support"]), int(row["gcd_degree"]))
                   for row in companion_rows]
    frontier_exact = [word for word in exact
                      if 14 <= len(word) <= CHECK_CAP - 14]
    shallow = [(word, Q) for word in frontier_exact]
    shallow.extend((word, depth) for word, depth in nested
                   if 14 <= len(word) <= 18 and Q < depth <= Q + 44)
    if len(shallow) < 2 and not (exact and companions):
        return []
    rng = random.Random(int(seed))
    rows = {}

    def add_pair(a, b, expected_depths, pairing_mode: str):
        weight = len(a) + len(b)
        if weight < int(min_check_weight) or weight > CHECK_CAP:
            return
        if common_gcd_degree(M, a, b) != Q:
            return
        key = _pair_key(a, b)
        if key in rows:
            return
        da, db = _word_gcd_degree(key[0]), _word_gcd_degree(key[1])
        if sorted((da, db)) != sorted(int(value) for value in expected_depths):
            return
        cycle = _cycle_profile(*key)
        rows[key] = {
            "A": [list(key[0])], "B": [list(key[1])],
            "m": M, "n": N, "k": K, "gcd_degree": Q,
            "branch_id": branch["branch_id"],
            "branch_index": branch["branch_index"],
            "factor_indices": branch["factor_indices"],
            "phylogenetic_arm": branch["phylogenetic_arm"],
            "inherited_q55_ancestors": branch["inherited_q55_ancestors"],
            "divisor_multiplier_symmetry": branch["divisor_multiplier_symmetry"],
            "pairing_mode": pairing_mode,
            "word_weights": [len(key[0]), len(key[1])],
            "individual_gcd_degrees": [da, db],
            "cycle_profile": cycle,
            "semantic_hash": _semantic_hash(*key),
        }

    reservoir_cap = max(int(count) * 4, int(count) + 512)

    # Deterministic harvest of the legal low shell.  Randomly selecting the
    # shallow-child arm is extremely wasteful when most child words weigh
    # 27--31: for a 32-cap, only the 14--17 shell can pair with a sparse
    # parent.  Enumerate those combinations before stochastic exploration so
    # a rare 15/17 or 14/16 pair cannot be missed by the 10% draw arm.
    if not EXACT_EXACT_ONLY:
        legal_exact = sorted(
            (word for word in exact if len(word) < CHECK_CAP),
            key=lambda word: (len(word), word))
        legal_descendants = []
        for word, depth in shallow:
            legal_descendants.append((word, int(depth),
                                      "complementary_descendant_shell"))
        for word, depth in companions:
            legal_descendants.append((word, int(depth), "shallow_child_control"))
        legal_descendants.sort(key=lambda item: (len(item[0]), item[1], item[0]))
        for a in legal_exact:
            for raw_b, depth, mode in legal_descendants:
                if len(a) + len(raw_b) < int(min_check_weight):
                    continue
                if len(a) + len(raw_b) > CHECK_CAP:
                    # Descendants are sorted by weight; later entries cannot
                    # become legal for this parent.
                    break
                add_pair(a, raw_b, (Q, depth), mode)
                if len(rows) >= reservoir_cap:
                    break
            if len(rows) >= reservoir_cap:
                break

    for _ in range(max(int(attempts), int(count) * 20)):
        if len(rows) >= reservoir_cap:
            break
        draw = rng.randrange(10)
        if EXACT_EXACT_ONLY:
            if len(frontier_exact) < 2:
                break
            a = rng.choice(frontier_exact)
            raw_b = rng.choice(frontier_exact)
            if a == raw_b:
                continue
            b = _shift(raw_b, rng.randrange(M))
            # Board only imposes the configured lower screening floor.  The
            # old literal 30 silently discarded legal 28/29-weight pairs;
            # those lighter pairs can have materially better overlap/girth.
            if len(a) + len(b) < int(min_check_weight):
                continue
            add_pair(a, b, (Q, Q), "exact_exact_frontier_shell")
            continue
        # Main arm: two independently shallow descendants whose extra factor
        # sets are complementary, so common gcd remains exactly q94.
        if len(shallow) >= 2 and draw < 6:
            a, da = rng.choice(shallow)
            raw_b, db = rng.choice(shallow)
            if a == raw_b and da == db:
                continue
            b = _shift(raw_b, rng.randrange(M))
            if len(a) + len(b) < int(min_check_weight):
                continue
            add_pair(a, b, (da, db), "complementary_descendant_shell")
        elif len(frontier_exact) >= 2 and draw < 9:
            a = rng.choice(frontier_exact)
            raw_b = rng.choice(frontier_exact)
            if a == raw_b:
                continue
            b = _shift(raw_b, rng.randrange(M))
            if len(a) + len(b) < int(min_check_weight):
                continue
            add_pair(a, b, (Q, Q), "exact_exact_frontier_shell")
        elif exact and companions:
            a = rng.choice(exact)
            raw_b, expected_depth = rng.choice(companions)
            b = _shift(raw_b, rng.randrange(M))
            add_pair(a, b, (Q, expected_depth), "shallow_child_control")
    ordered = sorted(rows.values(), key=_candidate_prior, reverse=True)
    # Preserve both girth/cycle leaders and heavy-shell leaders.
    cycle_take = ordered[:max(1, int(count) * 3 // 4)]
    heavy = sorted(ordered, key=lambda row: (
        sum(row["word_weights"]), -row["cycle_profile"]["four_cycle_energy"],
        row["semantic_hash"]), reverse=True)
    combined = {row["semantic_hash"]: row for row in cycle_take}
    for row in heavy:
        combined.setdefault(row["semantic_hash"], row)
        if len(combined) >= int(count):
            break
    selected = list(combined.values())[:int(count)]
    # Full multiplier audit is deferred until post-RIS finalists.  Computing
    # it for tens of thousands of doomed genomes costs more than proxy RIS.
    for row in selected:
        row["pair_multiplier_symmetry"] = 1
    return selected


def _screen_task(task):
    row, trials, seed, threads, target_d, ideal_trials = task
    a, b = tuple(row["A"][0]), tuple(row["B"][0])
    hx, hz = build_matrices(M, a, b)
    ideal_probe = _common_gcd_ideal_probe(
        a, b, hx, hz, trials=int(ideal_trials),
        seed=int(seed) ^ 0x6A09E667, target=int(target_d))
    if (ideal_probe.get("valid_X_logical") and
            int(ideal_probe.get("best_weight")) < int(target_d)):
        weight = int(ideal_probe["best_weight"])
        return {
            **row,
            "screen": {
                "d_upper": weight, "dx_upper": weight, "dz_upper": None,
                "trials_per_side": int(trials), "stopped_early": True,
                "mode": "common_gcd_ideal_x_logical_ris",
                "ideal_x_probe": ideal_probe,
            },
            "screen_witnesses": {"X": list(ideal_probe["witness"]), "Z": None},
        }
    ann_a = quotient(MODULUS, gcd(_poly(a), MODULUS))
    ann_b = quotient(MODULUS, gcd(_poly(b), MODULUS))
    ann_at = quotient(MODULUS, gcd(_poly(_reverse_support(a)), MODULUS))
    ann_bt = quotient(MODULUS, gcd(_poly(_reverse_support(b)), MODULUS))
    annihilator_specs = (
        ("Z", hx, hz, ann_a, 0, "ann(a)"),
        ("Z", hx, hz, ann_b, 1, "ann(b)"),
        ("X", hz, hx, ann_bt, 0, "ann(b^T)"),
        ("X", hz, hx, ann_at, 1, "ann(a^T)"),
    )
    for spec_index, (side, h_check, h_stabilizer, divisor, block, label) in enumerate(
            annihilator_specs):
        targeted = _targeted_span_logical(
            h_check, h_stabilizer, _one_block_divisor_span(int(divisor), block),
            trials=TARGETED_IDEAL_TRIALS,
            seed=int(seed) ^ (spec_index + 1) * 0x94D049BB,
            target=int(target_d))
        weight, witness = targeted.get("best_weight"), targeted.get("witness")
        if weight is not None and int(weight) < int(target_d):
            is_x = side == "X"
            return {
                **row,
                "screen": {
                    "d_upper": int(weight),
                    "dx_upper": int(weight) if is_x else None,
                    "dz_upper": None if is_x else int(weight),
                    "trials_per_side": int(trials), "stopped_early": True,
                    "mode": "dynamic_one_block_annihilator_logical_ris",
                    "annihilator": label, "annihilator_degree": degree(int(divisor)),
                    "killer_block": block, "targeted_trials": TARGETED_IDEAL_TRIALS,
                    "ideal_x_probe": ideal_probe,
                },
                "screen_witnesses": {
                    "X": list(witness) if is_x else None,
                    "Z": None if is_x else list(witness),
                },
            }
    for ideal_index, factor_indices in enumerate(KILLER_IDEAL_FACTOR_SETS):
        for side_index, (side, h_check, h_stabilizer) in enumerate(
                (("X", hz, hx), ("Z", hx, hz))):
            targeted = _targeted_ideal_logical(
                h_check, h_stabilizer, factor_indices,
                trials=TARGETED_IDEAL_TRIALS,
                seed=int(seed) ^ (ideal_index + 1) * 0xD1B54A35 ^ side_index * 0x9E3779B9,
                target=int(target_d))
            weight, witness = targeted.get("best_weight"), targeted.get("witness")
            if weight is not None and int(weight) < int(target_d):
                is_x = side == "X"
                return {
                    **row,
                    "screen": {
                        "d_upper": int(weight),
                        "dx_upper": int(weight) if is_x else None,
                        "dz_upper": None if is_x else int(weight),
                        "trials_per_side": int(trials),
                        "stopped_early": True,
                        "mode": "targeted_cyclic_ideal_logical_ris",
                        "killer_ideal_factor_indices": list(factor_indices),
                        "targeted_trials": TARGETED_IDEAL_TRIALS,
                        "ideal_x_probe": ideal_probe,
                    },
                    "screen_witnesses": {
                        "X": list(witness) if is_x else None,
                        "Z": None if is_x else list(witness),
                    },
                }
    for ideal_index, factor_indices in enumerate(ONE_BLOCK_KILLER_IDEAL_FACTOR_SETS):
        for block in (0, 1):
            span = _one_block_ideal_span(tuple(factor_indices), block)
            for side_index, (side, h_check, h_stabilizer) in enumerate(
                    (("X", hz, hx), ("Z", hx, hz))):
                targeted = _targeted_span_logical(
                    h_check, h_stabilizer, span, trials=1,
                    seed=int(seed) ^ (ideal_index + 1) * 0x94D049BB ^
                         block * 0xD1B54A35 ^ side_index * 0x9E3779B9,
                    target=int(target_d))
                weight, witness = targeted.get("best_weight"), targeted.get("witness")
                if weight is not None and int(weight) < int(target_d):
                    is_x = side == "X"
                    return {
                        **row,
                        "screen": {
                            "d_upper": int(weight),
                            "dx_upper": int(weight) if is_x else None,
                            "dz_upper": None if is_x else int(weight),
                            "trials_per_side": int(trials), "stopped_early": True,
                            "mode": "targeted_one_block_cyclic_ideal_logical_ris",
                            "killer_ideal_factor_indices": list(factor_indices),
                            "killer_block": block, "targeted_trials": 1,
                            "ideal_x_probe": ideal_probe,
                        },
                        "screen_witnesses": {
                            "X": list(witness) if is_x else None,
                            "Z": None if is_x else list(witness),
                        },
                    }
    result = cpp_fast.css_ris_parallel(
        hx, hz, trials=int(trials), seed=int(seed), pair_depth=24,
        combo_depth=2, target=int(target_d), stop_on_target=True,
        threads=int(threads))
    return {
        **row,
        "screen": {
            "d_upper": result.get("d_upper"),
            "dx_upper": result.get("dx_upper"),
            "dz_upper": result.get("dz_upper"),
            "trials_per_side": int(trials),
            "ideal_x_probe": ideal_probe,
            "stopped_early": bool(result["x"].get("stopped_early") or
                                  result["z"].get("stopped_early")),
        },
        "screen_witnesses": {"X": result["x"].get("witness"),
                             "Z": result["z"].get("witness")},
    }


def _parallel(fn, rows, *, trials: int, seed: int, workers: int,
              threads: int, target_d: int, ideal_trials: int, label: str):
    tasks = [(row, int(trials), int(seed) + index * 0x9E3779B9,
              int(threads), int(target_d), int(ideal_trials))
             for index, row in enumerate(rows)]
    if not tasks:
        return []
    out = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(fn, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            out.append(future.result())
            if done == 1 or done % 32 == 0 or done == len(futures):
                print(json.dumps({"stage": label, "done": done, "total": len(futures),
                                  "seconds": round(time.perf_counter() - started, 1)}),
                      flush=True)
    return out


def _replicated_screen(rows, *, replicas: int, trials: int, seed: int,
                       workers: int, threads: int, target_d: int,
                       ideal_trials: int, label: str):
    """Fresh-seed replicated screen; retain lightest observed replicate.

    Correlated single-stream winner's curse produced candidates clean at 200k
    yet refuted by the first fresh 20k stream.  Replication is therefore part
    of selection, not deferred validation ceremony.
    """
    replicas = max(1, int(replicas))
    tasks = []
    for index, row in enumerate(rows):
        for replica in range(replicas):
            tagged = {**row, "_screen_replica": replica}
            tasks.append((tagged, int(trials),
                          int(seed) + index * 0x9E3779B9 + replica * 0xD1B54A35,
                          int(threads), int(target_d), int(ideal_trials)))
    if not tasks:
        return []
    raw = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(_screen_task, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            raw.append(future.result())
            if done == 1 or done % 32 == 0 or done == len(futures):
                print(json.dumps({"stage": label, "done": done, "total": len(futures),
                                  "replicas": replicas,
                                  "seconds": round(time.perf_counter() - started, 1)}),
                      flush=True)
    grouped = {}
    for row in raw:
        grouped.setdefault(row["semantic_hash"], []).append(row)
    out = []
    for items in grouped.values():
        winner = min(items, key=lambda row: int(row["screen"].get("d_upper") or N + 1))
        winner = dict(winner)
        winner.pop("_screen_replica", None)
        replica_screens = [{
            "replica": int(item.get("_screen_replica", 0)),
            "d_upper": item["screen"].get("d_upper"),
            "dx_upper": item["screen"].get("dx_upper"),
            "dz_upper": item["screen"].get("dz_upper"),
            "stopped_early": item["screen"].get("stopped_early"),
        } for item in sorted(items, key=lambda row: int(row.get("_screen_replica", 0)))]
        winner["screen"] = {
            **winner["screen"],
            "replicas": replicas,
            "trials_per_side_per_replica": int(trials),
            "configured_total_trials_per_side": replicas * int(trials),
            "replica_screens": replica_screens,
            "stopped_early": any(bool(item["screen"].get("stopped_early"))
                                 for item in items),
        }
        out.append(winner)
    return out


def _survives(row: dict, target_d: int) -> bool:
    screen = row.get("screen", {})
    return (int(screen.get("d_upper") or -1) >= int(target_d) and
            not bool(screen.get("stopped_early", False)))


def _screen_quality(row: dict, target_d: int):
    screen = row.get("screen", {})
    return (
        int(_survives(row, target_d)),
        int(screen.get("d_upper") or -1),
        int(row.get("pair_multiplier_symmetry", 1)),
        *_candidate_prior(row),
    )


def _branch_cap(rows: list[dict], cap: int, target_d: int) -> list[dict]:
    out, counts = [], {}
    for row in sorted(rows, key=lambda item: _screen_quality(item, target_d), reverse=True):
        branch = row["branch_id"]
        if counts.get(branch, 0) >= int(cap):
            continue
        counts[branch] = counts.get(branch, 0) + 1
        out.append(row)
    return out


def _candidate_doc(row: dict, stage: str) -> dict | None:
    screen = row.get("screen", {})
    dx, dz = screen.get("dx_upper"), screen.get("dz_upper")
    wx = row.get("screen_witnesses", {}).get("X")
    wz = row.get("screen_witnesses", {}).get("Z")
    if dx is None or dz is None or not wx or not wz:
        return None
    code = normalize_code(M, row["A"][0], row["B"][0])
    hx, hz = build_supports(M, code.a, code.b)
    distance = min(int(dx), int(dz))
    return {
        "schema_version": "0.1",
        "name": f"[[{N},{K},d<={distance}]] Z_345 q94 phylogenetic GB",
        "code_type": "CSS", "n": N, "k": K,
        "checks": {"X": hx, "Z": hz},
        "distance": {
            "d": distance,
            "X": {"value": int(dx), "confidence": "upper_bound", "witness": list(wx)},
            "Z": {"value": int(dz), "confidence": "upper_bound", "witness": list(wz)},
        },
        "family": "generalized-bicycle",
        "provenance": {
            "authors": ["stage-only autonomous research"],
            "origin": "gb_m345_k188_phylo_campaign", "novelty": "unknown",
            "model": "GPT-5 Codex + phylogenetic factor-lattice search",
            "construction": (f"Z_345 GB; degree-94 factors={row['factor_indices']}; "
                             f"A={tuple(code.a)}; B={tuple(code.b)}"),
            "references": [],
            "notes": f"stage={stage}; randomized witness-backed upper bound",
        },
        "search": {
            "stage": stage, "semantic_hash": row["semantic_hash"],
            "branch_id": row["branch_id"],
            "phylogenetic_arm": row["phylogenetic_arm"],
            "cycle_profile": row["cycle_profile"],
            "pair_multiplier_symmetry": row["pair_multiplier_symmetry"],
            "common_gcd_ideal_probe": screen.get("ideal_x_probe"),
        },
        "regulation": {
            "stage_only": True, "submission_sent": False,
            "git_commit_performed": False, "official_gate_status": "not_run",
            "board_claim_allowed": False, "distance_is_not_proven": True,
            "literature_novelty": "unverified",
        },
    }


def _compact_structural(result: dict) -> dict:
    return {key: value for key, value in result.items() if key not in ("hx", "hz")}


def run(args):
    if not cpp_fast.available():
        raise RuntimeError(cpp_fast.accelerator_info())
    started = time.perf_counter()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_d = int(args.target_d or target_distance(args.leader_score))
    ideal_trials = int(getattr(args, "ideal_trials", COMMON_GCD_IDEAL_TRIALS))
    branches = enumerate_branches(Path(args.parent_report))
    if args.factor_subset:
        wanted = tuple(sorted(int(value) for value in
                              str(args.factor_subset).split(",") if value.strip()))
        branches = [row for row in branches
                    if tuple(sorted(row["factor_indices"])) == wanted]
        if not branches:
            raise ValueError(f"factor subset {wanted} is not a degree-{Q} branch")
    print(json.dumps({"stage": "branch_scout", "branches": len(branches),
                      "target": {"n": N, "k": K, "d": target_d,
                                 "score": K * target_d * target_d / N}}), flush=True)

    scout_tasks = [(branch, args.plain_scout_runs, args.exact_scout_runs,
                    args.quotient_scout_runs,
                    args.scout_trials, args.seed + index * 10007)
                   for index, branch in enumerate(branches)]
    scouts = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(_scout_task, task) for task in scout_tasks]
        for done, future in enumerate(as_completed(futures), 1):
            scouts.append(future.result())
            if done == 1 or done % 24 == 0 or done == len(futures):
                print(json.dumps({"stage": "branch_scout", "done": done,
                                  "total": len(futures)}), flush=True)

    # Sparse complementary descendants can define common q94 even when no
    # sparse exact-parent word appears in scout.  Admit every populated branch;
    # shape quotas and pair feasibility perform later pruning.
    viable = [row for row in scouts if row["scout_words"]]
    viable.sort(key=lambda row: (
        -int(row["complementary_pair_count"]),
        -int(row["complementary_pair_best_shell"] or 0),
        int(row["exact_best"] if row["exact_best"] is not None else M + 1),
        -int(bool(row["inherited_q55_ancestors"])),
        -int(row["divisor_multiplier_symmetry"]), row["branch_id"]))

    # Degree-94 has three distinct factor-shape phylogenies.  Short scouts can
    # make one shape look uniformly best although all its members share one
    # hidden logical mechanism.  Allocate beam evenly by shape; inside each
    # shape interleave inherited and out-of-sample branches.
    by_shape = {}
    for row in viable:
        shape = tuple(sorted(int(value) for value in row["factor_degrees"]))
        by_shape.setdefault(shape, []).append(row)
    selected_branches = []
    shapes = sorted(by_shape)
    shape_quota = max(1, math.ceil(int(args.mine_branches) / max(1, len(shapes))))
    for shape in shapes:
        group = by_shape[shape]
        inherited = [row for row in group if row["inherited_q55_ancestors"]]
        novel = [row for row in group if not row["inherited_q55_ancestors"]]
        interleaved = []
        for index in range(max(len(inherited), len(novel))):
            if index < len(inherited):
                interleaved.append(inherited[index])
            if index < len(novel):
                interleaved.append(novel[index])
        selected_branches.extend(interleaved[:shape_quota])
    selected_branches = list({row["branch_id"]: row for row in selected_branches}.values())
    if len(selected_branches) < int(args.mine_branches):
        used = {row["branch_id"] for row in selected_branches}
        selected_branches.extend(row for row in viable if row["branch_id"] not in used)
        selected_branches = selected_branches[:int(args.mine_branches)]
    print(json.dumps({"stage": "branch_select", "viable": len(viable),
                      "selected": len(selected_branches),
                      "inherited": sum(bool(x["inherited_q55_ancestors"])
                                       for x in selected_branches),
                      "shapes": _categorical_histogram(
                          str(tuple(sorted(x["factor_degrees"])))
                          for x in selected_branches)}), flush=True)

    mine_tasks = [(
        branch, args.detector_runs, args.detector_trials,
        args.lattice_detector_runs,
        args.child_runs, args.child_trials, args.max_companion_gcd,
        args.ideal_iterations, args.shell_iterations, args.max_words,
        args.hill_restarts, args.hill_steps, args.hill_bases,
        args.mutation_generations,
        args.max_exact_weight, args.seed + 1_000_003 + index * 10007)
        for index, branch in enumerate(selected_branches)]
    mined = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(_mine_task, task) for task in mine_tasks]
        for done, future in enumerate(as_completed(futures), 1):
            item = future.result()
            mined.append(item)
            print(json.dumps({"stage": "word_mine", "done": done,
                              "total": len(futures),
                              "branch": item["branch_id"],
                              "exact": item["exact_word_count"],
                              "nested": item["nested_word_count"],
                              "companions": item["companion_word_count"]}), flush=True)

    candidates = []
    for index, branch in enumerate(mined):
        candidates.extend(_pair_population(
            branch, count=args.pairs_per_branch, attempts=args.pair_attempts,
            min_check_weight=args.min_check_weight,
            seed=args.seed + 2_000_003 + index * 10007))
    raw_candidate_count = len(candidates)
    affine_unique = {}
    for row in candidates:
        key = _affine_pair_key(M, tuple(row["A"][0]), tuple(row["B"][0]))
        row["translation_semantic_hash"] = row["semantic_hash"]
        row["semantic_hash"] = _semantic_hash(*key)
        row["affine_unit_canonical"] = True
        previous = affine_unique.get(row["semantic_hash"])
        if previous is None or _candidate_prior(row) > _candidate_prior(previous):
            affine_unique[row["semantic_hash"]] = row
    candidates = list(affine_unique.values())
    print(json.dumps({"stage": "pair", "raw_candidates": raw_candidate_count,
                      "candidates": len(candidates),
                      "affine_duplicates_removed": raw_candidate_count - len(candidates),
                      "girth6": sum(x["cycle_profile"]["girth_at_least_6"]
                                    for x in candidates)}), flush=True)

    proxy = _replicated_screen(
        candidates, replicas=args.proxy_replicas, trials=args.proxy_trials,
        seed=args.seed + 3_000_003, workers=args.workers,
        threads=args.threads, target_d=target_d,
        ideal_trials=ideal_trials, label="proxy")
    proxy.sort(key=lambda row: _screen_quality(row, target_d), reverse=True)
    proxy_survivors = [row for row in proxy if _survives(row, target_d)]
    capped = _branch_cap(proxy_survivors, args.deep_per_branch, target_d)
    deep_rows, deep_phylogeny = select_phylogenetic_elites(
        capped, min(int(args.deep_count), len(capped)), pool_factor=8,
        exploitation_fraction=0.5)
    deep = _replicated_screen(
        deep_rows, replicas=args.deep_replicas, trials=args.deep_trials,
        seed=args.seed + 4_000_003, workers=args.workers,
        threads=args.threads, target_d=target_d,
        ideal_trials=ideal_trials, label="deep")
    deep.sort(key=lambda row: _screen_quality(row, target_d), reverse=True)
    deep_survivors = [row for row in deep if _survives(row, target_d)]
    confirm_rows, confirm_phylogeny = select_phylogenetic_elites(
        deep_survivors, min(int(args.confirm_count), len(deep_survivors)),
        pool_factor=8, exploitation_fraction=0.5)
    confirmed = _replicated_screen(
        confirm_rows, replicas=args.confirm_replicas, trials=args.confirm_trials,
        seed=args.seed + 5_000_003,
        workers=min(args.workers, max(1, args.confirm_count * args.confirm_replicas)),
        threads=args.threads, target_d=target_d,
        ideal_trials=ideal_trials, label="confirm")
    confirmed.sort(key=lambda row: _screen_quality(row, target_d), reverse=True)

    finals = []
    for rank, row in enumerate(confirmed, 1):
        row = dict(row)
        row["pair_multiplier_symmetry"] = _pair_multiplier_symmetry(
            tuple(row["A"][0]), tuple(row["B"][0]))
        doc = _candidate_doc(row, "confirm")
        if doc is None:
            continue
        path = out_dir / f"candidate_{rank}_{N}_{K}_{doc['distance']['d']}.json"
        path.write_text(json.dumps(doc, indent=2) + "\n")
        structural = _compact_structural(structural_validate(doc))
        finals.append({
            "rank": rank, "candidate_path": str(path.resolve()),
            "semantic_hash": row["semantic_hash"],
            "d_upper": int(doc["distance"]["d"]),
            "dx_upper": int(doc["distance"]["X"]["value"]),
            "dz_upper": int(doc["distance"]["Z"]["value"]),
            "score_upper": K * int(doc["distance"]["d"]) ** 2 / N,
            "survives_target": _survives(row, target_d),
            "trials_per_side_per_replica": int(args.confirm_trials),
            "replicas": int(args.confirm_replicas),
            "configured_total_trials_per_side": (int(args.confirm_trials) *
                                                  int(args.confirm_replicas)),
            "branch_id": row["branch_id"],
            "factor_indices": row["factor_indices"],
            "word_weights": row["word_weights"],
            "individual_gcd_degrees": row["individual_gcd_degrees"],
            "cycle_profile": row["cycle_profile"],
            "pair_multiplier_symmetry": row["pair_multiplier_symmetry"],
            "structural": structural,
        })

    report = {
        "schema_version": "1.0",
        "kind": "gb_m345_k188_whole_factor_phylogeny_campaign",
        "target": {"m": M, "n": N, "k": K, "gcd_degree": Q,
                   "target_d": target_d, "target_score": K * target_d * target_d / N,
                   "leader_score": float(args.leader_score),
                   "check_weight_range": [int(args.min_check_weight), CHECK_CAP]},
        "algebra": {
            "degree94_factor_subsets": len(branches),
            "group_order_345_is_cyclic_only": True,
            "reason": "gcd(345,phi(345))=1; all groups of order 345 are cyclic",
            "literal_girth_tradeoff": (
                "balanced total check weight >=28 forces a 4-cycle because "
                "2*s*(s-1)>344; campaign records exact cycle energy"),
        },
        "phylogenetic_hypothesis": {
            "exploitation": "extend strongest observed q55 divisors by exact degree-39 grafts",
            "exploration": "scout all 99 q94 factor subsets",
            "nested_subideal_escape": (
                "x+1 child detector forces odd quotient words; exact gcd filter removes "
                "other nested descendants"),
            "mutation": "exact-q94 parent XOR shifted nested descendant",
            "selection": "branch-diverse hard RIS survival plus cycle/symmetry Pareto priors",
            "common_gcd_ideal_gate": (
                "for each descendant compute g=gcd(a,b,x^ell-1), h=(x^ell-1)/g; "
                "probe the length-ell ideal <h> and exact-check every hit as "
                "(w,0) before spending CSS trials"),
        },
        "search": {
            "branches": len(branches), "viable_branches": len(viable),
            "mined_branches": len(mined), "candidates": len(candidates),
            "proxy_survivors": len(proxy_survivors),
            "deep_survivors": len(deep_survivors),
            "proxy_trials_per_side": int(args.proxy_trials),
            "proxy_replicas": int(args.proxy_replicas),
            "deep_trials_per_side": int(args.deep_trials),
            "deep_replicas": int(args.deep_replicas),
            "confirm_trials_per_side": int(args.confirm_trials),
            "confirm_replicas": int(args.confirm_replicas),
            "common_gcd_ideal_trials_per_screen": ideal_trials,
            "common_gcd_ideal_gate": (
                "derive g=gcd(a,b,x^ell-1), h=(x^ell-1)/g; probe <h> "
                "and exact-check (w,0) as an X-logical before CSS RIS"),
            "low_shell_hillclimb": {
                "restarts": int(args.hill_restarts),
                "steps": int(args.hill_steps),
                "bases_per_branch": int(args.hill_bases),
                "weight_range": [14, min(18, int(args.max_exact_weight))],
            },
            "deep_phylogeny": deep_phylogeny,
            "confirm_phylogeny": confirm_phylogeny,
            "seed": int(args.seed),
        },
        "scout": scouts,
        "mined": mined,
        "proxy_top": proxy[:64], "deep_top": deep[:64],
        "confirmed": confirmed, "final": finals,
        "accelerator": cpp_fast.accelerator_info(),
        "seconds_wall": time.perf_counter() - started,
        "regulation": {
            "stage_only": True, "submission_sent": False,
            "git_commit_performed": False,
            "distance_is_randomized_upper_bound": True,
            "official_gate_run": False, "board_claim_allowed": False,
        },
    }
    (out_dir / "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/gb_m345_k188_phylo_01"))
    parser.add_argument("--parent-report", type=Path, default=DEFAULT_PARENT_REPORT)
    parser.add_argument("--factor-subset",
                        help="optional comma-separated irreducible factor indices")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--leader-score", type=float, default=LEADER_SCORE)
    parser.add_argument("--target-d", type=int,
                        help="hard early-stop distance; default strictly beats leader score")
    parser.add_argument("--ideal-trials", type=int, default=COMMON_GCD_IDEAL_TRIALS,
                        help=("randomized pre-CSS trials in the common-gcd ideal "
                              "<h>; valid sub-target X logicals reject immediately"))
    parser.add_argument("--plain-scout-runs", type=int, default=2)
    parser.add_argument("--exact-scout-runs", type=int, default=4)
    parser.add_argument("--quotient-scout-runs", type=int, default=1,
                        help="restarts outside every immediate factor child")
    parser.add_argument("--scout-trials", type=int, default=512)
    parser.add_argument("--mine-branches", type=int, default=24)
    parser.add_argument("--detector-runs", type=int, default=16)
    parser.add_argument("--detector-trials", type=int, default=2048)
    parser.add_argument("--lattice-detector-runs", type=int, default=1,
                        help="parent-word restarts outside each immediate child")
    parser.add_argument("--child-runs", type=int, default=4,
                        help="exact RIS restarts for each immediate q94 child")
    parser.add_argument("--child-trials", type=int, default=2048)
    parser.add_argument("--max-companion-gcd", type=int, default=116,
                        help="reject saturated companion descendants above this gcd degree")
    parser.add_argument("--ideal-iterations", type=int, default=250_000)
    parser.add_argument("--shell-iterations", type=int, default=250_000,
                        help="dedicated 14--18 descendant-shell expansion")
    parser.add_argument("--max-words", type=int, default=2048)
    parser.add_argument("--hill-restarts", type=int, default=24)
    parser.add_argument("--hill-steps", type=int, default=1024)
    parser.add_argument("--hill-bases", type=int, default=8)
    parser.add_argument("--mutation-generations", type=int, default=3)
    parser.add_argument("--max-exact-weight", type=int, default=28)
    parser.add_argument("--min-check-weight", type=int, default=24)
    parser.add_argument("--pairs-per-branch", type=int, default=32)
    parser.add_argument("--pair-attempts", type=int, default=20_000)
    parser.add_argument("--proxy-trials", type=int, default=256)
    parser.add_argument("--proxy-replicas", type=int, default=2)
    parser.add_argument("--deep-per-branch", type=int, default=2)
    parser.add_argument("--deep-count", type=int, default=32)
    parser.add_argument("--deep-trials", type=int, default=20_000)
    parser.add_argument("--deep-replicas", type=int, default=4)
    parser.add_argument("--confirm-count", type=int, default=4)
    parser.add_argument("--confirm-trials", type=int, default=2_000_000)
    parser.add_argument("--confirm-replicas", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"out": str(args.out),
                      "seconds_wall": round(report["seconds_wall"], 1),
                      "final": report["final"]}, indent=2))


if __name__ == "__main__":
    main()
