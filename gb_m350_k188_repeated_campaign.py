from __future__ import annotations

"""Repeated-root Z_350, [[700,188,*]] generalized-bicycle campaign.

This is the representation jump after the square-free Z_345 q=94 lattice
collapsed under candidate-specific annihilator searches.  Since

    x^350 + 1 = (x^175 + 1)^2  over GF(2),

each irreducible of the odd core may occur with exponent 0, 1, or 2.  There
are 66 distinct degree-94 divisors.  The campaign explores all of them while
retaining the fast dynamic annihilator gate from the Z_345 campaign.

All distances are randomized, witness-backed upper bounds.  This script does
not submit or commit anything.
"""

import argparse
from functools import lru_cache
import itertools
import json
import math
from pathlib import Path
import random

import numpy as np

import gb_m345_k188_phylo_campaign as core
from distance_sketch import gf2_nullspace
from generalized_bicycle import (build_matrices, build_supports,
                                 common_gcd_degree, normalize_code)
from gf2_factor import degree, factor_xm_plus_one, gcd, mod, mul, quotient
from hyper_validator import structural_validate
from inverse_design_core import gf2_rank


M = 350
N = 700
Q = 94
K = 188
CHECK_CAP = 32
MODULUS = (1 << M) | 1
BASE_FACTORS = tuple(factor_xm_plus_one(175))
ROOT_MULTIPLICITY = 2
PAIR_EXACT_ONLY = False


def _configure_core() -> None:
    """Retarget generic fast machinery before workers are spawned."""
    core.M = M
    core.N = N
    core.Q = Q
    core.K = K
    core.CHECK_CAP = CHECK_CAP
    core.MODULUS = MODULUS
    # Learned Z_345 fixed ideals have no meaning in the repeated-root ring.
    # Candidate-specific annihilator gates remain enabled.
    core.KILLER_IDEAL_FACTOR_SETS = ()
    core.ONE_BLOCK_KILLER_IDEAL_FACTOR_SETS = ()
    core._two_block_ideal_span.cache_clear()
    core._one_block_ideal_span.cache_clear()
    core._one_block_divisor_span.cache_clear()


_configure_core()


def degree94_exponents() -> list[tuple[int, ...]]:
    """All distinct multiplicity vectors with total divisor degree Q."""
    degrees = tuple(degree(factor) for factor in BASE_FACTORS)
    return [tuple(exponents) for exponents in itertools.product(
            range(ROOT_MULTIPLICITY + 1), repeat=len(degrees))
            if sum(e * d for e, d in zip(exponents, degrees)) == Q]


def _divisor(exponents) -> int:
    out = 1
    for exponent, factor in zip(exponents, BASE_FACTORS):
        for _ in range(int(exponent)):
            out = mul(out, factor)
    return out


def _children(branch: dict) -> list[tuple[int, int, int]]:
    """(base-factor index, child divisor, child degree), one multiplicity step."""
    divisor = int(branch["divisor"])
    return [(index, mul(divisor, BASE_FACTORS[index]),
             Q + degree(BASE_FACTORS[index]))
            for index, exponent in enumerate(branch["factor_exponents"])
            if int(exponent) < ROOT_MULTIPLICITY]


@lru_cache(maxsize=131072)
def _gcd_exponents(poly: int) -> tuple[int, ...]:
    """Multiplicity signature of gcd(poly, x^350+1) over odd-core factors."""
    residue = gcd(int(poly), MODULUS)
    out = []
    for factor in BASE_FACTORS:
        exponent = 0
        while exponent < ROOT_MULTIPLICITY and mod(residue, factor) == 0:
            residue = quotient(residue, factor)
            exponent += 1
        out.append(exponent)
    if residue != 1:
        raise ArithmeticError("incomplete repeated-root gcd signature")
    return tuple(out)


def _word_signature(word) -> tuple[int, ...]:
    return _gcd_exponents(core._poly(word))


def enumerate_branches(_parent_report=None) -> list[dict]:
    branches = []
    for index, exponents in enumerate(degree94_exponents()):
        divisor = _divisor(exponents)
        expanded_degrees = [degree(factor) for exponent, factor in
                            zip(exponents, BASE_FACTORS) for _ in range(exponent)]
        branches.append({
            "branch_index": index,
            "branch_id": f"m{M}_q{Q}_e" + "".join(map(str, exponents)),
            # Kept for compatibility with the common report layer.
            "factor_indices": list(exponents),
            "factor_exponents": list(exponents),
            "factor_degrees": expanded_degrees,
            "base_factor_degrees": [degree(factor) for factor in BASE_FACTORS],
            "divisor": int(divisor),
            "divisor_weight": int(divisor).bit_count(),
            "inherited_q55_ancestors": [],
            "phylogenetic_arm": "repeated_root_novel_branch",
        })
    return branches


def _scout_task(task):
    branch, plain_runs, exact_runs, quotient_runs, trials, seed = task
    divisor = int(branch["divisor"])
    generator = core._ideal_generator(divisor)
    children = _children(branch)
    words: dict[tuple[int, ...], str] = {}
    counts = {"plain": 0, "immediate_child_detector": 0}

    for run in range(int(plain_runs)):
        word = core._classical(generator, trials=int(trials),
                               seed=int(seed) + run * 104729)
        if word is not None:
            words.setdefault(core._orbit(word), "plain")
            counts["plain"] += 1

    # A vector outside each child is exact with respect to that factor.
    # Cycle detectors across all children; exact gcd filtering is definitive.
    runs_per_child = max(1, int(quotient_runs))
    extra = max(0, int(exact_runs) - runs_per_child * len(children))
    for child_pos, (factor_index, child_divisor, _) in enumerate(children):
        detector = gf2_nullspace(core._ideal_generator(child_divisor))
        runs = runs_per_child + (extra if child_pos == 0 else 0)
        for run in range(runs):
            word = core._classical(
                generator, detectors=detector, trials=int(trials),
                seed=(int(seed) ^ ((factor_index + 1) * 0xD1B54A35)) + run * 104729)
            if word is not None:
                words.setdefault(core._orbit(word), f"factor_detector_{factor_index}")
                counts["immediate_child_detector"] += 1

    classified = [{"support": list(word), "weight": len(word),
                   "gcd_degree": core._word_gcd_degree(word), "source": source}
                  for word, source in words.items()]
    exact = [row for row in classified if row["gcd_degree"] == Q]
    nested = [row for row in classified if row["gcd_degree"] > Q]
    complementary = []
    for left, right in itertools.combinations(classified, 2):
        if left["weight"] + right["weight"] > CHECK_CAP:
            continue
        if common_gcd_degree(M, left["support"], right["support"]) == Q:
            complementary.append((left["weight"] + right["weight"],
                                  left["gcd_degree"], right["gcd_degree"]))
    return {
        **branch,
        "divisor_multiplier_symmetry": core._divisor_multiplier_symmetry(divisor),
        "scout_words": sorted(classified, key=lambda row: (row["weight"],
                                                            row["gcd_degree"])),
        "source_counts": counts,
        "exact_best": min((row["weight"] for row in exact), default=None),
        "nested_best": min((row["weight"] for row in nested), default=None),
        "exact_count": len(exact), "nested_count": len(nested),
        "complementary_pair_count": len(complementary),
        "complementary_pair_best_shell": max((row[0] for row in complementary),
                                               default=None),
        "complementary_pair_depths": sorted({tuple(sorted(row[1:]))
                                              for row in complementary}),
    }


def _mutate_exact_words(exact_words, nested_words, *, generations: int,
                        max_words: int, max_weight: int, seed: int):
    """Repeated-root-safe mutation; no odd-weight assumption."""
    rng = random.Random(int(seed))
    exact = {core._orbit(word) for word in exact_words
             if core._word_gcd_degree(word) == Q}
    nested = sorted({core._orbit(word) for word in nested_words},
                    key=lambda word: (len(word), word))
    frontier = sorted(exact, key=lambda word: (-len(word), word))[:256]
    hard_cap = max(int(max_words) * 4, len(exact) + 4096)
    for _ in range(max(0, int(generations))):
        if not frontier or not nested:
            break
        parents = frontier[:256]
        frontier = []
        attempts = max(4096, min(int(max_words) * 64,
                                 len(parents) * len(nested) * M))
        for _ in range(attempts):
            parent = rng.choice(parents)
            child = rng.choice(nested[:min(128, len(nested))])
            shifted = core._shift(child, rng.randrange(M))
            word = tuple(sorted(set(parent).symmetric_difference(shifted)))
            canonical = core._orbit(word)
            if (not canonical or len(canonical) > int(max_weight) or
                    canonical in exact or core._word_gcd_degree(canonical) != Q):
                continue
            exact.add(canonical)
            frontier.append(canonical)
            if len(exact) >= hard_cap:
                break
        if len(exact) >= hard_cap:
            break
    # Preserve candidate check shells first; then diverse heavier escape words.
    shell = sorted((word for word in exact if 14 <= len(word) <= 18),
                   key=lambda word: (len(word), word))
    selected = set(shell[:max(1, 3 * int(max_words) // 4)])
    for word in sorted(exact, key=lambda word: (-len(word), word)):
        selected.add(word)
        if len(selected) >= int(max_words):
            break
    return sorted(selected, key=lambda word: (len(word), word))


def _mine_task(task):
    (branch, detector_runs, detector_trials, lattice_detector_runs,
     child_runs, child_trials, max_companion_gcd, ideal_iterations,
     shell_iterations, max_words, mutation_generations, max_exact_weight, seed) = task
    divisor = int(branch["divisor"])
    generator = core._ideal_generator(divisor)
    children = _children(branch)
    pool = {core._orbit(tuple(row["support"]))
            for row in branch.get("scout_words", [])}

    # Spread parent searches over all multiplicity directions.
    for run in range(int(detector_runs)):
        factor_index, child_divisor, _ = children[run % len(children)]
        detector = gf2_nullspace(core._ideal_generator(child_divisor))
        word = core._classical(
            generator, detectors=detector, trials=int(detector_trials),
            seed=(int(seed) ^ ((factor_index + 1) * 0x9E3779B9)) + run * 104729)
        if word is not None:
            pool.add(core._orbit(word))

    for factor_index, child_divisor, _ in children:
        detector = gf2_nullspace(core._ideal_generator(child_divisor))
        for run in range(int(lattice_detector_runs)):
            word = core._classical(
                generator, detectors=detector, trials=int(detector_trials),
                seed=(int(seed) ^ ((factor_index + 1) * 0xD1B54A35)) + run * 104729)
            if word is not None:
                pool.add(core._orbit(word))

    bases = sorted(pool, key=lambda word: (len(word), word))[:48]
    if bases and int(ideal_iterations) > 0:
        mined = core.cpp_fast.cyclic_ideal_mine(
            bases, m=M, iterations=int(ideal_iterations), min_weight=3,
            max_weight=int(max_exact_weight), seed=int(seed) ^ 0xA5A55A5A,
            max_words=int(max_words))
        pool.update(core._orbit(tuple(int(x) for x in word)) for word in mined)

    if bases and int(shell_iterations) > 0:
        representatives = {}
        for word in bases:
            depth = core._word_gcd_degree(word)
            if Q <= depth <= int(max_companion_gcd):
                previous = representatives.get(depth)
                if previous is None or (len(word), word) < (len(previous), previous):
                    representatives[depth] = word
        for depth, base in sorted(representatives.items()):
            mined = core.cpp_fast.cyclic_ideal_mine(
                [base], m=M, iterations=int(shell_iterations), min_weight=14,
                max_weight=18, seed=(int(seed) ^ 0x6A09E667) + depth * 0x9E37,
                max_words=int(max_words))
            pool.update(core._orbit(tuple(int(x) for x in word)) for word in mined)

    exact = [word for word in pool if core._word_gcd_degree(word) == Q and
             len(word) <= int(max_exact_weight)]
    nested = [word for word in pool if Q < core._word_gcd_degree(word) <=
              int(max_companion_gcd) and len(word) < CHECK_CAP]
    exact = _mutate_exact_words(
        exact, nested, generations=int(mutation_generations),
        max_words=int(max_words), max_weight=int(max_exact_weight),
        seed=int(seed) ^ 0xC0FFEE)

    companions = {}
    for factor_index, child_divisor, child_q in children:
        if child_q > int(max_companion_gcd):
            continue
        child_generator = core._ideal_generator(child_divisor)
        # Plain plus one grandchild detector; exact gcd filter guards output.
        grand = [(j, mul(child_divisor, BASE_FACTORS[j]))
                 for j, exponent in enumerate(branch["factor_exponents"])
                 if int(exponent) + (j == factor_index) < ROOT_MULTIPLICITY]
        detectors = [None]
        if grand:
            detectors.append(gf2_nullspace(core._ideal_generator(grand[0][1])))
        for run in range(int(child_runs)):
            word = core._classical(
                child_generator, detectors=detectors[run % len(detectors)],
                trials=int(child_trials),
                seed=(int(seed) ^ ((factor_index + 1) * 0x94D049BB)) + run * 104729)
            if word is None:
                continue
            canonical = core._orbit(word)
            actual_q = core._word_gcd_degree(canonical)
            if actual_q != child_q or len(canonical) >= CHECK_CAP:
                continue
            companions[(canonical, actual_q)] = {
                "support": list(canonical), "weight": len(canonical),
                "gcd_degree": actual_q, "added_factor_index": factor_index,
                "added_factor_degree": degree(BASE_FACTORS[factor_index]),
            }
    return {
        **branch,
        "exact_words": [list(word) for word in exact],
        "nested_words": [list(word) for word in sorted(set(nested),
                                                        key=lambda x: (len(x), x))],
        "exact_word_count": len(exact), "nested_word_count": len(set(nested)),
        "companion_words": sorted(companions.values(),
                                  key=lambda row: (row["weight"], row["gcd_degree"])),
        "companion_word_count": len(companions),
        "exact_weight_histogram": core._histogram(len(word) for word in exact),
        "nested_weight_histogram": core._histogram(len(word) for word in nested),
        "companion_weight_histogram": core._histogram(
            row["weight"] for row in companions.values()),
        "companion_gcd_histogram": core._histogram(
            row["gcd_degree"] for row in companions.values()),
    }


def _pair_population(branch: dict, *, count: int, attempts: int,
                     min_check_weight: int, seed: int) -> list[dict]:
    # Repeated-root mining can produce an exact-q word of weight 24 paired
    # with an 8-weight shallow descendant.  Their 24+8 shell is legal at the
    # challenge cap; the former 12--20 filter silently deleted it.
    exact = [(tuple(word), Q) for word in branch.get("exact_words", [])
             if 1 <= len(word) < CHECK_CAP]
    nested = [(tuple(word), core._word_gcd_degree(word))
              for word in branch.get("nested_words", []) if 1 <= len(word) < CHECK_CAP]
    companions = [(tuple(row["support"]), int(row["gcd_degree"]))
                  for row in branch.get("companion_words", [])
                  if 1 <= len(row["support"]) < CHECK_CAP]
    if PAIR_EXACT_ONLY:
        nested = []
        companions = []
    unique = {}
    for word, depth in exact + nested + companions:
        unique.setdefault(core._orbit(word), int(depth))
    pool = [(word, depth) for word, depth in unique.items()]
    if len(pool) < 2:
        return []
    parent_signature = tuple(int(value) for value in branch["factor_exponents"])
    groups = {}
    for word, depth in pool:
        signature = _word_signature(word)
        groups.setdefault(signature, []).append((word, depth))
    compatible = []
    signatures = sorted(groups)
    for left_index, left in enumerate(signatures):
        for right in signatures[left_index:]:
            if all(min(a, b) == parent for a, b, parent in
                   zip(left, right, parent_signature)):
                compatible.append((left, right))
    if not compatible:
        return []
    rng = random.Random(int(seed))
    rows = {}
    reservoir_cap = max(int(count) * 4, int(count) + 512)
    for _ in range(max(int(attempts), int(count) * 20)):
        if len(rows) >= reservoir_cap:
            break
        left_signature, right_signature = rng.choice(compatible)
        a, da = rng.choice(groups[left_signature])
        raw_b, db = rng.choice(groups[right_signature])
        if a == raw_b and left_signature == right_signature:
            continue
        b = core._shift(raw_b, rng.randrange(M))
        total = len(a) + len(b)
        if total < int(min_check_weight) or total > CHECK_CAP:
            continue
        if common_gcd_degree(M, a, b) != Q:
            continue
        key = core._pair_key(a, b)
        if key in rows:
            continue
        actual = (core._word_gcd_degree(key[0]), core._word_gcd_degree(key[1]))
        actual_signatures = (_word_signature(key[0]), _word_signature(key[1]))
        cycle = core._cycle_profile(*key)
        rows[key] = {
            "A": [list(key[0])], "B": [list(key[1])],
            "m": M, "n": N, "k": K, "gcd_degree": Q,
            "branch_id": branch["branch_id"], "branch_index": branch["branch_index"],
            "factor_indices": branch["factor_indices"],
            "factor_exponents": branch["factor_exponents"],
            "phylogenetic_arm": branch["phylogenetic_arm"],
            "inherited_q55_ancestors": [],
            "divisor_multiplier_symmetry": branch["divisor_multiplier_symmetry"],
            "pairing_mode": ("exact_exact_frontier_shell" if actual == (Q, Q)
                             else "complementary_descendant_shell"),
            "word_weights": [len(key[0]), len(key[1])],
            "individual_gcd_degrees": list(actual),
            "individual_factor_exponents": [list(value) for value in actual_signatures],
            "cycle_profile": cycle,
            "semantic_hash": core._semantic_hash(*key),
            "pair_multiplier_symmetry": 1,
        }
    # Avoid spending the whole branch budget on one attractive cycle shell.
    # Repeated-root annihilator quality is strongly signature-dependent.
    grouped = {}
    for row in rows.values():
        signature_pair = tuple(sorted(tuple(value) for value in
                                      row["individual_factor_exponents"]))
        grouped.setdefault(signature_pair, []).append(row)
    for group in grouped.values():
        group.sort(key=core._candidate_prior, reverse=True)
    selected = []
    while len(selected) < int(count):
        changed = False
        for signature_pair in sorted(grouped):
            group = grouped[signature_pair]
            if group:
                selected.append(group.pop(0))
                changed = True
                if len(selected) >= int(count):
                    break
        if not changed:
            break
    return selected


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
        "name": f"[[{N},{K},d<={distance}]] Z_{M} repeated-root q{Q} GB",
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
            "origin": f"gb_m{M}_k{K}_repeated_campaign", "novelty": "unknown",
            "model": "GPT-5 Codex + repeated-root phylogenetic search",
            "construction": (f"Z_{M} GB; exponent vector={row['factor_exponents']}; "
                             f"A={tuple(code.a)}; B={tuple(code.b)}"),
            "references": [],
            "notes": f"stage={stage}; randomized witness-backed upper bound",
        },
        "search": {"stage": stage, "semantic_hash": row["semantic_hash"],
                   "branch_id": row["branch_id"],
                   "factor_exponents": row["factor_exponents"],
                   "cycle_profile": row["cycle_profile"]},
        "regulation": {"stage_only": True, "submission_sent": False,
                       "git_commit_performed": False,
                       "official_gate_status": "not_run",
                       "board_claim_allowed": False,
                       "distance_is_not_proven": True,
                       "literature_novelty": "unverified"},
    }


def _install_overrides() -> None:
    core.enumerate_branches = enumerate_branches
    core._scout_task = _scout_task
    core._mine_task = _mine_task
    core._mutate_exact_words = _mutate_exact_words
    core._pair_population = _pair_population
    core._candidate_doc = _candidate_doc


_install_overrides()


def _rank_preflight() -> dict:
    """Verify repeated-root k formula independently on every q=94 branch."""
    bad = []
    ranks = set()
    for branch in enumerate_branches():
        support = core._support(branch["divisor"])
        hx, hz = build_matrices(M, support, core._shift(support, 1))
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        computed_k = N - rx - rz
        ranks.add((rx, rz, computed_k))
        if computed_k != K or np.any((hx @ hz.T) & 1):
            bad.append({"branch_id": branch["branch_id"], "rx": rx,
                        "rz": rz, "computed_k": computed_k})
    if bad:
        raise ArithmeticError(f"repeated-root rank preflight failed: {bad[:3]}")
    return {"branches_checked": len(degree94_exponents()),
            "rank_tuples": [list(row) for row in sorted(ranks)], "ok": True}


def run(args):
    preflight = _rank_preflight()
    report = core.run(args)
    odd_core = M // ROOT_MULTIPLICITY
    report["kind"] = f"gb_m{M}_k{K}_repeated_root_phylogeny_campaign"
    report["algebra"] = {
        "odd_core": odd_core,
        "root_multiplicity": ROOT_MULTIPLICITY,
        "identity": (f"x^{M}+1=(x^{odd_core}+1)^{ROOT_MULTIPLICITY} "
                     "over GF(2)"),
        "base_factor_degrees": [degree(factor) for factor in BASE_FACTORS],
        "degree94_exponent_vectors": len(degree94_exponents()),
        "rank_preflight": preflight,
        "literal_girth_tradeoff": (
            "balanced total check weight >=28 forces 4-cycles; exact cycle energy used"),
    }
    report["phylogenetic_hypothesis"] = {
        "representation_jump": (
            f"leave square-free collapse basin for repeated-root m{M}"),
        "branches": "all degree-94 multiplicity vectors over factors of x^175+1",
        "nested_subideal_escape": "immediate multiplicity-child quotient detectors",
        "mutation": "exact-parent XOR shifted shallow repeated-root descendants",
        "selection": "dynamic annihilator kill-gate, replicated RIS, branch diversity",
    }
    Path(args.out, "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("results/gb_m350_k188_repeated_01"))
    parser.add_argument("--parent-report", type=Path, default=Path("unused.json"))
    parser.add_argument("--factor-subset",
                        help="optional comma-separated 9-entry multiplicity vector")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--leader-score", type=float, default=core.LEADER_SCORE)
    parser.add_argument("--target-d", type=int)
    parser.add_argument("--plain-scout-runs", type=int, default=1)
    parser.add_argument("--exact-scout-runs", type=int, default=0)
    parser.add_argument("--quotient-scout-runs", type=int, default=1)
    parser.add_argument("--scout-trials", type=int, default=256)
    parser.add_argument("--mine-branches", type=int, default=66)
    parser.add_argument("--detector-runs", type=int, default=8)
    parser.add_argument("--detector-trials", type=int, default=1024)
    parser.add_argument("--lattice-detector-runs", type=int, default=1)
    parser.add_argument("--child-runs", type=int, default=2)
    parser.add_argument("--child-trials", type=int, default=1024)
    parser.add_argument("--max-companion-gcd", type=int, default=118)
    parser.add_argument("--ideal-iterations", type=int, default=250000)
    parser.add_argument("--shell-iterations", type=int, default=100000)
    parser.add_argument("--max-words", type=int, default=2048)
    parser.add_argument("--mutation-generations", type=int, default=2)
    parser.add_argument("--max-exact-weight", type=int, default=24)
    parser.add_argument("--min-check-weight", type=int, default=28)
    parser.add_argument("--pairs-per-branch", type=int, default=32)
    parser.add_argument("--pair-attempts", type=int, default=20000)
    parser.add_argument("--proxy-trials", type=int, default=256)
    parser.add_argument("--proxy-replicas", type=int, default=2)
    parser.add_argument("--deep-per-branch", type=int, default=2)
    parser.add_argument("--deep-count", type=int, default=32)
    parser.add_argument("--deep-trials", type=int, default=20000)
    parser.add_argument("--deep-replicas", type=int, default=4)
    parser.add_argument("--confirm-count", type=int, default=4)
    parser.add_argument("--confirm-trials", type=int, default=2000000)
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
