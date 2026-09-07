"""Frobenius/univariate bicycle breakout campaign.

For an odd cyclic length ``m``, construct the GB pair ``(A, A^(2**ell))``
in GF(2)[x]/(x**m+1).  This is the univariate-bicycle restriction: the
second support is an exponent-doubling automorphism of the first, not an
independently sampled ideal word.  It is a representation jump from the
phase-pair campaign while retaining the same CSS/check-weight gates.

All screening values are witness-backed randomized upper bounds.  This file
never submits, commits, or treats a clean screening budget as a proof.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import random
import time

import cpp_fast
from generalized_bicycle import common_gcd_degree
from gf2_factor import degree, factor_xm_plus_one, gcd
from projection_breakout_campaign import (
    CURRENT_SCORE, PROJECTION, PROJECTED_TARGET, RAW_TARGET, _coalesce_tensor_branches,
    _ideal_mine, _sample_divisor, _screen_task, _parallel_screen,
    _passes_score_admission, _quality, _shift, _tensor_seed_branches,
    score_admission_distance, target_specs,
)


DEFAULT_M_VALUES = (301, 315, 327, 333, 337, 339, 341, 345)
CHECK_CAP = 32


def frobenius_support(word, power: int, m: int) -> tuple[int, ...]:
    """Apply a reduced Frobenius exponent multiplier to a support."""
    power, m = int(power), int(m)
    if power <= 0 or math.gcd(power, m) != 1:
        raise ValueError("Frobenius multiplier must be a unit modulo m")
    return tuple(sorted((power * int(value)) % m for value in word))


def _frobenius_steps(m: int, limit: int) -> list[tuple[str, int]]:
    """Distinct nonidentity powers x -> x^(2**ell) modulo m."""
    seen = {1}
    out = []
    value = 1
    for ell in range(1, max(1, int(limit)) + 1):
        value = (value * 2) % int(m)
        if value in seen:
            break
        seen.add(value)
        out.append((f"frobenius_{ell}", value))
    return out


def stabilizer_multipliers(word: tuple[int, ...], m: int,
                           limit: int | None = None) -> list[tuple[str, int]]:
    """Return nonidentity exponent multipliers preserving ``word``'s ideal.

    Frobenius powers are a small subgroup of the exponent automorphisms of a
    cyclic code.  For a mined word ``A``, retain every unit ``u`` for which
    ``gcd(A(x), A(x**u), x**m+1)`` has the same degree as ``gcd(A, x**m+1)``.
    This is an exact algebraic gate, not a distance proxy, and expands the
    search beyond the Frobenius orbit without changing n, k, or check weight.
    """
    m = int(m)
    base_q = common_gcd_degree(m, word, word)
    values = []
    for power in range(2, m):
        if math.gcd(power, m) != 1:
            continue
        transformed = frobenius_support(word, power, m)
        if common_gcd_degree(m, word, transformed) == base_q:
            values.append((f"stabilizer_{power}", power))
            if limit is not None and len(values) >= int(limit):
                break
    return values


def _automorphism_steps(word: tuple[int, ...], m: int, limit: int,
                        mode: str) -> list[tuple[str, int]]:
    if mode == "frobenius":
        return _frobenius_steps(m, limit)
    if mode == "stabilizer":
        # Do not truncate before random sampling: it would bias repeatedly
        # toward the smallest multipliers and erase much of the new branch.
        return stabilizer_multipliers(word, m)
    raise ValueError(f"unsupported automorphism mode: {mode}")


def _make_candidates(branch: dict, *, count: int, steps: int, min_word: int,
                     automorphism_mode: str, seed: int) -> list[dict]:
    """Sample relative phases induced by exact ideal-preserving maps."""
    m, q = int(branch["m"]), int(branch["gcd_degree"])
    words = [tuple(int(value) for value in word) for word in branch.get("words", [])
             if len(word) >= int(min_word) and 2 * len(word) <= CHECK_CAP]
    if not words:
        return []
    map_rows = [(word, map_id, power)
                for word in words
                for map_id, power in _automorphism_steps(
                    word, m, steps, automorphism_mode)]
    if not map_rows:
        return []

    def add_candidate(out: dict, word: tuple[int, ...], map_id: str,
                      power: int, phase: int):
        # Shifting A by s shifts B by power*s.  The resulting relative phase
        # is usually inequivalent to a common GB translation, so retain it.
        a = _shift(word, phase, m)
        b = frobenius_support(a, power, m)
        if len(a) + len(b) > CHECK_CAP or common_gcd_degree(m, a, b) != q:
            return
        key = (a, b)
        out.setdefault(key, {
            "A": [list(a)], "B": [list(b)], "m": m, "n": 2 * m,
            "k": 2 * q, "gcd_degree": q,
            "family": "univariate_frobenius_bicycle",
            "branch_id": branch["branch_id"],
            "factor_indices": branch["factor_indices"],
            "word_weights": [len(a), len(b)],
            "automorphism": map_id, "power_mod_m": power,
            "semantic_hash": hashlib.sha256(
                json.dumps([list(a), list(b)], separators=(",", ":")).encode()).hexdigest(),
        })

    # If the algebraic orbit fits the requested budget, cover it fully instead
    # of drawing with replacement.  This makes compact stabilizer branches
    # deterministic and avoids spending most candidate generation time trying
    # to rediscover the last few phase pairs.
    full_population = len(map_rows) * m
    out = {}
    if full_population <= int(count):
        for word, map_id, power in map_rows:
            for phase in range(m):
                add_candidate(out, word, map_id, power, phase)
        return list(out.values())

    rng = random.Random(int(seed))
    attempts = 0
    limit = max(100, int(count) * 100)
    while len(out) < int(count) and attempts < limit:
        attempts += 1
        word, map_id, power = rng.choice(map_rows)
        add_candidate(out, word, map_id, power, rng.randrange(m))
    return list(out.values())


def _mine_task(task):
    branch, params = task
    return _ideal_mine(branch, **params)


def _seed_branch(path: Path) -> dict:
    """Import only the algebraic parent support from a local CSS GB receipt."""
    doc = json.loads(Path(path).read_text())
    n = int(doc["n"])
    if n % 2 or doc.get("code_type") != "CSS":
        raise ValueError(f"{path} is not an even-length CSS parent")
    m = n // 2
    row = [int(value) for value in doc["checks"]["X"][0]]
    a = tuple(value for value in row if value < m)
    b = tuple(value - m for value in row if value >= m)
    q = common_gcd_degree(m, a, b)
    if not a or not b or q <= 0:
        raise ValueError(f"{path} has no usable cyclic GB parent support")
    polynomial = sum(1 << value for value in a)
    divisor = gcd(polynomial, (1 << m) | 1)
    if degree(divisor) != q:
        raise ValueError(f"{path} parent A does not have the shared exact divisor")
    return {
        "m": m, "n": n, "k": 2 * q, "gcd_degree": q,
        "divisor": int(divisor), "divisor_weight": int(divisor).bit_count(),
        "seed_support": list(a), "family": "imported_cyclic_parent",
        "factor_indices": [f"imported_parent:{Path(path).name}"],
        "branch_id": f"ub_parent_m{m}_k{2*q}_{Path(path).stem}",
    }


def run(args):
    if not cpp_fast.available():
        raise RuntimeError(cpp_fast.accelerator_info())
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m_values = (DEFAULT_M_VALUES if args.m_values is None else
                tuple(int(value) for value in str(args.m_values).split(",")
                      if str(value).strip()))
    factors = {m: factor_xm_plus_one(m) for m in m_values}
    specs = target_specs(m_values, factors, max_specs=None)
    if args.target_k is not None:
        wanted = {int(value) for value in str(args.target_k).split(",")
                  if str(value).strip()}
        specs = [spec for spec in specs if int(spec["k"]) in wanted]
    rng = random.Random(int(args.seed))
    branches, seen = [], set()
    for spec in specs:
        for index in range(max(0, int(args.random_branches))):
            sampled = _sample_divisor(factors[spec["m"]], spec["gcd_degree"], rng)
            if sampled is None:
                continue
            key = (spec["m"], spec["k"], tuple(sampled["factor_indices"]))
            if key in seen:
                continue
            seen.add(key)
            branches.append({**spec, **sampled, "family": "random_exact_divisor",
                             "branch_id": f"ub_m{spec['m']}_k{spec['k']}_r{index}",
                             "min_word": int(args.min_word),
                             "branch_seed": rng.randrange(1 << 63)})
    structured = []
    for spec in specs:
        for seed_branch in _tensor_seed_branches(spec["m"], {spec["gcd_degree"]}):
            structured.append({**spec, **seed_branch,
                               "min_word": int(args.min_word),
                               "branch_seed": rng.randrange(1 << 63)})
    for branch in _coalesce_tensor_branches(structured):
        branch["min_word"] = int(args.min_word)
        branch["branch_seed"] = rng.randrange(1 << 63)
    branches = _coalesce_tensor_branches(structured) + branches
    for seed_path in args.seed_code or []:
        branch = _seed_branch(Path(seed_path))
        if any(int(spec["m"]) == int(branch["m"]) and
               int(spec["k"]) == int(branch["k"]) for spec in specs):
            branch["min_word"] = int(args.min_word)
            branch["branch_seed"] = rng.randrange(1 << 63)
            branches.append(branch)
    unique = {}
    for branch in branches:
        key = (int(branch["m"]), int(branch["k"]), int(branch["divisor"]),
               tuple(tuple(item) for item in branch.get("seed_supports", [])),
               tuple(branch.get("factor_indices", [])))
        unique.setdefault(key, branch)
    branches = list(unique.values())
    score_target = float(args.score_target)
    print(json.dumps({"stage": "specs", "raw_target": RAW_TARGET,
                      "score_target": score_target,
                      "spec_count": len(specs), "branch_count": len(branches)}), flush=True)
    params = {"iterations": int(args.mine_iterations),
              "hill_restarts": int(args.hill_restarts),
              "hill_steps": int(args.hill_steps), "max_words": int(args.max_words),
              "min_word": int(args.min_word), "seed": 0}
    mined = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(_mine_task, (branch, {**params, "seed": int(branch["branch_seed"])}))
                   for branch in branches]
        for done, future in enumerate(as_completed(futures), 1):
            item = future.result()
            if len(item.get("words", [])) >= 1:
                mined.append(item)
            if done == 1 or done % 16 == 0 or done == len(futures):
                print(json.dumps({"stage": "mine", "done": done, "total": len(futures),
                                  "usable": len(mined)}), flush=True)
    candidates = []
    for index, branch in enumerate(mined):
        candidates.extend(_make_candidates(
            branch, count=int(args.per_branch), steps=int(args.frobenius_steps),
            min_word=int(args.min_word), automorphism_mode=args.automorphism_mode,
            seed=int(args.seed) + index * 1_000_003))
    print(json.dumps({"stage": "pair", "mined_branches": len(mined),
                      "candidates": len(candidates)}), flush=True)
    screened = _parallel_screen(candidates, trials=int(args.proxy_trials),
                                seed=int(args.seed) + 1009, workers=int(args.workers),
                                threads=int(args.threads), label="proxy",
                                target=lambda row: score_admission_distance(
                                    row, score_target))
    screened.sort(key=_quality, reverse=True)
    proxy_survivors = [row for row in screened
                       if _passes_score_admission(row, score_target)]
    deep = _parallel_screen(proxy_survivors[:max(0, int(args.deep_count))],
                            trials=int(args.deep_trials), seed=int(args.seed) + 2009,
                            workers=int(args.workers), threads=int(args.threads), label="deep",
                            target=lambda row: score_admission_distance(
                                row, score_target))
    deep.sort(key=_quality, reverse=True)
    deep_survivors = [row for row in deep
                      if _passes_score_admission(row, score_target)]
    final_runs = _parallel_screen(deep_survivors[:max(0, int(args.final_count))],
                                  trials=int(args.final_trials), seed=int(args.seed) + 3009,
                                  workers=int(args.workers), threads=int(args.threads), label="final",
                                  target=lambda row: score_admission_distance(
                                      row, score_target))
    report = {
        "schema_version": "1.0", "kind": "univariate_frobenius_bicycle_campaign",
        "target": {"projected_distance": PROJECTED_TARGET,
                   "raw_distance_threshold": RAW_TARGET, "projection_factor": PROJECTION,
                   "score_admission_target": score_target,
                   "check_weight_cap": CHECK_CAP},
        "search": {"m_values": list(m_values), "specs": specs, "branches": len(branches),
                   "mined_branches": len(mined), "candidate_count": len(candidates),
                   "frobenius_steps": int(args.frobenius_steps),
                   "automorphism_mode": args.automorphism_mode,
                   "proxy_trials_per_side": int(args.proxy_trials),
                   "score_admission_distances": sorted({
                       score_admission_distance(row, score_target) for row in candidates}),
                   "proxy_survivors": len(proxy_survivors),
                   "deep_trials_per_side": int(args.deep_trials),
                   "deep_survivors": len(deep_survivors),
                   "final_trials_per_side": int(args.final_trials), "seed": int(args.seed)},
        "mined": mined, "proxy_top": screened[:64], "deep_top": deep[:64],
        "final_runs": final_runs[:64], "seconds_wall": time.perf_counter() - started,
        "regulation": {"stage_only": True, "submission_sent": False,
                       "git_commit_performed": False,
                       "distance_is_randomized_upper_bound": True,
                       "official_gate_run": False, "board_claim_allowed": False},
    }
    (out / "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/univariate_frobenius_01"))
    parser.add_argument("--m-values", default=None)
    parser.add_argument("--target-k", default=None)
    parser.add_argument("--random-branches", type=int, default=2)
    parser.add_argument("--seed-code", type=Path, action="append", default=[],
                        help="local CSS GB parent; imports its A support as an algebraic seed")
    parser.add_argument("--mine-iterations", type=int, default=30_000)
    parser.add_argument("--hill-restarts", type=int, default=24)
    parser.add_argument("--hill-steps", type=int, default=768)
    parser.add_argument("--max-words", type=int, default=192)
    parser.add_argument("--min-word", type=int, default=15)
    parser.add_argument("--frobenius-steps", type=int, default=12)
    parser.add_argument("--automorphism-mode", choices=("frobenius", "stabilizer"),
                        default="frobenius",
                        help="exact divisor-preserving exponent maps for B")
    parser.add_argument("--score-target", type=float, default=CURRENT_SCORE,
                        help="projected score required for deep-stage admission")
    parser.add_argument("--per-branch", type=int, default=1024)
    parser.add_argument("--proxy-trials", type=int, default=1024)
    parser.add_argument("--deep-count", type=int, default=32)
    parser.add_argument("--deep-trials", type=int, default=100_000)
    parser.add_argument("--final-count", type=int, default=8)
    parser.add_argument("--final-trials", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"out": str(args.out), "search": report["search"],
                      "seconds_wall": round(report["seconds_wall"], 1)}, indent=2))


if __name__ == "__main__":
    main()
