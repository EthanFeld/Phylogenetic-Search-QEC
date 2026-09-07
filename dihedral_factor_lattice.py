from __future__ import annotations

"""Factor-coordinate rank solver for full-cap D_175 2BGA codes.

Each genome selects four divisors of x^175+1: rotation/reflection components
of a and b.  Evolution targets rank 256 while driving the four-way scalar
factor intersection to zero, escaping inherited cyclic logicals.
"""

import argparse
from functools import lru_cache
import json
from pathlib import Path
import random
import time

import dihedral_2bga_n700_campaign as base
from gf2_factor import degree, mul


FACTOR_DEGREES = tuple(degree(factor) for factor in base.FACTORS175)
FULL_MASK = (1 << len(FACTOR_DEGREES)) - 1


@lru_cache(maxsize=None)
def _divisor(mask: int) -> int:
    value = 1
    for index, factor in enumerate(base.FACTORS175):
        if (int(mask) >> index) & 1:
            value = mul(value, factor)
    return value


@lru_cache(maxsize=None)
def _support(mask: int) -> tuple[int, ...]:
    value = _divisor(int(mask))
    return tuple(index for index in range(base.ROT) if (value >> index) & 1)


def _mask_degree(mask: int) -> int:
    return sum(value for index, value in enumerate(FACTOR_DEGREES)
               if (int(mask) >> index) & 1)


def _common_degree(genome) -> int:
    common = FULL_MASK
    for mask in genome:
        common &= int(mask)
    return _mask_degree(common)


def _checks(genome):
    p, q, u, v = (_support(int(mask)) for mask in genome)
    a = tuple(sorted(p + tuple(base.ROT + value for value in q)))
    b = tuple(sorted(u + tuple(base.ROT + value for value in v)))
    return a, b


@lru_cache(maxsize=262144)
def _evaluate(genome):
    a, b = _checks(genome)
    weight = len(a) + len(b)
    rank = base._rank(a, b)
    return {"genome": list(genome), "rank": rank,
            "rank_error": abs(rank - base.TARGET_RANK),
            "common_degree": _common_degree(genome),
            "check_weight": weight}


def _mutate(genome, rng: random.Random):
    out = list(genome)
    operator = rng.randrange(5)
    if operator <= 2:
        component = rng.randrange(4)
        flips = 1 if operator == 0 else 2 if operator == 1 else 3
        for index in rng.sample(range(9), flips):
            out[component] ^= 1 << index
    elif operator == 3:
        component = rng.randrange(4)
        out[component] = rng.randrange(FULL_MASK)
    else:
        left, right = rng.sample(range(4), 2)
        out[left], out[right] = out[right], out[left]
    if any(mask == FULL_MASK for mask in out):
        return tuple(genome), "rejected_zero_component"
    return tuple(out), ("factor_toggle_" + str(operator) if operator < 3
                        else "component_replace" if operator == 3
                        else "component_swap")


def _select(population, beam: int):
    rows = list(population.values())
    selected = {}

    def add(items, count):
        for row in items[:int(count)]:
            selected[tuple(row["genome"])] = row

    # Two direct objectives, plus niches that bridge rank/common-factor basins.
    add(sorted(rows, key=lambda row: (row["rank_error"], row["common_degree"],
                                      row["check_weight"])), beam // 3)
    add(sorted(rows, key=lambda row: (row["common_degree"], row["rank_error"],
                                      row["check_weight"])), beam // 3)
    niches = {}
    for row in rows:
        niche = (min(row["common_degree"] // 3, 20),
                 min(row["rank_error"] // 3, 32))
        niches.setdefault(niche, []).append(row)
    for group in niches.values():
        group.sort(key=lambda row: (row["rank_error"] + row["common_degree"],
                                    row["check_weight"], row["genome"]))
    while len(selected) < int(beam):
        changed = False
        for niche in sorted(niches):
            if niches[niche]:
                row = niches[niche].pop(0)
                selected[tuple(row["genome"])] = row
                changed = True
                if len(selected) >= int(beam):
                    break
        if not changed:
            break
    return list(selected.values())[:int(beam)]


def run(args):
    rng = random.Random(int(args.seed))
    started = time.perf_counter()
    population = {}
    hits = {}
    history = []

    q47a = sum(1 << index for index in (1, 4, 5, 6))
    q47b = sum(1 << index for index in (2, 4, 5, 6))
    roots = [(q47a, q47b, q47a, q47b),
             (q47a, q47b, q47b, q47a),
             (q47b, q47a, q47a, q47b),
             (q47b, q47a, q47b, q47a)]

    def admit(genome, operator):
        genome = tuple(int(mask) for mask in genome)
        if genome in population or any(mask == FULL_MASK for mask in genome):
            return
        row = {**_evaluate(genome), "last_operator": operator}
        if row["check_weight"] > int(args.max_check_weight):
            return
        population[genome] = row
        if row["rank"] == base.TARGET_RANK and row["common_degree"] == 0:
            hits.setdefault(genome, row)

    for genome in roots:
        admit(genome, "q47_reciprocal_root")
    for _ in range(int(args.random_roots)):
        admit(tuple(rng.randrange(FULL_MASK) for _ in range(4)), "random_root")

    for generation in range(int(args.generations)):
        elites = _select(population, args.beam)
        for _ in range(int(args.mutations)):
            parent = rng.choice(elites)
            child, operator = _mutate(tuple(parent["genome"]), rng)
            admit(child, operator)
        best_common0 = min((row["rank_error"] for row in population.values()
                            if row["common_degree"] == 0), default=None)
        best_rank256 = min((row["common_degree"] for row in population.values()
                            if row["rank"] == base.TARGET_RANK), default=None)
        item = {"generation": generation + 1, "population": len(population),
                "hits": len(hits), "best_common0_rank_error": best_common0,
                "best_rank256_common_degree": best_rank256}
        history.append(item)
        print(json.dumps({"stage": "factor_lattice", **item}), flush=True)
        if len(hits) >= int(args.max_hits):
            break

    candidates = []
    for genome, row in list(hits.items())[:int(args.max_hits)]:
        a, b = _checks(genome)
        candidates.append({**row, "a": list(a), "b": list(b),
                           "k": base.K,
                           "semantic_hash": base._hash(a, b),
                           "reflection_count": len(_support(genome[1])) +
                                               len(_support(genome[3])),
                           "lineage": {"root_branch": "factor_lattice_common0"}})
    screened = base._screen(candidates, trials=args.proxy_trials,
                            seed=args.seed + 1000003, workers=args.workers,
                            threads=args.threads, target=args.target_d)
    screened.sort(key=lambda row: int(row["screen"].get("d_upper") or -1),
                  reverse=True)
    survivors = [row for row in screened
                 if int(row["screen"].get("d_upper") or -1) >= args.target_d and
                 not row["screen"]["stopped_early"]]
    report = {
        "kind": "dihedral_2bga_factor_lattice_rank_solver",
        "target": {"n": base.N, "k": base.K, "rank": base.TARGET_RANK,
                   "common_scalar_factor_degree": 0, "d": args.target_d},
        "search": {"population": len(population), "hits": len(hits),
                   "screened": len(screened), "survivors": len(survivors),
                   "history": history, "seconds_wall": time.perf_counter() - started,
                   "seed": args.seed},
        "survivors": survivors, "top": screened[:256],
        "regulation": {"stage_only": True, "submission_sent": False,
                       "git_commit_performed": False,
                       "distance_is_randomized_upper_bound": True,
                       "board_claim_allowed": False},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("results/dihedral_factor_lattice01.json"))
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--random-roots", type=int, default=4000)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--mutations", type=int, default=5000)
    parser.add_argument("--beam", type=int, default=512)
    parser.add_argument("--max-check-weight", type=int, default=128)
    parser.add_argument("--max-hits", type=int, default=256)
    parser.add_argument("--proxy-trials", type=int, default=256)
    parser.add_argument("--target-d", type=int, default=77)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"out": str(args.out), "search": report["search"],
                      "best": [(row["screen"].get("d_upper"), row["genome"])
                               for row in report["top"][:10]]}, indent=2))


if __name__ == "__main__":
    main()
