from __future__ import annotations

"""Full-cap [[700,188,*]] two-block group-algebra search over D_175.

Uses H_X=[L(a)|R(b)] and H_Z=[R(b)^T|L(a)^T].  Left and right regular
actions commute for arbitrary group-algebra elements, so CSS orthogonality is
automatic.  The initial phylogenetic roots are the two degree-47 cyclic ideals
inside C_175; they give rank 256 exactly but split into two components.  Mixed
rotation/reflection mutations seek connected rank-preserving descendants.

All distances are randomized witness-backed upper bounds.  No submit/commit.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

import numpy as np

import cpp_fast
from generalized_bicycle import common_gcd_degree
from gf2_factor import degree, factor_xm_plus_one, gcd, mul


ROT = 175
ORDER = 350
N = 700
K = 188
TARGET_RANK = (N - K) // 2
MOD175 = (1 << ROT) | 1
FACTORS175 = tuple(factor_xm_plus_one(ROT))


def _permutations():
    left = np.empty((ORDER, ORDER), dtype=np.int16)
    right = np.empty((ORDER, ORDER), dtype=np.int16)
    for g in range(ORDER):
        ga, gb = g % ROT, g // ROT
        inv_a = (-ga) % ROT if gb == 0 else ga
        inv_b = gb
        for h in range(ORDER):
            ha, hb = h % ROT, h // ROT
            left[g, h] = ((ga + (ha if gb == 0 else -ha)) % ROT +
                          ROT * ((gb + hb) & 1))
            right[g, h] = ((ha + (inv_a if hb == 0 else -inv_a)) % ROT +
                           ROT * ((hb + inv_b) & 1))
    return left, right


LEFT_PERMS, RIGHT_PERMS = _permutations()
COLS = np.arange(ORDER)


def _rep(support, *, left: bool) -> np.ndarray:
    out = np.zeros((ORDER, ORDER), dtype=np.uint8)
    permutations = LEFT_PERMS if left else RIGHT_PERMS
    for element in support:
        out[permutations[int(element)], COLS] ^= 1
    return out


def build_checks(a, b):
    A = _rep(a, left=True)
    B = _rep(b, left=False)
    return (np.concatenate([A, B], axis=1),
            np.concatenate([B.T, A.T], axis=1))


def _poly(word) -> int:
    value = 0
    for exponent in word:
        value ^= 1 << (int(exponent) % ROT)
    return value


def _cyclic_gcd_degree(word) -> int:
    return degree(gcd(_poly(word), MOD175))


def _orbit(word):
    word = tuple(sorted(int(value) % ROT for value in word))
    return min(tuple(sorted((value - anchor) % ROT for value in word))
               for anchor in word)


def degree47_branches():
    out = []
    for size in range(1, len(FACTORS175) + 1):
        for subset in itertools.combinations(range(len(FACTORS175)), size):
            if sum(degree(FACTORS175[index]) for index in subset) != 47:
                continue
            divisor = 1
            for index in subset:
                divisor = mul(divisor, FACTORS175[index])
            out.append({"branch_id": "c175_q47_" + "-".join(map(str, subset)),
                        "factor_indices": list(subset), "divisor": int(divisor),
                        "divisor_weight": int(divisor).bit_count()})
    return out


def _mine_branch(branch, *, iterations: int, min_weight: int,
                 max_weight: int, max_words: int, hill_restarts: int,
                 hill_steps: int, seed: int):
    base = tuple(index for index in range(ROT)
                 if (int(branch["divisor"]) >> index) & 1)
    mined = cpp_fast.cyclic_ideal_mine(
        [base], m=ROT, iterations=int(iterations), min_weight=int(min_weight),
        max_weight=int(max_weight), seed=int(seed), max_words=int(max_words))
    # Dense divisor support can hide the useful sparse shell from random XOR
    # sampling.  Hill-climb the same native ideal and union both populations;
    # every emitted word remains in the exact degree-47 ideal.
    climbed = cpp_fast.cyclic_ideal_hillclimb(
        base, m=ROT, restarts=int(hill_restarts), steps=int(hill_steps),
        min_weight=int(min_weight), max_weight=int(max_weight),
        seed=int(seed) ^ 0xD1B54A32D192ED03, max_words=int(max_words))
    words = {_orbit(tuple(int(value) for value in word))
             for word in itertools.chain(mined, climbed)}
    classified = [(word, _cyclic_gcd_degree(word)) for word in words]
    return {**branch,
            "words": [list(word) for word, _ in sorted(classified,
                                                       key=lambda row: (len(row[0]), row[1]))],
            "gcd_histogram": _histogram(depth for _, depth in classified),
            "weight_histogram": _histogram(len(word) for word, _ in classified)}


def _histogram(values):
    out = {}
    for value in values:
        key = str(int(value))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda item: int(item[0])))


def _seed_pairs(branch, *, count: int, attempts: int, seed: int):
    words = [tuple(word) for word in branch["words"]]
    rng = random.Random(int(seed))
    pairs = {}
    for _ in range(int(attempts)):
        if len(words) < 2 or len(pairs) >= int(count):
            break
        a, raw_b = rng.sample(words, 2)
        shift = rng.randrange(ROT)
        b = tuple(sorted((value + shift) % ROT for value in raw_b))
        if common_gcd_degree(ROT, a, b) != 47:
            continue
        key = (a, b) if a <= b else (b, a)
        pairs[key] = {"a": list(key[0]), "b": list(key[1]),
                      "branch_id": branch["branch_id"],
                      "factor_indices": branch["factor_indices"],
                      "root_gcd_degrees": [_cyclic_gcd_degree(key[0]),
                                           _cyclic_gcd_degree(key[1])]}
    return list(pairs.values())


def _cross_phylogeny_roots(branches, *, count: int, attempts: int,
                           max_check_weight: int, seed: int):
    """Mix reciprocal q47 lineages across four dihedral components."""
    if len(branches) < 2:
        return []
    banks = [[tuple(word) for word in branch["words"]] for branch in branches]
    # Low-budget smoke runs can legitimately mine no words.  Treat that as
    # an empty cross-lineage pool instead of raising from random.choice().
    if any(not bank for bank in banks):
        return []
    rng = random.Random(int(seed))
    out = {}
    for _ in range(int(attempts)):
        if len(out) >= int(count):
            break
        # Force both reciprocal factor lineages into each code; assignments
        # vary so determinant-like cancellations can restore rank256.
        assignment = rng.choice(((0, 1, 0, 1), (0, 1, 1, 0),
                                 (1, 0, 0, 1), (1, 0, 1, 0)))
        p, q, u, v = [rng.choice(banks[index]) for index in assignment]
        a = tuple(sorted(p + tuple(ROT + value for value in q)))
        b = tuple(sorted(u + tuple(ROT + value for value in v)))
        if len(a) + len(b) > int(max_check_weight):
            continue
        semantic = _hash(a, b)
        if semantic in out or _rank(a, b) != TARGET_RANK:
            continue
        out[semantic] = {
            "a": list(a), "b": list(b), "branch_id": "q47_reciprocal_cross",
            "factor_indices": [branches[0]["factor_indices"],
                               branches[1]["factor_indices"]],
            "root_gcd_degrees": [
                [_cyclic_gcd_degree(p), _cyclic_gcd_degree(q)],
                [_cyclic_gcd_degree(u), _cyclic_gcd_degree(v)]],
        }
    return list(out.values())


def _rank(a, b) -> int:
    A = _rep(a, left=True)
    B = _rep(b, left=False)
    return cpp_fast.gf2_rank(np.concatenate([A, B], axis=1))


def _hash(a, b) -> str:
    return hashlib.sha256(json.dumps([sorted(a), sorted(b)],
                                     separators=(",", ":")).encode()).hexdigest()


def _mutate(a, b, rng: random.Random, ideal_words=()):
    blocks = [set(a), set(b)]
    # Most moves remain inside the q47 ideal module; local moves provide rare
    # transverse jumps to adjacent rank strata.
    operator = rng.randrange(8)
    block = rng.randrange(2)
    if operator >= 3 and ideal_words:
        coset = rng.randrange(2)
        word = rng.choice(ideal_words)
        shift = rng.randrange(ROT)
        graft = {((int(value) + shift) % ROT) + coset * ROT for value in word}
        blocks[block].symmetric_difference_update(graft)
        label = "q47_ideal_component_xor"
    elif operator == 0:
        # Rotation/reflection lift at fixed exponent: phylogenetic sector graft.
        old = rng.choice(tuple(blocks[block]))
        new = (old + ROT) % ORDER
        if new not in blocks[block]:
            blocks[block].remove(old); blocks[block].add(new)
        label = "single_sector_graft"
    elif operator == 1:
        old = rng.choice(tuple(blocks[block]))
        new = (old + rng.choice((-2, -1, 1, 2))) % ROT + ROT * (old // ROT)
        if new not in blocks[block]:
            blocks[block].remove(old); blocks[block].add(new)
        label = "local_exponent_shift"
    elif operator == 2:
        old = rng.choice(tuple(blocks[block]))
        new = rng.randrange(ORDER)
        if new not in blocks[block]:
            blocks[block].remove(old); blocks[block].add(new)
        label = "random_replacement"
    elif operator == 3:
        # Two edits can cross rank barriers inaccessible to one-symbol moves.
        for _ in range(2):
            old = rng.choice(tuple(blocks[block]))
            new = rng.randrange(ORDER)
            if new not in blocks[block]:
                blocks[block].remove(old); blocks[block].add(new)
        label = "double_replacement"
    else:
        # Coordinated coset graft in both blocks.
        for target in (0, 1):
            old = rng.choice(tuple(blocks[target]))
            new = (old + ROT) % ORDER
            if new not in blocks[target]:
                blocks[target].remove(old); blocks[target].add(new)
        label = "coordinated_sector_graft"
    return tuple(sorted(blocks[0])), tuple(sorted(blocks[1])), label


def _reflection_count(row) -> int:
    return sum(value >= ROT for value in row["a"]) + sum(
        value >= ROT for value in row["b"])


def _evolve(seeds, word_banks, *, generations: int, mutations: int, beam: int,
            max_exact: int, max_check_weight: int, seed: int):
    rng = random.Random(int(seed))
    population = {}
    exact = {}
    history = []

    def admit(a, b, lineage, operator):
        if (not a or not b or len(a) + len(b) > int(max_check_weight) or
                len(a) + len(b) < 12):
            return
        semantic = _hash(a, b)
        if semantic in population:
            return
        rank = _rank(a, b)
        row = {"a": list(a), "b": list(b), "rank": rank,
               "k": N - 2 * rank, "semantic_hash": semantic,
               "lineage": lineage, "last_operator": operator}
        row["reflection_count"] = _reflection_count(row)
        row["check_weight"] = len(a) + len(b)
        population[semantic] = row
        if rank == TARGET_RANK and row["reflection_count"] > 0:
            exact.setdefault(semantic, row)

    for item in seeds:
        a, b = tuple(item["a"]), tuple(item["b"])
        admit(a, b, {"root_branch": item["branch_id"],
                     "factor_indices": item["factor_indices"]}, "cyclic_root")

    if not population:
        return [], history, population

    for generation in range(int(generations)):
        # Preserve exact connected genomes and near-rank stepping stones across
        # reflection-count niches.
        groups = {}
        for row in population.values():
            bucket = min(row["reflection_count"] // 2, 16)
            groups.setdefault(bucket, []).append(row)
        elites = []
        per_bucket = max(2, int(beam) // max(1, len(groups)))
        for bucket in sorted(groups):
            group = sorted(groups[bucket], key=lambda row: (
                abs(row["rank"] - TARGET_RANK), -row["reflection_count"],
                row["semantic_hash"]))
            elites.extend(group[:per_bucket])
        global_best = sorted(population.values(), key=lambda row: (
            abs(row["rank"] - TARGET_RANK), -row["reflection_count"],
            row["semantic_hash"]))[:max(1, int(beam) // 2)]
        elites = list({row["semantic_hash"]: row for row in
                       elites + global_best}.values())[:int(beam)]
        for _ in range(int(mutations)):
            parent = rng.choice(elites)
            branch = parent["lineage"]["root_branch"]
            a, b, operator = _mutate(tuple(parent["a"]), tuple(parent["b"]), rng,
                                     word_banks.get(branch, ()))
            admit(a, b, parent["lineage"], operator)
        history.append({"generation": generation + 1,
                        "population": len(population), "elites": len(elites),
                        "exact_connected": len(exact),
                        "best_rank_error": min(abs(row["rank"] - TARGET_RANK)
                                               for row in population.values()),
                        "max_exact_reflections": max(
                            (row["reflection_count"] for row in exact.values()),
                            default=0)})
        print(json.dumps({"stage": "rank_evolve", **history[-1]}), flush=True)
        if len(exact) > int(max_exact) * 4:
            # Keep evolving but bound retained exact genomes by reflection and
            # lineage diversity.
            keep = sorted(exact.values(), key=lambda row: (
                row["reflection_count"], row["semantic_hash"]), reverse=True)
            exact = {row["semantic_hash"]: row for row in keep[:int(max_exact) * 2]}
    # Round-robin weight/reflection/lineage niches.  Distance behavior varies
    # much more across these modules than rank or raw cycle score predicts.
    niches = {}
    for row in exact.values():
        niche = (row["lineage"]["root_branch"], row["check_weight"] // 4,
                 row["reflection_count"] // 4)
        niches.setdefault(niche, []).append(row)
    for group in niches.values():
        group.sort(key=lambda row: (row["reflection_count"],
                                    -row["check_weight"],
                                    row["semantic_hash"]), reverse=True)
    selected = []
    while len(selected) < int(max_exact):
        changed = False
        for niche in sorted(niches):
            if niches[niche]:
                selected.append(niches[niche].pop(0)); changed = True
                if len(selected) >= int(max_exact):
                    break
        if not changed:
            break
    return selected, history, population


def _screen_task(task):
    row, trials, seed, threads, target = task
    hx, hz = build_checks(row["a"], row["b"])
    result = cpp_fast.css_ris_parallel(
        hx, hz, trials=int(trials), seed=int(seed), pair_depth=24,
        combo_depth=2, target=int(target), stop_on_target=True,
        threads=int(threads))
    return {**row,
            "commutation_violations": int(np.count_nonzero((hx @ hz.T) & 1)),
            "screen": {"d_upper": result.get("d_upper"),
                       "dx_upper": result.get("dx_upper"),
                       "dz_upper": result.get("dz_upper"),
                       "trials_per_side": int(trials),
                       "stopped_early": bool(result["x"].get("stopped_early") or
                                             result["z"].get("stopped_early"))},
            "witnesses": {"X": result["x"].get("witness"),
                          "Z": result["z"].get("witness")}}


def _screen(rows, *, trials: int, seed: int, workers: int, threads: int,
            target: int):
    tasks = [(row, trials, seed + index * 0x9E3779B9, threads, target)
             for index, row in enumerate(rows)]
    out = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(_screen_task, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            out.append(future.result())
            if done == 1 or done % 32 == 0 or done == len(futures):
                print(json.dumps({"stage": "proxy", "done": done,
                                  "total": len(futures)}), flush=True)
    return out


def run(args):
    started = time.perf_counter()
    branches = []
    seeds = []
    word_banks = {}
    for index, branch in enumerate(degree47_branches()):
        mined = _mine_branch(
            branch, iterations=args.ideal_iterations,
            min_weight=args.word_min_weight, max_weight=args.word_max_weight,
            max_words=args.max_words, hill_restarts=args.hill_restarts,
            hill_steps=args.hill_steps, seed=args.seed + index * 10007)
        branches.append(mined)
        word_banks[mined["branch_id"]] = [tuple(word) for word in mined["words"]]
        roots = _seed_pairs(mined, count=args.roots_per_branch,
                            attempts=args.pair_attempts,
                            seed=args.seed + 1000003 + index * 10007)
        seeds.extend(roots)
        print(json.dumps({"stage": "cyclic_root", "branch": branch["branch_id"],
                          "words": len(mined["words"]), "roots": len(roots)}),
              flush=True)
    cross_roots = _cross_phylogeny_roots(
        branches, count=args.cross_roots, attempts=args.cross_attempts,
        max_check_weight=args.max_check_weight, seed=args.seed + 1500003)
    seeds.extend(cross_roots)
    word_banks["q47_reciprocal_cross"] = [
        tuple(word) for branch in branches for word in branch["words"]]
    print(json.dumps({"stage": "cross_phylogeny_root",
                      "exact_rank_roots": len(cross_roots)}), flush=True)
    exact, history, population = _evolve(
        seeds, word_banks, generations=args.generations, mutations=args.mutations,
        beam=args.beam, max_exact=args.max_exact,
        max_check_weight=args.max_check_weight, seed=args.seed + 2000003)
    screened = _screen(exact, trials=args.proxy_trials, seed=args.seed + 3000003,
                       workers=args.workers, threads=args.threads,
                       target=args.target_d)
    screened.sort(key=lambda row: int(row["screen"].get("d_upper") or -1),
                  reverse=True)
    survivors = [row for row in screened
                 if int(row["screen"].get("d_upper") or -1) >= args.target_d and
                 not row["screen"]["stopped_early"]]
    report = {
        "kind": "dihedral_2bga_n700_k188_phylogenetic_campaign",
        "construction": {"group": "D_175", "group_order": ORDER,
                         "checks": "HX=[L(a)|R(b)], HZ=[R(b)^T|L(a)^T]",
                         "target_rank": TARGET_RANK,
                         "source": "Lin-Pryadko 2BGA construction"},
        "target": {"n": N, "k": K, "d": args.target_d,
                   "score": K * args.target_d ** 2 / N},
        "search": {"roots": len(seeds), "population": len(population),
                   "exact_connected": len(exact), "screened": len(screened),
                   "survivors": len(survivors), "history": history,
                   "seconds_wall": time.perf_counter() - started,
                   "seed": args.seed},
        "branches": branches,
        "survivors": survivors,
        "top": screened[:256],
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
                        default=Path("results/dihedral_2bga_n700_01.json"))
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--ideal-iterations", type=int, default=500000)
    parser.add_argument("--word-min-weight", type=int, default=7)
    parser.add_argument("--word-max-weight", type=int, default=32)
    parser.add_argument("--max-words", type=int, default=4096)
    parser.add_argument("--hill-restarts", type=int, default=512)
    parser.add_argument("--hill-steps", type=int, default=1024)
    parser.add_argument("--roots-per-branch", type=int, default=64)
    parser.add_argument("--pair-attempts", type=int, default=100000)
    parser.add_argument("--cross-roots", type=int, default=512)
    parser.add_argument("--cross-attempts", type=int, default=50000)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--mutations", type=int, default=4000)
    parser.add_argument("--beam", type=int, default=256)
    parser.add_argument("--max-exact", type=int, default=512)
    parser.add_argument("--max-check-weight", type=int, default=64)
    parser.add_argument("--proxy-trials", type=int, default=512)
    parser.add_argument("--target-d", type=int, default=77)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"out": str(args.out),
                      "search": report["search"],
                      "best": [(row["screen"].get("d_upper"),
                                row["reflection_count"], row["semantic_hash"])
                               for row in report["top"][:10]]}, indent=2))


if __name__ == "__main__":
    main()
