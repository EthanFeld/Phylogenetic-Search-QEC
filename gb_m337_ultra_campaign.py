from __future__ import annotations

"""High-score Z_337 generalized-bicycle factor-lattice campaign.

The saturated Z_341, k=182 pocket repeatedly collapsed below its apparent
distance after longer RIS runs.  This campaign walks *up* the observed Z_337
phylogeny: start with the 674/86/90 divisor, add a configurable number of
irreducible factors, then mine sparse words in each child ideal.  It supports
k=170 and higher-k escape attempts without relabeling nested subideals.  Only
check-cap-feasible branches (wt(a)+wt(b)<=32) receive RIS.

Distance output is always a witness-backed randomized upper bound.  The
killer bank is an allocation sieve, never a distance proof.  This file does
not submit, commit, or change challenge files.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
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
from gf2_factor import degree, factor_xm_plus_one, gcd, mod, mul
from hyper_validator import structural_validate
from phylogenetic_search import select_phylogenetic_elites


M = 337
N = 2 * M
K = 170
GCD_DEGREE = K // 2
CHECK_CAP = 32
TARGET_D = 80
DEFAULT_CHALLENGE_ROOT = Path("challenge_data")


def _support(poly: int) -> tuple[int, ...]:
    return tuple(index for index in range(M) if (int(poly) >> index) & 1)


def _poly(values) -> int:
    out = 0
    for value in values:
        out ^= 1 << (int(value) % M)
    return out


def _shift(word: tuple[int, ...], amount: int) -> tuple[int, ...]:
    return tuple(sorted((int(value) + int(amount)) % M for value in word))


def _orbit(word: tuple[int, ...]) -> tuple[int, ...]:
    return min(_shift(tuple(word), amount) for amount in range(M))


def _pair_key(a: tuple[int, ...], b: tuple[int, ...]):
    return min((_shift(a, amount), _shift(b, amount)) for amount in range(M))


def _hash(a, b) -> str:
    return hashlib.sha256(json.dumps([list(a), list(b)], separators=(",", ":")).encode()).hexdigest()


def _read_ab(path: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    doc = json.loads(path.read_text())
    if int(doc["n"]) != N:
        raise ValueError(f"{path} has wrong block length")
    first = doc["checks"]["X"][0]
    a = tuple(int(value) for value in first if int(value) < M)
    b = tuple(int(value) - M for value in first if int(value) >= M)
    if common_gcd_degree(M, a, b) * 2 != int(doc["k"]):
        raise ValueError(f"{path} does not decode as expected GB support")
    return a, b


def _root_divisor(path: Path) -> int:
    a, b = _read_ab(path)
    return gcd(gcd(_poly(a), _poly(b)), (1 << M) | 1)


def _factor_indices(poly: int, factors: list[int]) -> tuple[int, ...]:
    return tuple(index for index, factor in enumerate(factors) if mod(poly, factor) == 0)


def _ideal_generator(divisor: int) -> np.ndarray:
    """Generator of all degree-<m cyclic-ideal words divisible by divisor."""
    support = _support(divisor)
    width = M - degree(divisor)
    out = np.zeros((width, M), dtype=np.uint8)
    for shift in range(width):
        out[shift, [shift + value for value in support]] = 1
    return out


def _classical_word(generator: np.ndarray, seed: int, trials: int, *,
                    detectors=None, target: int = CHECK_CAP + 1,
                    stop_on_target: bool = False):
    result = cpp_fast.classical_ris(
        generator, detectors=detectors, trials=int(trials), seed=int(seed),
        pair_depth=24, target=int(target), stop_on_target=bool(stop_on_target))
    witness = result.get("witness")
    return (None if witness is None else tuple(int(value) for value in witness),
            result.get("best_weight"))


def _scout_task(task):
    index, factor_pair, divisor, restarts, trials, seed = task
    generator = _ideal_generator(int(divisor))
    words = []
    for restart in range(int(restarts)):
        word, weight = _classical_word(generator, int(seed) + restart * 7919, trials)
        if word is not None and weight is not None:
            words.append((word, int(weight)))
    best = min((weight for _, weight in words), default=None)
    return {
        "branch_index": int(index), "added_factor_indices": list(factor_pair),
        "divisor": int(divisor), "divisor_weight": int(divisor).bit_count(),
        "scout_weights": [weight for _, weight in words], "scout_best": best,
    }


def _mine_task(task):
    (branch, runs, trials, shell_iterations, shell_max_words,
     outside_runs, outside_trials, outside_max_word, seed) = task
    generator = _ideal_generator(int(branch["divisor"]))
    orbits: dict[tuple[int, ...], int] = {}
    weights = []
    for run in range(int(runs)):
        word, weight = _classical_word(generator, int(seed) + run * 104729, trials)
        if word is None or weight is None or int(weight) > CHECK_CAP // 2:
            continue
        canonical = _orbit(word)
        orbits[canonical] = min(orbits.get(canonical, int(weight)), int(weight))
        weights.append(int(weight))

    # The lightest parent word can live in a deeper child subideal.  Shell
    # XORs can never escape that child.  Search the intended parent quotient
    # directly: null(child_generator) detects words outside each immediate
    # child rowspace.  Retain asymmetric words too; only wt(a)+wt(b)<=32 is a
    # challenge constraint, not wt(a),wt(b)<=16 individually.
    outside_words = []
    if int(outside_runs) > 0 and int(outside_max_word) > 0:
        factors = factor_xm_plus_one(M)
        divisor = int(branch["divisor"])
        for factor_index, factor in enumerate(factors):
            if mod(divisor, factor) == 0:
                continue
            child = mul(divisor, factor)
            if degree(child) >= M:
                continue
            detectors = gf2_nullspace(_ideal_generator(child))
            for run in range(int(outside_runs)):
                word, weight = _classical_word(
                    generator,
                    int(seed) ^ (factor_index + 1) * 0x9E3779B9 ^ run * 104729,
                    int(outside_trials), detectors=detectors,
                    target=int(outside_max_word) + 1, stop_on_target=True)
                if (word is None or weight is None or
                        int(weight) > int(outside_max_word)):
                    continue
                canonical = _orbit(word)
                orbits[canonical] = min(orbits.get(canonical, int(weight)), int(weight))
                outside_words.append(canonical)

    # Classical RIS deliberately returns the lightest ideal word.  On the
    # strongest branch that trapped discovery on a 12+12 check shell.  XOR
    # shifted low-word parents inside the same exact ideal to populate 14--16
    # shells; these are new check geometries, not more phases of one word.
    low_parents = sorted(orbits, key=lambda word: (len(word), word))
    shell_words = []
    if low_parents and int(shell_max_words) > 0:
        # Separate quotas matter: abundant weight-14 words otherwise fill the
        # native buffer before its deterministic pair shell reaches weight 16.
        per_shell = max(1, int(shell_max_words) // 2)
        for shell_weight in (14, CHECK_CAP // 2):
            shell_words.extend(cpp_fast.cyclic_ideal_mine(
                low_parents, m=M,
                iterations=max(0, int(shell_iterations) // 2),
                min_weight=shell_weight, max_weight=shell_weight,
                seed=(int(seed) ^ 0xA5A55A5A) + shell_weight * 0x9E37,
                max_words=per_shell))
        for word in shell_words:
            canonical = _orbit(tuple(int(value) for value in word))
            orbits.setdefault(canonical, len(word))
    words = sorted(orbits, key=lambda word: (len(word), word))
    return {**branch, "word_orbits": [list(word) for word in words],
            "word_count": len(words),
            "word_weights": sorted(len(word) for word in words),
            "shell_word_count": sum(len(word) > 12 for word in words),
            "outside_subideal_word_count": len(set(outside_words)),
            "outside_subideal_gcd_degrees": sorted({
                common_gcd_degree(M, word, word) for word in outside_words})}


def _kernel_syndrome_zero(a, b, side: str, witness: tuple[int, ...]) -> bool:
    """Fast GB syndrome test for a known logical; no matrix construction."""
    vector = np.zeros(N, dtype=np.uint8)
    vector[list(witness)] = 1
    left, right = vector[:M], vector[M:]
    syndrome = np.zeros(M, dtype=np.uint8)
    if side == "X":  # X logicals live in ker(H_Z).
        for value in b:
            syndrome ^= np.roll(left, int(value))
        for value in a:
            syndrome ^= np.roll(right, int(value))
    elif side == "Z":  # Z logicals live in ker(H_X).
        for value in a:
            syndrome ^= np.roll(left, -int(value))
        for value in b:
            syndrome ^= np.roll(right, -int(value))
    else:
        raise ValueError("side must be X or Z")
    return not bool(np.any(syndrome))


def _history_killers(paths, target_d: int = TARGET_D) -> list[dict]:
    """Load independently witnessed below-target logicals from deep receipts."""
    out = []
    for path in (Path(value) for value in paths):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        items = []
        refutation = payload.get("refutation") if isinstance(payload, dict) else None
        if isinstance(refutation, dict):
            items.append(refutation)
        distance = payload.get("distance") if isinstance(payload, dict) else None
        if isinstance(distance, dict):
            for side in ("X", "Z"):
                item = distance.get(side)
                if isinstance(item, dict):
                    items.append({**item, "side": side,
                                  "weight": item.get("weight", item.get("value"))})
        for item in items:
            side, weight, witness = item.get("side"), item.get("weight"), item.get("witness")
            if side not in ("X", "Z") or weight is None or witness is None:
                continue
            if int(weight) < int(target_d):
                out.append({"source": path.name, "side": side,
                            "weight": int(weight),
                            "witness": [int(value) for value in witness]})
    return out


def _initial_killers(board_path: Path, history_paths=(),
                     target_d: int = TARGET_D) -> list[dict]:
    doc = json.loads(board_path.read_text())
    out = []
    for side in ("X", "Z"):
        item = doc["distance"][side]
        if int(item["value"]) < int(target_d):
            out.append({"source": board_path.name, "side": side,
                        "weight": int(item["value"]),
                        "witness": [int(value) for value in item["witness"]]})
    out.extend(_history_killers(history_paths, target_d))
    unique = {}
    for item in out:
        key = (item["side"], tuple(item["witness"]))
        unique.setdefault(key, item)
    return list(unique.values())


def _sieve(rows: list[dict], killers: list[dict]) -> tuple[list[dict], int]:
    survivors = []
    rejected = 0
    for row in rows:
        a, b = tuple(row["A"][0]), tuple(row["B"][0])
        hits = [killer["source"] + ":" + killer["side"]
                for killer in killers
                if _kernel_syndrome_zero(a, b, killer["side"],
                                         tuple(killer["witness"]))]
        if hits:
            rejected += 1
            continue
        row = dict(row)
        row["killer_bank_hits"] = []
        survivors.append(row)
    return survivors, rejected


def _candidate_population(branches: list[dict], count: int, seed: int,
                          min_word_weight: int = 12,
                          gcd_degree: int = GCD_DEGREE,
                          target_d: int = TARGET_D) -> list[dict]:
    rng = random.Random(int(seed))
    candidates: dict[tuple[tuple[int, ...], tuple[int, ...]], dict] = {}
    viable = [branch for branch in branches
              if sum(len(word) >= int(min_word_weight)
                     for word in branch.get("word_orbits", [])) >= 2]
    attempts = 0
    limit = max(100, int(count) * 100)
    while viable and len(candidates) < int(count) and attempts < limit:
        attempts += 1
        branch = rng.choice(viable)
        words = [tuple(word) for word in branch["word_orbits"]
                 if len(word) >= int(min_word_weight)]
        if len(words) < 2:
            continue
        # Prefer newly expanded high-weight shells while retaining low-shell
        # ancestry.  16+16 gets 17x weight of 12+12 under this schedule.
        sampling_weights = [1 + max(0, len(word) - 12) ** 2 for word in words]
        a = rng.choices(words, weights=sampling_weights, k=1)[0]
        b = rng.choices(words, weights=sampling_weights, k=1)[0]
        if a == b:
            continue
        b = _shift(b, rng.randrange(M))
        if len(a) + len(b) > CHECK_CAP:
            continue
        if common_gcd_degree(M, a, b) != int(gcd_degree):
            continue
        key = _pair_key(a, b)
        candidates.setdefault(key, {
            "A": [list(key[0])], "B": [list(key[1])],
            "n": N, "k": 2 * int(gcd_degree), "group_order": M, "group_index": 0,
            "lineage": f"d90_to_k{2 * int(gcd_degree)}_factor_child",
            "branch_index": branch["branch_index"],
            "added_factor_indices": branch["added_factor_indices"],
            "divisor_weight": branch["divisor_weight"],
            "word_weights": [len(key[0]), len(key[1])],
            "individual_gcd_degrees": [common_gcd_degree(M, key[0], key[0]),
                                       common_gcd_degree(M, key[1], key[1])],
            "target_d": int(target_d),
            "semantic_hash": _hash(*key),
        })
    return list(candidates.values())


def _screen_task(task):
    row, seed, trials, threads = task
    a, b = tuple(row["A"][0]), tuple(row["B"][0])
    hx, hz = build_matrices(M, a, b)
    result = cpp_fast.css_ris_parallel(
        hx, hz, trials=int(trials), seed=int(seed), pair_depth=24,
        target=int(row.get("target_d", TARGET_D)), stop_on_target=True,
        threads=int(threads))
    out = dict(row)
    out["screen"] = {"d_upper": result.get("d_upper"),
                     "dx_upper": result.get("dx_upper"),
                     "dz_upper": result.get("dz_upper"),
                     "trials_per_side": int(trials),
                     "stopped_early": bool(result["x"].get("stopped_early") or
                                           result["z"].get("stopped_early"))}
    out["screen_witnesses"] = {"X": result["x"].get("witness"),
                                "Z": result["z"].get("witness")}
    return out


def _confirm_task(task):
    row, seed, trials, threads = task
    a, b = tuple(row["A"][0]), tuple(row["B"][0])
    hx, hz = build_matrices(M, a, b)
    result = cpp_fast.css_ris_parallel(
        hx, hz, trials=int(trials), seed=int(seed), pair_depth=24,
        target=int(row.get("target_d", TARGET_D)), stop_on_target=True,
        threads=int(threads))
    return {**row, "confirm": result}


def _parallel(fn, rows, *, seed, trials, workers, threads, label):
    tasks = [(row, int(seed) + index * 0x9E3779B9, int(trials), int(threads))
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
                                  "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    return out


def _quality(row: dict) -> tuple[int, int, int, int, int, str]:
    screen = row.get("screen", {})
    d_upper = int(screen.get("d_upper") or -1)
    target_d = int(row.get("target_d", TARGET_D))
    depths = [int(value) for value in row.get("individual_gcd_degrees", [])]
    common_depth = int(row.get("k", K)) // 2
    complementary = int(len(depths) == 2 and min(depths) == common_depth
                        and max(depths) > common_depth)
    # A single high randomized minimum is a severe winner's-curse signal.
    # First retain candidates that have not refuted the frontier, then prefer
    # complementary ideal depths and the empirically safer dense check shell.
    # Mixed (85,127) halves had no catastrophic d=22 misses in the audit. On
    # the shell-expansion audit,
    # 16+16 survived 100k in 7/7 cases while both catastrophic d=22 misses
    # came from 14+14.  Distance only breaks ties inside a shell.
    return (int(d_upper >= target_d and not screen.get("stopped_early", False)),
            complementary,
            sum(int(value) for value in row.get("word_weights", [])),
            d_upper,
            min(int(screen.get("dx_upper") or -1),
                int(screen.get("dz_upper") or -1)),
            str(row.get("semantic_hash", "")))


def _candidate_doc(row: dict, result: dict) -> dict | None:
    dx, dz = result["x"].get("best_weight"), result["z"].get("best_weight")
    wx, wz = result["x"].get("witness"), result["z"].get("witness")
    if dx is None or dz is None or wx is None or wz is None:
        return None
    code = normalize_code(M, row["A"][0], row["B"][0])
    hx, hz = build_supports(M, code.a, code.b)
    d = min(int(dx), int(dz))
    return {
        "schema_version": "0.1",
        "name": f"[[{code.n},{code.k},d<={d}]] Z_337 factor-child generalized bicycle",
        "code_type": "CSS", "n": code.n, "k": code.k,
        "checks": {"X": hx, "Z": hz},
        "distance": {"d": d,
                     "X": {"value": int(dx), "confidence": "upper_bound", "witness": list(wx)},
                     "Z": {"value": int(dz), "confidence": "upper_bound", "witness": list(wz)}},
        "provenance": {
            "authors": ["stage-only autonomous research"], "origin": "local_campaign",
            "novelty": "unknown", "model": "GPT-5 Codex",
            "construction": (f"Z_337 generalized bicycle; a={list(code.a)}, b={list(code.b)}, "
                             f"gcd degree={code.gcd_degree}; individual gcd degrees="
                             f"{row.get('individual_gcd_degrees')}; "
                             f"d90-to-k{code.k} factor child "
                             f"{row['added_factor_indices']}"),
            "references": [],
            "notes": "Native RIS witnesses independently checked locally; submission_sent=false.",
        },
        "family": "generalized-bicycle",
        "regulation": {"stage_only": True, "submission_sent": False,
                       "git_commit_performed": False, "official_gate_status": "not_run",
                       "board_claim_allowed": False, "distance_is_not_proven": True,
                       "literature_novelty": "unverified"},
    }


def _compact_structural(value: dict) -> dict:
    return {key: item for key, item in value.items() if key not in ("hx", "hz")}


def run(args):
    if not cpp_fast.available():
        raise RuntimeError(cpp_fast.accelerator_info())
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    board_root = Path(args.challenge_root) / "codes"
    ancestor_path = board_root / "674-86-90.json"
    known_path = board_root / "674-170-76.json"
    factors = factor_xm_plus_one(M)
    ancestor = _root_divisor(ancestor_path)
    ancestor_degree = degree(ancestor)
    used = set(_factor_indices(ancestor, factors))
    remaining = [index for index in range(len(factors)) if index not in used]
    added_factor_count = int(args.added_factors)
    if added_factor_count < 1 or added_factor_count > len(remaining):
        raise ValueError("added-factors is outside the remaining divisor lattice")
    added_degrees = sorted(degree(factors[index]) for index in remaining)
    target_gcd_degree = ancestor_degree + sum(added_degrees[:added_factor_count])
    target_k = 2 * target_gcd_degree
    target_d = args.target_d
    if target_d is None:
        target_d = max(1, math.ceil(math.sqrt(float(args.leader_score) * N / target_k)))
        while target_k * target_d * target_d / N <= float(args.leader_score):
            target_d += 1
    branch_tasks = []
    for branch_index, added in enumerate(
            __import__("itertools").combinations(remaining, added_factor_count)):
        divisor = ancestor
        for factor_index in added:
            divisor = mul(divisor, factors[factor_index])
        if degree(divisor) != target_gcd_degree:
            continue
        branch_tasks.append((branch_index, added, divisor, args.scout_restarts,
                             args.scout_trials, args.seed + branch_index * 4099))
    print(json.dumps({"stage": "factor_scout", "branches": len(branch_tasks),
                      "target": {"n": N, "k": target_k, "d": target_d}}), flush=True)
    scout = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(_scout_task, task) for task in branch_tasks]
        for done, future in enumerate(as_completed(futures), 1):
            scout.append(future.result())
            if done == 1 or done % 24 == 0 or done == len(futures):
                print(json.dumps({"stage": "factor_scout", "done": done,
                                  "total": len(futures)}), flush=True)
    viable = [row for row in scout if row.get("scout_best") is not None and
              int(row["scout_best"]) <= CHECK_CAP // 2]
    viable.sort(key=lambda row: (int(row["scout_best"]), int(row["divisor_weight"]),
                                 row["added_factor_indices"]))
    mine_tasks = [(row, args.word_runs, args.word_trials,
                   args.shell_iterations, args.shell_max_words,
                   args.outside_subideal_runs, args.outside_subideal_trials,
                   args.outside_max_word,
                   args.seed + 500_003 + index * 4099)
                  for index, row in enumerate(viable)]
    mined = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(_mine_task, task) for task in mine_tasks]
        for done, future in enumerate(as_completed(futures), 1):
            mined.append(future.result())
            print(json.dumps({"stage": "word_mine", "done": done, "total": len(futures),
                              "orbits": mined[-1]["word_count"]}), flush=True)
    killers = _initial_killers(known_path, args.killer_history, target_d)
    generations = []
    for generation in range(max(1, int(args.generations))):
        population = _candidate_population(
            mined, args.population, args.seed + generation * 1_000_003,
            min_word_weight=args.min_word_weight,
            gcd_degree=target_gcd_degree, target_d=target_d)
        filtered, sieve_rejected = _sieve(population, killers)
        screened = _parallel(_screen_task, filtered, seed=args.seed + 100_003 + generation * 1_000_003,
                             trials=args.proxy_trials, workers=args.workers,
                             threads=args.threads, label=f"proxy_g{generation + 1}")
        screened.sort(key=_quality, reverse=True)
        proxy_survivors = [row for row in screened
                           if int(row.get("screen", {}).get("d_upper") or -1) >= target_d
                           and not row.get("screen", {}).get("stopped_early", False)]
        beam = proxy_survivors[:min(len(proxy_survivors), int(args.beam_size))]
        deep = _parallel(_screen_task, beam, seed=args.seed + 200_003 + generation * 1_000_003,
                         trials=args.deep_trials, workers=args.workers,
                         threads=args.threads, label=f"deep_g{generation + 1}")
        deep.sort(key=_quality, reverse=True)
        deep_survivors = [row for row in deep
                          if int(row.get("screen", {}).get("d_upper") or -1) >= target_d
                          and not row.get("screen", {}).get("stopped_early", False)]
        selected, phylogeny = select_phylogenetic_elites(
            deep_survivors, min(int(args.confirm_count), len(deep_survivors)), pool_factor=8,
            exploitation_fraction=0.5)
        confirmed = _parallel(_confirm_task, selected,
                              seed=args.seed + 300_003 + generation * 1_000_003,
                              trials=args.confirm_trials, workers=min(args.workers, max(1, args.confirm_count)),
                              threads=args.threads, label=f"confirm_g{generation + 1}")
        confirmed.sort(key=lambda row: (int(row["confirm"].get("d_upper") or -1),
                                        _quality(row)), reverse=True)
        final = []
        for rank, row in enumerate(confirmed, 1):
            result = row["confirm"]
            doc = _candidate_doc(row, result)
            if doc is None:
                continue
            structural = _compact_structural(structural_validate(doc))
            d = int(doc["distance"]["d"])
            path = out / f"candidate_g{generation + 1}_{rank}_{d}.json"
            path.write_text(json.dumps(doc, indent=2) + "\n")
            final.append({"rank": rank, "candidate_path": str(path.resolve()),
                          "semantic_hash": row["semantic_hash"], "d": d,
                          "dx": int(doc["distance"]["X"]["value"]),
                          "dz": int(doc["distance"]["Z"]["value"]),
                          "score_upper": int(doc["k"]) * d * d / N,
                          "trials_per_side": int(args.confirm_trials),
                          "stopped_early": bool(result["x"].get("stopped_early") or
                                                result["z"].get("stopped_early")),
                          "branch": row["added_factor_indices"], "structural": structural})
            if d < target_d:
                for side, native_side in (("X", "x"), ("Z", "z")):
                    witness = result[native_side].get("witness")
                    weight = result[native_side].get("best_weight")
                    if witness is not None and weight is not None and int(weight) < target_d:
                        killers.append({"source": row["semantic_hash"], "side": side,
                                        "weight": int(weight), "witness": list(witness)})
        generations.append({
            "generation": generation + 1, "population": len(population),
            "sieve_rejected": sieve_rejected, "screened": len(screened),
            "proxy_survivors": len(proxy_survivors),
            "beam": len(beam), "deep": len(deep),
            "deep_survivors": len(deep_survivors), "phylogeny": phylogeny,
            "proxy_top": screened[:32], "deep_top": deep[:32], "final": final,
            "killer_bank_size_after": len(killers),
        })
        (out / "progress.json").write_text(json.dumps({"generations": generations,
                                                         "killer_bank": killers}, indent=2) + "\n")
    report = {
        "schema_version": "1.0", "kind": "gb_m337_factor_phylogeny_campaign",
        "submission_sent": False, "git_commit_performed": False,
        "target": {"m": M, "n": N, "k": target_k,
                   "gcd_degree": target_gcd_degree,
                   "check_weight_cap": CHECK_CAP,
                   "leader_beating_score": target_k * target_d * target_d / N,
                   "score_to_beat": float(args.leader_score),
                   "leader_beating_d": target_d,
                   "minimum_word_shell": int(args.min_word_weight)},
        "phylogenetic_hypothesis": {
            "parent": f"[[674,86,d<=90]] degree-{ancestor_degree} divisor",
            "transition": (f"add {added_factor_count} irreducible factors to form "
                           f"degree-{target_gcd_degree}/k={target_k} children"),
            "selection": ("sparse exact-parent/complementary-child words; phase-diverse "
                          "cross-orbit pairings; asymmetric halves allowed under total cap"),
            "outside_subideal_search": {
                "runs_per_child": int(args.outside_subideal_runs),
                "trials_per_run": int(args.outside_subideal_trials),
                "maximum_retained_word_weight": int(args.outside_max_word),
            },
            "rejected_transition": "[[674,128,d<=80]] children lacked <=16 sparse words under w<=32",
        },
        "factor_scout": scout, "viable_branches": [{key: value for key, value in row.items()
                                                       if key != "divisor"} for row in mined],
        "killer_bank": killers, "generations": generations,
        "accelerator": cpp_fast.accelerator_info(), "seconds_wall": time.perf_counter() - started,
        "regulation": {"stage_only": True, "submission_sent": False,
                       "git_commit_performed": False,
                       "distance_is_randomized_upper_bound": True,
                       "official_gate_run": False},
    }
    (out / "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/gb_m337_ultra_01"))
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--added-factors", type=int, default=2,
                        help="irreducible factors added above the degree-43 ancestor")
    parser.add_argument("--target-d", type=int,
                        help="early-stop frontier distance; derived from leader-score by default")
    parser.add_argument("--leader-score", type=float, default=1582.226,
                        help="score which the derived integer target must strictly exceed")
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--scout-restarts", type=int, default=3)
    parser.add_argument("--scout-trials", type=int, default=64)
    parser.add_argument("--word-runs", type=int, default=48)
    parser.add_argument("--word-trials", type=int, default=128)
    parser.add_argument("--shell-iterations", type=int, default=500_000)
    parser.add_argument("--shell-max-words", type=int, default=4096)
    parser.add_argument("--outside-subideal-runs", type=int, default=0,
                        help="detector-constrained RIS restarts per immediate child")
    parser.add_argument("--outside-subideal-trials", type=int, default=4096)
    parser.add_argument("--outside-max-word", type=int, default=20,
                        help="retain parent-quotient words up to this asymmetric weight")
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--min-word-weight", type=int, default=12,
                        help="discard mined A/B words below this shell (use 16 for frontier runs)")
    parser.add_argument("--proxy-trials", type=int, default=64)
    parser.add_argument("--beam-size", type=int, default=16)
    parser.add_argument("--deep-trials", type=int, default=512)
    parser.add_argument("--confirm-count", type=int, default=2)
    parser.add_argument("--confirm-trials", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--killer-history", type=Path, action="append", default=[
        Path("results/gb_m337_ultra_01/validation_ris_2m.json"),
        Path("results/gb_m337_ultra_01/validation_g1_2_ris20m.json"),
    ])
    parser.add_argument("--challenge-root", type=Path, default=DEFAULT_CHALLENGE_ROOT)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"out": str(args.out), "seconds_wall": round(result["seconds_wall"], 1),
                      "final": result["generations"][-1]["final"]}, indent=2))


if __name__ == "__main__":
    main()
