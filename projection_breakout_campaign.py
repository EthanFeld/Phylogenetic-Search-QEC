"""Projection-aware generalized-bicycle breakout campaign.

The current high-rate cyclic branches repeatedly settle near 75--85 after
long RIS runs.  This campaign changes two axes at once: cyclic length and
exact GF(2) divisor degree.  It searches lower-rate divisor pockets where a
larger raw distance is plausible, then ranks only by a conservative planning
projection ``0.8 * d_raw``.  The projection is a heuristic; every distance
value remains a witness-backed randomized upper bound.

The raw target for projected distance 100 is ceil(100 / 0.8) = 125.  No
candidate is promoted to a claim by this file.  It performs local structural
gates, writes an auditable stage-only report, and never commits or submits.
"""

from __future__ import annotations

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

import cpp_fast
from generalized_bicycle import build_matrices, build_supports, common_gcd_degree
from gf2_factor import degree, factor_xm_plus_one, gcd, mul
from hyper_validator import structural_validate
from phylogenetic_search import select_phylogenetic_elites


PROJECTION = 0.80
PROJECTED_TARGET = 100
RAW_TARGET = math.ceil(PROJECTED_TARGET / PROJECTION)
CHECK_CAP = 32
CURRENT_SCORE = 1582.226
# The first run sampled a handful of lengths and mostly rediscovered the same
# weak tensor pocket.  Sweep every admissible odd length in the score-relevant
# range.  ``--m-values`` remains available for a small controlled replay.
DEFAULT_M_VALUES = tuple(range(251, 351, 2))


def score_admission_distance(row: dict, score_target: float = CURRENT_SCORE) -> int:
    """Raw RIS bound needed to clear a score target after safety deflation.

    ``RAW_TARGET`` is intentionally strict: it plans for a projected d=100.
    It is not the right admission gate for every (n, k) point.  This threshold
    applies the same projection factor to the actual score line, so a compact,
    high-rate branch that could beat the board is not discarded before its
    deeper validator gets a chance to find (or reject) it.
    """
    n, k = int(row["n"]), int(row["k"])
    if n <= 0 or k <= 0:
        raise ValueError("score admission requires positive n and k")
    if float(score_target) <= 0:
        raise ValueError("score target must be positive")
    return math.ceil(math.sqrt(float(score_target) * n / k) / PROJECTION)


def _passes_score_admission(row: dict, score_target: float) -> bool:
    observed = int(row.get("screen", {}).get("d_upper") or -1)
    return observed >= score_admission_distance(row, score_target)


def raw_score_distance(row: dict, score_target: float = CURRENT_SCORE) -> int:
    """Raw distance needed to exceed a score line without planning margin."""
    return math.ceil(math.sqrt(float(score_target) * int(row["n"]) / int(row["k"])))


def _support(poly: int, m: int) -> tuple[int, ...]:
    return tuple(i for i in range(int(m)) if (int(poly) >> i) & 1)


def _poly(values) -> int:
    result = 0
    for value in values:
        result ^= 1 << int(value)
    return result


def _shift(word, amount: int, m: int) -> tuple[int, ...]:
    return tuple(sorted((int(x) + int(amount)) % int(m) for x in word))


def _orbit(word, m: int) -> tuple[int, ...]:
    return min(_shift(word, amount, m) for amount in range(int(m)))


def _pair_key(a, b, m: int):
    return min((_shift(a, amount, m), _shift(b, amount, m))
               for amount in range(int(m)))


def _factor_groups(factors: list[int]):
    grouped = {}
    for index, poly in enumerate(factors):
        grouped.setdefault(degree(poly), []).append((index, poly))
    return sorted(grouped.items())


def _small_factor_products(component: int):
    """Return sparse local products used as tensor-product seed words."""
    factors = factor_xm_plus_one(int(component))
    products = {1: ((), 1)}
    for index, factor in enumerate(factors):
        products.setdefault(int(factor), ((index,), int(factor)))
    # Pair products cover the usual (1+x)f(y) shells.  A bounded set of
    # three-factor products supplies mixed seeds that the old search omitted:
    # they remain cheap locally, yet often create a different sparse orbit
    # after CRT lifting.  Exact gcd is checked after lifting.
    for left in range(len(factors)):
        for right in range(left + 1, len(factors)):
            products.setdefault(mul(factors[left], factors[right]),
                                ((left, right), mul(factors[left], factors[right])))
    light = sorted(range(len(factors)),
                   key=lambda index: (_support(factors[index], component).__len__(), index))[:24]
    for indices in itertools.combinations(light, 3):
        value = mul(mul(factors[indices[0]], factors[indices[1]]), factors[indices[2]])
        products.setdefault(value, (tuple(indices), value))
    out = []
    for poly, (indices, value) in products.items():
        support = _support(int(value), int(component))
        if len(support) <= 16:
            out.append({"indices": list(indices), "poly": int(value),
                        "support": list(support)})
    return out


def _tensor_seed_branches(m: int, wanted_degrees: set[int]) -> list[dict]:
    """Generate sparse CRT/tensor seeds for coprime factorizations of m."""
    m = int(m)
    modulus = (1 << m) | 1
    branches = {}
    for left in range(2, int(math.sqrt(m)) + 1):
        if m % left or math.gcd(left, m // left) != 1:
            continue
        right = m // left
        crt = {(x % left, x % right): x for x in range(m)}
        xs, ys = _small_factor_products(left), _small_factor_products(right)
        for px in xs:
            for py in ys:
                seed_support = tuple(sorted(crt[(x, y)]
                                            for x in px["support"]
                                            for y in py["support"]))
                if not seed_support or len(seed_support) > 16:
                    continue
                seed_poly = _poly(seed_support)
                divisor = gcd(seed_poly, modulus)
                q = degree(divisor)
                if q not in wanted_degrees:
                    continue
                key = (q, seed_support)
                branches.setdefault(key, {
                    "family": "crt_tensor_seed",
                    "seed_support": list(seed_support),
                    "divisor": int(divisor), "divisor_weight": int(divisor).bit_count(),
                    "gcd_degree": int(q), "component_orders": [left, right],
                    "factor_indices": [f"C{left}:{','.join(map(str, px['indices']))}",
                                       f"C{right}:{','.join(map(str, py['indices']))}"],
                    "tensor_seed_weight": len(seed_support),
                })
    return list(branches.values())


def _coalesce_tensor_branches(branches: list[dict]) -> list[dict]:
    """Merge CRT seeds with the same exact divisor into a mixed seed family.

    Equal divisor, rather than equal divisor degree, is essential: the cyclic
    shift span of every merged seed then stays in one ideal.  This is the
    out-of-sample branch generator; tensor products supply only the parents.
    """
    grouped = {}
    for branch in branches:
        key = (int(branch["m"]), int(branch["k"]), int(branch["divisor"]))
        grouped.setdefault(key, []).append(branch)
    out = []
    for (_, _, divisor), members in grouped.items():
        first = members[0]
        supports = sorted({tuple(int(x) for x in item["seed_support"])
                           for item in members}, key=lambda item: (len(item), item))
        recipes = [item["factor_indices"] for item in members]
        out.append({
            **first,
            "family": "crt_multiseed_exact_divisor",
            "seed_supports": [list(item) for item in supports],
            "seed_recipes": recipes,
            "factor_indices": [f"exact_divisor_multiseed:{len(supports)}"],
            "branch_id": (f"multiseed_m{first['m']}_k{first['k']}_"
                          f"q{first['gcd_degree']}_{len(supports)}s_"
                          f"{hashlib.sha256(str(divisor).encode()).hexdigest()[:10]}"),
        })
    return out


def _sample_divisor(factors: list[int], target_degree: int, rng: random.Random):
    """Sample an exact-degree factor subset without rejection blowups."""
    groups = _factor_groups(factors)
    choices = [(d, items) for d, items in groups]

    @lru_cache(maxsize=None)
    def feasible(position: int, remaining: int) -> bool:
        if position == len(choices):
            return remaining == 0
        d, items = choices[position]
        for count in range(min(len(items), remaining // d) + 1):
            if feasible(position + 1, remaining - count * d):
                return True
        return False

    if not feasible(0, int(target_degree)):
        return None
    selected = []
    remaining = int(target_degree)
    for position, (d, items) in enumerate(choices):
        options = [count for count in range(min(len(items), remaining // d) + 1)
                   if feasible(position + 1, remaining - count * d)]
        # Favor a few larger irreducibles, while retaining branch diversity.
        weights = [1.0 + 0.35 * count for count in options]
        count = rng.choices(options, weights=weights, k=1)[0]
        if count:
            selected.extend(rng.sample(items, count))
        remaining -= count * d
    divisor = 1
    indices = []
    for index, poly in selected:
        divisor = mul(divisor, poly)
        indices.append(int(index))
    return {
        "factor_indices": sorted(indices),
        "degree": int(degree(divisor)),
        "divisor": int(divisor),
        "divisor_weight": int(divisor).bit_count(),
    }


def _feasible_k(m: int, factors: list[int], k: int) -> bool:
    # For H_X=[A|B], H_Z=[B^T|A^T], k=2*deg(gcd(A,B,x^m+1)).
    # The divisor degree is k/2; m-k/2 is the complementary ideal dimension.
    q = int(k) // 2
    if q < 0 or int(k) % 2:
        return False
    groups = _factor_groups(factors)
    reachable = {0}
    for d, items in groups:
        old = list(reachable)
        for base in old:
            for count in range(1, len(items) + 1):
                value = base + count * d
                if value <= q:
                    reachable.add(value)
    return q in reachable


def target_specs(m_values, factors_by_m, *, max_specs: int | None = None):
    """Choose score-relevant k values, with emphasis on lower-rate branches."""
    specs = []
    for m in sorted(set(int(x) for x in m_values)):
        factors = factors_by_m[m]
        minimum = math.ceil(CURRENT_SCORE * (2 * m) / (PROJECTED_TARGET ** 2))
        # Include the first feasible rate above the current score line, a
        # lower-rate exploratory point, and high-rate controls when present.
        # Quantum k is even here.  Starting an every-other range on an odd
        # score threshold silently excludes every feasible k (notably the
        # established m=337/k=170 branch), so round the lower bound upward.
        first_k = max(64, int(minimum))
        first_k += first_k & 1
        feasible = [k for k in range(first_k, 183, 2)
                    if _feasible_k(m, factors, k)]
        if not feasible:
            continue
        picks = {feasible[0], feasible[min(len(feasible) // 2, len(feasible) - 1)],
                 feasible[-1]}
        if len(feasible) >= 4:
            picks.add(feasible[1])
        for k in sorted(picks):
            specs.append({"m": m, "k": k, "gcd_degree": k // 2,
                          "projected_score_at_100": k * PROJECTED_TARGET ** 2 / (2 * m)})
    if max_specs is not None:
        specs = sorted(specs, key=lambda row: (-row["projected_score_at_100"],
                                               row["m"], row["k"]))[:int(max_specs)]
    return sorted(specs, key=lambda row: (row["m"], row["k"]))


def _ideal_mine(branch: dict, *, iterations: int, hill_restarts: int,
                hill_steps: int, max_words: int, min_word: int, seed: int):
    m = int(branch["m"])
    # A canonical divisor can be dense even when the same ideal has a sparse
    # CRT/tensor generator.  With an exact-divisor seed family, mine its
    # *combined* shift span: this generates non-separable words while keeping
    # every word in the same algebraic ideal.
    raw_bases = branch.get("seed_supports") or [branch.get("seed_support", [])]
    bases = [tuple(int(x) for x in base) for base in raw_bases if base]
    if not bases:
        bases = [_support(int(branch["divisor"]), m)]
    words = cpp_fast.cyclic_ideal_mine(
        bases, m=m, iterations=int(iterations), min_weight=int(min_word),
        max_weight=16, seed=int(seed), max_words=int(max_words))
    # Hillclimb each of a small, weight-prioritized subset.  The multi-seed
    # miner above finds cross-seed combinations; this step fills local shells.
    for index, base in enumerate(sorted(bases, key=lambda item: (len(item), item))[:8]):
        words.extend(cpp_fast.cyclic_ideal_hillclimb(
            base, m=m, restarts=int(hill_restarts), steps=int(hill_steps),
            min_weight=int(min_word), max_weight=16,
            seed=(int(seed) ^ 0x9E3779B9) + index * 0x9E37,
            max_words=int(max_words)))
    unique = sorted(set(tuple(int(x) for x in word) for word in words),
                    key=lambda word: (len(word), word))
    # Keep one representative for each cyclic orbit, plus all short words if
    # the branch is sparse enough.  Relative shifts are generated later.
    reps = {}
    for word in unique:
        reps.setdefault(_orbit(word, m), word)
    return {**branch, "words": [list(word) for word in reps.values()],
            "word_count": len(reps),
            "word_weights": sorted(len(word) for word in reps.values())}


def _mine_task(task):
    branch, params = task
    return _ideal_mine(branch, **params)


def _pair_population(branch: dict, *, count: int, min_word: int, seed: int):
    m = int(branch["m"])
    q = int(branch["gcd_degree"])
    words = [tuple(int(x) for x in word) for word in branch.get("words", [])
             if len(word) >= int(min_word)]
    if len(words) < 2:
        return []
    # Low-weight factors were responsible for the d<=22 tensor collapse.
    # Restrict pairings to the heaviest check-compatible shell, then retain a
    # small band below it for diversity.  A check has at most 32 positions.
    allowed = [(left, right) for left in words for right in words
               if len(left) + len(right) <= CHECK_CAP]
    if not allowed:
        return []
    best_shell = max(len(left) + len(right) for left, right in allowed)
    allowed = [(left, right) for left, right in allowed
               if len(left) + len(right) >= best_shell - 2]
    rng = random.Random(int(seed))
    out = {}
    attempts = 0
    limit = max(100, int(count) * 80)
    while len(out) < int(count) and attempts < limit:
        attempts += 1
        a, raw_b = rng.choice(allowed)
        b = _shift(raw_b, rng.randrange(m), m)
        if common_gcd_degree(m, a, b) != q:
            continue
        key = _pair_key(a, b, m)
        out.setdefault(key, {
            "A": [list(key[0])], "B": [list(key[1])],
            "m": m, "n": 2 * m, "k": 2 * q,
            "gcd_degree": q, "factor_indices": branch["factor_indices"],
            "branch_id": branch["branch_id"],
            "family": branch.get("family", "random_factor_divisor"),
            "divisor_weight": branch["divisor_weight"],
            "word_weights": [len(key[0]), len(key[1])],
            "semantic_hash": hashlib.sha256(
                json.dumps([list(key[0]), list(key[1])],
                           separators=(",", ":")).encode()).hexdigest(),
        })
    return list(out.values())


def _make_candidates(mined: list[dict], *, pairs_per_branch: int, seed: int):
    rows = []
    for index, branch in enumerate(mined):
        rows.extend(_pair_population(
            branch, count=int(pairs_per_branch), min_word=int(branch["min_word"]),
            seed=int(seed) + index * 100003))
    return rows


def _screen_task(task):
    row, trials, seed, threads, target = task
    m = int(row["m"])
    a, b = tuple(row["A"][0]), tuple(row["B"][0])
    hx, hz = build_matrices(m, a, b)
    result = cpp_fast.css_ris_parallel(
        # Pair-24 is the calibrated admission screen.  The triple-12 shell
        # is useful as a supplemental probe, but it can miss a two-row word
        # whose rows rank outside its small light beam (observed on m345/q55).
        hx, hz, trials=int(trials), seed=int(seed), pair_depth=24,
        combo_depth=2,
        target=int(target), stop_on_target=True, threads=int(threads))
    out = dict(row)
    out["screen"] = {
        "d_upper": result.get("d_upper"), "dx_upper": result.get("dx_upper"),
        "dz_upper": result.get("dz_upper"), "trials_per_side": int(trials),
        "mode": result.get("mode"),
    }
    out["screen_witnesses"] = {"X": result["x"].get("witness"),
                               "Z": result["z"].get("witness")}
    return out


def _parallel_screen(rows, *, trials: int, seed: int, workers: int,
                     threads: int, label: str, target=RAW_TARGET):
    if not rows:
        return []
    tasks = [(row, int(trials), int(seed) + index * 0x9E3779B9,
              int(threads), int(target(row) if callable(target) else target))
             for index, row in enumerate(rows)]
    result = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(_screen_task, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            result.append(future.result())
            if done == 1 or done % 64 == 0 or done == len(futures):
                print(json.dumps({"stage": label, "done": done,
                                  "total": len(futures),
                                  "seconds": round(time.perf_counter() - started, 1)}),
                      flush=True)
    return result


def _quality(row: dict):
    screen = row.get("screen", {})
    d = int(screen.get("d_upper") or -1)
    k, n = int(row["k"]), int(row["n"])
    projected_d = PROJECTION * d
    projected_score = k * projected_d ** 2 / n
    # Reward raw distance first; score breaks ties.  The final hash makes
    # ordering reproducible without pretending the projection is a proof.
    return (d, projected_score, k, str(row.get("semantic_hash", "")))


def _branch_capped_rows(rows: list[dict], per_branch: int) -> list[dict]:
    """Keep only a few proxy leaders from each algebraic lineage.

    A single short RIS run can rank a whole correlated branch highly even when
    all members share an undiscovered low logical.  Capping each branch before
    phylogenetic selection makes costly holdouts test separate divisors/orbits
    instead of repeatedly testing one lucky collapse pocket.
    """
    cap = max(1, int(per_branch))
    taken: dict[str, int] = {}
    out = []
    for row in sorted(rows, key=_quality, reverse=True):
        key = str(row.get("branch_id") or row.get("semantic_hash") or "unknown")
        if taken.get(key, 0) >= cap:
            continue
        taken[key] = taken.get(key, 0) + 1
        out.append(row)
    return out


def _candidate_doc(row: dict, result: dict, *, stage: str) -> dict | None:
    dx = result["x"].get("best_weight")
    dz = result["z"].get("best_weight")
    wx = result["x"].get("witness")
    wz = result["z"].get("witness")
    if None in (dx, dz, wx, wz):
        return None
    m = int(row["m"])
    # An exhausted randomized run can report the sentinel n+1 with an empty
    # vector.  That is not a witness and must never become a candidate file.
    if int(dx) > 2 * m or int(dz) > 2 * m or not wx or not wz:
        return None
    a, b = tuple(row["A"][0]), tuple(row["B"][0])
    hx, hz = build_supports(m, a, b)
    d = min(int(dx), int(dz))
    projected = PROJECTION * d
    return {
        "schema_version": "0.1",
        "name": f"[[{2*m},{row['k']},d<={d}]] projection breakout GB",
        "code_type": "CSS", "n": 2 * m, "k": int(row["k"]),
        "checks": {"X": hx, "Z": hz},
        "distance": {
            "d": d,
            "X": {"value": int(dx), "confidence": "upper_bound", "witness": list(wx)},
            "Z": {"value": int(dz), "confidence": "upper_bound", "witness": list(wz)},
        },
        "projection": {
            "factor": PROJECTION, "projected_distance": projected,
            "projected_target": PROJECTED_TARGET,
            "raw_target": RAW_TARGET,
            "status": "planning_heuristic_only",
        },
        "provenance": {
            "authors": ["stage-only autonomous research"],
            "origin": "projection_breakout_campaign", "novelty": "unknown",
            "model": "GPT-5 Codex + native cyclic ideal/RIS kernels",
            "construction": (f"cyclic generalized bicycle m={m}; "
                             f"factor_indices={row['factor_indices']}; "
                             f"gcd degree={row['gcd_degree']}; "
                             f"A={a}; B={b}"),
            "references": [], "notes": f"stage={stage}; raw RIS witness only",
        },
        "family": "generalized-bicycle",
        "search": {"stage": stage, "factor_indices": row["factor_indices"],
                    "semantic_hash": row["semantic_hash"]},
        "regulation": {
            "stage_only": True, "submission_sent": False,
            "git_commit_performed": False, "official_gate_status": "not_run",
            "board_claim_allowed": False, "distance_is_not_proven": True,
            "projection_is_not_a_distance_claim": True,
            "literature_novelty": "unverified",
        },
    }


def _deep_task(task):
    row, trials, seed, threads = task
    return _screen_task((row, trials, seed, threads, RAW_TARGET))


def run(args):
    if not cpp_fast.available():
        raise RuntimeError(cpp_fast.accelerator_info())
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m_values = DEFAULT_M_VALUES if args.m_values is None else tuple(
        int(x) for x in str(args.m_values).split(",") if str(x).strip())

    factors_by_m = {m: factor_xm_plus_one(m) for m in m_values}
    specs = target_specs(m_values, factors_by_m, max_specs=args.max_specs)
    score_target = float(args.score_target)
    threshold_fn = raw_score_distance if args.raw_score_gate else score_admission_distance
    threshold = lambda row: threshold_fn(row, score_target)
    passes_admission = lambda row: int(row.get("screen", {}).get("d_upper") or -1) >= threshold(row)
    if args.target_k is not None:
        wanted_k = {int(value) for value in str(args.target_k).split(",")
                    if str(value).strip()}
        specs = [spec for spec in specs if int(spec["k"]) in wanted_k]
    print(json.dumps({"stage": "specs", "spec_count": len(specs),
                      "projection": PROJECTION, "raw_target": RAW_TARGET,
                      "score_target": score_target,
                      "raw_score_gate": bool(args.raw_score_gate),
                      "specs": specs}, indent=2), flush=True)

    rng = random.Random(int(args.seed))
    branches = []
    seen = set()
    for spec in specs:
        m, k, q = spec["m"], spec["k"], spec["gcd_degree"]
        for branch_index in range(int(args.branches_per_spec)):
            sampled = None
            for _ in range(100):
                sampled = _sample_divisor(factors_by_m[m], q, rng)
                if sampled is None:
                    break
                key = (m, k, tuple(sampled["factor_indices"]))
                if key not in seen:
                    seen.add(key)
                    break
            if sampled is None:
                continue
            branch = {**spec, **sampled,
                      "branch_id": f"m{m}_k{k}_b{branch_index}",
                      "min_word": int(args.min_word),
                      "branch_seed": rng.randrange(1 << 63)}
            branches.append(branch)
    # Inject sparse algebraic seeds before random branches.  These are the
    # same type of structured escape that produced the earlier C11xC31 branch,
    # generalized over every coprime factorization in the sweep.
    structured = []
    for spec in specs:
        for seed_branch in _tensor_seed_branches(spec["m"], {spec["gcd_degree"]}):
            item = {**spec, **seed_branch,
                    "branch_id": (f"tensor_m{spec['m']}_k{spec['k']}_"
                                  f"{seed_branch['component_orders']}_"
                                  f"{seed_branch['tensor_seed_weight']}w"),
                    "min_word": int(args.min_word),
                    "branch_seed": rng.randrange(1 << 63)}
            key = (item["m"], item["k"], tuple(item["seed_support"]))
            if key not in seen:
                seen.add(key)
                structured.append(item)
    if structured:
        structured = _coalesce_tensor_branches(structured)
        # Structured seeds deserve one slot per target before stochastic fill.
        branches = structured + branches
        unique_branches = {}
        for branch in branches:
            key = (branch["m"], branch["k"],
                   tuple(tuple(item) for item in branch.get("seed_supports", [])),
                   tuple(branch.get("seed_support", [])),
                   tuple(branch.get("factor_indices", [])))
            unique_branches.setdefault(key, branch)
        branches = list(unique_branches.values())
    print(json.dumps({"stage": "branches", "branch_count": len(branches)}), flush=True)

    mine_params = {"iterations": int(args.mine_iterations),
                   "hill_restarts": int(args.hill_restarts),
                   "hill_steps": int(args.hill_steps),
                   "max_words": int(args.max_words),
                   "min_word": int(args.min_word), "seed": 0}
    mined = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = []
        for branch in branches:
            params = {**mine_params, "seed": int(branch["branch_seed"])}
            futures.append(pool.submit(_mine_task, (branch, params)))
        for done, future in enumerate(as_completed(futures), 1):
            item = future.result()
            mined.append(item)
            if done == 1 or done % 16 == 0 or done == len(futures):
                print(json.dumps({"stage": "mine", "done": done,
                                  "total": len(futures),
                                  "with_two_orbits": sum(len(x.get("words", [])) >= 2
                                                         for x in mined)}), flush=True)
    mined = [item for item in mined if len(item.get("words", [])) >= 2]
    candidates = _make_candidates(mined, pairs_per_branch=args.pairs_per_branch,
                                  seed=args.seed + 101)
    print(json.dumps({"stage": "pair", "mined_branches": len(mined),
                      "candidates": len(candidates)}), flush=True)
    screened = _parallel_screen(candidates, trials=args.proxy_trials,
                                seed=args.seed + 1009, workers=args.workers,
                                threads=args.threads, label="proxy",
                                target=threshold)
    screened.sort(key=_quality, reverse=True)
    (out / "progress.json").write_text(json.dumps({
        "projection": PROJECTION, "raw_target": RAW_TARGET,
        "specs": specs, "branches": branches, "mined": mined,
        "proxy_top": screened[:64]}, indent=2) + "\n")

    # A broad proxy is only an admission test.  Spend expensive trials on
    # candidates which have not yet exposed a witness below the raw target.
    proxy_survivors = [row for row in screened
                       if passes_admission(row)]
    branch_capped_proxy = _branch_capped_rows(
        proxy_survivors, int(args.deep_per_branch))
    deep_rows, deep_selection = select_phylogenetic_elites(
        branch_capped_proxy, max(0, int(args.deep_count)))
    deep_selection["pre_cap_candidates"] = len(proxy_survivors)
    deep_selection["post_branch_cap_candidates"] = len(branch_capped_proxy)
    deep_selection["per_branch_cap"] = int(args.deep_per_branch)
    deep = _parallel_screen(deep_rows, trials=args.deep_trials,
                            seed=args.seed + 2009, workers=args.workers,
                            threads=args.threads, label="deep",
                            target=threshold)
    deep.sort(key=_quality, reverse=True)
    deep_survivors = [row for row in deep
                      if passes_admission(row)]
    final_rows, final_selection = select_phylogenetic_elites(
        deep_survivors, max(0, int(args.final_count)))
    final_runs = _parallel_screen(final_rows, trials=args.final_trials,
                                  seed=args.seed + 3009, workers=args.workers,
                                  threads=args.threads, label="final",
                                  target=threshold)
    final_runs.sort(key=_quality, reverse=True)
    final = []
    for index, row in enumerate(final_runs):
        result = {"x": {"best_weight": row["screen"]["dx_upper"],
                         "witness": row["screen_witnesses"]["X"]},
                  "z": {"best_weight": row["screen"]["dz_upper"],
                         "witness": row["screen_witnesses"]["Z"]}}
        doc = _candidate_doc(row, result, stage="final")
        if doc is None:
            continue
        if not passes_admission(row):
            continue
        structural = structural_validate(doc)
        path = out / f"candidate_{index}_{doc['n']}_{doc['k']}_{doc['distance']['d']}.json"
        path.write_text(json.dumps(doc, indent=2) + "\n")
        d = int(doc["distance"]["d"])
        projected = PROJECTION * d
        final.append({
            "candidate": str(path.resolve()), "n": doc["n"], "k": doc["k"],
            "raw_d_upper": d, "projected_distance": projected,
            "raw_score_upper": doc["k"] * d * d / doc["n"],
            "projected_score": doc["k"] * projected * projected / doc["n"],
            "meets_score_admission": passes_admission(row),
            "structural": {key: value for key, value in structural.items()
                            if key not in ("hx", "hz")},
            "branch": row["branch_id"], "factor_indices": row["factor_indices"],
        })

    report = {
        "schema_version": "1.0", "kind": "projection_aware_gb_breakout_campaign",
        "target": {"projected_distance": PROJECTED_TARGET,
                    "raw_distance_threshold": RAW_TARGET,
                    "projection_factor": PROJECTION,
                    "score_admission_target": score_target,
                    "raw_score_gate": bool(args.raw_score_gate),
                    "current_score_line": CURRENT_SCORE,
                    "check_weight_cap": CHECK_CAP},
        "search": {"m_values": list(m_values), "specs": specs,
                   "branches": len(branches), "mined_branches": len(mined),
                   "candidate_count": len(candidates),
                   "proxy_trials_per_side": int(args.proxy_trials),
                   "score_admission_distances": sorted({
                       threshold(row) for row in candidates}),
                   "proxy_survivors": len(proxy_survivors),
                   "deep_count": int(args.deep_count),
                   "deep_trials_per_side": int(args.deep_trials),
                   "deep_survivors": len(deep_survivors),
                   "deep_selection": deep_selection,
                   "final_count": int(args.final_count),
                   "final_trials_per_side": int(args.final_trials),
                   "final_selection": final_selection,
                   "seed": int(args.seed)},
        "mined": mined, "proxy_top": screened[:64], "deep_top": deep[:64],
        "final_runs": final_runs[:64],
        "final": final, "seconds_wall": time.perf_counter() - started,
        "regulation": {"stage_only": True, "submission_sent": False,
                       "git_commit_performed": False,
                       "distance_is_randomized_upper_bound": True,
                       "projection_is_heuristic": True,
                       "official_gate_run": False,
                       "board_claim_allowed": False},
    }
    (out / "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("results/projection_breakout_01"))
    parser.add_argument("--m-values", default=None,
                        help="comma-separated odd cyclic lengths")
    parser.add_argument("--max-specs", type=int, default=0,
                        help="0 means all score-relevant specs")
    parser.add_argument("--target-k", default=None,
                        help="optional comma-separated k values after feasibility filtering")
    parser.add_argument("--score-target", type=float, default=CURRENT_SCORE,
                        help="projected score required for deep-stage admission")
    parser.add_argument("--raw-score-gate", action="store_true",
                        help="use actual board-score distance rather than 0.8 planning margin")
    parser.add_argument("--branches-per-spec", type=int, default=3)
    parser.add_argument("--mine-iterations", type=int, default=15000)
    parser.add_argument("--hill-restarts", type=int, default=16)
    parser.add_argument("--hill-steps", type=int, default=512)
    parser.add_argument("--max-words", type=int, default=128)
    parser.add_argument("--min-word", type=int, default=12,
                        help="minimum A/B support weight before pair generation")
    parser.add_argument("--pairs-per-branch", type=int, default=96)
    parser.add_argument("--proxy-trials", type=int, default=384)
    parser.add_argument("--deep-count", type=int, default=48)
    parser.add_argument("--deep-per-branch", type=int, default=2,
                        help="max proxy leaders per algebraic branch before phylo deep beam")
    parser.add_argument("--deep-trials", type=int, default=5000)
    parser.add_argument("--final-count", type=int, default=8)
    parser.add_argument("--final-trials", type=int, default=2_000_000,
                        help="maximum final RIS trials per side; target-stop enabled")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.max_specs <= 0:
        args.max_specs = None
    report = run(args)
    print(json.dumps({"out": str(args.out), "proxy_candidates": len(report["proxy_top"]),
                      "final": report["final"],
                      "seconds_wall": round(report["seconds_wall"], 1)}, indent=2))


if __name__ == "__main__":
    main()
