from __future__ import annotations

"""Canonical, fixed-budget ML mutation campaign.

Stages: fixed-budget classifier -> stratified seed selection -> canonical
mutation -> cached GCD-family RIS -> verified CSS RIS.  Candidate distances
remain randomized witness upper bounds.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import random
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`.*",
    category=UserWarning,
)

import cpp_fast
from generalized_bicycle import build_matrices, build_supports, common_gcd_degree
from gf2_factor import degree, factor_xm_plus_one, gcd, mod, mul, quotient
from hyper_validator import _logical_check, structural_validate
from ideal_x_logical_search import ideal_basis
from repair_and_rank import _features


TARGET_SCORE = 1542.0
MODEL_BUDGET = 256
DEFAULT_SURVIVAL_HORIZON = 65_536


def _family_cache_key(key: tuple[int, int]) -> str:
    return f"{int(key[0])}:{hex(int(key[1]))}"


def _load_witness_cache(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    families = payload.get("families", payload) if isinstance(payload, dict) else {}
    return families if isinstance(families, dict) else {}


def _write_witness_cache(path: Path, rows: list[dict]) -> int:
    families = _load_witness_cache(path)
    added = 0
    for row in rows:
        gate = row.get("ideal_gate") or {}
        witness = gate.get("ideal_witness")
        key = row.get("gcd_family_key")
        if not witness or not isinstance(key, list) or len(key) != 2:
            continue
        cache_key = _family_cache_key((int(key[0]), int(str(key[1]), 16)))
        weight = int(gate.get("raw_ideal_weight") or len(witness))
        old = families.get(cache_key)
        if old is None or weight < int(old.get("weight", 10**9)):
            families[cache_key] = {
                "family_key": [int(key[0]), str(key[1])],
                "weight": weight, "witness": [int(x) for x in witness],
                "trials_run": int(gate.get("ideal_trials_run") or 0),
                "backend": gate.get("ideal_backend") or gate.get("ideal_mode"),
                "source_candidate_id": row.get("candidate_id"),
                "cache_schema": "family_witness_v1",
            }
            added += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": "1.0", "families": families},
                               indent=2) + "\n", encoding="utf-8")
    return added


def _poly(values, m: int) -> int:
    out = 0
    for value in values:
        out ^= 1 << (int(value) % int(m))
    return out


def _rotate(mask: int, shift: int, m: int) -> int:
    m = int(m)
    shift %= m
    ring = (1 << m) - 1
    if not shift:
        return int(mask) & ring
    return ((int(mask) << shift) | (int(mask) >> (m - shift))) & ring


def _support(mask: int) -> tuple[int, ...]:
    out = []
    mask = int(mask)
    while mask:
        bit = mask & -mask
        out.append(bit.bit_length() - 1)
        mask ^= bit
    return tuple(out)


@lru_cache(maxsize=200_000)
def _canonical_mask(mask: int, m: int) -> int:
    """Canonical representative under cyclic coordinate translation."""
    bits = _support(mask)
    if not bits:
        return 0
    return min(_rotate(mask, -anchor, m) for anchor in bits)


def _canonical_pair(a_mask: int, b_mask: int, m: int) -> tuple[int, int]:
    # Independent block translations are coordinate equivalences for GB codes.
    return _canonical_mask(a_mask, m), _canonical_mask(b_mask, m)


def _hash_pair(a: tuple[int, ...], b: tuple[int, ...]) -> str:
    return hashlib.sha256(json.dumps([list(a), list(b)], separators=(",", ":")).encode()).hexdigest()


def _required_distance(n: int, k: int) -> int:
    return int(math.sqrt(TARGET_SCORE * int(n) / int(k))) + 1


def _family_key_masks(a_mask: int, b_mask: int, m: int) -> tuple[int, int]:
    modulus = (1 << int(m)) | 1
    g = gcd(gcd(int(a_mask), int(b_mask)), modulus)
    return int(m), int(g)


def _family_key(row: dict) -> tuple[int, int]:
    return _family_key_masks(_poly(row["A"][0], row["m"]),
                             _poly(row["B"][0], row["m"]), int(row["m"]))


def _mul_mod(a: int, b: int, m: int) -> int:
    out = 0
    left = int(a)
    while left:
        bit = left & -left
        out ^= _rotate(int(b), bit.bit_length() - 1, m)
        left ^= bit
    return out & ((1 << int(m)) - 1)


def _random_combo(a_mask: int, b_mask: int, m: int, rng: random.Random,
                  terms_min: int = 2, terms_max: int = 5) -> int:
    atoms = (int(a_mask), int(b_mask))
    out = 0
    for _ in range(rng.randint(terms_min, terms_max)):
        out ^= _rotate(atoms[rng.randrange(2)], rng.randrange(int(m)), int(m))
    return out


def _random_mask(m: int, weight: int, rng: random.Random) -> int:
    """GF(2) sparse mask; duplicate picks XOR away."""
    out = 0
    for _ in range(int(weight)):
        out ^= 1 << rng.randrange(int(m))
    return out


@lru_cache(maxsize=32)
def _modulus_factors(m: int) -> tuple[int, ...]:
    return tuple(factor_xm_plus_one(int(m)))


@lru_cache(maxsize=128)
def _degree_divisor_candidates(m: int, target_degree: int) -> tuple[int, ...]:
    modulus = (1 << int(m)) | 1
    states: dict[int, list[int]] = {0: [1]}
    for factor in _modulus_factors(int(m)):
        fd = degree(factor)
        # Snapshot values immutably.  Live lists let a factor added at this
        # pass become a source for the same pass, producing products that do
        # not divide x^m+1 (especially with repeated factors).
        previous = tuple((current, tuple(values))
                         for current, values in states.items())
        for current, values in previous:
            if current + fd > int(target_degree):
                continue
            bucket = states.setdefault(current + fd, [])
            for value in values:
                product = mul(value, factor)
                if (degree(product) == current + fd
                        and mod(modulus, product) == 0
                        and product not in bucket and len(bucket) < 256):
                    bucket.append(product)
    return tuple(value for value in states.get(int(target_degree), [])
                 if degree(value) == int(target_degree)
                 and mod(modulus, value) == 0)


def _random_degree_divisor(m: int, target_degree: int, rng: random.Random,
                           avoid: int | None = None) -> int | None:
    """Sample a different modulus divisor with requested GF(2) degree."""
    choices = [value for value in _degree_divisor_candidates(int(m), int(target_degree))
               if avoid is None or int(value) != int(avoid)]
    if choices:
        return int(rng.choice(choices))
    return None


def _rewire(mask: int, m: int, rng: random.Random, edits: int = 1) -> int:
    bits = set(_support(mask))
    if not bits:
        return 0
    for _ in range(int(edits)):
        if not bits:
            break
        bits.remove(rng.choice(tuple(bits)))
        bits.add(rng.randrange(int(m)))
    return sum(1 << x for x in bits)


def _add_child(children: dict, seen_equiv: set, a_mask: int, b_mask: int,
               seed: dict, mode: str, attempts: int) -> None:
    m = int(seed["m"])
    a, b = _support(a_mask), _support(b_mask)
    # structural_validate enforces the challenge's max CSS row weight of 32;
    # keep this gate here so no invalid candidate reaches RIS/GCD validation.
    max_support = 32
    if not a or not b or not (20 <= len(a) + len(b) <= max_support):
        return
    canon = _canonical_pair(int(a_mask), int(b_mask), m)
    if canon in seen_equiv:
        return
    base_a = _poly(seed["representation"]["A"][0], m)
    base_b = _poly(seed["representation"]["B"][0], m)
    parent_canon = _canonical_pair(base_a, base_b, m)
    if canon == parent_canon:
        return
    q = int(seed["k"]) // 2
    if common_gcd_degree(m, a, b) != q:
        return
    seen_equiv.add(canon)
    key = (a, b)
    children.setdefault(key, {
        "A": [list(a)], "B": [list(b)], "n": int(seed["n"]),
        "k": int(seed["k"]), "m": m,
        "parent_code_id": seed["code_id"],
        "parent_semantic_hash": seed.get("semantic_hash"),
        "parent_lineage_group": seed.get("repaired_group") or seed.get("semantic_hash"),
        "parent_split": seed.get("split_repaired"),
        "family": seed.get("family", "generalized-bicycle"),
        "representation_kind": seed.get("representation_kind", "ring_genome"),
        "algebraic_group": seed.get("algebraic_group") or seed.get("repaired_group"),
        "semantic_group": seed.get("semantic_group"),
        "representation": {"A": [list(a)], "B": [list(b)]},
        "fixed_budget_p": seed.get("fixed_budget_p"),
        "fixed_budget_priority": seed.get("fixed_budget_priority"),
        "fixed_budget_model": seed.get("fixed_budget_model"),
        "fixed_budget": seed.get("fixed_budget"),
        "mutation_mode": mode, "mutation_attempt": int(attempts),
        "equivalence_key": [hex(canon[0]), hex(canon[1])],
    })


def _mutate_seed(seed: dict, count: int, rng_seed: int,
                 mate_pool: list[dict] | None = None) -> list[dict]:
    rep = seed.get("representation") or {}
    if not rep.get("A") or not rep.get("B") or int(seed["n"]) % 2:
        return []
    seed = {**seed, "m": int(seed["n"]) // 2}
    m = int(seed["m"])
    q = int(seed["k"]) // 2
    a_mask, b_mask = _poly(rep["A"][0], m), _poly(rep["B"][0], m)
    rng = random.Random(int(rng_seed))
    children, seen_equiv = {}, set()
    attempts = 0
    modes = ("affine_combo", "triple_combo", "divisor_factor",
             "support_rewire", "cross_swap", "gcd_escape")

    parent_g = _family_key_masks(a_mask, b_mask, m)[1]
    # Work in the quotient ring for operators that edit supports.  Editing
    # raw A/B masks usually destroys divisibility by the parent GCD and is
    # therefore rejected by the exact family gate.
    quotient_a = quotient(a_mask, parent_g)
    quotient_b = quotient(b_mask, parent_g)
    # Keep escape and sparse exploration bounded.  Previously these two
    # stages could consume the whole child budget, starving the structured
    # operators below (especially cross_swap/rewire).  Reserve at least half
    # the budget for the full operator cycle.
    escape_target = max(1, int(count) // 4)
    sparse_target = max(1, int(count) // 4)
    cross_target = max(1, int(count) // 4)
    sparse_limit = min(int(count), escape_target + sparse_target)
    cross_limit = min(int(count), escape_target + sparse_target + cross_target)
    escape_attempts = 0
    while len(children) < escape_target and escape_attempts < max(500, int(count) * 250):
        escape_attempts += 1
        new_g = _random_degree_divisor(m, q, rng, parent_g)
        if new_g is None:
            continue
        na = _mul_mod(new_g, _random_mask(m, rng.randint(2, 7), rng), m)
        nb = _mul_mod(new_g, _random_mask(m, rng.randint(2, 7), rng), m)
        _add_child(children, seen_equiv, na, nb, seed, "gcd_escape", escape_attempts)

    # Sparse ideal pool: random short combinations often cancel back to valid
    # row weight, while arbitrary combinations become too dense. Every pool
    # word remains divisible by parent gcd; exact gcd gate below removes rank
    # changes.
    sparse_pool = {a_mask, b_mask}
    for _ in range(max(2_000, int(count) * 250)):
        word = _random_combo(a_mask, b_mask, m, rng, 1, 7)
        if 8 <= word.bit_count() <= 24:
            sparse_pool.add(word)
        if len(sparse_pool) >= 512:
            break
    sparse_pool = tuple(sparse_pool)
    for _ in range(max(2_000, int(count) * 400)):
        if len(children) >= sparse_limit:
            break
        attempts += 1
        if not sparse_pool:
            break
        na, nb = rng.choice(sparse_pool), rng.choice(sparse_pool)
        _add_child(children, seen_equiv, na, nb, seed, "sparse_ideal_combo", attempts)

    # Cross-seed recombination supplies low-weight A/B masks that a single
    # parent's ideal rarely contains.  Restrict mates to the same (n,k), so
    # the generated code's claimed dimension remains meaningful; exact GCD
    # and support gates still decide every child.
    mates = []
    for mate in mate_pool or ():
        if mate.get("code_id") == seed.get("code_id"):
            continue
        if int(mate.get("n", -1)) != int(seed["n"]):
            continue
        if int(mate.get("k", -1)) != int(seed["k"]):
            continue
        rep_m = mate.get("representation") or {}
        if rep_m.get("A") and rep_m.get("B"):
            mates.append((
                _poly(rep_m["A"][0], m),
                _poly(rep_m["B"][0], m),
                mate,
            ))
    cross_attempts = 0
    while mates and len(children) < cross_limit and cross_attempts < max(2_000, int(count) * 500):
        cross_attempts += 1
        mate_a, mate_b, _ = rng.choice(mates)
        choice = rng.randrange(4)
        if choice == 0:
            na, nb = a_mask, mate_b
        elif choice == 1:
            na, nb = mate_a, b_mask
        elif choice == 2:
            na, nb = b_mask, mate_a
        else:
            na, nb = mate_b, a_mask
        _add_child(children, seen_equiv, na, nb, seed, "cross_seed", cross_attempts)

    attempts = 0
    while len(children) < int(count) and attempts < max(300, int(count) * 100):
        attempts += 1
        mode = modes[attempts % len(modes)]
        if mode == "affine_combo":
            na = a_mask ^ _rotate(b_mask, rng.randrange(m), m)
            nb = b_mask ^ _rotate(a_mask, rng.randrange(m), m)
        elif mode == "triple_combo":
            na = _random_combo(a_mask, b_mask, m, rng, 3, 6)
            nb = _random_combo(a_mask, b_mask, m, rng, 3, 6)
        elif mode == "divisor_factor":
            g = _family_key_masks(a_mask, b_mask, m)[1]
            qa = _random_mask(m, rng.randint(2, 7), rng)
            qb = _random_mask(m, rng.randint(2, 7), rng)
            na, nb = _mul_mod(g, qa, m), _mul_mod(g, qb, m)
        elif mode == "support_rewire":
            qna = quotient_a ^ _rotate(quotient_b, rng.randrange(m), m)
            qnb = quotient_b ^ _rotate(quotient_a, rng.randrange(m), m)
            na, nb = _mul_mod(parent_g, qna, m), _mul_mod(parent_g, qnb, m)
        else:
            if mode == "gcd_escape":
                new_g = _random_degree_divisor(m, q, rng, parent_g)
                if new_g is None:
                    continue
                na = _mul_mod(new_g, _random_mask(m, rng.randint(2, 7), rng), m)
                nb = _mul_mod(new_g, _random_mask(m, rng.randint(2, 7), rng), m)
                _add_child(children, seen_equiv, na, nb, seed, mode, attempts)
                continue
            # Cross-swap quotient components, preserving the parent ideal.
            na = _mul_mod(parent_g,
                          quotient_a ^ _rotate(quotient_b, rng.randrange(m), m), m)
            nb = _mul_mod(parent_g,
                          quotient_b ^ _rotate(quotient_a, rng.randrange(m), m), m)
        _add_child(children, seen_equiv, na, nb, seed, mode, attempts)
    return list(children.values())[:int(count)]


def _split_of(row: dict) -> str | None:
    return row.get("split_repaired") or row.get("split")


def _curve_labels(curves: list[dict], codes: list[dict], horizon: int) -> dict[tuple[str, str], int]:
    """Build long-budget labels; missing horizon remains right-censored."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for point in curves:
        cid = point.get("code_id")
        budget = point.get("trials_per_side")
        value = point.get("best_distance")
        if cid and budget is not None and value is not None:
            grouped[(str(cid), str(point.get("side", "overall")))].append(point)
    labels: dict[tuple[str, str], int] = {}
    for code in codes:
        cid = str(code.get("code_id"))
        eligible = []
        for side in ("overall", "X", "Z"):
            points = [p for p in grouped.get((cid, side), [])
                      if int(p["trials_per_side"]) >= int(horizon)]
            if points:
                eligible.append(min(int(p["best_distance"]) for p in points))
        if eligible:
            labels[(_split_of(code) or "", cid)] = int(min(eligible))
    return labels


def _train_fixed_model(codes: list[dict], runs: list[dict], budget: int,
                       curves: list[dict] | None = None,
                       survival_horizon: int | None = None) -> tuple[object, dict, list[str]]:
    curves = curves or []
    horizon = int(survival_horizon or 0)
    long_labels = _curve_labels(curves, codes, horizon) if horizon > 0 else {}
    family_index = {x: i for i, x in enumerate(sorted({c.get("family", "unknown")
                                                        for c in codes if _split_of(c) == "train"}))}
    kind_index = {x: i for i, x in enumerate(sorted({c.get("representation_kind", "unknown")
                                                      for c in codes if _split_of(c) == "train"}))}
    by_code: dict[tuple[str, str], int] = {}
    observed_budget_counts: dict[str, int] = {}
    for run in runs:
        t, d = run.get("trials_per_side"), run.get("distance_observed")
        split = run.get("split_repaired") or run.get("split")
        if split not in ("train", "validation", "test") or not t or d is None:
            continue
        if int(t) != int(budget):
            continue
        cid = run.get("code_id")
        if cid:
            key = (split, cid)
            observed_budget_counts[split] = observed_budget_counts.get(split, 0) + 1
            # Any verified low witness refutes. Keep minimum distance at exact budget.
            by_code[key] = min(by_code.get(key, 10**9), int(d))
    if long_labels:
        by_code = long_labels
    train = [c for c in codes if _split_of(c) == "train" and ("train", str(c["code_id"])) in by_code]
    if not train:
        raise ValueError("no fixed-budget training rows")
    feature_maps, names_set = [], set()
    for code in train:
        vector, row_names = _features(code, family_index, kind_index)
        feature_maps.append(dict(zip(row_names, vector)))
        names_set.update(row_names)
    names = sorted(names_set)
    vectors = [[row.get(name, 0.0) for name in names] for row in feature_maps]
    x = np.asarray(vectors, dtype=float)
    y = np.asarray([int(code["k"]) * by_code[("train", str(code["code_id"]))] ** 2 / int(code["n"]) >= TARGET_SCORE for code in train], dtype=np.uint8)
    model = ExtraTreesClassifier(n_estimators=300, random_state=20260905,
                                 min_samples_leaf=4, max_features=0.75,
                                 class_weight="balanced_subsample", n_jobs=-1)
    model.fit(x, y)
    pred = model.predict_proba(x)[:, 1] if len(model.classes_) == 2 else np.full(len(x), float(y.mean()))
    metrics = {"budget_per_side": int(budget),
               "survival_horizon_per_side": int(horizon) if horizon else None,
               "label_policy": "long_horizon_min_witness" if long_labels else "exact_budget_min_witness",
               "censored_rows_excluded": {split: sum(1 for c in codes
                   if _split_of(c) == split and (split, str(c["code_id"])) not in by_code)
                   for split in ("train", "validation", "test")},
               "observed_exact_budget_runs": observed_budget_counts,
               "train_rows": len(train),
               "positive_rows": int(y.sum()), "positive_rate": float(y.mean()),
               "model": "ExtraTreesClassifier",
               "target": ("score_ge_1542_at_horizon" if long_labels
                           else "fixed_budget_score_ge_1542"),
               "feature_names": names}
    if len(set(y)) == 2:
        metrics.update({"train_roc_auc": float(roc_auc_score(y, pred)),
                        "train_average_precision": float(average_precision_score(y, pred))})
    for split in ("validation", "test"):
        eval_rows = [c for c in codes if _split_of(c) == split and (split, str(c["code_id"])) in by_code]
        if not eval_rows:
            continue
        eval_maps = []
        for code in eval_rows:
            values, row_names = _features(code, family_index, kind_index)
            eval_maps.append(dict(zip(row_names, values)))
        eval_x = np.asarray([[row.get(name, 0.0) for name in names] for row in eval_maps], dtype=float)
        eval_y = np.asarray([int(code["k"]) * by_code[(split, str(code["code_id"]))] ** 2 / int(code["n"]) >= TARGET_SCORE
                             for code in eval_rows], dtype=np.uint8)
        eval_pred = model.predict_proba(eval_x)[:, 1] if len(model.classes_) == 2 else np.full(len(eval_x), float(y.mean()))
        metrics[f"{split}_rows"] = len(eval_rows)
        metrics[f"{split}_positive_rows"] = int(eval_y.sum())
        if len(set(eval_y)) == 2:
            metrics[f"{split}_roc_auc"] = float(roc_auc_score(eval_y, eval_pred))
            metrics[f"{split}_average_precision"] = float(average_precision_score(eval_y, eval_pred))
    return (model, {"family_index": family_index, "kind_index": kind_index, **metrics}, names)


def _score_model_row(row: dict, model, model_meta: dict, names: list[str]) -> tuple[float, float]:
    values, row_names = _features(row, model_meta["family_index"], model_meta["kind_index"])
    vector = [dict(zip(row_names, values)).get(name, 0.0) for name in names]
    p = float(model.predict_proba(np.asarray([vector], dtype=float))[:, 1][0]) if len(model.classes_) == 2 else 0.0
    horizon = int(model_meta.get("survival_horizon_per_side") or model_meta.get("budget_per_side") or MODEL_BUDGET)
    cost = max(1, int(row.get("max_trials_per_side") or horizon))
    return p, p / math.sqrt(cost / max(1, horizon))


def _rank_seeds(queue: list[dict], codes: dict, model, model_meta: dict, names: list[str], split: str, count: int) -> list[dict]:
    def seed_stratum(row: dict) -> tuple:
        # n/k/family alone is too coarse: repaired corpora can contain thousands
        # of near-duplicates from one algebraic lineage.  Force first-pass
        # coverage across independent algebraic groups; only then consume
        # additional seeds from prolific groups.
        return (int(row["n"]) // 2, int(row["k"]), row.get("family"),
                row.get("algebraic_group") or row.get("semantic_group") or row.get("code_id"))

    family_index, kind_index = model_meta["family_index"], model_meta["kind_index"]
    rows, seen, strata = [], set(), set()
    for item in queue:
        cid = item.get("code_id")
        code = codes.get(cid)
        if not code or _split_of(code) != split or cid in seen:
            continue
        rep = code.get("representation") or {}
        if code.get("representation_kind") != "ring_genome" or not rep.get("A") or not rep.get("B"):
            continue
        p, priority = _score_model_row(code, model, model_meta, names)
        m = int(code["n"]) // 2
        stratum = seed_stratum(code)
        cost = max(1, int(item.get("max_trials_per_side") or MODEL_BUDGET))
        rows.append({**code, "fixed_budget_p": p, "fixed_budget_priority": priority,
                     "fixed_budget_model": model_meta["model"], "fixed_budget": MODEL_BUDGET,
                     "queue_observed_score": item.get("observed_score_upper"),
                     "queue_predicted_score": item.get("predicted_score_mean")})
    rows.sort(key=lambda x: (-x["fixed_budget_priority"], -x["fixed_budget_p"], x["code_id"]))
    selected = []
    # First spend one seed per coarse code geometry.  Without this pass, a
    # prolific high-scoring (n,k,family) bucket can still consume all seeds,
    # even when algebraic-group diversity is present inside that bucket.
    coarse_seen = set()
    for row in list(rows):
        coarse = (int(row["n"]) // 2, int(row["k"]), row.get("family"))
        if coarse in coarse_seen:
            continue
        selected.append(row); rows.remove(row); coarse_seen.add(coarse)
        strata.add(seed_stratum(row))
        if len(selected) >= int(count):
            return selected
    # Round-robin strata: prevents one prolific family consuming all seeds.
    while rows and len(selected) < int(count):
        added = False
        for row in list(rows):
            stratum = seed_stratum(row)
            if stratum in strata and len(strata) < int(count):
                continue
            selected.append(row); rows.remove(row); strata.add(stratum); added = True
            if len(selected) >= int(count):
                break
        if not added:
            selected.append(rows.pop(0))
    return selected


def _ideal_family(rep: dict, trials: int, seed: int, cached: dict | None = None,
                  gpu: bool = False) -> dict:
    m = int(rep["m"]); a, b = tuple(rep["A"][0]), tuple(rep["B"][0])
    modulus = (1 << m) | 1
    g = gcd(gcd(_poly(a, m), _poly(b, m)), modulus)
    q = degree(g)
    out = {"family_key": [m, hex(int(g))], "gcd_degree": q,
           "ideal_trials_requested": int(trials)}
    if cached and cached.get("witness"):
        cached_weight = int(cached.get("weight") or len(cached["witness"]))
        cached_trials = int(cached.get("trials_run") or 0)
        required = _required_distance(int(rep["n"]), int(rep["k"]))
        # Low witness permanently refutes family. Reuse without spending more.
        if cached_weight < required:
            out.update({"raw_ideal_weight": cached_weight,
                        "ideal_witness": [int(x) for x in cached["witness"]],
                        "ideal_trials_run": cached_trials,
                        "ideal_mode": "persistent_family_cache_refutation",
                        "cache_hit": True, "status": "cache_hit"})
            return out
        # Non-refuting cache is only a prefix. Extend with fresh seed namespace.
        if cached_trials >= int(trials):
            out.update({"raw_ideal_weight": cached_weight,
                        "ideal_witness": [int(x) for x in cached["witness"]],
                        "ideal_trials_run": cached_trials,
                        "ideal_mode": "persistent_family_cache_complete",
                        "cache_hit": True, "status": "cache_hit"})
            return out
        seed = int(seed) ^ 0xD1B54A32D192ED03 ^ cached_trials
        out["cache_extended_from"] = cached_trials
        out["cache_hit"] = False
    out["cache_hit"] = False
    if q != int(rep["k"]) // 2:
        out["status"] = "gcd_dimension_changed"; return out
    try:
        basis = ideal_basis(quotient(modulus, g), m, q)
    except (ValueError, ArithmeticError) as exc:
        out.update({"status": "ideal_basis_rejected", "error": str(exc)}); return out
    backend = "cpu"
    gpu_error = None
    if gpu:
        try:
            import gpu_fast
            if gpu_fast.available():
                ris = gpu_fast.classical_ris(
                    basis, trials=int(trials), seed=int(seed), pair_depth=24,
                    target=None, stop_on_target=False, threads=448)
                backend = "cuda_randomized_echelon_ris"
            else:
                raise RuntimeError("CUDA unavailable")
        except Exception as exc:
            gpu_error = str(exc)
            ris = cpp_fast.classical_ris(basis, trials=int(trials), seed=int(seed),
                                         pair_depth=24, target=None,
                                         stop_on_target=False)
    else:
        ris = cpp_fast.classical_ris(basis, trials=int(trials), seed=int(seed),
                                     pair_depth=24, target=None,
                                     stop_on_target=False)
    new_weight = ris.get("best_weight")
    new_witness = ris.get("witness")
    old_weight = int(cached.get("weight")) if cached and cached.get("weight") is not None else None
    old_trials = int(cached.get("trials_run") or 0) if cached else 0
    if old_weight is not None and (new_weight is None or old_weight <= int(new_weight)):
        best_weight, best_witness = old_weight, [int(x) for x in cached["witness"]]
    else:
        best_weight, best_witness = new_weight, new_witness
    out.update({"raw_ideal_weight": best_weight, "ideal_witness": best_witness,
                "ideal_trials_run": old_trials + int(ris.get("trials_run") or 0),
                "ideal_mode": ris.get("backend") or ris.get("mode"),
                "ideal_backend": backend, "ideal_gpu_requested": bool(gpu),
                "ideal_gpu_error": gpu_error, "ideal_seconds": ris.get("seconds")})
    out["status"] = "ideal_hit" if ris.get("witness") else "ideal_no_hit"
    return out


def _attach_ideal(row: dict, base: dict) -> dict:
    ideal = dict(base)
    ideal["family_cache_reused"] = True
    weight, witness = ideal.get("raw_ideal_weight"), ideal.get("ideal_witness")
    if weight is not None and witness is not None:
        hx, hz = build_matrices(int(row["m"]), tuple(row["A"][0]), tuple(row["B"][0]))
        check = _logical_check(witness, hz, hx)
        ideal["ideal_css_check"] = {k: v for k, v in check.items() if k != "row_rank"}
        ideal["ideal_score_upper"] = int(row["k"]) * int(weight) ** 2 / int(row["n"])
        target = _required_distance(row["n"], row["k"])
        ideal["status"] = "gcd_refuted" if check.get("ok") and int(weight) < target else "ideal_survivor"
    else:
        ideal["ideal_css_check"] = {"ok": False, "reason": "no witness"}
        ideal["status"] = "ideal_no_hit"
    return ideal


def _verified_css(row: dict, trials: int, seed: int, pair_depth: int,
                  combo_depth: int, prior_trials: int, stage: str,
                  gpu: bool = False, gpu_rank_cap: int = 0) -> dict:
    m = int(row["m"]); a, b = tuple(row["A"][0]), tuple(row["B"][0])
    hx, hz = build_matrices(m, a, b)
    target = _required_distance(row["n"], row["k"])
    backend = "cpu_cpp_ris"
    reduction = "full_rref"
    rank_cap = None
    if gpu:
        try:
            import gpu_fast
            if not gpu_fast.available():
                raise RuntimeError(f"CUDA unavailable: {gpu_fast.load_error()}")
            rank_cap = int(gpu_rank_cap) if int(gpu_rank_cap) > 0 else None
            native = gpu_fast.css_ris(
                hx, hz, trials=int(trials), seed=int(seed),
                pair_depth=int(pair_depth), target=None, stop_on_target=False,
                threads=448, batch_size=8192, rank_cap=rank_cap)
            backend = "cuda_randomized_echelon_css_ris"
            reduction = "forward_echelon_truncated" if rank_cap else "forward_echelon"
        except Exception:
            # GPU proxy is opportunistic; CPU remains fallback. Deep stage stays CPU.
            native = cpp_fast.css_ris_parallel(
                hx, hz, trials=int(trials), seed=int(seed),
                pair_depth=int(pair_depth), combo_depth=int(combo_depth),
                target=None, stop_on_target=False, threads=1)
            backend = "cpu_cpp_ris_fallback"
            reduction = "full_rref"
            rank_cap = None
    else:
        native = cpp_fast.css_ris_parallel(hx, hz, trials=int(trials), seed=int(seed),
                                           pair_depth=int(pair_depth), combo_depth=int(combo_depth),
                                           target=None, stop_on_target=False, threads=1)
    checks = {}
    for side, kernel, stabilizer in (("x", hz, hx), ("z", hx, hz)):
        item = native.get(side, {})
        witness = item.get("witness")
        check = _logical_check(witness, kernel, stabilizer) if witness else {"ok": False, "reason": "missing witness"}
        checks[side.upper()] = {k: v for k, v in check.items() if k != "row_rank"}
        item["css_check"] = checks[side.upper()]
    css_ok = bool(checks["X"].get("ok") and checks["Z"].get("ok"))
    actual = max(int(native.get("x", {}).get("trials_run") or 0), int(native.get("z", {}).get("trials_run") or 0))
    dx = native.get("dx_upper") if checks["X"].get("ok") else None
    dz = native.get("dz_upper") if checks["Z"].get("ok") else None
    d = min(dx, dz) if dx is not None and dz is not None else None
    return {"stage": stage, "trial_budget_requested_per_side": int(trials),
            "trials_per_side_actual": actual, "prior_trials_per_side": int(prior_trials),
            "cumulative_trials_per_side": int(prior_trials) + actual,
            "trial_accounting": "cumulative_compute_independent_seeds",
            "target_distance": target, "x": native.get("x"), "z": native.get("z"),
            "dx_upper": dx, "dz_upper": dz, "d_upper": d, "css_ok": css_ok,
            "survives_target": bool(css_ok and d is not None and int(d) >= target),
            "screen_refuted": bool(native.get("screen_refuted")),
            "total_sector_shots": native.get("total_sector_shots"),
            "total_seconds": native.get("total_seconds"),
            "backend": backend, "reduction": reduction, "rank_cap": rank_cap}


def _family_task(task):
    (key, rows, ideal_trials, proxy_trials, seed, pair_depth, cached, gpu_ideal,
     gpu_css, gpu_css_rank_cap) = task
    base = _ideal_family(rows[0], ideal_trials, seed, cached, gpu=gpu_ideal)
    out = []
    for row in rows:
        ideal = _attach_ideal(row, base)
        result = {**row, "candidate_id": _hash_pair(tuple(row["A"][0]), tuple(row["B"][0])),
                  "gcd_family_key": [key[0], hex(key[1])], "ideal_gate": ideal}
        if ideal.get("status") != "gcd_refuted":
            result["proxy"] = _verified_css(row, proxy_trials, seed ^ 0x5A5A5A5A,
                                             pair_depth, 2, 0, "proxy",
                                             gpu=gpu_css, gpu_rank_cap=gpu_css_rank_cap)
            result["status"] = "proxy_survivor" if result["proxy"]["survives_target"] else "proxy_refuted"
        else:
            result["status"] = "gcd_refuted"
        out.append(result)
    return out


def _parallel_family_screen(rows: list[dict], args, witness_cache: dict) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault(_family_key(row), []).append(row)
    tasks = [(key, group, int(args.ideal_trials), int(args.proxy_trials),
              int(args.seed) + i * 0x9E3779B9, int(args.pair_depth),
              witness_cache.get(_family_cache_key(key)),
              bool(getattr(args, "gpu_ideal", False)),
              bool(getattr(args, "gpu_css", False)),
              int(getattr(args, "gpu_css_rank_cap", 0)))
             for i, (key, group) in enumerate(sorted(groups.items(), key=lambda x: x[0]))]
    out = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(_family_task, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            out.extend(future.result())
            if done == 1 or done % 16 == 0 or done == len(tasks):
                print(json.dumps({"stage": "gcd_family_proxy", "families_done": done,
                                  "families_total": len(tasks), "candidates": len(out)}), flush=True)
    return sorted(out, key=lambda row: row["candidate_id"])


def _split_overlap(codes: list[dict]) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for code in codes:
        group = code.get("algebraic_group") or code.get("repaired_group") or code.get("semantic_group")
        split = _split_of(code)
        if group and split:
            groups[str(group)].add(str(split))
    return {group: splits for group, splits in groups.items() if len(splits) > 1}


def _deep_task(task):
    row, trials, seed, pair_depth = task
    prior = int((row.get("proxy") or {}).get("cumulative_trials_per_side", 0))
    deep = _verified_css(row, trials, seed, pair_depth, 3, prior, "deep")
    return {**row, "deep": deep,
            "status": "deep_survivor" if deep["survives_target"] else "deep_refuted"}


def _parallel_deep(rows: list[dict], args) -> list[dict]:
    tasks = [(row, int(args.deep_trials), int(args.seed) ^ 0xC0FFEE ^ i * 0xD1B54A35,
              int(args.deep_pair_depth)) for i, row in enumerate(rows)]
    out = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(_deep_task, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            out.append(future.result())
            if done == 1 or done % 16 == 0 or done == len(tasks):
                print(json.dumps({"stage": "deep", "done": done, "total": len(tasks)}), flush=True)
    return sorted(out, key=lambda row: row["candidate_id"])


def _candidate_doc(row: dict, out: Path, index: int) -> dict | None:
    deep = row.get("deep") or {}
    if not deep.get("css_ok") or not deep.get("x", {}).get("witness") or not deep.get("z", {}).get("witness"):
        return None
    path = out / f"candidate_{index:04d}_{row['n']}_{row['k']}_{deep['d_upper']}.json"
    doc = {"schema_version": "0.2", "name": f"v2 mutation {row['candidate_id']}",
           "code_type": "CSS", "n": int(row["n"]), "k": int(row["k"]),
           "checks": {"X": build_supports(int(row["m"]), tuple(row["A"][0]), tuple(row["B"][0]))[0],
                      "Z": build_supports(int(row["m"]), tuple(row["A"][0]), tuple(row["B"][0]))[1]},
           "distance": {"d": int(deep["d_upper"]),
                        "X": {"value": int(deep["dx_upper"]), "confidence": "upper_bound", "witness": deep["x"]["witness"]},
                        "Z": {"value": int(deep["dz_upper"]), "confidence": "upper_bound", "witness": deep["z"]["witness"]}},
           "provenance": {"parent_code_id": row["parent_code_id"], "parent_lineage_group": row.get("parent_lineage_group"),
                          "parent_split": row.get("parent_split"), "mutation_mode": row.get("mutation_mode"),
                          "gcd_family_key": row.get("gcd_family_key")},
           "regulation": {"stage_only": True, "distance_is_not_proven": True,
                          "submission_sent": False, "git_commit_performed": False}}
    # Final export gate: deep RIS output is not admissible corpus data unless
    # the emitted document independently passes exact shape/CSS/witness checks.
    structural = structural_validate(doc)
    if not structural.get("ok"):
        return None
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return {"candidate_id": row["candidate_id"], "candidate_path": str(path.resolve()),
            "n": row["n"], "k": row["k"], "d_upper": deep["d_upper"],
            "score_upper": int(row["k"]) * int(deep["d_upper"]) ** 2 / int(row["n"]),
            "seed_split": row.get("parent_split"), "parent_lineage_group": row.get("parent_lineage_group")}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    codes_list = [json.loads(x) for x in Path(args.codes).read_text(encoding="utf-8").splitlines() if x.strip()]
    codes = {x["code_id"]: x for x in codes_list}
    runs = [json.loads(x) for x in Path(args.runs).read_text(encoding="utf-8").splitlines() if x.strip()]
    queue = [json.loads(x) for x in Path(args.queue).read_text(encoding="utf-8").splitlines() if x.strip()]
    overlap = _split_overlap(codes_list)
    if overlap and not args.allow_split_overlap:
        sample = ", ".join(f"{key}:{sorted(value)}" for key, value in list(overlap.items())[:3])
        raise ValueError(f"algebraic groups cross splits ({len(overlap)}); use regroup_corpus.py first; sample={sample}")
    curves = []
    if args.curves and Path(args.curves).exists():
        curves = [json.loads(x) for x in Path(args.curves).read_text(encoding="utf-8").splitlines() if x.strip()]
    model, model_meta, names = _train_fixed_model(
        codes_list, runs, int(args.model_budget), curves=curves,
        survival_horizon=int(args.survival_horizon) if int(args.survival_horizon) > 0 else None)
    if getattr(args, "gpu_ideal", False) and int(args.workers) != 1:
        raise ValueError("--gpu-ideal requires --workers 1; one CUDA owner only")
    if getattr(args, "gpu_css", False) and int(args.workers) != 1:
        raise ValueError("--gpu-css requires --workers 1; one CUDA owner only")
    seeds = _rank_seeds(queue, codes, model, model_meta, names, args.seed_split, int(args.seeds))
    candidates = []
    for index, seed in enumerate(seeds):
        candidates.extend(_mutate_seed(seed, int(args.mutations_per_seed),
                                       int(args.seed) + index * 0x9E3779B9,
                                       mate_pool=seeds))
    # Dedup exact supports and equivalence classes globally before any RIS.
    unique, seen_actual, seen_equiv = [], set(), set()
    for row in candidates:
        actual = (tuple(row["A"][0]), tuple(row["B"][0]))
        equiv = tuple(row["equivalence_key"])
        if actual in seen_actual or equiv in seen_equiv:
            continue
        seen_actual.add(actual); seen_equiv.add(equiv); unique.append(row)
    # Mutation can alter structure/family. Recompute score on each child;
    # parent score remains provenance only.
    for row in unique:
        p, priority = _score_model_row(row, model, model_meta, names)
        row["child_survival_p"] = p
        row["child_survival_priority"] = priority
    _write_jsonl(out / "candidates.jsonl", unique)
    cache_path = Path(args.witness_cache) if args.witness_cache else None
    witness_cache = {} if args.refresh_witness_cache else _load_witness_cache(cache_path)
    proxy = _parallel_family_screen(unique, args, witness_cache)
    cache_updates = _write_witness_cache(cache_path, proxy) if cache_path is not None else 0
    _write_jsonl(out / "proxy_results.jsonl", proxy)
    survivors = [x for x in proxy if x.get("status") == "proxy_survivor"]
    survivors.sort(key=lambda x: (-(x.get("child_survival_priority") or 0),
                                  -(x.get("proxy", {}).get("d_upper") or 0), x["candidate_id"]))
    selected, per_parent, per_family = [], {}, {}
    family_cap = max(1, int(args.deep_per_gcd_family))
    for row in survivors:
        parent = row["parent_code_id"]
        family = str(row.get("gcd_family_key"))
        if per_parent.get(parent, 0) >= int(args.deep_per_parent):
            continue
        if per_family.get(family, 0) >= family_cap:
            continue
        selected.append(row)
        per_parent[parent] = per_parent.get(parent, 0) + 1
        per_family[family] = per_family.get(family, 0) + 1
        if len(selected) >= int(args.deep_count):
            break
    # Fill unused deep slots after diversity pass.
    if len(selected) < int(args.deep_count):
        for row in survivors:
            if row in selected:
                continue
            parent = row["parent_code_id"]
            if per_parent.get(parent, 0) >= int(args.deep_per_parent):
                continue
            selected.append(row); per_parent[parent] = per_parent.get(parent, 0) + 1
            if len(selected) >= int(args.deep_count):
                break
    deep = _parallel_deep(selected, args)
    _write_jsonl(out / "deep_results.jsonl", deep)
    final = []
    for index, row in enumerate(deep, 1):
        if row.get("status") == "deep_survivor":
            doc = _candidate_doc(row, out, index)
            if doc: final.append(doc)
    report = {"schema_version": "2.0", "kind": "canonical_fixed_budget_ml_mutation_campaign",
              "target_score": TARGET_SCORE, "seed_split": args.seed_split,
              "seed_count": len(seeds), "candidate_count_generated": len(candidates),
              "candidate_count_unique_canonical": len(unique), "equivalence_rejected": len(candidates) - len(unique),
              "gcd_family_count": len({_family_key(x) for x in unique}),
              "proxy_survivors": len(survivors), "deep_selected": len(selected), "deep_survivors": len(final),
              "mutation_modes": {mode: sum(x.get("mutation_mode") == mode for x in unique) for mode in sorted({x.get("mutation_mode") for x in unique})},
              "fixed_budget_model": model_meta, "search": {"ideal_trials": args.ideal_trials, "proxy_trials": args.proxy_trials,
                  "deep_trials": args.deep_trials, "model_budget": args.model_budget, "workers": args.workers,
                  "survival_horizon": int(args.survival_horizon), "deep_per_gcd_family": int(args.deep_per_gcd_family)},
              "gpu_ideal": bool(getattr(args, "gpu_ideal", False)),
              "gpu_css": bool(getattr(args, "gpu_css", False)),
              "gpu_css_rank_cap": int(getattr(args, "gpu_css_rank_cap", 0)),
              "split_integrity": {"overlap_groups": len(overlap), "allow_overlap": bool(args.allow_split_overlap)},
              "witness_cache": {"path": str(cache_path.resolve()) if cache_path else None,
                                "loaded_families": len(witness_cache), "updated_families": cache_updates,
                                "cache_hits": sum(bool((x.get("ideal_gate") or {}).get("cache_hit")) for x in proxy)},
              "final": final, "regulation": {"stage_only": True, "distance_is_not_proven": True,
                  "trial_accounting": "actual_per_side_and_cumulative_compute_recorded", "submission_sent": False,
                  "git_commit_performed": False}}
    (out / "campaign.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "model.json").write_text(json.dumps(model_meta, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--queue", type=Path, default=Path("results/corpus_repaired_algebraic/search_queue.jsonl"))
    p.add_argument("--codes", type=Path, default=Path("results/corpus_repaired_algebraic/codes.jsonl"))
    p.add_argument("--runs", type=Path, default=Path("results/corpus_repaired_algebraic/runs.jsonl"))
    p.add_argument("--curves", type=Path, default=Path("results/corpus_repaired_algebraic/curves.jsonl"))
    p.add_argument("--out", type=Path, default=Path("results/ml_mutation_search_v2_train"))
    p.add_argument("--seed-split", choices=("train", "validation", "test"), default="train")
    p.add_argument("--seeds", type=int, default=16)
    p.add_argument("--mutations-per-seed", type=int, default=48)
    p.add_argument("--model-budget", type=int, default=MODEL_BUDGET)
    p.add_argument("--survival-horizon", type=int, default=DEFAULT_SURVIVAL_HORIZON,
                   help="per-side curve horizon for long-budget labels; 0=exact-budget model")
    p.add_argument("--ideal-trials", type=int, default=256)
    p.add_argument("--proxy-trials", type=int, default=256)
    p.add_argument("--deep-trials", type=int, default=4096)
    p.add_argument("--deep-count", type=int, default=32)
    p.add_argument("--deep-per-parent", type=int, default=4)
    p.add_argument("--deep-per-gcd-family", type=int, default=2)
    p.add_argument("--pair-depth", type=int, default=24)
    p.add_argument("--deep-pair-depth", type=int, default=32)
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    p.add_argument("--gpu-ideal", action="store_true",
                   help="route ideal-family RIS to CUDA; requires --workers 1")
    p.add_argument("--gpu-css", action="store_true",
                   help="route proxy CSS RIS to CUDA; requires --workers 1")
    p.add_argument("--gpu-css-rank-cap", type=int, default=64,
                   help="GPU proxy rank cap; 0=full forward echelon")
    p.add_argument("--allow-split-overlap", action="store_true",
                   help="allow algebraic families crossing train/validation/test")
    p.add_argument("--witness-cache", type=Path, default=Path("results/ml_mutation_witness_cache.json"))
    p.add_argument("--refresh-witness-cache", action="store_true")
    args = p.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
