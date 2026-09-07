from __future__ import annotations

"""Row-truncated cyclic-parent search for ``[[682,132,d]]``.

The strict q=66 generalized-bicycle family repeatedly exposes the same
sub-90 logicals.  This branch starts with a q<66 cyclic parent, whose full
checks have rank >275, and samples rank-275 subspaces of each check rowspace.
CSS commutation is inherited exactly because every retained row comes from a
commuting parent.  The row subspaces are randomized genomes, so cyclic
translation no longer identifies the whole candidate.

Distance is a randomized RIS upper bound.  This file is stage-only: it never
commits, submits, or labels a clean screen as an exact-distance proof.
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
from generalized_bicycle import build_matrices, common_gcd_degree
from gf2_factor import degree, factor_xm_plus_one, mul
from hyper_validator import structural_validate


M, N, TARGET_K, TARGET_RANK = 341, 682, 132, 275
CHECK_CAP = 32
FACTORS = tuple(factor_xm_plus_one(M))
DEGREES = tuple(degree(factor) for factor in FACTORS)


def _support(poly: int) -> tuple[int, ...]:
    return tuple(index for index in range(M) if (int(poly) >> index) & 1)


def _factor_subsets(q: int, *, max_weight: int, limit: int,
                    seed: int) -> list[dict]:
    """Find diverse low-support divisors of exact degree q.

    Exhaustive enumeration is cheap for the first sparse tail but can become
    large for q near 66.  Reservoir sampling keeps the caller bounded while
    retaining the lowest observed divisor weights and multiple factor shapes.
    """
    indices_by_degree = {}
    for index, value in enumerate(DEGREES):
        indices_by_degree.setdefault(int(value), []).append(index)
    choices = [(d, tuple(values)) for d, values in sorted(indices_by_degree.items())]
    rng = random.Random(int(seed))
    found = []
    seen = set()

    def visit(position: int, remaining: int, subset: tuple[int, ...], divisor: int):
        if position == len(choices):
            if remaining != 0 or 0 not in subset:
                return
            weight = int(divisor).bit_count()
            if weight > int(max_weight):
                return
            key = tuple(sorted(subset))
            if key in seen:
                return
            seen.add(key)
            found.append({"factor_indices": list(key), "divisor": int(divisor),
                          "divisor_weight": weight, "degree": int(q)})
            return
        d, options = choices[position]
        max_take = min(len(options), remaining // d)
        for take in range(max_take + 1):
            for chosen in itertools.combinations(options, take):
                value = int(divisor)
                for index in chosen:
                    value = mul(value, FACTORS[index])
                visit(position + 1, remaining - take * d,
                      (*subset, *chosen), value)

    visit(0, int(q), (), 1)
    rng.shuffle(found)
    found.sort(key=lambda row: (row["divisor_weight"], row["factor_indices"]))
    # Distance search needs legal sparse shells first: a q=61 parent with a
    # 12-weight divisor can yield 12+12 checks, while a shape-first quota can
    # silently spend the entire bank on 18-weight divisors (already over cap).
    # Retain the lightest roots, with deterministic seed tie-breaking.
    found.sort(key=lambda row: (row["divisor_weight"],
                                hashlib.sha256(
                                    f"{seed}:{row['factor_indices']}".encode()).hexdigest(),
                                row["factor_indices"]))
    return found[:int(limit)]


def _basis_rank(rows: list[int]) -> int:
    basis = {}
    rank = 0
    for raw in rows:
        value = int(raw)
        while value:
            pivot = value.bit_length() - 1
            prior = basis.get(pivot)
            if prior is None:
                basis[pivot] = value
                rank += 1
                break
            value ^= prior
    return rank


def _row_ints(matrix: np.ndarray) -> list[int]:
    return [sum(int(value) << index for index, value in enumerate(row)
                if int(value)) for row in np.asarray(matrix, dtype=np.uint8)]


def _select_rows(matrix: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, dict]:
    rows = _row_ints(matrix)
    rng = random.Random(int(seed))
    order = list(range(len(rows)))
    rng.shuffle(order)
    basis = {}
    selected = []
    for index in order:
        value = rows[index]
        while value:
            pivot = value.bit_length() - 1
            prior = basis.get(pivot)
            if prior is None:
                basis[pivot] = value
                selected.append(index)
                break
            value ^= prior
        if len(selected) == int(count):
            break
    if len(selected) != int(count):
        raise ArithmeticError("parent rowspace cannot supply target rank")
    selected.sort()
    return matrix[np.asarray(selected, dtype=np.int64)], {
        "seed": int(seed), "source_rows": int(matrix.shape[0]),
        "selected_rows": len(selected), "selected_indices": selected,
        "rank": _basis_rank([rows[index] for index in selected]),
    }


def _mine_words(parent: dict, *, iterations: int, max_words: int,
                min_weight: int, max_weight: int, seed: int) -> list[tuple[int, ...]]:
    base = _support(parent["divisor"])
    words = cpp_fast.cyclic_ideal_mine(
        [base], m=M, iterations=int(iterations), min_weight=int(min_weight),
        max_weight=int(max_weight), seed=int(seed), max_words=int(max_words))
    words += cpp_fast.cyclic_ideal_hillclimb(
        base, m=M, restarts=32, steps=1024, min_weight=int(min_weight),
        max_weight=int(max_weight), seed=int(seed) ^ 0x9E3779B9,
        max_words=int(max_words))
    # The divisor itself is a guaranteed exact-q word.  The native miner is
    # stochastic and may return no shell when the ideal's short spectrum is
    # sparse, so retain this algebraic seed explicitly.
    if int(min_weight) <= len(base) <= int(max_weight):
        words.append(base)
    unique = {tuple(sorted(int(value) for value in word)) for word in words}
    return sorted(unique, key=lambda word: (len(word), word))


def _hash_checks(hx: np.ndarray, hz: np.ndarray) -> str:
    payload = np.concatenate([hx, hz], axis=0).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _candidate_task(task):
    parent, pair_index, a, b, seed, trials, threads = task
    hx_full, hz_full = build_matrices(M, a, b)
    if int(cpp_fast.gf2_rank(hx_full)) < TARGET_RANK:
        return None
    if int(cpp_fast.gf2_rank(hz_full)) < TARGET_RANK:
        return None
    hx, sx = _select_rows(hx_full, TARGET_RANK, int(seed) ^ 0xA5A5A5A5)
    hz, sz = _select_rows(hz_full, TARGET_RANK, int(seed) ^ 0x5A5A5A5A)
    if int(np.count_nonzero((hx @ hz.T) & 1)):
        raise ArithmeticError("row selection violated inherited commutation")
    result = cpp_fast.css_ris_parallel(
        hx, hz, trials=int(trials), seed=int(seed), pair_depth=24,
        combo_depth=2, target=90, stop_on_target=True, threads=int(threads))
    checks = {
        "X": [np.flatnonzero(row).astype(int).tolist() for row in hx],
        "Z": [np.flatnonzero(row).astype(int).tolist() for row in hz],
    }
    out = {
        "n": N, "k": TARGET_K, "A": [list(a)], "B": [list(b)],
        "parent": parent, "pair_index": int(pair_index),
        "semantic_hash": _hash_checks(hx, hz), "checks": checks,
        "row_selection": {"X": sx, "Z": sz},
        "screen": result,
    }
    return out


def _doc(row: dict, stage: str) -> dict | None:
    result = row["screen"]
    dx, dz = result.get("dx_upper"), result.get("dz_upper")
    wx, wz = result["x"].get("witness"), result["z"].get("witness")
    if None in (dx, dz, wx, wz) or not wx or not wz:
        return None
    d = min(int(dx), int(dz))
    doc = {
        "schema_version": "0.1",
        "name": f"[[{N},{TARGET_K},d<={d}]] q<66 row-truncated cyclic parent",
        "code_type": "CSS", "n": N, "k": TARGET_K,
        "checks": row["checks"],
        "distance": {
            "d": d,
            "X": {"value": int(dx), "confidence": "upper_bound",
                  "witness": list(wx)},
            "Z": {"value": int(dz), "confidence": "upper_bound",
                  "witness": list(wz)},
        },
        "family": "row-truncated-generalized-bicycle",
        "provenance": {
            "authors": ["stage-only autonomous research"],
            "origin": "gb_m341_k132_rowtruncate",
            "model": "GPT-5 Codex + phylogenetic row-subspace search",
            "construction": ("X/Z checks are independent rank-275 row subsets "
                             "of a commuting q<66 cyclic GB parent"),
            "parent": row["parent"], "pair": {"A": row["A"], "B": row["B"]},
            "row_selection": row["row_selection"],
            "screen": {key: value for key, value in result.items()
                       if key not in ("x", "z")},
            "notes": f"stage={stage}; randomized witness-backed upper bound",
        },
        "search": {"semantic_hash": row["semantic_hash"],
                    "pair_index": row["pair_index"]},
        "regulation": {
            "stage_only": True, "submission_sent": False,
            "git_commit_performed": False, "official_gate_status": "not_run",
            "board_claim_allowed": False, "distance_is_not_proven": True,
            "literature_novelty": "unverified",
        },
    }
    structural = structural_validate(doc)
    if not structural["ok"]:
        return None
    return doc


def run(args):
    if not cpp_fast.available():
        raise RuntimeError(cpp_fast.accelerator_info())
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parents = []
    for q in (61, 56, 51):
        parents.extend(_factor_subsets(q, max_weight=args.divisor_weight_max,
                                       limit=args.parents_per_q,
                                       seed=args.seed + q))
    # Avoid identical divisor roots while retaining q strata.
    unique = {}
    for parent in parents:
        unique[(parent["degree"], tuple(parent["factor_indices"]))] = parent
    parents = list(unique.values())
    print(json.dumps({"stage": "parents", "count": len(parents),
                      "degrees": sorted({p["degree"] for p in parents}),
                      "weights": sorted({p["divisor_weight"] for p in parents})}),
          flush=True)

    tasks = []
    pair_meta = []
    for pindex, parent in enumerate(parents):
        words = _mine_words(
            parent, iterations=args.mine_iterations, max_words=args.max_words,
            min_weight=args.min_word, max_weight=args.max_word,
            seed=args.seed + 100003 * pindex)
        if len(words) < 2:
            continue
        # Pair only exact-parent words; their GB parent rank is determined by q.
        # Multiple pairs per root sample independent check geometry.
        rng = random.Random(args.seed + 700001 * pindex)
        pairs = []
        for _ in range(args.pairs_per_parent * 80):
            a, b = rng.sample(words, 2)
            shift = rng.randrange(M)
            b = tuple(sorted((int(x) + shift) % M for x in b))
            if len(a) + len(b) > CHECK_CAP or len(a) + len(b) < args.min_check_weight:
                continue
            if common_gcd_degree(M, a, b) != int(parent["degree"]):
                continue
            key = (tuple(a), tuple(b))
            if key in pairs:
                continue
            pairs.append(key)
            if len(pairs) >= args.pairs_per_parent:
                break
        for pair_index, (a, b) in enumerate(pairs):
            pair_meta.append((parent, pair_index, a, b))
            tasks.append((parent, pair_index, a, b,
                          args.seed + 0x9E3779B9 * len(tasks),
                          args.proxy_trials, args.threads))
    print(json.dumps({"stage": "pairs", "tasks": len(tasks)}), flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(_candidate_task, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            row = future.result()
            if row is not None:
                rows.append(row)
            if done == 1 or done % 32 == 0 or done == len(futures):
                print(json.dumps({"stage": "proxy", "done": done,
                                  "total": len(futures), "usable": len(rows)}),
                      flush=True)
    rows.sort(key=lambda row: (int(row["screen"].get("d_upper") or -1),
                               int(row["screen"].get("dx_upper") or -1),
                               int(row["screen"].get("dz_upper") or -1)),
              reverse=True)
    final = []
    for index, row in enumerate(rows[:args.final_count], 1):
        doc = _doc(row, "proxy_final")
        if doc is None:
            continue
        path = out / f"candidate_{index}_{N}_{TARGET_K}_{doc['distance']['d']}.json"
        path.write_text(json.dumps(doc, indent=2) + "\n")
        final.append({"path": str(path.resolve()), "d": doc["distance"]["d"],
                      "dx": doc["distance"]["X"]["value"],
                      "dz": doc["distance"]["Z"]["value"],
                      "score": TARGET_K * doc["distance"]["d"] ** 2 / N,
                      "parent": row["parent"], "A": row["A"], "B": row["B"]})
    report = {
        "schema_version": "1.0", "kind": "gb_m341_k132_rowtruncate_campaign",
        "target": {"n": N, "k": TARGET_K, "required_d_to_beat_682_182_76": 90,
                    "check_weight_cap": CHECK_CAP},
        "search": {"parents": len(parents), "tasks": len(tasks),
                   "usable": len(rows), "proxy_trials_per_side": args.proxy_trials,
                   "final_count": args.final_count,
                   "seconds_wall": time.perf_counter() - started},
        "final": final, "top": rows[:min(64, len(rows))],
        "regulation": {"stage_only": True, "submission_sent": False,
                       "git_commit_performed": False, "official_gate_run": False,
                       "board_claim_allowed": False,
                       "distance_is_randomized_upper_bound": True},
    }
    (out / "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/gb_m341_k132_rowtruncate01"))
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--parents-per-q", type=int, default=2)
    parser.add_argument("--divisor-weight-max", type=int, default=21)
    parser.add_argument("--mine-iterations", type=int, default=100000)
    parser.add_argument("--max-words", type=int, default=512)
    parser.add_argument("--min-word", type=int, default=10)
    parser.add_argument("--max-word", type=int, default=16)
    parser.add_argument("--pairs-per-parent", type=int, default=16)
    parser.add_argument("--min-check-weight", type=int, default=24)
    parser.add_argument("--proxy-trials", type=int, default=20000)
    parser.add_argument("--final-count", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"out": str(args.out), "search": report["search"],
                      "final": report["final"]}, indent=2))


if __name__ == "__main__":
    main()
