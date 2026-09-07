from __future__ import annotations

"""Efficient ML-seed mutation -> GCD gate -> staged RIS campaign.

All distances are randomized witness upper bounds.  The GCD gate runs before
CSS RIS; generated children never enter model training, avoiding lineage leak.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random

import cpp_fast
from generalized_bicycle import build_matrices, build_supports, common_gcd_degree
from gf2_factor import degree, gcd, quotient
from hyper_validator import _logical_check
from ideal_x_logical_search import ideal_basis


TARGET_SCORE = 1542.0


def _poly(values: tuple[int, ...], m: int) -> int:
    out = 0
    for value in values:
        out ^= 1 << (int(value) % int(m))
    return out


def _rotate(mask: int, shift: int, m: int) -> int:
    shift %= int(m)
    ring = (1 << int(m)) - 1
    if not shift:
        return int(mask) & ring
    return ((int(mask) << shift) | (int(mask) >> (m - shift))) & ring


def _support(mask: int) -> tuple[int, ...]:
    out = []
    while mask:
        bit = mask & -mask
        out.append(bit.bit_length() - 1)
        mask ^= bit
    return tuple(out)


def _hash_pair(a: tuple[int, ...], b: tuple[int, ...]) -> str:
    return hashlib.sha256(json.dumps([list(a), list(b)], separators=(",", ":")).encode()).hexdigest()


def _required_distance(n: int, k: int) -> int:
    return int((TARGET_SCORE * int(n) / int(k)) ** 0.5) + 1


def _add_child(children: dict[tuple[tuple[int, ...], tuple[int, ...]], dict],
               a_mask: int, b_mask: int, seed: dict, m: int, mode: str) -> None:
    a = _support(a_mask)
    b = _support(b_mask)
    if not a or not b or len(a) + len(b) > 32 or len(a) + len(b) < 20:
        return
    rep = seed.get("representation") or {}
    base_a = tuple(rep.get("A", [[]])[0])
    base_b = tuple(rep.get("B", [[]])[0])
    if (a, b) == (base_a, base_b):
        return
    q = int(seed["k"]) // 2
    if common_gcd_degree(m, a, b) != q:
        return
    key = (a, b)
    children.setdefault(key, {
        "A": [list(a)], "B": [list(b)], "n": int(seed["n"]), "k": int(seed["k"]),
        "m": int(m), "parent_code_id": seed["code_id"],
        "parent_semantic_hash": seed.get("semantic_hash"),
        "parent_split": seed.get("split_repaired"), "mutation_mode": mode,
    })


def _mutate_seed(seed: dict, count: int, rng_seed: int) -> list[dict]:
    rep = seed.get("representation") or {}
    if not rep.get("A") or not rep.get("B") or int(seed["n"]) % 2:
        return []
    m = int(seed["n"]) // 2
    base_a = tuple(int(x) for x in rep["A"][0])
    base_b = tuple(int(x) for x in rep["B"][0])
    a_mask, b_mask = _poly(base_a, m), _poly(base_b, m)
    rng = random.Random(int(rng_seed))
    children: dict[tuple[tuple[int, ...], tuple[int, ...]], dict] = {}

    # Relative rotations are cheap exploration; XOR combinations stay in the
    # parent ideal, so exact common-GCD filtering is the only algebraic gate.
    shifts = list(range(m))
    rng.shuffle(shifts)
    for shift in shifts[:min(m, max(8, count))]:
        _add_child(children, _rotate(a_mask, shift, m), _rotate(b_mask, rng.randrange(m), m), seed, m, "relative_rotation")
        _add_child(children, a_mask ^ _rotate(b_mask, shift, m), b_mask, seed, m, "a_xor_shift_b")
        _add_child(children, a_mask, b_mask ^ _rotate(a_mask, shift, m), seed, m, "b_xor_shift_a")
        if len(children) >= int(count):
            break

    attempts = 0
    while len(children) < int(count) and attempts < max(64, int(count) * 12):
        attempts += 1
        left = a_mask if rng.randrange(2) == 0 else b_mask
        right = a_mask if rng.randrange(2) == 0 else b_mask
        na = left ^ _rotate(right, rng.randrange(m), m)
        left = a_mask if rng.randrange(2) == 0 else b_mask
        right = a_mask if rng.randrange(2) == 0 else b_mask
        nb = left ^ _rotate(right, rng.randrange(m), m)
        _add_child(children, na, nb, seed, m, "joint_xor_shift")
    return list(children.values())[:int(count)]


def _candidate_hash(row: dict) -> str:
    return _hash_pair(tuple(row["A"][0]), tuple(row["B"][0]))


def _ideal_gate(row: dict, trials: int, seed: int) -> dict:
    m = int(row["m"])
    a, b = tuple(row["A"][0]), tuple(row["B"][0])
    modulus = (1 << m) | 1
    g = gcd(gcd(_poly(a, m), _poly(b, m)), modulus)
    q = degree(g)
    result = {"gcd_degree": q, "ideal_trials": int(trials)}
    if q != int(row["k"]) // 2:
        result["status"] = "gcd_dimension_changed"
        return result
    h = quotient(modulus, g)
    try:
        basis = ideal_basis(h, m, q)
    except (ValueError, ArithmeticError) as exc:
        result.update({"status": "ideal_basis_rejected", "error": str(exc)})
        return result
    target = _required_distance(row["n"], row["k"])
    ris = cpp_fast.classical_ris(basis, trials=int(trials), seed=int(seed), pair_depth=12,
                                 target=target, stop_on_target=True)
    weight = ris.get("best_weight")
    result.update({"raw_ideal_weight": weight, "ideal_trials_run": ris.get("trials_run"),
                   "ideal_witness": ris.get("witness"), "ideal_stopped_early": ris.get("stopped_early")})
    if weight is None:
        result["status"] = "ideal_no_hit"
        return result
    hx, hz = build_matrices(m, a, b)
    check = _logical_check(ris["witness"], hz, hx)
    result["ideal_css_check"] = {key: value for key, value in check.items() if key != "row_rank"}
    result["ideal_score_upper"] = int(row["k"]) * int(weight) ** 2 / int(row["n"])
    result["status"] = "gcd_refuted" if check.get("ok") and int(weight) < target else "ideal_survivor"
    return result


def _css_screen(row: dict, trials: int, seed: int, threads: int, pair_depth: int, combo_depth: int, stage: str) -> dict:
    m = int(row["m"])
    a, b = tuple(row["A"][0]), tuple(row["B"][0])
    hx, hz = build_matrices(m, a, b)
    target = _required_distance(row["n"], row["k"])
    result = cpp_fast.css_ris_parallel(hx, hz, trials=int(trials), seed=int(seed),
                                       pair_depth=int(pair_depth), combo_depth=int(combo_depth),
                                       target=target, stop_on_target=True, threads=int(threads))
    d = result.get("d_upper")
    return {"stage": stage, "d_upper": d, "dx_upper": result.get("dx_upper"),
            "dz_upper": result.get("dz_upper"), "x": result.get("x"), "z": result.get("z"),
            "trials_per_side": int(trials), "target_distance": target,
            "stopped_early": bool(result.get("x", {}).get("stopped_early") or
                                   result.get("z", {}).get("stopped_early")),
            "survives_target": bool(d is not None and int(d) >= target and not result.get("screen_refuted"))}


def _screen_task(task: tuple[dict, int, int, int, int, int]) -> dict:
    row, ideal_trials, proxy_trials, seed, threads, pair_depth = task
    ideal = _ideal_gate(row, ideal_trials, seed ^ 0xA5A5A5A5)
    out = {**row, "candidate_id": _candidate_hash(row), "ideal_gate": ideal}
    if ideal.get("status") == "gcd_refuted":
        out["status"] = "gcd_refuted"
        return out
    out["proxy"] = _css_screen(row, proxy_trials, seed ^ 0x5A5A5A5A, threads, pair_depth, 2, "proxy")
    out["status"] = "proxy_survivor" if out["proxy"]["survives_target"] else "proxy_refuted"
    return out


def _deep_task(task: tuple[dict, int, int, int, int]) -> dict:
    row, trials, seed, threads, pair_depth = task
    out = {**row, "deep": _css_screen(row, trials, seed, threads, pair_depth, 3, "deep")}
    out["status"] = "deep_survivor" if out["deep"]["survives_target"] else "deep_refuted"
    return out


def _parallel_screen(rows: list[dict], *, ideal_trials: int, proxy_trials: int,
                     seed: int, workers: int, threads: int, pair_depth: int) -> list[dict]:
    tasks = [(row, int(ideal_trials), int(proxy_trials), int(seed) + i * 0x9E3779B9,
              int(threads), int(pair_depth)) for i, row in enumerate(rows)]
    out = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(_screen_task, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            row = future.result()
            # Candidate hash uniquely identifies the task; preserve stable order
            # after collection below.
            out[done - 1] = row
            if done == 1 or done % 32 == 0 or done == len(tasks):
                print(json.dumps({"stage": "proxy", "done": done, "total": len(tasks)}), flush=True)
    return sorted((row for row in out if row is not None), key=lambda row: row["candidate_id"])


def _parallel_deep(rows: list[dict], *, trials: int, seed: int, workers: int,
                   threads: int, pair_depth: int) -> list[dict]:
    tasks = [(row, int(trials), int(seed) + i * 0xD1B54A35, int(threads), int(pair_depth))
             for i, row in enumerate(rows)]
    out = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(_deep_task, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            out.append(future.result())
            if done == 1 or done % 16 == 0 or done == len(tasks):
                print(json.dumps({"stage": "deep", "done": done, "total": len(tasks)}), flush=True)
    return sorted(out, key=lambda row: row["candidate_id"])


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _candidate_doc(row: dict) -> dict | None:
    deep = row.get("deep", {})
    dx, dz = deep.get("dx_upper"), deep.get("dz_upper")
    wx = (deep.get("x") or {}).get("witness")
    wz = (deep.get("z") or {}).get("witness")
    if dx is None or dz is None or not wx or not wz:
        return None
    m = int(row["m"])
    hx, hz = build_supports(m, tuple(row["A"][0]), tuple(row["B"][0]))
    d = min(int(dx), int(dz))
    return {
        "schema_version": "0.1", "name": f"ML mutation {row['candidate_id']}",
        "code_type": "CSS", "n": int(row["n"]), "k": int(row["k"]),
        "checks": {"X": hx, "Z": hz},
        "distance": {"d": d, "X": {"value": int(dx), "confidence": "upper_bound", "witness": wx},
                     "Z": {"value": int(dz), "confidence": "upper_bound", "witness": wz}},
        "provenance": {"parent_code_id": row["parent_code_id"], "parent_semantic_hash": row.get("parent_semantic_hash"),
                        "mutation_mode": row.get("mutation_mode"), "gcd_degree": row["ideal_gate"].get("gcd_degree")},
        "regulation": {"stage_only": True, "submission_sent": False,
                        "git_commit_performed": False, "distance_is_not_proven": True},
    }


def run(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    queue = [json.loads(line) for line in Path(args.queue).read_text(encoding="utf-8").splitlines() if line.strip()]
    codes = {row["code_id"]: row for row in (json.loads(line) for line in Path(args.codes).read_text(encoding="utf-8").splitlines() if line.strip())}
    seeds = []
    seen_parents = set()
    for item in queue:
        code = codes.get(item["code_id"])
        if not code or code.get("split_repaired") != "train" or code["code_id"] in seen_parents:
            continue
        rep = code.get("representation") or {}
        if rep.get("A") and rep.get("B") and code.get("representation_kind") == "ring_genome":
            seeds.append({**code, "required_distance_for_1542": item.get("required_distance_for_1542")})
            seen_parents.add(code["code_id"])
        if len(seeds) >= int(args.seeds):
            break
    candidates = []
    for index, seed in enumerate(seeds):
        candidates.extend(_mutate_seed(seed, int(args.mutations_per_seed), int(args.seed) + index * 0x9E3779B9))
    # Stable fresh-child split. Children are never reused as ML training rows.
    for row in candidates:
        row["fresh_lineage_holdout"] = int(row["candidate_id"][-2:], 16) % 5 == 0 if row.get("candidate_id") else False
    # candidate_id is assigned after generation; do it now.
    for row in candidates:
        row["candidate_id"] = _candidate_hash(row)
        row["fresh_lineage_holdout"] = int(row["candidate_id"][-2:], 16) % 5 == 0
    candidates = list({row["candidate_id"]: row for row in candidates}.values())
    _write_jsonl(out / "candidates.jsonl", candidates)
    proxy = _parallel_screen(candidates, ideal_trials=args.ideal_trials, proxy_trials=args.proxy_trials,
                             seed=args.seed, workers=args.workers, threads=args.threads, pair_depth=args.pair_depth)
    _write_jsonl(out / "proxy_results.jsonl", proxy)
    survivors = [row for row in proxy if row["status"] == "proxy_survivor"]
    survivors.sort(key=lambda row: (row.get("fresh_lineage_holdout", False),
                                    -(row.get("proxy", {}).get("d_upper") or 0)))
    selected = []
    per_parent = {}
    for row in survivors:
        parent = row["parent_code_id"]
        if per_parent.get(parent, 0) >= int(args.deep_per_parent):
            continue
        per_parent[parent] = per_parent.get(parent, 0) + 1
        selected.append(row)
        if len(selected) >= int(args.deep_count):
            break
    deep = _parallel_deep(selected, trials=args.deep_trials, seed=args.seed ^ 0xC0FFEE,
                          workers=args.workers, threads=args.threads, pair_depth=args.deep_pair_depth)
    _write_jsonl(out / "deep_results.jsonl", deep)
    final = []
    for index, row in enumerate(deep, 1):
        if row["status"] != "deep_survivor":
            continue
        doc = _candidate_doc(row)
        if doc is None:
            continue
        path = out / f"candidate_{index:04d}_{row['n']}_{row['k']}_{row['deep']['d_upper']}.json"
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        final.append({"candidate_id": row["candidate_id"], "candidate_path": str(path.resolve()),
                      "n": row["n"], "k": row["k"], "d_upper": row["deep"]["d_upper"],
                      "score_upper": int(row["k"]) * int(row["deep"]["d_upper"]) ** 2 / int(row["n"]),
                      "fresh_lineage_holdout": row["fresh_lineage_holdout"]})
    report = {
        "schema_version": "1.0", "kind": "ml_seed_mutation_staged_ris_campaign",
        "target_score": TARGET_SCORE, "seed_count": len(seeds), "candidate_count": len(candidates),
        "proxy_survivors": len(survivors), "deep_selected": len(selected), "deep_survivors": len(final),
        "fresh_lineage_holdout_proxy": sum(row.get("fresh_lineage_holdout", False) for row in survivors),
        "fresh_lineage_holdout_deep": sum(row.get("fresh_lineage_holdout", False) for row in deep),
        "search": {"ideal_trials": args.ideal_trials, "proxy_trials": args.proxy_trials,
                   "deep_trials": args.deep_trials, "workers": args.workers, "threads": args.threads,
                   "pair_depth": args.pair_depth, "deep_pair_depth": args.deep_pair_depth},
        "final": final,
        "regulation": {"stage_only": True, "distance_is_not_proven": True,
                        "submission_sent": False, "git_commit_performed": False},
    }
    (out / "campaign.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=Path("results/corpus_repaired/search_queue.jsonl"))
    parser.add_argument("--codes", type=Path, default=Path("results/corpus_repaired/codes.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/ml_mutation_search_01"))
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--mutations-per-seed", type=int, default=32)
    parser.add_argument("--ideal-trials", type=int, default=128)
    parser.add_argument("--proxy-trials", type=int, default=128)
    parser.add_argument("--deep-trials", type=int, default=4096)
    parser.add_argument("--deep-count", type=int, default=32)
    parser.add_argument("--deep-per-parent", type=int, default=4)
    parser.add_argument("--pair-depth", type=int, default=12)
    parser.add_argument("--deep-pair-depth", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
