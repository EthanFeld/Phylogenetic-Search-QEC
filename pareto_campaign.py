from __future__ import annotations

"""Parallel, stage-only search campaign for observed QLDPC Pareto points.

No submission/network side effects.  Every retained distance is an RIS upper
bound with explicit X/Z witnesses; official verification remains a final gate.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

import cpp_fast
from distance_sketch import fast_refute_supports
from inverse_design_core import (
    TargetSpec,
    build_product_checks,
    candidate_groups_of_order,
    design_from_specs,
    css_specs,
)
from phylogenetic_search import compact_history


DEFAULT_TARGETS = ((150, 30), (180, 36), (200, 40), (220, 44),
                   (250, 50), (300, 60), (350, 70))


def _supports(matrix):
    return [np.flatnonzero(row).astype(int).tolist() for row in matrix]


def ml_group_indices(plan_path, targets, max_check_weight, limit=8,
                     status="missing"):
    """Select group branches from the persisted ML coverage plan.

    ML ranks coverage gaps only; it never certifies a candidate.  Keeping this
    selector here makes campaign reproduction explicit and prevents a stale
    hand-copied group list from silently steering a run.
    """
    path = Path(plan_path)
    plan = json.loads(path.read_text())
    if status == "missing":
        rows = plan.get("recommended_missing_branches", [])
    elif status == "seeded":
        rows = plan.get("recommended_seeded_branches", [])
    else:
        rows = (plan.get("recommended_missing_branches", []) +
                plan.get("recommended_seeded_branches", []))
    wanted = {(int(n), int(k)) for n, k in targets}
    selected = []
    seen = set()
    for row in rows:
        key = (int(row.get("n", -1)), int(row.get("k", -1)))
        if key not in wanted or int(row.get("max_check_weight", -1)) != int(max_check_weight):
            continue
        gi = int(row["group_index"])
        if gi not in seen:
            seen.add(gi)
            selected.append(gi)
        if len(selected) >= int(limit):
            break
    if not selected:
        raise ValueError("ML plan has no matching branches for selected targets/weight")
    return selected


def _one(task):
    target, seed, cfg = task
    n, k = target
    backend = str(cfg.get("backend", "cpp"))
    spec = TargetSpec(n, k, int(cfg["max_check_weight"]), None)
    started = time.perf_counter()
    search = design_from_specs(
        spec, seed=int(seed), seed_draws=int(cfg["seed_draws"]),
        seed_screen_trials=int(cfg["seed_screen_trials"]),
        product_screen_trials=int(cfg["product_screen_trials"]),
        deep_trials=int(cfg["deep_trials"]),
        deep_repeats=int(cfg["deep_repeats"]),
        probe_keep=int(cfg["probe_keep"]), seed_keep=int(cfg["seed_keep"]),
        max_deep_products=int(cfg["max_deep_products"]), backend=backend,
        mutation_elites=int(cfg["mutation_elites"]),
        mutations_per_elite=int(cfg["mutations_per_elite"]),
        mutation_radius=int(cfg["mutation_radius"]),
        product_mutation_elites=int(cfg["product_mutation_elites"]),
        product_mutations_per_elite=int(cfg["product_mutations_per_elite"]),
        product_mutation_radius=int(cfg["product_mutation_radius"]),
        product_mutation_rounds=int(cfg["product_mutation_rounds"]),
        history_rows=cfg.get("_history_rows", ()),
        group_indices=cfg.get("group_indices"),
        candidate_pool_size=int(cfg["confirm_candidates"]),
    )
    candidate = search.get("candidate")
    if candidate is None:
        return {"target": {"n": n, "k": k}, "seed": int(seed),
                "status": "no_candidate", "seconds": time.perf_counter() - started}

    confirmed=[]
    candidate_pool=search.get("candidate_pool") or [candidate]
    for pool_index, pool_candidate in enumerate(candidate_pool):
        groups = candidate_groups_of_order(int(pool_candidate["group_order"]))
        group = groups[int(pool_candidate["group_index"])]
        A = tuple(tuple(int(v) for v in row) for row in pool_candidate["A"])
        B = tuple(tuple(int(v) for v in row) for row in pool_candidate["B"])
        hx, hz = build_product_checks(A, B, group)
        checks_x, checks_z = _supports(hx), _supports(hz)
        # d=1 forces both sectors to run; collect independent witnesses.
        ris = fast_refute_supports(
            checks_x, checks_z, n, 1,
            seed=(int(seed) + pool_index * 0x9E3779B9) & 0xffffffff,
            trials=int(cfg["confirm_trials"]), max_seconds=None,
            backend=backend,
        )
        sx, sz = ris["sectors"].get("x", {}), ris["sectors"].get("z", {})
        dx, dz = sx.get("best_weight"), sz.get("best_weight")
        if dx is None or dz is None:
            continue
        confirmed.append((min(int(dx), int(dz)), pool_candidate, group, A, B, hx, hz, ris, sx, sz))
    if not confirmed:
        return {"target": {"n": n, "k": k}, "seed": int(seed),
                "status": "no_witness", "search": {
                    "screened": search.get("screened_product_count"),
                    "deepened": search.get("deepened_product_count"),
                    "confirmed_candidates": 0},
                "seconds": time.perf_counter() - started}
    _, candidate, group, A, B, hx, hz, ris, sx, sz = max(
        confirmed, key=lambda item: (item[0],
                                     item[1].get("distance_estimate", {}).get("d_upper") or -1,
                                     item[1].get("semantic_hash", "")))
    dx, dz = int(sx["best_weight"]), int(sz["best_weight"])
    d = min(dx, dz)
    sp = css_specs(hx, hz)
    return {
        "status": "ok", "target": {"n": n, "k": k}, "seed": int(seed),
        "n": int(n), "k": int(k), "d_upper": d,
        "dx_upper": int(dx), "dz_upper": int(dz),
        "max_check_weight": int(sp["max_check_weight"]),
        "kd2_over_n_upper": float(k * d * d / n),
        "semantic_hash": candidate["semantic_hash"],
        "group": group.name, "group_index": int(candidate["group_index"]),
        "group_order": int(candidate["group_order"]),
        "A": candidate["A"], "B": candidate["B"],
        "witness_x": sx.get("witness"), "witness_z": sz.get("witness"),
        "search_observed": {
            "screen_d": candidate["screen"].get("d_upper"),
            "deep_d": candidate.get("distance_estimate", {}).get("d_upper"),
            "screened": search.get("screened_product_count"),
            "deepened": search.get("deepened_product_count"),
            "candidate_pool": len(candidate_pool),
            "confirmed_candidates": len(confirmed),
        },
        "phylogeny": search.get("phylogeny"),
        "history_used": bool(search.get("history_injected_count", 0)),
        "confirm_trials_per_sector": int(cfg["confirm_trials"]),
        "confirm_mode": ris.get("mode"),
        "seconds": time.perf_counter() - started,
    }


def _dominates(a, b):
    if a.get("status") != "ok" or b.get("status") != "ok":
        return False
    no_worse = (a["n"] <= b["n"] and a["k"] >= b["k"] and
                a["d_upper"] >= b["d_upper"] and
                a["max_check_weight"] <= b["max_check_weight"])
    strict = (a["n"] < b["n"] or a["k"] > b["k"] or
              a["d_upper"] > b["d_upper"] or
              a["max_check_weight"] < b["max_check_weight"])
    return no_worse and strict


def frontier(rows):
    # Semantic dedupe first: preserve best observed d and shortest provenance.
    best = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = row["semantic_hash"]
        old = best.get(key)
        if old is None or (row["d_upper"], -row["seconds"]) > (old["d_upper"], -old["seconds"]):
            best[key] = row
    unique = list(best.values())
    out = [row for row in unique if not any(_dominates(other, row)
                                           for other in unique if other is not row)]
    return sorted(out, key=lambda r: (r["n"], -r["k"], -r["d_upper"], r["semantic_hash"]))


def run_campaign(out: Path, *, seeds_per_target=8, workers=4,
                 seed_draws=192, seed_screen_trials=16,
                 product_screen_trials=24, deep_trials=240, deep_repeats=1,
                 max_deep_products=12, probe_keep=32, seed_keep=6,
                 max_check_weight=9,
                 confirm_trials=4000, mutation_elites=0,
                 confirm_candidates=1,
                 mutations_per_elite=0, mutation_radius=1,
                 product_mutation_elites=0,
                 product_mutations_per_elite=0, product_mutation_rounds=1,
                 product_mutation_radius=1,
                 phylogeny_exploitation_fraction=0.5,
                 seed_offset=0,
                 history_path=None,
                 group_indices=None,
                 backend="cpp",
                 targets=DEFAULT_TARGETS):
    requested_backend = str(backend).lower()
    if requested_backend not in {"cpp", "auto", "numpy"}:
        raise ValueError("backend must be one of: cpp, auto, numpy")
    native_info = cpp_fast.accelerator_info()
    if requested_backend == "cpp" and not native_info["available"]:
        raise RuntimeError(
            "backend=cpp requested but qldpc_fast is unavailable: "
            f"{native_info['load_error']}"
        )
    effective_backend = (
        "cpp" if requested_backend in {"cpp", "auto"} and native_info["available"]
        else "numpy"
    )
    out.mkdir(parents=True, exist_ok=True)
    targets = tuple((int(n), int(k)) for n, k in targets)
    history_rows=[]
    if history_path is not None:
        history_path=Path(history_path)
        if history_path.exists():
            history_data=json.loads(history_path.read_text())
            raw_rows=history_data.get('finds',history_data.get('results',[]))
            for n,k in targets:
                history_rows.extend(compact_history(raw_rows,n,k))
        else:
            raise FileNotFoundError(f'history file not found: {history_path}')
    cfg = {"seed_draws": seed_draws, "seed_screen_trials": seed_screen_trials,
           "product_screen_trials": product_screen_trials,
           "deep_trials": deep_trials, "max_deep_products": max_deep_products,
           "deep_repeats": int(deep_repeats),
           "probe_keep": probe_keep, "seed_keep": seed_keep,
           "max_check_weight": int(max_check_weight),
           "confirm_trials": confirm_trials,
           "confirm_candidates": int(confirm_candidates),
           "mutation_elites": mutation_elites,
           "mutations_per_elite": mutations_per_elite,
           "mutation_radius": int(mutation_radius),
           "product_mutation_elites": product_mutation_elites,
           "product_mutations_per_elite": product_mutations_per_elite,
           "product_mutation_radius": int(product_mutation_radius),
           "product_mutation_rounds": product_mutation_rounds,
           "phylogeny_exploitation_fraction": float(phylogeny_exploitation_fraction),
           "group_indices": (list(group_indices) if group_indices is not None else None),
           "backend": effective_backend,
           "seed_offset": int(seed_offset),
           "history_path": str(history_path) if history_path is not None else None,
           "history_count": len(history_rows),
           "_history_rows": history_rows}
    tasks = []
    for ti, target in enumerate(targets):
        for si in range(int(seeds_per_target)):
            # Stable, disjoint streams across target sizes and reruns.
            tasks.append((target, int(seed_offset) + 910000 + ti * 10000 + si * 97, cfg))
    started = time.perf_counter(); rows = []
    if int(workers) <= 1:
        rows = [_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = [pool.submit(_one, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), 1):
                row = future.result(); rows.append(row)
                print(json.dumps({"done": i, "total": len(tasks),
                                  "status": row.get("status"),
                                  "target": row.get("target"),
                                  "d": row.get("d_upper")}, sort_keys=True), flush=True)
    rows.sort(key=lambda r: (r.get("target", {}).get("n", 0), r.get("seed", 0)))
    pareto = frontier(rows)
    result = {
        "campaign": {"targets": [dict(n=n, k=k) for n, k in targets],
                      "tasks": len(tasks), "seeds_per_target": int(seeds_per_target),
                      "workers": int(workers),
                      "config": {k:v for k,v in cfg.items() if not k.startswith('_')},
                      "seconds_wall": time.perf_counter() - started,
                      "backend": ("cpp_native_ris" if effective_backend == "cpp"
                                  else "numpy_packed_ris"),
                      "backend_requested": requested_backend,
                      "accelerator": native_info},
        "results": rows,
        "pareto_observed": pareto,
    }
    (out / "campaign.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "pareto.json").write_text(json.dumps(pareto, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/campaign_01"))
    ap.add_argument("--seeds-per-target", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--confirm-trials", type=int, default=4000)
    ap.add_argument("--confirm-candidates", type=int, default=1,
                    help="confirm top N deep candidates per task")
    ap.add_argument("--only-n", type=int, nargs="*", default=None,
                    help="restrict campaign to listed n values")
    ap.add_argument("--seed-draws", type=int, default=192)
    ap.add_argument("--seed-screen-trials", type=int, default=16)
    ap.add_argument("--product-screen-trials", type=int, default=24)
    ap.add_argument("--deep-trials", type=int, default=240)
    ap.add_argument("--deep-repeats", type=int, default=1,
                    help="independent RIS passes per deep candidate")
    ap.add_argument("--max-deep-products", type=int, default=12)
    ap.add_argument("--max-check-weight", type=int, default=9,
                    help="maximum check weight; legal odd branches 9, 15, 21, 27 (verifier cap 32)")
    ap.add_argument("--mutation-elites", type=int, default=0)
    ap.add_argument("--mutations-per-elite", type=int, default=0)
    ap.add_argument("--mutation-radius", type=int, default=1,
                    help="successive one-symbol shells for seed mutations")
    ap.add_argument("--product-mutation-elites", type=int, default=0)
    ap.add_argument("--product-mutations-per-elite", type=int, default=0)
    ap.add_argument("--product-mutation-radius", type=int, default=1,
                    help="successive one-symbol shells for product mutations")
    ap.add_argument("--product-mutation-rounds", type=int, default=1)
    ap.add_argument("--phylo-exploitation-fraction", type=float, default=0.5,
                    help="fraction of phylogenetic elite slots reserved for top quality")
    ap.add_argument("--group-indices", type=int, nargs="*", default=None,
                    help="optional global group indices to search; omit for all")
    ap.add_argument("--ml-plan", type=Path,
                    default=Path("results/ml_branch_plan.json"),
                    help="persisted ML branch plan")
    ap.add_argument("--ml-branch-limit", type=int, default=0,
                    help="select top ML-ranked group branches (0 disables)")
    ap.add_argument("--ml-branch-status", choices=("missing", "seeded", "both"),
                    default="missing")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--probe-keep", type=int, default=32)
    ap.add_argument("--seed-keep", type=int, default=6)
    ap.add_argument("--history", type=Path,
                    default=Path("results/regulated_findings.json"),
                    help="prior archive/campaign JSON used for same-target seed injection")
    ap.add_argument("--backend", choices=("cpp", "auto", "numpy"), default="cpp",
                    help="distance/search backend: cpp requires qldpc_fast; auto falls back")
    args = ap.parse_args()
    targets = DEFAULT_TARGETS
    if args.only_n:
        wanted = set(args.only_n)
        targets = tuple(t for t in DEFAULT_TARGETS if t[0] in wanted)
        if not targets:
            raise SystemExit("--only-n did not match a default target")
    if args.ml_branch_limit and args.group_indices is not None:
        raise SystemExit("choose --group-indices or --ml-branch-limit, not both")
    group_indices = args.group_indices
    if args.ml_branch_limit:
        group_indices = ml_group_indices(
            args.ml_plan, targets, args.max_check_weight,
            limit=args.ml_branch_limit, status=args.ml_branch_status)
    result = run_campaign(args.out, seeds_per_target=args.seeds_per_target,
                          workers=args.workers, confirm_trials=args.confirm_trials,
                          confirm_candidates=args.confirm_candidates,
                          seed_draws=args.seed_draws,
                          seed_screen_trials=args.seed_screen_trials,
                          product_screen_trials=args.product_screen_trials,
                          deep_trials=args.deep_trials,
                          deep_repeats=args.deep_repeats,
                          max_deep_products=args.max_deep_products,
                          max_check_weight=args.max_check_weight,
                          mutation_elites=args.mutation_elites,
                          mutations_per_elite=args.mutations_per_elite,
                          mutation_radius=args.mutation_radius,
                          product_mutation_elites=args.product_mutation_elites,
                          product_mutations_per_elite=args.product_mutations_per_elite,
                          product_mutation_radius=args.product_mutation_radius,
                          product_mutation_rounds=args.product_mutation_rounds,
                          phylogeny_exploitation_fraction=args.phylo_exploitation_fraction,
                          seed_offset=args.seed_offset,
                          history_path=args.history,
                          group_indices=group_indices,
                          backend=args.backend,
                          probe_keep=args.probe_keep,
                          seed_keep=args.seed_keep,
                          targets=targets)
    if args.ml_branch_limit:
        result["campaign"]["ml_branch_selection"] = {
            "plan": str(args.ml_plan),
            "status": args.ml_branch_status,
            "limit": int(args.ml_branch_limit),
            "selected_group_indices": list(group_indices),
        }
        (args.out / "campaign.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"tasks": result["campaign"]["tasks"],
                      "pareto_observed": [{key: row[key] for key in
                                           ("n", "k", "d_upper", "semantic_hash")}
                                          for row in result["pareto_observed"]],
                      "seconds_wall": result["campaign"]["seconds_wall"]}, indent=2))


if __name__ == "__main__":
    main()
