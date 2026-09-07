from __future__ import annotations

"""Rank-preserving coefficient shells around D_175 factor genomes."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

import dihedral_2bga_n700_campaign as base
import dihedral_factor_lattice as lattice
from gf2_factor import degree, gcd


def _actual_common_degree(parts) -> int:
    common = base.MOD175
    for word in parts:
        common = gcd(common, base._poly(word))
    return degree(common)


def _row(parent, parts, operation):
    p, q, u, v = parts
    a = tuple(sorted(p + tuple(base.ROT + value for value in q)))
    b = tuple(sorted(u + tuple(base.ROT + value for value in v)))
    semantic = base._hash(a, b)
    changed_components = operation.get("components", [operation.get("component")])
    decorated_components = sorted(set(parent.get("decorated_components", [])) |
                                  {int(value) for value in changed_components
                                   if value is not None})
    return {"a": list(a), "b": list(b), "rank": base.TARGET_RANK,
            "k": base.K, "semantic_hash": semantic,
            "genome": parent["genome"],
            "factor_common_degree": parent.get("common_degree", 0),
            "actual_component_common_degree": _actual_common_degree(parts),
            "check_weight": len(a) + len(b),
            "reflection_count": len(q) + len(v),
            "coefficient_operation": operation,
            "decorated_components": decorated_components,
            "lineage": {"root_branch": "factor_lattice_coefficient_shell",
                        "parent_semantic_hash": parent.get("semantic_hash")}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path,
                        default=Path("results/dihedral_coefficient_shell01.json"))
    parser.add_argument("--parents", type=int, default=32)
    parser.add_argument("--max-candidates", type=int, default=2048)
    parser.add_argument("--max-check-weight", type=int, default=160)
    parser.add_argument("--multiplier-terms", type=int, default=3,
                        choices=(2, 3),
                        help="2 uses 1+x^s; 3 uses parity-safe 1+x^s+x^t")
    parser.add_argument("--samples-per-component", type=int, default=512)
    parser.add_argument("--components-per-step", type=int, default=1,
                        choices=(1, 2))
    parser.add_argument("--proxy-trials", type=int, default=512)
    parser.add_argument("--target-d", type=int, default=77)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    started = time.perf_counter()
    payload = json.loads(args.source.read_text())
    rng = random.Random(int(args.seed))
    parents = payload["top"][:int(args.parents)]
    candidates = {}
    attempted = 0
    exact = 0
    for parent_index, parent in enumerate(parents):
        if not parent.get("decorated_components") and parent.get("coefficient_operation"):
            parent = {**parent, "decorated_components": [
                int(parent["coefficient_operation"]["component"])]}
        genome = tuple(parent["genome"])
        if parent.get("a") and parent.get("b"):
            roots = [
                tuple(value for value in parent["a"] if value < base.ROT),
                tuple(value - base.ROT for value in parent["a"] if value >= base.ROT),
                tuple(value for value in parent["b"] if value < base.ROT),
                tuple(value - base.ROT for value in parent["b"] if value >= base.ROT),
            ]
        else:
            roots = [lattice._support(mask) for mask in genome]
        available = [component for component in range(4)
                     if component not in parent.get("decorated_components", [])]
        component_groups = list(itertools.combinations(
            available, int(args.components_per_step)))
        for components in component_groups:
            if args.multiplier_terms == 2 and len(components) == 1:
                multiplier_sets = [((shift,),) for shift in range(1, base.ROT)]
            else:
                multiplier_sets = [tuple(
                    tuple(rng.sample(range(1, base.ROT),
                                     args.multiplier_terms - 1))
                    for _ in components)
                    for _ in range(int(args.samples_per_component))]
            for shifts_by_component in multiplier_sets:
                attempted += 1
                parts = list(roots)
                valid = True
                for component, shifts in zip(components, shifts_by_component):
                    root = set(roots[component])
                    decorated = set(root)
                    for shift in shifts:
                        decorated.symmetric_difference_update(
                            (value + shift) % base.ROT for value in root)
                    parts[component] = tuple(sorted(decorated))
                    valid &= bool(parts[component])
                if not valid:
                    continue
                row = _row(parent, parts, {
                    "type": (f"{len(components)}_component_"
                             f"{args.multiplier_terms}_term_xor"),
                    "components": list(components),
                    "shifts": [list(value) for value in shifts_by_component],
                    "parent_index": parent_index})
                if row["semantic_hash"] == parent.get("semantic_hash"):
                    continue
                if row["check_weight"] > int(args.max_check_weight):
                    continue
                if base._rank(row["a"], row["b"]) != base.TARGET_RANK:
                    continue
                exact += 1
                candidates.setdefault(row["semantic_hash"], row)
    rows = sorted(candidates.values(), key=lambda row: (
        -len(row["decorated_components"]),
        row["actual_component_common_degree"], row["check_weight"],
        -row["reflection_count"], row["semantic_hash"]))[:int(args.max_candidates)]
    print(json.dumps({"stage": "coefficient_shell", "attempted": attempted,
                      "exact": exact, "selected": len(rows)}), flush=True)
    screened = base._screen(rows, trials=args.proxy_trials,
                            seed=args.seed + 1000003, workers=args.workers,
                            threads=args.threads, target=args.target_d)
    screened.sort(key=lambda row: int(row["screen"].get("d_upper") or -1),
                  reverse=True)
    survivors = [row for row in screened
                 if int(row["screen"].get("d_upper") or -1) >= args.target_d and
                 not row["screen"]["stopped_early"]]
    report = {
        "kind": "dihedral_2bga_rank_preserving_coefficient_shell",
        "source": str(args.source.resolve()),
        "target": {"n": base.N, "k": base.K, "d": args.target_d},
        "search": {"parents": len(parents), "attempted": attempted,
                   "exact_rank": exact, "selected": len(rows),
                   "survivors": len(survivors),
                   "seconds_wall": time.perf_counter() - started},
        "survivors": survivors, "top": screened[:256],
        "regulation": {"stage_only": True, "submission_sent": False,
                       "git_commit_performed": False,
                       "distance_is_randomized_upper_bound": True,
                       "board_claim_allowed": False},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "search": report["search"],
                      "best": [(row["screen"].get("d_upper"),
                                row["actual_component_common_degree"],
                                row["check_weight"], row["semantic_hash"])
                               for row in report["top"][:10]]}, indent=2))


if __name__ == "__main__":
    main()
