from __future__ import annotations

"""Near-record mutation campaign for ``[[682,132,d]]`` Z_341 GB codes.

The broad sparse-divisor sweeps found d=90 candidates, but fresh 20M/side
RIS refuted both.  This campaign keeps their exact q=66 algebraic lineage and
searches locally around their A/B words instead of reopening saturated random
branches.

All distance values are randomized upper bounds.  This file is stage-only:
it never submits, commits, or treats a RIS result as a proof.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json
from pathlib import Path
import re
import random
import time

import cpp_fast
import gb_m341_k132_target as target
import gb_m345_k188_phylo_campaign as core


M, N, Q, K = 341, 682, 66, 132
CHECK_CAP = 32
ROOT_CANDIDATES = (
    Path("results/gb_m341_k132_sparse_allw15_01/full_2m/candidate_9_682_132_92_90.json"),
    Path("results/gb_m341_k132_sparse_w17_auto01/full_2m/candidate_2_candidate_2_682_132_92_90.json"),
)


@lru_cache(maxsize=500_000)
def _common_q(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return core.common_gcd_degree(M, a, b)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _first_row_words(doc: dict) -> tuple[tuple[int, ...], tuple[int, ...]]:
    row = [int(value) for value in doc["checks"]["X"][0]]
    a = tuple(sorted(value for value in row if value < M))
    b = tuple(sorted(value - M for value in row if value >= M))
    if not a or not b or len(a) + len(b) > CHECK_CAP:
        raise ValueError(f"invalid root first row in {doc.get('name')}")
    return a, b


def _xor_shift(left, right, shift: int) -> tuple[int, ...]:
    out = set(int(value) % M for value in left)
    for value in right:
        value = (int(value) + int(shift)) % M
        if value in out:
            out.remove(value)
        else:
            out.add(value)
    return tuple(sorted(out))


def _valid_word(word, *, min_weight: int = 14, max_weight: int = 18,
                exact: bool = True) -> bool:
    weight = len(word)
    if not int(min_weight) <= weight <= int(max_weight):
        return False
    depth = core._word_gcd_degree(tuple(word))
    return depth == Q if exact else depth > Q


def _branch_report(candidate_path: Path, doc: dict) -> tuple[dict, dict]:
    # Historical roots lived one level below their campaign directory;
    # current target campaigns place candidate JSON directly beside
    # campaign.json.  Walk upward so both layouts resolve deterministically.
    campaign_path = None
    for parent in (candidate_path.parent, *candidate_path.parents):
        probe = parent / "campaign.json"
        if probe.exists():
            campaign_path = probe
            break
    if campaign_path is None:
        raise FileNotFoundError(f"campaign.json not found above {candidate_path}")
    campaign = _load_json(campaign_path)
    search = doc.get("search", {})
    branch_id = search.get("branch_id") or doc.get("provenance", {}).get("origin")
    mined = campaign.get("mined", [])
    branch = next((item for item in mined
                   if item.get("branch_id") == branch_id), None)
    if branch is None:
        # Mutation output intentionally stores compact lineage metadata rather
        # than copying multi-megabyte mined pools.  Reconstruct enough branch
        # context from candidate metadata; roots/native mining refill pools.
        construction = doc.get("provenance", {}).get("construction", "")
        factors = search.get("factor_indices")
        if factors is None:
            match = re.search(r"factor indices=\[([^\]]+)\]", construction)
            factors = ([int(value.strip()) for value in match.group(1).split(",")
                        if value.strip()] if match else [])
        branch = {
            "branch_id": branch_id or "record_local_mutation",
            "branch_index": 0,
            "factor_indices": list(factors),
            "factor_degrees": list(search.get("factor_degrees", [])),
            "inherited_q55_ancestors": [],
            "divisor_multiplier_symmetry": 1,
            "phylogenetic_arm": search.get("phylogenetic_arm",
                                           "record_local_mutation"),
            "exact_words": [], "nested_words": [],
        }
    return campaign, branch


def _add_words(pool: set[tuple[int, ...]], words, *, exact: bool,
               min_weight: int = 14, max_weight: int = 18) -> int:
    added = 0
    for raw in words:
        word = core._orbit(tuple(int(value) for value in raw))
        if _valid_word(word, min_weight=min_weight,
                       max_weight=max_weight, exact=exact):
            if word not in pool:
                pool.add(word)
                added += 1
    return added


def _mine_lineage(doc: dict, branch: dict, *, args: argparse.Namespace,
                  seed: int) -> tuple[dict, dict, dict]:
    root_a, root_b = _first_row_words(doc)
    exact: set[tuple[int, ...]] = set()
    nested: set[tuple[int, ...]] = set()
    _add_words(exact, [root_a, root_b], exact=True)
    _add_words(exact, branch.get("exact_words", []), exact=True)
    _add_words(nested, branch.get("nested_words", []), exact=False,
               min_weight=10, max_weight=20)

    # Native miners search same ideal using many cyclic-shift combinations.
    # Feed root words first: old campaign miners were seeded by generic scout
    # words, not by the best d=90 genome itself.
    native_bases = [root_a, root_b]
    native_bases.extend(sorted(exact, key=lambda word: (len(word), word))[:24])
    native_mined = cpp_fast.cyclic_ideal_mine(
        native_bases, m=M, iterations=int(args.native_iterations),
        min_weight=14, max_weight=18, seed=int(seed) ^ 0xA5A55A5A,
        max_words=int(args.native_max_words))
    _add_words(exact, native_mined, exact=True)
    native_hill = []
    for index, base in enumerate((root_a, root_b)):
        native_hill.extend(cpp_fast.cyclic_ideal_hillclimb(
            base, m=M, restarts=int(args.hill_restarts),
            steps=int(args.hill_steps), min_weight=14, max_weight=18,
            seed=(int(seed) ^ 0x6A09E667) + index * 0x9E3779B9,
            max_words=int(args.hill_max_words)))
    _add_words(exact, native_hill, exact=True)

    # Local phylogenetic mutation: exact parent XOR shifted descendant.  A
    # second exact parent arm explores recombinations missed by one-child
    # mutation.  Exact gcd gate prevents accidental branch collapse.
    exact_sources = sorted(exact, key=lambda word: (len(word), word))
    nested_sources = sorted(nested, key=lambda word: (len(word), word))
    rng = random.Random(int(seed) ^ 0xC0FFEE)
    mutation_added = 0
    attempts = max(0, int(args.mutation_attempts))
    for _ in range(attempts):
        parent = rng.choice(exact_sources or [root_a])
        if nested_sources and rng.random() < float(args.nested_fraction):
            child = rng.choice(nested_sources)
        else:
            child = rng.choice(exact_sources or [root_b])
        word = _xor_shift(parent, child, rng.randrange(M))
        if not _valid_word(word):
            continue
        word = core._orbit(word)
        if word not in exact:
            exact.add(word)
            exact_sources.append(word)
            mutation_added += 1

    # Couple root-adjacent words across the two best branches.  Cross-lineage
    # words enter only when their polynomial still has exact common degree 66.
    stats = {
        "branch_id": branch["branch_id"],
        "root_a_weight": len(root_a), "root_b_weight": len(root_b),
        "initial_exact": len(branch.get("exact_words", [])),
        "initial_nested": len(branch.get("nested_words", [])),
        "native_mined": len(native_mined), "native_hillclimb": len(native_hill),
        "mutation_attempts": attempts, "mutation_added": mutation_added,
        "exact_shell": len(exact), "nested_shell": len(nested),
    }
    return branch, {"exact": exact, "nested": nested,
                    "root": (root_a, root_b)}, stats


def _make_row(branch: dict, a, b, *, mode: str) -> dict | None:
    if len(a) + len(b) < 28 or len(a) + len(b) > CHECK_CAP:
        return None
    key = core._pair_key(a, b)
    if _common_q(key[0], key[1]) != Q:
        return None
    da, db = core._word_gcd_degree(key[0]), core._word_gcd_degree(key[1])
    return {
        "A": [list(key[0])], "B": [list(key[1])], "m": M,
        "n": N, "k": K, "gcd_degree": Q,
        "branch_id": branch["branch_id"],
        "branch_index": branch.get("branch_index", 0),
        "factor_indices": branch["factor_indices"],
        "factor_degrees": branch.get("factor_degrees", []),
        "phylogenetic_arm": "record_local_mutation",
        "inherited_q55_ancestors": branch.get("inherited_q55_ancestors", []),
        "divisor_multiplier_symmetry": branch.get("divisor_multiplier_symmetry", 1),
        "pairing_mode": mode,
        "word_weights": [len(key[0]), len(key[1])],
        "individual_gcd_degrees": [da, db],
        "cycle_profile": core._cycle_profile(*key),
        "semantic_hash": core._semantic_hash(*key),
        "pair_multiplier_symmetry": 1,
    }


def _pair_lineages(lineages: list[tuple[dict, dict, dict]], *, args,
                   seed: int) -> tuple[list[dict], dict]:
    rows: dict[str, dict] = {}
    rng = random.Random(int(seed))
    pools = []
    for branch, words, _stats in lineages:
        exact = sorted(words["exact"], key=lambda word: (len(word), word))
        # Weight 28/30/32 only; this excludes low shells which cannot improve
        # board score and preserves legal check cap.
        exact = [word for word in exact if 14 <= len(word) <= 18]
        pools.append((branch, exact))
        if len(exact) >= 2:
            for _ in range(min(int(args.deterministic_pairs), len(exact) * 2)):
                a = exact[_ % len(exact)]
                b = exact[(_ * 17 + 1) % len(exact)]
                row = _make_row(branch, a, b, mode="record_local_exact")
                if row is not None:
                    rows[row["semantic_hash"]] = row

    # Same-lineage pairings dominate; cross-lineage pairings test whether a
    # rare common q=66 ideal survives after recombination.
    for branch, exact in pools:
        if len(exact) < 2:
            continue
        # Sparse fallback roots do not justify a campaign-sized random loop:
        # repeated pairs only repeat the same expensive polynomial gcd test.
        local_attempts = min(
            int(args.pair_attempts),
            max(256, len(exact) * len(exact) * 64),
        )
        for _ in range(local_attempts):
            a = rng.choice(exact)
            b = rng.choice(exact)
            if a == b:
                continue
            b = core._shift(b, rng.randrange(M))
            row = _make_row(branch, a, b, mode="record_local_exact")
            if row is not None:
                rows.setdefault(row["semantic_hash"], row)

    for left_index in range(len(pools)):
        for right_index in range(left_index + 1, len(pools)):
            left_branch, left = pools[left_index]
            right_branch, right = pools[right_index]
            if not left or not right:
                continue
            cross_attempts = min(
                int(args.cross_pair_attempts),
                max(256, len(left) * len(right) * 64),
            )
            for _ in range(cross_attempts):
                a = rng.choice(left)
                b = core._shift(rng.choice(right), rng.randrange(M))
                # Keep metadata attached to left branch; exact gcd is the
                # authoritative lineage gate, not the label.
                row = _make_row(left_branch, a, b, mode="cross_lineage_exact")
                if row is not None:
                    rows.setdefault(row["semantic_hash"], row)

    # Verifier-equivalence includes exponent-unit multipliers, not only
    # translations/block swap.  Canonicalize after cheap semantic dedup so
    # affine-equivalent genomes do not consume RIS budget as fake diversity.
    pre_affine = list(rows.values())
    affine_unique = {}
    for row in pre_affine:
        affine = core._affine_pair_key(
            M, tuple(row["A"][0]), tuple(row["B"][0]))
        row["translation_semantic_hash"] = row["semantic_hash"]
        row["semantic_hash"] = core._semantic_hash(*affine)
        row["affine_unit_canonical"] = True
        previous = affine_unique.get(row["semantic_hash"])
        if previous is None or core._candidate_prior(row) > core._candidate_prior(previous):
            affine_unique[row["semantic_hash"]] = row
    rows = affine_unique

    ordered = sorted(rows.values(), key=lambda row: (
        int(row["cycle_profile"]["girth_at_least_6"]),
        -int(row["cycle_profile"]["four_cycle_energy"]),
        sum(row["word_weights"]), row["semantic_hash"]), reverse=True)
    if len(ordered) > int(args.max_candidates):
        # Retain cycle leaders plus deterministic random diversity.
        keep = ordered[:int(args.max_candidates) // 2]
        tail = ordered[int(args.max_candidates) // 2:]
        rng.shuffle(tail)
        keep.extend(tail[:int(args.max_candidates) - len(keep)])
        ordered = keep
    return ordered, {"raw_pairs": len(pre_affine),
                     "affine_unique": len(rows),
                     "retained_pairs": len(ordered),
                     "girth6": sum(bool(row["cycle_profile"]["girth_at_least_6"])
                                    for row in ordered)}


def _write_docs(rows: list[dict], out_dir: Path, stage: str) -> list[dict]:
    finals = []
    for rank, row in enumerate(rows, 1):
        doc = target._candidate_doc(row, stage)
        if doc is None:
            continue
        path = out_dir / f"candidate_{rank}_{N}_{K}_{doc['distance']['d']}.json"
        path.write_text(json.dumps(doc, indent=2) + "\n")
        finals.append({
            "rank": rank, "candidate_path": str(path.resolve()),
            "semantic_hash": row["semantic_hash"],
            "d_upper": int(doc["distance"]["d"]),
            "dx_upper": int(doc["distance"]["X"]["value"]),
            "dz_upper": int(doc["distance"]["Z"]["value"]),
            "score_upper": K * int(doc["distance"]["d"]) ** 2 / N,
            "branch_id": row["branch_id"],
            "word_weights": row["word_weights"],
            "cycle_profile": row["cycle_profile"],
            "regulation": doc["regulation"],
        })
    return finals


def run(args: argparse.Namespace) -> dict:
    if not cpp_fast.available():
        raise RuntimeError(cpp_fast.accelerator_info())
    started = time.perf_counter()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = [Path(path) for path in args.root]
    lineages = []
    lineage_stats = []
    for index, path in enumerate(roots):
        doc = _load_json(path)
        _campaign, branch = _branch_report(path, doc)
        result = _mine_lineage(doc, branch, args=args,
                               seed=int(args.seed) + index * 0x9E3779B9)
        lineages.append(result)
        lineage_stats.append(result[2])
        print(json.dumps({"stage": "lineage_mutation", **result[2]}), flush=True)

    candidates, pair_stats = _pair_lineages(lineages, args=args, seed=args.seed)
    print(json.dumps({"stage": "pair", **pair_stats}), flush=True)
    ideal_trials = int(getattr(args, "ideal_trials",
                               core.COMMON_GCD_IDEAL_TRIALS))
    proxy = core._replicated_screen(
        candidates, replicas=int(args.proxy_replicas), trials=int(args.proxy_trials),
        seed=int(args.seed) + 0x3000003, workers=int(args.workers),
        threads=int(args.threads), target_d=int(args.target_d),
        ideal_trials=ideal_trials, label="mutation_proxy")
    proxy.sort(key=lambda row: core._screen_quality(row, int(args.target_d)), reverse=True)
    proxy_survivors = [row for row in proxy
                       if core._survives(row, int(args.target_d))]
    # Cap each lineage, but keep cross-lineage diversity alive.
    capped = core._branch_cap(proxy_survivors, int(args.deep_per_branch),
                              int(args.target_d))
    deep_rows = capped[:int(args.deep_count)]
    deep = core._replicated_screen(
        deep_rows, replicas=int(args.deep_replicas), trials=int(args.deep_trials),
        seed=int(args.seed) + 0x4000003, workers=int(args.workers),
        threads=int(args.threads), target_d=int(args.target_d),
        ideal_trials=ideal_trials, label="mutation_deep")
    deep.sort(key=lambda row: core._screen_quality(row, int(args.target_d)), reverse=True)
    deep_survivors = [row for row in deep
                      if core._survives(row, int(args.target_d))]
    finals = _write_docs(deep_survivors, out_dir, "deep")
    report = {
        "schema_version": "1.0",
        "kind": "gb_m341_k132_record_local_mutation_campaign",
        "target": {"m": M, "n": N, "k": K, "gcd_degree": Q,
                   "target_d": int(args.target_d),
                   "target_score": K * int(args.target_d) ** 2 / N,
                   "check_weight_range": [28, CHECK_CAP]},
        "hypothesis": {
            "lineage": "mutate two independent 2M-clean d=90 q66 roots",
            "mutation": "exact q66 parent XOR shifted exact/nested same-ideal word",
            "native_search": "C++ cyclic ideal miner + hillclimb seeded by record-adjacent A/B",
            "cross_lineage": "admit only pairs passing exact common_gcd_degree=66",
            "selection": "cycle-aware RIS proxy, branch-capped deep RIS",
            "validation": "all distance values remain randomized upper bounds",
        },
        "roots": [str(path.resolve()) for path in roots],
        "lineage_stats": lineage_stats,
        "pair": pair_stats,
        "search": {
            "candidates": len(candidates),
            "proxy_survivors": len(proxy_survivors),
            "deep_survivors": len(deep_survivors),
            "proxy_trials_per_side": int(args.proxy_trials),
            "proxy_replicas": int(args.proxy_replicas),
            "deep_trials_per_side": int(args.deep_trials),
            "deep_replicas": int(args.deep_replicas),
            "ideal_trials_per_screen": ideal_trials,
            "seed": int(args.seed),
        },
        "proxy_top": proxy[:64], "deep_top": deep[:64], "final": finals,
        "accelerator": cpp_fast.accelerator_info(),
        "seconds_wall": time.perf_counter() - started,
        "regulation": {
            "stage_only": True, "submission_sent": False,
            "git_commit_performed": False, "official_gate_run": False,
            "board_claim_allowed": False,
            "distance_is_randomized_upper_bound": True,
        },
    }
    (out_dir / "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("results/gb_m341_k132_record_local_mutation_01"))
    parser.add_argument("--root", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--target-d", type=int, default=90)
    parser.add_argument("--native-iterations", type=int, default=600_000)
    parser.add_argument("--native-max-words", type=int, default=4096)
    parser.add_argument("--hill-restarts", type=int, default=384)
    parser.add_argument("--hill-steps", type=int, default=4096)
    parser.add_argument("--hill-max-words", type=int, default=2048)
    parser.add_argument("--mutation-attempts", type=int, default=400_000)
    parser.add_argument("--nested-fraction", type=float, default=0.45)
    parser.add_argument("--deterministic-pairs", type=int, default=512)
    parser.add_argument("--pair-attempts", type=int, default=120_000)
    parser.add_argument("--cross-pair-attempts", type=int, default=80_000)
    parser.add_argument("--max-candidates", type=int, default=3000)
    parser.add_argument("--proxy-trials", type=int, default=384)
    parser.add_argument("--proxy-replicas", type=int, default=2)
    parser.add_argument("--deep-count", type=int, default=48)
    parser.add_argument("--deep-per-branch", type=int, default=24)
    parser.add_argument("--deep-trials", type=int, default=20_000)
    parser.add_argument("--deep-replicas", type=int, default=2)
    parser.add_argument("--ideal-trials", type=int,
                        default=core.COMMON_GCD_IDEAL_TRIALS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not args.root:
        args.root = list(ROOT_CANDIDATES)
    report = run(args)
    print(json.dumps({"out": str(args.out), "pair": report["pair"],
                      "search": report["search"], "final": report["final"],
                      "seconds_wall": round(report["seconds_wall"], 1)}, indent=2))


if __name__ == "__main__":
    main()
