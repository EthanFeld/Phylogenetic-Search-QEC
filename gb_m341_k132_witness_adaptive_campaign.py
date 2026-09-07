from __future__ import annotations

"""Witness-adaptive q=66 mutation search for ``[[682,132,d]]``.

Previous d=95 proxy winners collapsed to d=89/d=88.  Their 20M witnesses
provide a useful adversarial signal: candidate mutations that leave those
vectors in the checked kernel are deprioritized.  Syndrome pressure is only a
search heuristic; every finalist still goes through fresh RIS and verifier
checks.

Stage-only.  No submission, commit, or PR.
"""

import argparse
import json
from pathlib import Path
import random
import time

import cpp_fast
import gb_m341_k132_mutation_campaign as local
import gb_m341_k132_target as target
import gb_m345_k188_phylo_campaign as core
from generalized_bicycle import cyclic_shift


M, N, Q, K = 341, 682, 66, 132
ROOTS = (
    Path("results/gb_m341_k132_sparse_allw15_01/full_2m/candidate_9_682_132_92_90.json"),
    Path("results/gb_m341_k132_sparse_w17_auto01/full_2m/candidate_2_candidate_2_682_132_92_90.json"),
)
KILLERS = (
    Path("results/gb_m341_k132_record_local_mutation_01/validation_20m/candidate_1_d95_89.json"),
    Path("results/gb_m341_k132_record_local_mutation_01/validation_20m/candidate_2_d95_retry_88.json"),
)


def _mask(word) -> int:
    mask = 0
    for value in word:
        mask ^= 1 << (int(value) % M)
    return mask


def _killer_blocks(doc: dict) -> tuple[int, int, int]:
    witness = doc["distance"]["X"]["witness"]
    # Refined receipts carry X/Z witnesses.  Select caller separately.
    x = _mask(doc["distance"]["X"]["witness"])
    z = _mask(doc["distance"]["Z"]["witness"])
    return x, z, int(doc["distance"]["d"])


def _word_syndrome(word, witness_mask: int, sign: int) -> int:
    syndrome = 0
    for value in word:
        syndrome ^= cyclic_shift(witness_mask, int(sign) * int(value), M)
    return syndrome


def _pressure(a, b, killers: list[tuple[int, int, int]]) -> dict:
    """Return known-killer syndrome pressure for candidate A/B pair.

    X logicals are checked against H_Z=(B^T|A^T); Z logicals against
    H_X=(A|B).  Nonzero syndrome means known killer no longer lies in kernel.
    """
    values = []
    for index, (xmask, zmask, _distance) in enumerate(killers):
        sx = (_word_syndrome(b, xmask, +1) ^
              _word_syndrome(a, xmask, +1)).bit_count()
        # The block order differs: X witness uses block-0 B^T and block-1 A^T.
        # Recompute with block masks below when full witness is supplied.
        values.append((index, "X", int(sx)))
        sz = (_word_syndrome(a, zmask, -1) ^
              _word_syndrome(b, zmask, -1)).bit_count()
        values.append((index, "Z", int(sz)))
    per_code = {}
    for index, side, value in values:
        per_code.setdefault(index, {})[side] = value
    flat = [int(value) for _index, _side, value in values]
    return {
        "per_killer": per_code,
        "sum": sum(flat), "min": min(flat, default=0),
        "nonzero": sum(value > 0 for value in flat),
    }


def _pressure_with_blocks(a, b, killer_blocks) -> dict:
    """Correct block-aware syndrome pressure."""
    values = []
    for index, (x0, x1, z0, z1, _distance) in enumerate(killer_blocks):
        sx = (_word_syndrome(b, x0, +1) ^
              _word_syndrome(a, x1, +1)).bit_count()
        sz = (_word_syndrome(a, z0, -1) ^
              _word_syndrome(b, z1, -1)).bit_count()
        values.extend(((index, "X", int(sx)), (index, "Z", int(sz))))
    per_code = {}
    for index, side, value in values:
        per_code.setdefault(str(index), {})[side] = value
    flat = [value for _index, _side, value in values]
    return {"per_killer": per_code, "sum": sum(flat),
            "min": min(flat, default=0),
            "nonzero": sum(value > 0 for value in flat)}


def _killer_block_data(doc: dict) -> tuple[int, int, int, int, int]:
    x = doc["distance"]["X"]["witness"]
    z = doc["distance"]["Z"]["witness"]
    x0 = _mask(value for value in x if int(value) < M)
    x1 = _mask(int(value) - M for value in x if int(value) >= M)
    z0 = _mask(value for value in z if int(value) < M)
    z1 = _mask(int(value) - M for value in z if int(value) >= M)
    return x0, x1, z0, z1, int(doc["distance"]["d"])


def _raw_pairs(lineages, *, killers, args, seed: int) -> tuple[list[dict], dict]:
    rng = random.Random(int(seed))
    raw = []
    for branch, words, _stats in lineages:
        pool = sorted((word for word in words["exact"] if 14 <= len(word) <= 18),
                      key=lambda word: (len(word), word))
        if len(pool) < 2:
            continue
        for _ in range(int(args.pair_attempts)):
            a = rng.choice(pool)
            b = core._shift(rng.choice(pool), rng.randrange(M))
            if a == b or len(a) + len(b) < 28 or len(a) + len(b) > 32:
                continue
            if core.common_gcd_degree(M, a, b) != Q:
                continue
            pressure = _pressure_with_blocks(a, b, killers)
            raw.append((pressure, branch, a, b, "same_lineage"))

    # Cross-lineage pair arm.  Exact gcd gate is mandatory; most pairs fail,
    # but survivors test whether the two factor phylogenies can recombine.
    if len(lineages) >= 2:
        left_branch, left_words, _ = lineages[0]
        right_branch, right_words, _ = lineages[1]
        left = [word for word in left_words["exact"] if 14 <= len(word) <= 18]
        right = [word for word in right_words["exact"] if 14 <= len(word) <= 18]
        for _ in range(int(args.cross_pair_attempts)):
            if not left or not right:
                break
            a = rng.choice(left)
            b = core._shift(rng.choice(right), rng.randrange(M))
            if len(a) + len(b) < 28 or len(a) + len(b) > 32:
                continue
            if core.common_gcd_degree(M, a, b) != Q:
                continue
            raw.append((_pressure_with_blocks(a, b, killers), left_branch,
                        a, b, "cross_lineage"))

    # Pressure first, cycle profile later.  This avoids 341-shift Python
    # scans across hundreds of thousands of doomed pair attempts.
    raw.sort(key=lambda item: (item[0]["nonzero"], item[0]["min"],
                               item[0]["sum"], rng.random()), reverse=True)
    selected = raw[:int(args.raw_keep)]
    rows = {}
    for pressure, branch, a, b, mode in selected:
        row = local._make_row(branch, a, b, mode=mode)
        if row is None:
            continue
        key = row["semantic_hash"]
        row["witness_pressure"] = pressure
        previous = rows.get(key)
        if previous is None or pressure["sum"] > previous["witness_pressure"]["sum"]:
            rows[key] = row
    pre_affine = list(rows.values())
    affine_unique = {}
    for row in pre_affine:
        affine = core._affine_pair_key(
            M, tuple(row["A"][0]), tuple(row["B"][0]))
        row["translation_semantic_hash"] = row["semantic_hash"]
        row["semantic_hash"] = core._semantic_hash(*affine)
        row["affine_unit_canonical"] = True
        previous = affine_unique.get(row["semantic_hash"])
        if previous is None or row["witness_pressure"]["sum"] > previous["witness_pressure"]["sum"]:
            affine_unique[row["semantic_hash"]] = row
    rows = list(affine_unique.values())
    rows.sort(key=lambda row: (
        row["witness_pressure"]["nonzero"],
        row["witness_pressure"]["min"],
        row["witness_pressure"]["sum"],
        int(row["cycle_profile"]["girth_at_least_6"]),
        -int(row["cycle_profile"]["four_cycle_energy"]),
        row["semantic_hash"]), reverse=True)
    return rows[:int(args.max_candidates)], {
        "raw_legal_pairs": len(raw), "raw_keep": len(selected),
        "pre_affine_unique": len(pre_affine),
        "retained": min(len(rows), int(args.max_candidates)),
        "pressure_nonzero_all_killers": sum(
            row["witness_pressure"]["nonzero"] == 2 * len(killers)
            for row in rows),
    }


def _write_docs(rows, out_dir: Path, stage: str) -> list[dict]:
    finals = []
    for rank, row in enumerate(rows, 1):
        doc = target._candidate_doc(row, stage)
        if doc is None:
            continue
        path = out_dir / f"candidate_{rank}_{N}_{K}_{doc['distance']['d']}.json"
        path.write_text(json.dumps(doc, indent=2) + "\n")
        finals.append({"rank": rank, "candidate_path": str(path.resolve()),
                       "d_upper": int(doc["distance"]["d"]),
                       "dx_upper": int(doc["distance"]["X"]["value"]),
                       "dz_upper": int(doc["distance"]["Z"]["value"]),
                       "score_upper": K * int(doc["distance"]["d"]) ** 2 / N,
                       "branch_id": row["branch_id"],
                       "word_weights": row["word_weights"],
                       "witness_pressure": row["witness_pressure"],
                       "cycle_profile": row["cycle_profile"],
                       "regulation": doc["regulation"]})
    return finals


def run(args: argparse.Namespace) -> dict:
    if not cpp_fast.available():
        raise RuntimeError(cpp_fast.accelerator_info())
    started = time.perf_counter()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = [Path(value) for value in args.root]
    if not roots:
        roots = list(ROOTS)
    killer_paths = [Path(value) for value in args.killer]
    if not killer_paths:
        killer_paths = list(KILLERS)
    killers = [_killer_block_data(json.loads(path.read_text()))
               for path in killer_paths]

    lineages = []
    lineage_stats = []
    for index, path in enumerate(roots):
        doc = local._load_json(path)
        _campaign, branch = local._branch_report(path, doc)
        result = local._mine_lineage(
            doc, branch, args=args,
            seed=int(args.seed) + index * 0x9E3779B9)
        lineages.append(result)
        lineage_stats.append(result[2])
        print(json.dumps({"stage": "adaptive_lineage", **result[2]}), flush=True)

    candidates, pair_stats = _raw_pairs(lineages, killers=killers, args=args,
                                         seed=args.seed)
    print(json.dumps({"stage": "adaptive_pair", **pair_stats}), flush=True)
    proxy = core._replicated_screen(
        candidates, replicas=int(args.proxy_replicas), trials=int(args.proxy_trials),
        seed=int(args.seed) + 0x3000003, workers=int(args.workers),
        threads=int(args.threads), target_d=int(args.target_d),
        ideal_trials=int(args.ideal_trials), label="adaptive_proxy")
    proxy.sort(key=lambda row: (
        core._screen_quality(row, int(args.target_d)),
        row.get("witness_pressure", {}).get("sum", 0)), reverse=True)
    survivors = [row for row in proxy
                 if core._survives(row, int(args.target_d))]
    # Persist the expensive broad screen before starting deep fanout.  Long
    # campaigns are often intentionally interrupted after proxy ranking; a
    # checkpoint keeps that work auditable and prevents silent data loss.
    (out_dir / "proxy_checkpoint.json").write_text(json.dumps({
        "schema_version": "1.0",
        "kind": "gb_m341_k132_witness_adaptive_proxy_checkpoint",
        "target_d": int(args.target_d), "proxy": proxy,
        "survivors": len(survivors),
        "regulation": {"stage_only": True, "submission_sent": False,
                       "git_commit_performed": False,
                       "distance_is_randomized_upper_bound": True},
    }, indent=2) + "\n")
    capped = core._branch_cap(survivors, int(args.deep_per_branch),
                              int(args.target_d))
    deep = core._replicated_screen(
        capped[:int(args.deep_count)], replicas=int(args.deep_replicas),
        trials=int(args.deep_trials), seed=int(args.seed) + 0x4000003,
        workers=int(args.workers), threads=int(args.threads),
        target_d=int(args.target_d), ideal_trials=int(args.ideal_trials),
        label="adaptive_deep")
    deep.sort(key=lambda row: (
        core._screen_quality(row, int(args.target_d)),
        row.get("witness_pressure", {}).get("sum", 0)), reverse=True)
    deep_survivors = [row for row in deep
                      if core._survives(row, int(args.target_d))]
    finals = _write_docs(deep_survivors, out_dir, "deep")
    report = {
        "schema_version": "1.0",
        "kind": "gb_m341_k132_witness_adaptive_campaign",
        "target": {"m": M, "n": N, "k": K, "gcd_degree": Q,
                   "target_d": int(args.target_d),
                   "check_weight_range": [28, 32]},
        "hypothesis": {
            "failure_mode": "20M gates found d=89 and d=88 killers in d=95 proxy roots",
            "adaptation": "maximize nonzero syndrome against known X/Z killer witnesses",
            "meaning": "syndrome pressure deprioritizes known collapse vectors; not distance proof",
            "lineage": "q66 exact ideal mutations plus cross-lineage exact-gcd recombination",
        },
        "roots": [str(path.resolve()) for path in roots],
        "killers": [str(path.resolve()) for path in killer_paths],
        "lineage_stats": lineage_stats, "pair": pair_stats,
        "search": {"candidates": len(candidates),
                   "proxy_survivors": len(survivors),
                   "deep_survivors": len(deep_survivors),
                   "proxy_trials_per_side": int(args.proxy_trials),
                   "deep_trials_per_side": int(args.deep_trials),
                   "deep_replicas": int(args.deep_replicas),
                   "common_gcd_ideal_trials_per_screen": int(args.ideal_trials),
                   "seed": int(args.seed)},
        "proxy_top": proxy[:64], "deep_top": deep[:64], "final": finals,
        "accelerator": cpp_fast.accelerator_info(),
        "seconds_wall": time.perf_counter() - started,
        "regulation": {"stage_only": True, "submission_sent": False,
                       "git_commit_performed": False,
                       "official_gate_run": False, "board_claim_allowed": False,
                       "distance_is_randomized_upper_bound": True},
    }
    (out_dir / "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("results/gb_m341_k132_witness_adaptive_01"))
    # Empty defaults let explicit CLI roots/killers replace historical seeds;
    # fallback is applied inside run().
    parser.add_argument("--root", type=Path, action="append", default=[])
    parser.add_argument("--killer", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--target-d", type=int, default=90)
    parser.add_argument("--native-iterations", type=int, default=700_000)
    parser.add_argument("--native-max-words", type=int, default=8192)
    parser.add_argument("--hill-restarts", type=int, default=768)
    parser.add_argument("--hill-steps", type=int, default=8192)
    parser.add_argument("--hill-max-words", type=int, default=4096)
    parser.add_argument("--mutation-attempts", type=int, default=800_000)
    parser.add_argument("--nested-fraction", type=float, default=0.50)
    parser.add_argument("--pair-attempts", type=int, default=120_000)
    parser.add_argument("--cross-pair-attempts", type=int, default=100_000)
    parser.add_argument("--raw-keep", type=int, default=3000)
    parser.add_argument("--max-candidates", type=int, default=1800)
    parser.add_argument("--proxy-trials", type=int, default=256)
    parser.add_argument("--proxy-replicas", type=int, default=1)
    parser.add_argument("--deep-count", type=int, default=48)
    parser.add_argument("--deep-per-branch", type=int, default=24)
    parser.add_argument("--deep-trials", type=int, default=20_000)
    parser.add_argument("--deep-replicas", type=int, default=2)
    parser.add_argument("--ideal-trials", type=int, default=1024,
                        help="common-gcd ideal refutation trials per screen")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"out": str(args.out), "pair": report["pair"],
                      "search": report["search"], "final": report["final"],
                      "seconds_wall": round(report["seconds_wall"], 1)}, indent=2))


if __name__ == "__main__":
    main()
