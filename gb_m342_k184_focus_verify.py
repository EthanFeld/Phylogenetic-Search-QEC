from __future__ import annotations

"""Focused m342 follow-up: re-pair selected mined branches and deepen RIS."""

import argparse
import json
from pathlib import Path
import time

import gb_m342_k184_repeated_campaign as wrapper


ENGINE = wrapper.campaign
ENGINE.PAIR_EXACT_ONLY = False


def _dedup(rows):
    unique = {}
    for row in rows:
        key = ENGINE.core._affine_pair_key(
            ENGINE.M, tuple(row["A"][0]), tuple(row["B"][0]))
        row["translation_semantic_hash"] = row["semantic_hash"]
        row["semantic_hash"] = ENGINE.core._semantic_hash(*key)
        row["affine_unit_canonical"] = True
        previous = unique.get(row["semantic_hash"])
        if previous is None or ENGINE.core._candidate_prior(row) > ENGINE.core._candidate_prior(previous):
            unique[row["semantic_hash"]] = row
    return list(unique.values())


def _histogram(rows, field):
    out = {}
    for row in rows:
        value = str(row.get("screen", {}).get(field))
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items(), key=lambda item: item[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path,
                        default=Path("results/gb_m342_k184_focus01"))
    parser.add_argument("--branch-ids", default=(
        "m342_q92_e010111020000,m342_q92_e200010200200,"
        "m342_q92_e200010021100,m342_q92_e200101021000"))
    parser.add_argument("--pairs-per-branch", type=int, default=512)
    parser.add_argument("--pair-attempts", type=int, default=500_000)
    parser.add_argument("--min-check-weight", type=int, default=28)
    parser.add_argument("--target-d", type=int, default=77)
    parser.add_argument("--proxy-trials", type=int, default=1024)
    parser.add_argument("--proxy-replicas", type=int, default=2)
    parser.add_argument("--deep-trials", type=int, default=20_000)
    parser.add_argument("--deep-replicas", type=int, default=4)
    parser.add_argument("--confirm-count", type=int, default=8)
    parser.add_argument("--confirm-trials", type=int, default=2_000_000)
    parser.add_argument("--confirm-replicas", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--pair-seed", type=int)
    parser.add_argument("--proxy-seed", type=int)
    parser.add_argument("--deep-seed", type=int)
    parser.add_argument("--confirm-seed", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    started = time.perf_counter()
    pair_seed = int(args.pair_seed if args.pair_seed is not None else args.seed)
    proxy_seed = int(args.proxy_seed if args.proxy_seed is not None
                     else args.seed + 1_000_003)
    deep_seed = int(args.deep_seed if args.deep_seed is not None
                    else args.seed + 2_000_003)
    confirm_seed = int(args.confirm_seed if args.confirm_seed is not None
                       else args.seed + 3_000_003)
    source = json.loads(args.source.read_text())
    wanted = set(value.strip() for value in args.branch_ids.split(",") if value.strip())
    branches = [row for row in source["mined"] if row["branch_id"] in wanted]
    if not branches:
        raise ValueError("no requested branch IDs found in source report")
    candidates = []
    counts = {}
    for index, branch in enumerate(branches):
        rows = ENGINE._pair_population(
            branch, count=args.pairs_per_branch, attempts=args.pair_attempts,
            min_check_weight=args.min_check_weight,
            seed=pair_seed + index * 7919)
        counts[branch["branch_id"]] = len(rows)
        candidates.extend(rows)
    raw_count = len(candidates)
    candidates = _dedup(candidates)
    print(json.dumps({"stage": "focus_pair", "raw": raw_count,
                      "unique": len(candidates), "branches": counts}), flush=True)

    proxy = ENGINE.core._replicated_screen(
        candidates, replicas=args.proxy_replicas, trials=args.proxy_trials,
        seed=proxy_seed, workers=args.workers, threads=args.threads,
        target_d=args.target_d, label="m342_focus_proxy")
    proxy.sort(key=lambda row: ENGINE.core._screen_quality(row, args.target_d),
               reverse=True)
    proxy_survivors = [row for row in proxy
                       if ENGINE.core._survives(row, args.target_d)]
    print(json.dumps({"stage": "focus_proxy", "survivors": len(proxy_survivors),
                      "hist": _histogram(proxy, "d_upper")}), flush=True)

    deep = ENGINE.core._replicated_screen(
        proxy_survivors, replicas=args.deep_replicas, trials=args.deep_trials,
        seed=deep_seed, workers=args.workers, threads=args.threads,
        target_d=args.target_d, label="m342_focus_deep")
    deep.sort(key=lambda row: ENGINE.core._screen_quality(row, args.target_d),
              reverse=True)
    deep_survivors = [row for row in deep
                      if ENGINE.core._survives(row, args.target_d)]
    print(json.dumps({"stage": "focus_deep", "survivors": len(deep_survivors),
                      "hist": _histogram(deep, "d_upper")}), flush=True)

    confirmed = []
    if deep_survivors:
        selected = deep_survivors[:max(1, int(args.confirm_count))]
        confirmed = ENGINE.core._replicated_screen(
            selected, replicas=args.confirm_replicas,
            trials=args.confirm_trials, seed=confirm_seed,
            workers=min(args.workers, max(1, len(selected) * args.confirm_replicas)),
            threads=args.threads, target_d=args.target_d,
            label="m342_focus_confirm")
        confirmed.sort(key=lambda row: ENGINE.core._screen_quality(row, args.target_d),
                       reverse=True)
    finals = []
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(confirmed, 1):
        doc = ENGINE._candidate_doc(row, "focus_confirm")
        if doc is None:
            continue
        path = out_dir / f"candidate_{rank}_{ENGINE.N}_{ENGINE.K}_{doc['distance']['d']}.json"
        path.write_text(json.dumps(doc, indent=2) + "\n")
        finals.append({
            "rank": rank, "candidate_path": str(path.resolve()),
            "semantic_hash": row["semantic_hash"],
            "branch_id": row["branch_id"],
            "word_weights": row["word_weights"],
            "individual_gcd_degrees": row["individual_gcd_degrees"],
            "d_upper": doc["distance"]["d"],
            "dx_upper": doc["distance"]["X"]["value"],
            "dz_upper": doc["distance"]["Z"]["value"],
            "score_upper": ENGINE.K * doc["distance"]["d"] ** 2 / ENGINE.N,
            "structural": ENGINE.core.structural_validate(doc),
        })
    report = {
        "kind": "gb_m342_k184_focus_verification",
        "source": str(args.source.resolve()),
        "target": {"m": ENGINE.M, "n": ENGINE.N, "k": ENGINE.K,
                   "target_d": args.target_d,
                   "target_score": ENGINE.K * args.target_d ** 2 / ENGINE.N},
        "search": {"raw_candidates": raw_count, "candidates": len(candidates),
                   "proxy_survivors": len(proxy_survivors),
                   "deep_survivors": len(deep_survivors),
                   "branch_candidate_counts": counts,
                   "proxy_trials": args.proxy_trials,
                   "proxy_replicas": args.proxy_replicas,
                   "deep_trials": args.deep_trials,
                   "deep_replicas": args.deep_replicas,
                   "confirm_trials": args.confirm_trials,
                   "confirm_replicas": args.confirm_replicas,
                   "seconds_wall": time.perf_counter() - started},
        "proxy_top": proxy[:64], "deep_top": deep[:64],
        "confirmed": confirmed, "final": finals,
        "regulation": {"stage_only": True, "submission_sent": False,
                       "git_commit_performed": False,
                       "official_gate_run": False,
                       "distance_is_randomized_upper_bound": True,
                       "board_claim_allowed": False},
    }
    (out_dir / "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(out_dir), "final": finals}, indent=2))


if __name__ == "__main__":
    main()
