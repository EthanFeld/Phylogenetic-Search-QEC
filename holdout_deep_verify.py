from __future__ import annotations

"""Deep RIS validation for fresh mutation-lineage holdouts."""

import argparse
import json
import os
from pathlib import Path

from ml_mutation_search import _candidate_doc, _parallel_deep, _write_jsonl


def run(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    proxy = [json.loads(line) for line in Path(args.proxy).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    survivors = [row for row in proxy
                 if row.get("status") == "proxy_survivor" and row.get("fresh_lineage_holdout")]
    survivors.sort(key=lambda row: (-(row.get("proxy", {}).get("d_upper") or 0), row["candidate_id"]))
    selected = []
    per_parent = {}
    for row in survivors:
        parent = row["parent_code_id"]
        if per_parent.get(parent, 0) >= int(args.per_parent):
            continue
        per_parent[parent] = per_parent.get(parent, 0) + 1
        selected.append(row)
        if len(selected) >= int(args.count):
            break
    deep = _parallel_deep(selected, trials=args.trials, seed=args.seed,
                          workers=args.workers, threads=args.threads,
                          pair_depth=args.pair_depth)
    _write_jsonl(out / "deep_results.jsonl", deep)
    final = []
    for index, row in enumerate(deep, 1):
        if row.get("status") != "deep_survivor":
            continue
        doc = _candidate_doc(row)
        if doc is None:
            continue
        path = out / f"candidate_{index:04d}_{row['n']}_{row['k']}_{row['deep']['d_upper']}.json"
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        final.append({
            "candidate_id": row["candidate_id"], "candidate_path": str(path.resolve()),
            "n": int(row["n"]), "k": int(row["k"]),
            "d_upper": int(row["deep"]["d_upper"]),
            "score_upper": int(row["k"]) * int(row["deep"]["d_upper"]) ** 2 / int(row["n"]),
        })
    report = {
        "schema_version": "1.0", "kind": "fresh_lineage_holdout_deep_ris",
        "target_score": 1542.0, "proxy_holdout_survivors": len(survivors),
        "deep_selected": len(selected), "deep_survivors": len(final),
        "trials_per_side": int(args.trials), "workers": int(args.workers),
        "threads": int(args.threads), "pair_depth": int(args.pair_depth),
        "final": final,
        "claim_policy": "RIS distances are randomized witness upper bounds; final GCD gate required.",
    }
    (out / "campaign.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--per-parent", type=int, default=4)
    parser.add_argument("--trials", type=int, default=4096)
    parser.add_argument("--pair-depth", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
