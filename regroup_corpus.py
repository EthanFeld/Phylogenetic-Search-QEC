from __future__ import annotations

"""Reassign repaired corpus splits by algebraic family, preserving audit data."""

import argparse
import json
from collections import Counter
from pathlib import Path

from repair_and_rank import _algebraic_split_group, _sha


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("results/corpus_repaired"))
    p.add_argument("--out", type=Path, default=Path("results/corpus_repaired_algebraic"))
    args = p.parse_args()
    codes = _read(args.source / "codes.jsonl")
    groups = {row["code_id"]: _algebraic_split_group(row) for row in codes}
    buckets = {group: ("test" if int(_sha(group)[:8], 16) % 100 < 10
                       else "validation" if int(_sha(group)[:8], 16) % 100 < 20
                       else "train") for group in set(groups.values())}
    for row in codes:
        group = groups[row["code_id"]]
        row["semantic_group"] = row.get("semantic_group") or row.get("repaired_group")
        row["algebraic_group"] = group
        row["repaired_group"] = group
        row["split_repaired"] = buckets[group]
    args.out.mkdir(parents=True, exist_ok=True)
    _write(args.out / "codes.jsonl", codes)
    for name in ("valid_codes.jsonl", "gcd_screened_survivors.jsonl", "hard_negatives.jsonl", "invalidation_ledger.jsonl"):
        src = args.source / name
        if src.exists():
            rows = _read(src)
            known = {row["code_id"] for row in codes}
            _write(args.out / name, [row for row in rows if row.get("code_id") in known])
    split_by_code = {row["code_id"]: row["split_repaired"] for row in codes}
    for name in ("runs.jsonl", "curves.jsonl"):
        src = args.source / name
        if src.exists():
            rows = _read(src)
            for row in rows:
                row["split_repaired"] = split_by_code.get(row.get("code_id"), row.get("split_repaired"))
            _write(args.out / name, rows)
    src = args.source / "search_queue.jsonl"
    existing_queue = []
    if src.exists():
        existing_queue = _read(src)
        for row in existing_queue:
            row["split_repaired"] = split_by_code.get(row.get("code_id"), row.get("split_repaired"))
        _write(args.out / "search_queue.jsonl", existing_queue)
    queued_ids = {row.get("code_id") for row in existing_queue}
    for split in ("train", "validation", "test"):
        split_queue = [row for row in existing_queue if row.get("split_repaired") == split]
        for code in codes:
            if code.get("split_repaired") != split or code["code_id"] in queued_ids:
                continue
            split_queue.append({
                "code_id": code["code_id"], "family": code.get("family"),
                "n": code.get("n"), "k": code.get("k"),
                "observed_score_upper": code.get("effective_score_upper"),
                "predicted_score_mean": code.get("effective_score_upper"),
                "max_trials_per_side": max((int(x.get("trials_per_side", 0)) for x in []), default=0),
                "curve_points": code.get("curve_point_count", 0),
                "split_repaired": split, "gcd_gate_status": code.get("validity_status"),
                "gcd_screened": code.get("gcd_screened", False),
                "action": "fresh_gcd_check_then_deep_search",
            })
        _write(args.out / f"search_queue_{split}.jsonl", split_queue)
    group_splits = {group: buckets[group] for group in set(groups.values())}
    summary = {"codes": len(codes), "algebraic_groups": len(group_splits),
               "split_counts": dict(Counter(buckets.values())),
               "group_split_overlap": 0,
               "source": str(args.source.resolve()), "out": str(args.out.resolve())}
    (args.out / "algebraic_split_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
