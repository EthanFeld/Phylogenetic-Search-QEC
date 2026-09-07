from __future__ import annotations

"""Deepen the Z-ranked candidates from gb_asymmetric_campaign."""

import argparse
import json
from pathlib import Path

from gb_aggressive_campaign import _json_default, _parallel_map, _task_deep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path,
                    default=Path("results/gb_asymmetric_campaign_01"))
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--trials", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()
    data = json.loads((args.source / "z_ranked.json").read_text())
    rows = data["rows"][:max(1, int(args.count))]
    refined = _parallel_map(
        _task_deep, rows, seed=args.seed, trials=args.trials,
        workers=args.workers, threads=args.threads, label="z_deep")
    refined.sort(key=lambda r: (
        -1 if r.get("dz_deep") is None else int(r["dz_deep"]),
        -1 if r.get("dx_deep") is None else int(r["dx_deep"]),
        -1 if r.get("d_deep") is None else int(r["d_deep"]),
        r.get("semantic_hash", "")), reverse=True)
    for rank, row in enumerate(refined, 1):
        row["deep_z_rank"] = rank
    out = {
        "kind": "gb_asymmetric_z_deep_campaign",
        "submission_sent": False,
        "git_commit_performed": False,
        "count": len(refined),
        "trials_per_side": int(args.trials),
        "rows": refined,
    }
    (args.source / "z_deep.json").write_text(
        json.dumps(out, indent=2, default=_json_default) + "\n")
    print(json.dumps({
        "count": len(refined),
        "top": [{"rank": r["deep_z_rank"],
                 "hash": r["semantic_hash"][:8],
                 "dx": r.get("dx_deep"), "dz": r.get("dz_deep"),
                 "d": r.get("d_deep")}
                for r in refined[:16]]}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
