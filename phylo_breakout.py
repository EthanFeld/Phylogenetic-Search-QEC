"""Detect saturation-to-breakthrough patterns in the empirical search tree.

The score archive is empirical and its distance values are randomized upper
bounds.  This module only diagnoses search-policy transitions.  It never
promotes a candidate to a submission claim.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def _campaign_name(path: str) -> str:
    return Path(path).parent.name


def _weight(row: dict) -> int:
    return int(row.get("max_check_weight", 0) or 0)


def analyze(archive: dict) -> dict:
    rows = [row for row in archive.get("finds", [])
            if row.get("status") == "ok" and row.get("n") is not None]
    by_target_weight: dict[tuple[int, int], dict[int, list[dict]]] = defaultdict(
        lambda: defaultdict(list))
    campaigns_by_target: dict[tuple[int, int], set[str]] = defaultdict(set)
    for row in rows:
        target = (int(row["n"]), int(row["k"]))
        by_target_weight[target][_weight(row)].append(row)
        for source in row.get("observed_in", []):
            campaigns_by_target[target].add(_campaign_name(source))

    ladders = []
    breakthroughs = []
    for target, by_weight in sorted(by_target_weight.items()):
        levels = []
        for weight, members in sorted(by_weight.items()):
            best = max(int(member.get("d_upper", 0)) for member in members)
            winner = max(members, key=lambda member: int(member.get("d_upper", 0)))
            branches = sorted({
                (int(member.get("group_order", 0)),
                 int(member.get("group_index", 0)),
                 str(member.get("group", "unknown")))
                for member in members
            })
            levels.append({"max_check_weight": weight, "best_d_upper": best,
                           "rows": len(members), "branch_count": len(branches),
                           "branches": [list(branch) for branch in branches],
                           "winner_branch": [int(winner.get("group_order", 0)),
                                             int(winner.get("group_index", 0)),
                                             str(winner.get("group", "unknown"))]})
        ladders.append({"target": {"n": target[0], "k": target[1]},
                        "levels": levels,
                        "campaign_count": len(campaigns_by_target[target])})
        for previous, current in zip(levels, levels[1:]):
            delta = current["best_d_upper"] - previous["best_d_upper"]
            if delta > 0:
                breakthroughs.append({
                    "target": {"n": target[0], "k": target[1]},
                    "from": {"max_check_weight": previous["max_check_weight"],
                             "best_d_upper": previous["best_d_upper"]},
                    "to": {"max_check_weight": current["max_check_weight"],
                           "best_d_upper": current["best_d_upper"]},
                    "delta_d_upper": delta,
                    "operator": "check-weight/structural-branch jump",
                    "winner_branch": current["winner_branch"],
                })

    # Current GB discovery has an explicit saturation receipt.  Keep it in the
    # same report so the transfer from empirical tree to algebraic escape is
    # reproducible rather than hidden in prose.
    saturation = {}
    saturation_path = Path("results/gb_collapse_guard_01/discovery_saturation.json")
    if saturation_path.exists():
        try:
            saturation = json.loads(saturation_path.read_text())
        except (OSError, json.JSONDecodeError):
            saturation = {"status": "unreadable", "path": str(saturation_path)}

    return {
        "schema_version": "1.0",
        "kind": "phylogenetic_saturation_breakout_analysis",
        "distance_policy": "empirical RIS upper bounds; diagnosis only",
        "archive_rows": len(rows),
        "target_ladders": ladders,
        "breakthroughs": breakthroughs,
        "transferable_rules": [
            {
                "observed_pattern": "same family/weight plateaus while repeated local mutation changes little",
                "escape": "jump representation/branch; reserve diversity slots",
                "repo_action": "sample new algebraic divisors instead of more words from one ideal",
            },
            {
                "observed_pattern": "distance rises only after check-weight or monomial scope expands",
                "escape": "change structural family while retaining exact k/check gates",
                "repo_action": "divisor-driven cyclic branches; later cross-factored/mixed-monomial branch",
            },
            {
                "observed_pattern": "high-k attractors can look strong under short randomized screening",
                "escape": "exact/independent verification gates and robust multi-seed ranking",
                "repo_action": "keep collapse blacklist, holdout RIS, and official local gate",
            },
        ],
        "current_gb_saturation": saturation,
        "regulation": {
            "stage_only": True,
            "submission_sent": False,
            "git_commit_performed": False,
            "model_scores_do_not_certify_distance": True,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path,
                        default=Path("results/regulated_findings.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/phylogenetic_breakout_analysis.json"))
    args = parser.parse_args()
    report = analyze(json.loads(args.archive.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "breakthroughs": report["breakthroughs"],
                      "archive_rows": report["archive_rows"]}, indent=2))


if __name__ == "__main__":
    main()

