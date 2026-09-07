from __future__ import annotations

"""Build compact cross-campaign phylogenetic search telemetry."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from phylogenetic_search import upgma


def _literature_tree(catalog: dict) -> dict:
    """Make a provenance taxonomy without mixing paper rows into the score tree."""
    taxonomy = list(catalog.get("taxonomy", []))
    families = list(catalog.get("families", []))
    nodes = []
    edges = []
    for node in taxonomy:
        item = dict(node)
        item.setdefault("kind", "branch")
        nodes.append(item)
        if item.get("parent") is not None:
            edges.append({"parent": item["parent"], "child": item["id"]})
    for family in families:
        item = {
            "id": family.get("id"),
            "parent": family.get("parent"),
            "kind": "family",
            "label": family.get("label", family.get("id")),
            "challenge_family_tag": family.get("challenge_family_tag"),
            "repo_fit": family.get("repo_fit"),
            "search_priority": family.get("search_priority"),
            "primary_sources": family.get("primary_sources", []),
            "reference_instances": family.get("reference_instances", []),
        }
        nodes.append(item)
        if item.get("parent") is not None:
            edges.append({"parent": item["parent"], "child": item["id"]})
        for index, parameter_set in enumerate(family.get("reference_instances", [])):
            reference_id = f"{family.get('id')}::reference::{index}"
            nodes.append({
                "id": reference_id,
                "parent": family.get("id"),
                "kind": "reference_instance",
                "label": str(parameter_set),
                "parameter_set": str(parameter_set),
                "reference_only": True,
                "primary_sources": family.get("primary_sources", []),
            })
            edges.append({"parent": family.get("id"), "child": reference_id})
    roots = [node["id"] for node in nodes if node.get("parent") is None]
    return {
        "algorithm": "explicit-parent-provenance-taxonomy",
        "root": roots[0] if len(roots) == 1 else roots,
        "nodes": nodes,
        "edges": edges,
        "policy": "literature nodes are reference-only and never score or displace empirical leaves",
    }


def build_report(archive: dict, max_leaves: int = 64,
                 literature: dict | None = None,
                 branch_plan: dict | None = None) -> dict:
    rows = [row for row in archive.get("finds", [])
            if row.get("status") == "ok" and row.get("A") and row.get("B")]
    ranked = sorted(rows, key=lambda row: (
        -int(row.get("d_upper", -1)),
        -float(row.get("kd2_over_n_upper", 0.0)),
        int(row.get("n", 0)),
        str(row.get("semantic_hash", ""))))
    leaves = ranked[:max(0, int(max_leaves))]
    tree = upgma(leaves, max_leaves=max_leaves)
    tree.pop("cophenetic", None)
    tree["leaf_policy"] = "top_rows_by_d_then_kd2_over_n"

    groups = defaultdict(list)
    for row in rows:
        key = (int(row["n"]), int(row["k"]), row.get("group", "unknown"))
        groups[key].append(row)
    group_summary = []
    for (n, k, group), members in sorted(groups.items()):
        official = [m.get("evidence", {}).get("official") for m in members]
        passed = [x for x in official if x and x.get("passed")]
        group_summary.append({
            "n": n, "k": k, "group": group, "rows": len(members),
            "best_d_upper": max(int(m["d_upper"]) for m in members),
            "mean_d_upper": sum(int(m["d_upper"]) for m in members) / len(members),
            "best_kd2_over_n_upper": max(float(m.get("kd2_over_n_upper", 0.0))
                                          for m in members),
            "official_passes": len(passed),
            "board_ready": sum(bool(x.get("board_advancing")) for x in passed),
        })

    traced = [row for row in rows if row.get("phylogeny")]
    selected_min = []
    selected_mean = []
    for row in traced:
        for group in row["phylogeny"].get("group_traces", []):
            for trace in group.get("rounds", []):
                stats = trace.get("selected_pairwise_distance", {})
                if stats.get("min") is not None:
                    selected_min.append(float(stats["min"]))
                if stats.get("mean") is not None:
                    selected_mean.append(float(stats["mean"]))

    top_leaves = [{key: row.get(key) for key in (
        "semantic_hash", "n", "k", "d_upper", "kd2_over_n_upper", "group",
        "group_index", "group_order", "history_used") } for row in leaves]
    literature = literature or {}
    return {
        "schema_version": "1.1",
        "kind": "cross_campaign_phylogenetic_search_report",
        "source": "results/regulated_findings.json",
        "selection": {
            "max_leaves": int(max_leaves),
            "score_order": ["d_upper", "kd2_over_n_upper", "smaller_n"],
            "tree_algorithm": "UPGMA",
            "distance": "normalized support symmetric-difference; group/order mismatch=1",
        },
        "counts": {
            "rows": len(rows), "tree_leaves": len(leaves),
            "rows_with_lineage_telemetry": len(traced),
            "history_seeded_rows": sum(bool(row.get("history_used")) for row in rows),
            "groups": len(group_summary),
        },
        "tree": tree,
        "top_leaves": top_leaves,
        "group_summary": group_summary,
        "lineage_telemetry": {
            "greedy_pd_rounds": len(selected_min),
            "selected_pairwise_min_mean": (sum(selected_min) / len(selected_min)
                                            if selected_min else 0.0),
            "selected_pairwise_mean_mean": (sum(selected_mean) / len(selected_mean)
                                             if selected_mean else 0.0),
        },
        "literature": literature,
        "branch_discovery": branch_plan or {},
        "regulation": archive.get("regulation", {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path,
                        default=Path("results/regulated_findings.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/phylogenetic_tree.json"))
    parser.add_argument("--literature-catalog", type=Path,
                        default=Path("literature_code_families.json"))
    parser.add_argument("--max-leaves", type=int, default=64)
    parser.add_argument("--ml-plan", type=Path,
                        default=Path("results/ml_branch_plan.json"))
    args = parser.parse_args()
    catalog = {}
    if args.literature_catalog.exists():
        catalog = json.loads(args.literature_catalog.read_text())
    literature = {
        "source": args.literature_catalog.as_posix(),
        "retrieved_on": catalog.get("retrieved_on"),
        "compliance": catalog.get("compliance", {}),
        "families": catalog.get("families", []),
        "taxonomy": _literature_tree(catalog),
    }
    branch_plan = {}
    if args.ml_plan.exists():
        branch_plan = json.loads(args.ml_plan.read_text())
    report = build_report(json.loads(args.archive.read_text()), args.max_leaves,
                          literature=literature, branch_plan=branch_plan)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "counts": report["counts"],
                      "tree": report["tree"]["pairwise_distance"]}, indent=2))


if __name__ == "__main__":
    main()
