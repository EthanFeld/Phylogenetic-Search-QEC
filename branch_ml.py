"""Torch branch-priority model for qLDPC phylogenetic campaigns.

Model ranks search opportunities, never certifies codes. It uses archive rows
for empirical priors, then adds uncertainty and undercoverage bonuses so
missing branches get sampled before exploitation campaigns.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

from inverse_design_core import candidate_groups_of_order


INSTANCE_RE = re.compile(r"\[\[(\d+)\s*,\s*(\d+)\s*,\s*(?:<=\s*)?(\d+)\]\]")
WEIGHTS = (9, 15, 21, 27)
_GROUP_CACHE = {}


def _instances(catalog):
    out = []
    for family in catalog.get("families", []):
        for text in family.get("reference_instances", []):
            match = INSTANCE_RE.search(str(text))
            if match:
                n, k, d = map(int, match.groups())
                out.append({"family": family.get("id"), "n": n, "k": k,
                            "d": d, "tag": family.get("challenge_family_tag"),
                            "priority": family.get("search_priority"),
                            "repo_fit": family.get("repo_fit"),
                            "source": family.get("primary_sources", [])})
    return out


def _features(n, k, weight, group_order, group_index, abelian, count):
    return [float(n) / 1000.0, float(k) / 100.0, float(weight) / 32.0,
            float(group_order) / 128.0, float(group_index) / 32.0,
            float(abelian), math.log1p(float(count)) / 4.0]


def _fit_ensemble(x, y, seeds=5):
    if torch is None or len(x) < 8:
        return None
    xt = torch.tensor(np.asarray(x), dtype=torch.float32)
    yt = torch.tensor(np.asarray(y).reshape(-1, 1) / 32.0, dtype=torch.float32)
    models = []
    for seed in range(int(seeds)):
        torch.manual_seed(1701 + seed)
        model = nn.Sequential(nn.Linear(7, 24), nn.Tanh(), nn.Linear(24, 16),
                              nn.Tanh(), nn.Linear(16, 1))
        opt = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-4)
        for _ in range(220):
            loss = ((model(xt) - yt) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        models.append(model.eval())
    return models


def _predict(models, features, fallback):
    if not models:
        return float(fallback), 0.0
    xt = torch.tensor(np.asarray([features]), dtype=torch.float32)
    values = [float(model(xt).detach().cpu().item() * 32.0) for model in models]
    return float(np.mean(values)), float(np.std(values))


def _group_for(order, index):
    order = int(order)
    if order not in _GROUP_CACHE:
        _GROUP_CACHE[order] = candidate_groups_of_order(order)
    groups = _GROUP_CACHE[order]
    index = int(index)
    return groups[index] if 0 <= index < len(groups) else None


def _usable_rows(archive):
    """Keep empirical rows, excluding candidates officially refuted later."""
    rows = []
    for row in archive.get("finds", []):
        if row.get("status") != "ok" or not row.get("n") or not row.get("k"):
            continue
        official = row.get("evidence", {}).get("official") or {}
        if (row.get("regulation", {}).get("official_gate_status") == "refuted" or
                official.get("refuted", False)):
            continue
        rows.append(row)
    return rows


def _refuted_branch_keys(archive):
    keys = set()
    for row in archive.get("finds", []):
        official = row.get("evidence", {}).get("official") or {}
        if (row.get("regulation", {}).get("official_gate_status") != "refuted" and
                not official.get("refuted", False)):
            continue
        try:
            keys.add((int(row["n"]), int(row["k"]),
                      int(row.get("max_check_weight", 0)),
                      int(row.get("group_order", 0)),
                      int(row.get("group_index", 0))))
        except (KeyError, TypeError, ValueError):
            continue
    return keys


def _branch_records(rows):
    """Aggregate candidates by complete branch; search-count stays out of ML."""
    grouped = defaultdict(list)
    for row in rows:
        key = (int(row["n"]), int(row["k"]), int(row.get("max_check_weight", 0)),
               int(row.get("group_order", 0)), int(row.get("group_index", 0)))
        grouped[key].append(row)
    records = []
    for key, values in sorted(grouped.items()):
        n, k, weight, order, index = key
        group = _group_for(order, index)
        if group is None:
            continue
        records.append({
            "key": key,
            "features": _features(n, k, weight, order, index,
                                   group.is_abelian, 0),
            "label": max(float(row.get("d_upper", 0)) for row in values),
            "count": len(values),
            "group": group.name,
        })
    return records


def _rank(values):
    order = np.argsort(np.asarray(values), kind="stable")
    result = np.empty(len(order), dtype=float)
    result[order] = np.arange(len(order), dtype=float)
    return result


def _oos_branch_validation(records, folds=5):
    """Grouped holdout: each complete branch hidden from its training fold."""
    if len(records) < 10:
        return {"method": "grouped-branch-holdout", "status": "insufficient_data",
                "branch_count": len(records)}
    folds = max(2, min(int(folds), len(records)))
    actual, predicted, held = [], [], []
    for fold in range(folds):
        test = [r for r in records if
                (r["key"][0] + 3 * r["key"][1] + 5 * r["key"][2] +
                 7 * r["key"][3] + 11 * r["key"][4]) % folds == fold]
        train = [r for r in records if r not in test]
        if not test or len(train) < 8:
            continue
        models = _fit_ensemble([r["features"] for r in train],
                               [r["label"] for r in train], seeds=3)
        for record in test:
            mean, uncertainty = _predict(models, record["features"], record["label"])
            actual.append(record["label"])
            predicted.append(mean)
            held.append({"key": list(record["key"]), "group": record["group"],
                         "actual_best_d": round(record["label"], 4),
                         "predicted_d": round(mean, 4),
                         "uncertainty": round(uncertainty, 4)})
    if len(actual) < 2:
        return {"method": "grouped-branch-holdout", "status": "insufficient_data",
                "branch_count": len(records), "tested_branches": len(actual)}
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    rank_actual, rank_predicted = _rank(actual), _rank(predicted)
    rank_corr = float(np.corrcoef(rank_actual, rank_predicted)[0, 1])
    top = {}
    for size in (1, 5, 10):
        size = min(size, len(actual))
        predicted_top = set(np.argsort(predicted)[-size:])
        actual_top = set(np.argsort(actual)[-size:])
        top[str(size)] = round(len(predicted_top & actual_top) / size, 4)
    held.sort(key=lambda r: (-r["predicted_d"], r["key"]))
    return {
        "method": "grouped-branch-holdout",
        "status": "ok",
        "folds": folds,
        "branch_count": len(records),
        "tested_branches": len(actual),
        "mae": round(float(np.mean(np.abs(actual - predicted))), 4),
        "rmse": round(float(np.sqrt(np.mean((actual - predicted) ** 2))), 4),
        "rank_correlation": round(rank_corr, 4) if np.isfinite(rank_corr) else 0.0,
        "top_recall": top,
        "predicted_top_examples": held[:12],
    }


def build_plan(archive, catalog):
    rows = _usable_rows(archive)
    refuted_keys = _refuted_branch_keys(archive)
    branch_records = _branch_records(rows)
    x = [record["features"] for record in branch_records]
    y = [record["label"] for record in branch_records]
    models = _fit_ensemble(x, y)

    targets = sorted({(int(row["n"]), int(row["k"])) for row in rows
                      if int(row["n"]) in (300, 350)})
    branch_rows = []
    for n, k in targets:
        order = n // 5
        groups = candidate_groups_of_order(order)
        for weight in WEIGHTS:
            for gi, group in enumerate(groups):
                observed = [row for row in rows if int(row["n"]) == n and
                            int(row["k"]) == k and
                            int(row.get("max_check_weight", 0)) == weight and
                            int(row.get("group_order", -1)) == order and
                            int(row.get("group_index", -1)) == gi]
                count = len(observed)
                best = max((int(row.get("d_upper", 0)) for row in observed), default=0)
                key = (n, k, weight, order, gi)
                mean, uncertainty = _predict(
                    models, _features(n, k, weight, order, gi, group.is_abelian, 0),
                    best or (k / 10.0))
                undercoverage = 1.0 / math.sqrt(1.0 + count)
                branch_rows.append({
                    "branch": "regular-group", "n": n, "k": k,
                    "max_check_weight": weight, "group_index": gi,
                    "group": group.name, "observed_rows": count,
                    "refuted_rows": int(key in refuted_keys),
                    "observed_best_d": best, "ml_predicted_d": round(mean, 4),
                    "ml_uncertainty": round(uncertainty, 4),
                    "undercoverage": round(undercoverage, 4),
                    "priority_score": round(mean + uncertainty + 4.0 * undercoverage, 4),
                    "status": ("refuted_seen" if key in refuted_keys else
                               "missing" if count == 0 else "seeded"),
                })
    branch_rows.sort(key=lambda row: (-row["priority_score"], row["n"],
                                      row["max_check_weight"], row["group_index"]))

    family_rows = []
    instances = _instances(catalog)
    for family in catalog.get("families", []):
        refs = [item for item in instances if item["family"] == family.get("id")]
        max_ref_score = max((item["k"] * item["d"] ** 2 / item["n"]
                             for item in refs), default=0.0)
        observed = [row for row in rows
                    if family.get("challenge_family_tag") in
                    {"lifted-product", "generalized-bicycle", "2bga-coset"}
                    and row.get("n") in {item["n"] for item in refs}]
        family_rows.append({
            "family": family.get("id"), "label": family.get("label"),
            "repo_fit": family.get("repo_fit"),
            "search_priority": family.get("search_priority"),
            "reference_count": len(refs),
            "reference_best_kd2_over_n": round(max_ref_score, 4),
            "matching_archive_rows": len(observed),
            "ml_branch_score": round(max_ref_score + (3.0 if not observed else 0.0) +
                                     (1.0 if family.get("search_priority") == "highest" else 0.0), 4),
            "action": "implement_or_seed" if not observed else "exploit_after_branch_fill",
            "sources": family.get("primary_sources", []),
        })
    family_rows.sort(key=lambda row: (-row["ml_branch_score"], row["family"] or ""))
    return {
        "schema_version": "1.0", "kind": "ml_guided_qldpc_branch_plan",
        "model": "torch-ensemble-MLP" if models else "coverage-prior-fallback",
        "warning": "Heuristic prioritization only; grouped holdout is diagnostic; no model score certifies distance or validity.",
        "training": {"rows": len(rows), "branch_rows": len(branch_records),
                      "ensemble_size": len(models or []),
                      "features": ["n", "k", "weight", "group_order", "group_index",
                                   "abelian", "log1p_count"]},
        "oos_validation": _oos_branch_validation(branch_records),
        "recommended_missing_branches": [row for row in branch_rows
                                          if row["status"] == "missing"][:32],
        "recommended_seeded_branches": [row for row in branch_rows
                                        if row["status"] == "seeded"][:32],
        "literature_family_prior": family_rows,
        "literature_instances": instances,
        "refuted_branch_count": sum(row["status"] == "refuted_seen"
                                     for row in branch_rows),
        "refuted_branches": [row for row in branch_rows
                             if row["status"] == "refuted_seen"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("results/regulated_findings.json"))
    parser.add_argument("--catalog", type=Path, default=Path("literature_code_families.json"))
    parser.add_argument("--out", type=Path, default=Path("results/ml_branch_plan.json"))
    args = parser.parse_args()
    plan = build_plan(json.loads(args.archive.read_text()), json.loads(args.catalog.read_text()))
    args.out.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "model": plan["model"],
                      "missing": len(plan["recommended_missing_branches"]),
                      "families": len(plan["literature_family_prior"])}, indent=2))


if __name__ == "__main__":
    main()
