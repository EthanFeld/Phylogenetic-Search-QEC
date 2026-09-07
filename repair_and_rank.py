from __future__ import annotations

"""Repair qLDPC corpus validity, remove split leakage, rank search seeds.

GCD/ideal refutations become auditable hard negatives. They are excluded from
the live queue, retained in ``hard_negatives.jsonl``, and may train the model
through their effective witness-backed upper bound.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from gf2_factor import gcd as gf2_gcd


DEFAULT_CORPUS = Path("results/corpus")
DEFAULT_RESULTS = Path("results")
DEFAULT_OUT = Path("results/corpus_repaired")
TARGET_SCORE = 1542.0
MODEL_SEEDS = tuple(range(8))


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _support_poly(values: Any, modulus_degree: int) -> int:
    out = 0
    if isinstance(values, list) and values and isinstance(values[0], list):
        values = values[0]
    for value in values if isinstance(values, list) else []:
        out ^= 1 << (int(value) % int(modulus_degree))
    return out


def _algebraic_split_group(code: dict[str, Any]) -> str:
    """Group equivalent algebraic families before train/val/test hashing."""
    rep = code.get("representation") or {}
    n, k = int(code.get("n") or 0), int(code.get("k") or 0)
    kind = str(code.get("representation_kind") or code.get("family") or "unknown")
    a_values, b_values = rep.get("A"), rep.get("B")
    if a_values is None or b_values is None:
        a_values, b_values = rep.get("a"), rep.get("b")
    if a_values is not None and b_values is not None and n > 0:
        degree = n // 2 if n % 2 == 0 else n
        a, b = _support_poly(a_values, degree), _support_poly(b_values, degree)
        if a and b:
            modulus = (1 << degree) | 1
            family_gcd = gf2_gcd(gf2_gcd(a, b), modulus)
            return f"algebraic:{kind}:{n}:{k}:{hex(int(family_gcd))}"
    semantic = code.get("semantic_hash") or code.get("code_id")
    return f"fallback:{kind}:{n}:{k}:{semantic}"


def _norm_path(path: Path) -> str:
    return os.path.normcase(str(path.resolve())).replace("/", "\\")


def _source_path(results_root: Path, source: Any) -> Path | None:
    if not isinstance(source, str) or not source:
        return None
    path = Path(source)
    return path if path.is_absolute() else results_root / path


def _resolve(value: Any, results_root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    p = Path(value)
    if p.exists():
        return p.resolve()
    p = results_root / value
    if p.exists():
        return p.resolve()
    matches = list(results_root.rglob(Path(value).name))
    return matches[0].resolve() if len(matches) == 1 else None


def _is_gcd_refutation(report_path: Path, report: dict[str, Any]) -> bool:
    ref = report.get("refutation")
    source = ref.get("source") if isinstance(ref, dict) else None
    text = f"{report_path.name} {report_path.parent.name} {source or ''}".lower()
    return source == "common_gcd_ideal" or "gcd" in text or "ideal" in text


def build_invalidation_ledger(corpus: Path, results_root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    codes = _read_jsonl(corpus / "codes.jsonl")
    path_index: dict[str, set[str]] = defaultdict(set)
    for code in codes:
        for source in code.get("source_paths", []):
            source_path = _source_path(results_root, source)
            if source_path is not None:
                path_index[_norm_path(source_path)].add(code["code_id"])
    invalidations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reports_seen = 0
    linked = 0
    unmatched: list[str] = []
    for path in results_root.rglob("*.json"):
        try:
            in_corpus = path.resolve().is_relative_to(corpus.resolve())
        except AttributeError:
            in_corpus = str(path.resolve()).startswith(str(corpus.resolve()))
        if in_corpus:
            continue
        report = _load_json(path)
        status = report.get("status") if isinstance(report, dict) else None
        if not isinstance(report, dict) or not isinstance(status, str) or status not in {"refuted", "invalid", "failed"}:
            continue
        ref = report.get("refutation")
        if not isinstance(ref, dict):
            continue
        reports_seen += 1
        links: set[str] = set()
        for key in ("refined_candidate_path", "candidate_path", "source"):
            candidate = _resolve(report.get(key), results_root)
            if candidate is not None:
                links.update(path_index.get(_norm_path(candidate), set()))
        if not links:
            unmatched.append(path.as_posix())
            continue
        linked += 1
        kind = "gcd" if _is_gcd_refutation(path, report) else "witness"
        item = {
            "report_path": path.as_posix(),
            "kind": kind,
            "status": report.get("status"),
            "target_distance": report.get("target"),
            "side": ref.get("side"),
            "weight": ref.get("weight"),
            "source": ref.get("source"),
            "g_degree_K": ref.get("g_degree_K"),
            "h_degree": ref.get("h_degree"),
        }
        for code_id in links:
            invalidations[code_id].append(item)
    for values in invalidations.values():
        values.sort(key=lambda x: (x.get("kind", ""), x.get("weight") or 10**9, x["report_path"]))
    audit = {
        "reports_seen": reports_seen,
        "linked_reports": linked,
        "unmatched_reports": len(unmatched),
        "unmatched_paths": unmatched[:100],
        "codes_invalidated": len(invalidations),
        "gcd_invalidated_codes": sum(any(x["kind"] == "gcd" for x in v) for v in invalidations.values()),
        "witness_invalidated_codes": sum(any(x["kind"] == "witness" for x in v) for v in invalidations.values()),
    }
    return invalidations, audit


def _flat_support_stats(rep: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}

    def stats(prefix: str, rows: Any) -> None:
        if not isinstance(rows, list):
            return
        if rows and all(isinstance(x, (int, float)) for x in rows):
            rows = [rows]
        rows = [r for r in rows if isinstance(r, list)]
        if not rows:
            return
        weights = np.asarray([len(r) for r in rows], dtype=float)
        vals = np.asarray([x for r in rows for x in r if isinstance(x, (int, float))], dtype=float)
        out[f"{prefix}_rows"] = float(len(rows))
        out[f"{prefix}_weight_mean"] = float(weights.mean())
        out[f"{prefix}_weight_std"] = float(weights.std())
        out[f"{prefix}_weight_min"] = float(weights.min())
        out[f"{prefix}_weight_max"] = float(weights.max())
        if len(vals):
            out[f"{prefix}_value_mean"] = float(vals.mean())
            out[f"{prefix}_value_std"] = float(vals.std())
    for key in ("X", "Z", "A", "B", "a", "b"):
        stats(key, rep.get(key))
    if isinstance(rep.get("a"), list) and isinstance(rep.get("b"), list):
        a, b = set(rep["a"]), set(rep["b"])
        out["ab_intersection"] = float(len(a & b))
        out["ab_union"] = float(len(a | b))
        out["ab_symmetric_difference"] = float(len(a ^ b))
    return out


def _features(code: dict[str, Any], family_index: dict[str, int], kind_index: dict[str, int]) -> tuple[list[float], list[str]]:
    numeric = {
        "n": code.get("n"), "k": code.get("k"), "rate": code.get("rate"),
        "check_rows_x": code.get("check_rows_x"), "check_rows_z": code.get("check_rows_z"),
        "check_weight_min": code.get("check_weight_min"), "check_weight_max": code.get("check_weight_max"),
        "check_weight_mean": code.get("check_weight_mean"), "check_weight_total": code.get("check_weight_total"),
        "matrix_density": code.get("matrix_density"), "rank": code.get("rank"),
        "gcd_degree": code.get("gcd_degree"), "reflection_count": code.get("reflection_count"),
        "commutation_violations": code.get("commutation_violations"),
        # Curve telemetry is optional. Keep explicit zero-valued features for
        # unseen children while preserving signal for previously measured
        # corpus codes.
        "observed_distance": code.get("observed_distance"),
        "observed_distance_x": code.get("observed_distance_x"),
        "observed_distance_z": code.get("observed_distance_z"),
        "projected_final_distance": code.get("projected_final_distance"),
        "projection_low": code.get("projection_low"),
        "projection_high": code.get("projection_high"),
        "projection_points": code.get("projection_points"),
        "curve_point_count": code.get("curve_point_count"),
        "early_stop_seen": int(bool(code.get("early_stop_seen"))),
    }
    numeric.update(_flat_support_stats(code.get("representation") or {}))
    names = list(numeric)
    values = [float(numeric[k]) if numeric[k] is not None and math.isfinite(float(numeric[k])) else 0.0 for k in names]
    family = code.get("family", "unknown")
    kind = code.get("representation_kind", "unknown")
    for label, index, size, prefix in ((family, family_index, len(family_index), "family"), (kind, kind_index, len(kind_index), "kind")):
        values.extend([1.0 if index.get(label) == i else 0.0 for i in range(size)])
        names.extend([f"{prefix}_{key}" for key in sorted(index, key=index.get)])
    return values, names


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict)) else v for k, v in row.items() if k in fields})


def _score(n: Any, k: Any, d: Any) -> float | None:
    if not all(isinstance(x, (int, float)) and x > 0 for x in (n, k, d)):
        return None
    return float(k) * float(d) ** 2 / float(n)


def apply_external_gcd(invalidations: dict[str, list[dict[str, Any]]], screen_path: Path | None,
                       ladder_path: Path | None, codes: list[dict[str, Any]]) -> dict[str, int]:
    if screen_path is None or not screen_path.exists():
        return {"screen_rows": 0, "ladder_rows": 0, "target_refuting": 0, "survivors": 0}
    screen_rows = _read_jsonl(screen_path)
    screen_by_path = {_norm_path(Path(row["candidate_path"])): row for row in screen_rows if row.get("candidate_path")}
    ladder_rows: dict[str, dict[str, Any]] = {}
    if ladder_path is not None and ladder_path.exists():
        ladder = _load_json(ladder_path) or {}
        for row in ladder.get("candidates", []):
            if row.get("candidate_path"):
                ladder_rows[_norm_path(Path(row["candidate_path"]))] = row
    code_by_id = {row["code_id"]: row for row in codes}
    applied = 0
    target_refuting = 0
    survivors = 0
    for path_key, screen in screen_by_path.items():
        code_id = screen.get("code_id")
        if code_id not in code_by_id:
            continue
        ladder = ladder_rows.get(path_key)
        weight = (ladder or {}).get("final_best_ideal_weight") or screen.get("ideal_weight")
        if not isinstance(weight, (int, float)):
            continue
        code = code_by_id[code_id]
        score = _score(code.get("n"), code.get("k"), int(weight))
        refutes_target = score is not None and score < TARGET_SCORE
        ladder_series = (ladder or {}).get("series", [])
        witness = (ladder_series[-1].get("new_witness") if ladder_series else None)
        if witness is None:
            witness = ((screen.get("report") or {}).get("found_X_logical") or {}).get("block_support")
        item = {
            "report_path": str(ladder_path or screen_path), "kind": "gcd",
            "status": "refuted" if refutes_target else "screened_survivor",
            "target_distance": code.get("observed_distance"), "side": "X",
            "weight": int(weight), "source": "common_gcd_ideal_ladder" if ladder else "common_gcd_ideal",
            "trials_per_side": (max((int(x.get("trials_per_side", 0)) for x in (ladder or {}).get("series", [])), default=0)
                                if ladder else screen.get("trials")),
            "ideal_score_upper": score, "target_refuting": bool(refutes_target),
            "witness": witness,
        }
        invalidations[code_id].append(item)
        applied += 1
        target_refuting += int(refutes_target)
        survivors += int(not refutes_target)
    return {"screen_rows": len(screen_rows), "ladder_rows": len(ladder_rows),
            "target_refuting": target_refuting, "survivors": survivors, "applied": applied}


def run(corpus: Path = DEFAULT_CORPUS, results_root: Path = DEFAULT_RESULTS, out: Path = DEFAULT_OUT,
        gcd_screens: list[Path] | None = None, gcd_ladders: list[Path] | None = None) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    progress = {"stage": "audit", "percent": 10, "updated_utc": datetime.now(timezone.utc).isoformat()}
    (out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    codes = _read_jsonl(corpus / "codes.jsonl")
    runs = _read_jsonl(corpus / "runs.jsonl")
    curves = _read_jsonl(corpus / "curves.jsonl")
    invalidations, audit = build_invalidation_ledger(corpus, results_root)
    external_audit = {"screen_rows": 0, "ladder_rows": 0, "target_refuting": 0, "survivors": 0, "applied": 0}
    for index, screen_path in enumerate(gcd_screens or []):
        ladder_path = (gcd_ladders or [])[index] if index < len(gcd_ladders or []) else None
        current = apply_external_gcd(invalidations, screen_path, ladder_path, codes)
        for key in external_audit:
            external_audit[key] += current.get(key, 0)
    audit["external_gcd"] = external_audit
    audit["gcd_invalidated_codes"] = sum(
        any(x["kind"] == "gcd" and x.get("target_refuting", True) for x in values)
        for values in invalidations.values())
    progress.update({"stage": "gcd_ledger", "percent": 35, "audit": audit})
    (out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")

    sem_groups: dict[str, str] = {}
    algebraic_groups: dict[str, str] = {}
    for code in codes:
        sem = code.get("semantic_hash")
        sem_groups[code["code_id"]] = f"semantic:{sem}" if sem else f"code:{code['code_id']}"
        algebraic_groups[code["code_id"]] = _algebraic_split_group(code)
    group_bucket: dict[str, str] = {}
    for group in sorted(set(algebraic_groups.values())):
        bucket = int(_sha(group)[:8], 16) % 100
        group_bucket[group] = "test" if bucket < 10 else "validation" if bucket < 20 else "train"

    repaired: list[dict[str, Any]] = []
    for code in codes:
        reports = invalidations.get(code["code_id"], [])
        gcd_reports = [x for x in reports if x["kind"] == "gcd"]
        all_weights = [int(x["weight"]) for x in reports if isinstance(x.get("weight"), (int, float))]
        effective_d = code.get("observed_distance")
        if all_weights:
            distance_candidates = ([effective_d] if isinstance(effective_d, (int, float)) else []) + all_weights
            effective_d = min(distance_candidates) if distance_candidates else None
        target_refuting_gcd = [x for x in gcd_reports if x.get("target_refuting", True)]
        status = ("invalidated_gcd" if target_refuting_gcd else "gcd_screened_survivor" if gcd_reports
                  else "invalidated_witness" if reports else "not_refuted")
        item = dict(code)
        item.update({
            "validity_status": status,
            "gcd_invalidated": bool(target_refuting_gcd),
            "gcd_screened": bool(gcd_reports),
            "hard_negative": bool(reports),
            "invalidation_count": len(reports),
            "gcd_invalidation_count": len(gcd_reports),
            "invalidation_reports": reports,
            "effective_distance_upper": effective_d,
            "effective_score_upper": _score(code.get("n"), code.get("k"), effective_d),
            "semantic_group": sem_groups[code["code_id"]],
            "algebraic_group": algebraic_groups[code["code_id"]],
            "repaired_group": algebraic_groups[code["code_id"]],
            "split_repaired": group_bucket[algebraic_groups[code["code_id"]]],
        })
        repaired.append(item)
    repaired.sort(key=lambda x: x["code_id"])
    valid = [x for x in repaired if x["validity_status"] == "not_refuted"]
    screened = [x for x in repaired if x["validity_status"] == "gcd_screened_survivor"]
    hard = [x for x in repaired if x["hard_negative"]]
    valid_ids = {x["code_id"] for x in valid}
    _write_jsonl(out / "codes.jsonl", repaired)
    _write_jsonl(out / "valid_codes.jsonl", valid)
    _write_jsonl(out / "gcd_screened_survivors.jsonl", screened)
    _write_jsonl(out / "hard_negatives.jsonl", hard)
    _write_jsonl(out / "invalidation_ledger.jsonl", [
        {"code_id": x["code_id"], "validity_status": x["validity_status"], "reports": x["invalidation_reports"]}
        for x in repaired if x["invalidation_reports"]
    ])
    _write_jsonl(out / "runs.jsonl", [dict(x, split_repaired=group_bucket[algebraic_groups[x["code_id"]]]) for x in runs if x["code_id"] in valid_ids])
    _write_jsonl(out / "curves.jsonl", [dict(x, split_repaired=group_bucket[algebraic_groups[x["code_id"]]]) for x in curves if x["code_id"] in valid_ids])
    coverage_by_code: dict[str, list[int]] = defaultdict(list)
    for row in curves:
        if row.get("code_id") not in valid_ids:
            continue
        trials = row.get("trials_per_side")
        if isinstance(trials, (int, float)) and trials > 0:
            coverage_by_code[row["code_id"]].append(int(trials))
    progress.update({"stage": "leakage_repaired", "percent": 55, "valid_codes": len(valid), "hard_negatives": len(hard), "split_groups": len(set(sem_groups.values()))})
    (out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")

    trainable = [x for x in repaired if x.get("effective_score_upper") is not None]
    family_index = {name: i for i, name in enumerate(sorted({x.get("family", "unknown") for x in trainable}))}
    kind_index = {name: i for i, name in enumerate(sorted({x.get("representation_kind", "unknown") for x in trainable}))}
    feature_rows = []
    feature_maps: list[tuple[dict[str, float], dict[str, Any]]] = []
    name_set: set[str] = set()
    for code in trainable:
        row, row_names = _features(code, family_index, kind_index)
        row_map = dict(zip(row_names, row))
        name_set.update(row_map)
        feature_maps.append((row_map, code))
    names = sorted(name_set)
    feature_rows = [(code, [row_map.get(name, 0.0) for name in names]) for row_map, code in feature_maps]
    X = np.asarray([x[1] for x in feature_rows], dtype=float)
    y = np.asarray([x[0]["effective_score_upper"] for x in feature_rows], dtype=float)
    test_mask = np.asarray([x[0]["split_repaired"] == "test" for x in feature_rows])
    train_mask = np.asarray([x[0]["split_repaired"] == "train" for x in feature_rows])
    models = []
    predictions = np.zeros((len(feature_rows), len(MODEL_SEEDS)), dtype=float)
    for column, seed in enumerate(MODEL_SEEDS):
        model = ExtraTreesRegressor(n_estimators=300, random_state=seed, min_samples_leaf=3, max_features=0.75, n_jobs=-1)
        model.fit(X[train_mask], y[train_mask])
        predictions[:, column] = model.predict(X)
        models.append(model)
    val_mask = np.asarray([x[0]["split_repaired"] == "validation" for x in feature_rows])
    eval_mask = test_mask | val_mask
    metrics = {
        "train_rows": int(train_mask.sum()), "validation_rows": int(val_mask.sum()), "test_rows": int(test_mask.sum()),
        "mae": float(mean_absolute_error(y[eval_mask], predictions[eval_mask].mean(axis=1))) if eval_mask.any() else None,
        "rmse": float(math.sqrt(mean_squared_error(y[eval_mask], predictions[eval_mask].mean(axis=1)))) if eval_mask.any() else None,
    }
    model_meta = {"feature_names": names, "family_index": family_index, "kind_index": kind_index, "target": "effective_score_upper", "metrics": metrics, "seeds": list(MODEL_SEEDS)}
    (out / "model_metadata.json").write_text(json.dumps(model_meta, indent=2) + "\n", encoding="utf-8")

    queue = []
    for (code, _), preds in zip(feature_rows, predictions):
        if code["validity_status"] not in {"not_refuted", "gcd_screened_survivor"}:
            continue
        n, k = code.get("n"), code.get("k")
        if not isinstance(n, int) or not isinstance(k, int) or k <= 0:
            continue
        threshold_d = math.sqrt(TARGET_SCORE * n / k)
        observed = code.get("effective_score_upper") or 0.0
        mean, std = float(preds.mean()), float(preds.std())
        prob = float(np.mean(preds >= TARGET_SCORE))
        # Trial coverage is used to prioritize candidates needing deeper runs.
        coverage = coverage_by_code.get(code["code_id"], [])
        max_trials = max((int(x) for x in coverage), default=0)
        distinct_trials = len(set(coverage))
        queue.append({
            "code_id": code["code_id"], "family": code.get("family"), "n": n, "k": k,
            "required_distance_for_1542": math.ceil(threshold_d),
            "observed_score_upper": round(observed, 4),
            "predicted_score_mean": round(mean, 4), "predicted_score_std": round(std, 4),
            "predicted_qualify_fraction": round(prob, 4),
            "predicted_margin_score": round(mean - TARGET_SCORE, 4),
            "max_trials_per_side": max_trials, "curve_points": len(coverage),
            "trial_coverage_status": "multi_budget" if distinct_trials > 1 else "single_budget" if coverage else "none",
            "split_repaired": code["split_repaired"], "source_first": code.get("source_first"),
            "gcd_gate_status": code["validity_status"], "gcd_screened": code.get("gcd_screened", False),
            "action": "fresh_gcd_check_then_deep_search",
        })
    queue.sort(key=lambda x: (-x["predicted_qualify_fraction"], -x["predicted_margin_score"], x["n"], -x["k"]))
    _write_jsonl(out / "search_queue.jsonl", queue[:1000])
    _write_csv(out / "search_queue.csv", queue[:1000], list(queue[0]) if queue else ["code_id"])
    _write_csv(out / "codes.csv", repaired, ["code_id", "split_repaired", "validity_status", "gcd_invalidated", "hard_negative", "n", "k", "rate", "family", "representation_kind", "observed_distance", "effective_distance_upper", "observed_distance_x", "observed_distance_z", "effective_score_upper", "invalidation_count", "gcd_invalidation_count", "source_first"])

    progress = {
        "stage": "baseline_ml_queue_ready", "percent": 70,
        "updated_utc": datetime.now(timezone.utc).isoformat(), "audit": audit,
        "codes": len(repaired), "valid_codes": len(valid), "gcd_screened_survivors": len(screened), "hard_negatives": len(hard),
        "queue_rows": min(1000, len(queue)), "metrics": metrics,
        "best_valid_observed_score": max((x.get("effective_score_upper") or 0 for x in valid), default=None),
        "best_gcd_screened_score": max((x.get("effective_score_upper") or 0 for x in screened), default=None),
        "next": "mutate top queue seeds, rerun GCD/ideal validity, staged RIS, fresh-lineage holdout",
    }
    (out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    (out / "repair_report.json").write_text(json.dumps({"schema_version": "1.0", "target_score": TARGET_SCORE, "audit": audit, "validity_counts": Counter(x["validity_status"] for x in repaired), "split_counts": Counter(x["split_repaired"] for x in repaired), "model": model_meta, "progress": progress}, indent=2, default=dict) + "\n", encoding="utf-8")
    return progress


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--gcd-screen", type=Path, action="append", default=[])
    ap.add_argument("--gcd-ladder", type=Path, action="append", default=[])
    args = ap.parse_args()
    progress = run(args.corpus, args.results, args.out, args.gcd_screen, args.gcd_ladder)
    print(json.dumps(progress, indent=2, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
