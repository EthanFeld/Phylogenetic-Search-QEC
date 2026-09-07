from __future__ import annotations

"""Build an ML-ready corpus from heterogeneous qLDPC search artifacts.

Outputs four linked tables below ``results/corpus``:

* ``codes.jsonl``: one row per exact code representation, with structural
  features, construction genome, provenance, and deterministic split.
* ``runs.jsonl``: every observed search/validation run linked by ``code_id``.
* ``curves.jsonl``: best distance versus actual trial budget, per sector and
  combined CSS distance.
* ``manifest.json`` and CSV mirrors: schema, counts, projection policy.

Distance values are witness-backed randomized upper bounds unless an artifact
explicitly says otherwise. A projected terminal distance is a model estimate,
not a certificate.
"""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROOT = Path("results")
DEFAULT_OUT = Path("results/corpus")
DEFAULT_PROJECTION_TRIALS = 20_000_000
SCHEMA_VERSION = "1.0"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _int(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _float(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _clean_supports(value: Any) -> list[list[int]] | None:
    if not isinstance(value, list):
        return None
    out: list[list[int]] = []
    for row in value:
        if not isinstance(row, list):
            return None
        vals = [_int(x) for x in row]
        if any(x is None for x in vals):
            return None
        out.append(sorted(set(int(x) for x in vals)))
    return out


def _is_checks(value: Any) -> bool:
    return isinstance(value, dict) and _clean_supports(value.get("X")) is not None and _clean_supports(value.get("Z")) is not None


def _path_label(parts: tuple[str, ...]) -> str:
    return ".".join(parts) if parts else "$"


def _candidate_rep(obj: dict[str, Any], context: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    checks = obj.get("checks")
    if _is_checks(checks):
        return "checks", {"X": _clean_supports(checks["X"]), "Z": _clean_supports(checks["Z"])}
    if isinstance(obj.get("a"), list) and isinstance(obj.get("b"), list):
        return "cyclic_genome", {"a": obj["a"], "b": obj["b"]}
    if isinstance(obj.get("A"), list) and isinstance(obj.get("B"), list):
        return "ring_genome", {"A": obj["A"], "B": obj["B"]}
    return None


def _candidate_context(obj: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    ctx = dict(parent)
    for key in ("n", "k", "m", "family", "code_type", "group", "group_order", "group_index"):
        if key in obj and obj[key] is not None:
            ctx[key] = obj[key]
    target = obj.get("target")
    if isinstance(target, dict):
        for key in ("n", "k", "m"):
            if key in target and target[key] is not None:
                ctx[key] = target[key]
    if "n" not in ctx and _int(ctx.get("m")) is not None:
        ctx["n"] = 2 * int(ctx["m"])
    return ctx


def _family(obj: dict[str, Any], ctx: dict[str, Any], path: Path) -> str:
    for source in (obj, ctx):
        for key in ("family", "code_family", "construction_family"):
            if source.get(key):
                return str(source[key])
    text = f"{path.name} {obj.get('name', '')} {obj.get('code_type', '')}".lower()
    if "dihedral" in text or "2bga" in text:
        return "dihedral-2bga"
    if isinstance(obj.get("a"), list) and isinstance(obj.get("b"), list):
        return "generalized-bicycle"
    if isinstance(obj.get("A"), list) and isinstance(obj.get("B"), list):
        return "group-ring-product"
    return "unknown"


def _distance(obj: dict[str, Any]) -> dict[str, int | None]:
    dx = _int(obj.get("dx_upper"))
    dz = _int(obj.get("dz_upper"))
    d = _int(obj.get("d_upper"))
    distance = obj.get("distance")
    if isinstance(distance, dict):
        d = _int(distance.get("d")) if _int(distance.get("d")) is not None else d
        x, z = distance.get("X"), distance.get("Z")
        if isinstance(x, dict):
            dx = _int(x.get("value")) if _int(x.get("value")) is not None else dx
        if isinstance(z, dict):
            dz = _int(z.get("value")) if _int(z.get("value")) is not None else dz
    screen = obj.get("screen")
    if isinstance(screen, dict):
        for key, name in (("d_upper", "d"), ("dx_upper", "dx"), ("dz_upper", "dz")):
            value = _int(screen.get(key))
            if name == "d" and value is not None:
                d = value
            elif name == "dx" and value is not None:
                dx = value
            elif name == "dz" and value is not None:
                dz = value
    if d is None and dx is not None and dz is not None:
        d = min(dx, dz)
    return {"dx": dx, "dz": dz, "d": d}


def _witnesses(obj: dict[str, Any]) -> dict[str, list[int] | None]:
    out: dict[str, list[int] | None] = {"X": None, "Z": None}
    for side, key in (("X", "witness_x"), ("Z", "witness_z")):
        if isinstance(obj.get(key), list):
            out[side] = [_int(x) for x in obj[key] if _int(x) is not None]
    distance = obj.get("distance")
    if isinstance(distance, dict):
        for side in ("X", "Z"):
            value = distance.get(side)
            if isinstance(value, dict) and isinstance(value.get("witness"), list):
                out[side] = [_int(x) for x in value["witness"] if _int(x) is not None]
    for container_key in ("screen_witnesses", "witnesses"):
        container = obj.get(container_key)
        if isinstance(container, dict):
            for side in ("X", "Z"):
                value = container.get(side)
                if isinstance(value, list):
                    out[side] = [_int(x) for x in value if _int(x) is not None]
                elif isinstance(value, dict) and isinstance(value.get("witness"), list):
                    out[side] = [_int(x) for x in value["witness"] if _int(x) is not None]
    return out


def _trial_info(obj: dict[str, Any]) -> dict[str, Any]:
    screen = obj.get("screen") if isinstance(obj.get("screen"), dict) else {}
    search = obj.get("search") if isinstance(obj.get("search"), dict) else {}
    source = screen or search
    per_side = None
    for key in ("trials_per_side", "trials_per_sector", "trials", "configured_total_trials_per_side",
                "requested_trials_per_side", "total_trials_per_side"):
        per_side = _int(source.get(key))
        if per_side is None:
            per_side = _int(obj.get(key))
        if per_side is not None:
            break
    if per_side is None:
        per_side = _int(obj.get("confirm_trials_per_sector"))
    x_trials = _int(source.get("x_trials"))
    z_trials = _int(source.get("z_trials"))
    if x_trials is None:
        x_trials = _int(source.get("x_trials_total"))
    if z_trials is None:
        z_trials = _int(source.get("z_trials_total"))
    if x_trials is None:
        x_trials = _int(obj.get("x_trials"))
    if z_trials is None:
        z_trials = _int(obj.get("z_trials"))
    if x_trials is None:
        x_trials = _int(obj.get("x_trials_total"))
    if z_trials is None:
        z_trials = _int(obj.get("z_trials_total"))
    if x_trials is not None or z_trials is not None:
        per_side = max(x_trials or 0, z_trials or 0)
    if per_side is None and x_trials is not None and z_trials is not None:
        per_side = max(x_trials, z_trials)
    total = (x_trials + z_trials) if x_trials is not None and z_trials is not None else (2 * per_side if per_side is not None else None)
    requested = None
    for key in ("requested_trials_per_side", "requested_trials_per_side_total",
                "requested_budget_per_side", "target_trials_per_side"):
        requested = _int(source.get(key))
        if requested is None:
            requested = _int(obj.get(key))
        if requested is not None:
            if key.endswith("_total"):
                requested = max(1, requested // 2)
            break
    actual = max(x_trials or 0, z_trials or 0, per_side or 0)
    stopped = source.get("stopped_early")
    if stopped is None:
        stopped = obj.get("stopped_early")
    status = str(obj.get("status", "")).lower()
    if stopped is None and status in {"refuted", "target_refuted"}:
        stopped = bool(requested is not None and actual < requested)
    return {
        "trials_per_side": per_side,
        "x_trials": x_trials,
        "z_trials": z_trials,
        "trials_total": total,
        "requested_trials_per_side": requested,
        "incomplete_budget": bool(requested is not None and actual < requested),
        "early_stopped": bool(stopped) if stopped is not None else None,
        "seed": _int(source.get("seed")) if source else _int(obj.get("seed")),
    }


def _structural(rep: dict[str, Any], obj: dict[str, Any], n: int | None) -> dict[str, Any]:
    x, z = rep.get("X"), rep.get("Z")
    all_weights: list[int] = []
    rows_x = rows_z = None
    if isinstance(x, list) and isinstance(z, list):
        rows_x, rows_z = len(x), len(z)
        all_weights = [len(row) for row in x + z]
    elif rep.get("a") is not None and rep.get("b") is not None:
        a, b = rep["a"], rep["b"]
        all_weights = [len(a), len(b)] if isinstance(a, list) and isinstance(b, list) else []
    elif rep.get("A") is not None and rep.get("B") is not None:
        a, b = rep["A"], rep["B"]
        if isinstance(a, list) and isinstance(b, list):
            all_weights = [len(row) for row in a + b if isinstance(row, list)]
    total = sum(all_weights) if all_weights else None
    return {
        "check_rows_x": rows_x,
        "check_rows_z": rows_z,
        "check_weight_min": min(all_weights) if all_weights else _int(obj.get("check_weight")),
        "check_weight_max": max(all_weights) if all_weights else _int(obj.get("check_weight")),
        "check_weight_mean": (sum(all_weights) / len(all_weights)) if all_weights else None,
        "check_weight_total": total,
        "matrix_density": (total / (n * (rows_x + rows_z))) if n and rows_x is not None and rows_z is not None and rows_x + rows_z else None,
        "rank": _int(obj.get("rank")),
        "gcd_degree": _int(obj.get("gcd_degree")),
        "reflection_count": _int(obj.get("reflection_count")),
        "commutation_violations": _int(obj.get("commutation_violations")),
    }


def _representation_key(family: str, ctx: dict[str, Any], rep: dict[str, Any]) -> str:
    # Exclude non-semantic labels. Exact duplicate artifacts then collapse.
    if "X" in rep:
        body = {"family": family, "n": _int(ctx.get("n")), "X": rep["X"], "Z": rep["Z"]}
    elif "a" in rep:
        body = {"family": family, "m": _int(ctx.get("m")) or (_int(ctx.get("n")) // 2 if _int(ctx.get("n")) else None), "a": rep["a"], "b": rep["b"]}
    else:
        body = {"family": family, "n": _int(ctx.get("n")), "k": _int(ctx.get("k")), "A": rep["A"], "B": rep["B"], "group_order": _int(ctx.get("group_order")), "group_index": _int(ctx.get("group_index"))}
    return _sha(body)


def _walk_candidates(obj: Any, path: Path, root: Path, out: list[dict[str, Any]], context: dict[str, Any] | None = None, parts: tuple[str, ...] = ()) -> None:
    if context is None:
        context = {}
    if isinstance(obj, dict):
        ctx = _candidate_context(obj, context)
        candidate = _candidate_rep(obj, ctx)
        if candidate:
            kind, rep = candidate
            n = _int(ctx.get("n"))
            k = _int(ctx.get("k"))
            if n is not None or k is not None or kind == "checks":
                family = _family(obj, ctx, path)
                code_id = _representation_key(family, ctx, rep)
                # Keep only candidate-local telemetry. Retaining root JSON
                # objects here would pin multi-hundred-MB validation trees in
                # memory across all 2k+ source files.
                keep = {
                    "semantic_hash", "d_upper", "dx_upper", "dz_upper",
                    "distance", "screen", "search", "witness_x", "witness_z",
                    "screen_witnesses", "witnesses", "status", "regulation",
                    "seed", "rank", "gcd_degree", "reflection_count",
                    "commutation_violations", "check_weight",
                }
                snapshot = {key: obj[key] for key in keep if key in obj}
                out.append({
                    "code_id": code_id,
                    "representation_kind": kind,
                    "representation": rep,
                    "object": snapshot,
                    "context": ctx,
                    "family": family,
                    "source_path": _rel(path, root),
                    "source_key": _path_label(parts),
                })
        # Matrices/genomes/witnesses dominate file size and contain no candidates.
        skip = {"checks", "A", "B", "a", "b", "witness_x", "witness_z", "witnesses", "screen_witnesses"}
        for key, value in obj.items():
            if key in skip:
                continue
            _walk_candidates(value, path, root, out, ctx, parts + (str(key),))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            _walk_candidates(value, path, root, out, context, parts + (str(index),))


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _obs_from_candidate(item: dict[str, Any]) -> dict[str, Any]:
    obj = item["object"]
    d = _distance(obj)
    trial = _trial_info(obj)
    return {
        "distance_x": d["dx"], "distance_z": d["dz"], "distance_observed": d["d"],
        "witness_x": _witnesses(obj)["X"], "witness_z": _witnesses(obj)["Z"],
        **trial,
        "status": str(obj.get("status", obj.get("regulation", {}).get("official_gate_status", "observed"))) if isinstance(obj.get("regulation", {}), dict) else str(obj.get("status", "observed")),
        "observation_type": "candidate_record",
    }


def _curve_from_screen(item: dict[str, Any], code_id: str, run_id: str) -> list[dict[str, Any]]:
    obj = item["object"]
    screen = obj.get("screen")
    if not isinstance(screen, dict):
        screen = {}
    budget = _int(screen.get("trials_per_side")) or _int(screen.get("configured_total_trials_per_side")) or _int(screen.get("trials")) or _int(obj.get("confirm_trials_per_sector"))
    if budget is None:
        return []
    d = _distance(obj)
    for key in ("d_upper", "dx_upper", "dz_upper"):
        value = _int(screen.get(key))
        if key == "d_upper" and value is not None:
            d["d"] = value
        elif key == "dx_upper" and value is not None:
            d["dx"] = value
        elif key == "dz_upper" and value is not None:
            d["dz"] = value
    if d["d"] is None and d["dx"] is not None and d["dz"] is not None:
        d["d"] = min(d["dx"], d["dz"])
    points = []
    for side, value in (("X", d["dx"]), ("Z", d["dz"]), ("overall", d["d"])):
        if value is not None:
            points.append({"code_id": code_id, "run_id": run_id, "side": side, "trials_per_side": budget, "trials_total": 2 * budget, "best_distance": value, "early_stopped": bool(screen.get("stopped_early", False)), "source_kind": "screen"})
    return points


def _resolve_link(value: Any, root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    p = Path(value)
    if p.exists():
        return p
    candidate = root / value
    if candidate.exists():
        return candidate
    by_name = list(root.rglob(p.name)) if p.name else []
    return by_name[0] if len(by_name) == 1 else None


def _fit_projection(points: list[dict[str, Any]], target: int, pooled_slope: float | None) -> dict[str, Any]:
    ordered = sorted(points, key=lambda p: (int(p["trials_per_side"]), int(p["best_distance"])))
    if not ordered:
        return {"projected_final_distance": None, "projection_low": None, "projection_high": None, "projection_method": "no_curve", "projection_points": 0, "projection_target_trials_per_side": target}
    observed = min(float(p["best_distance"]) for p in ordered)
    if len(ordered) >= 2:
        xs = [math.log1p(max(1, int(p["trials_per_side"]))) for p in ordered]
        ys = [float(p["best_distance"]) for p in ordered]
        xbar = sum(xs) / len(xs)
        ybar = sum(ys) / len(ys)
        denom = sum((x - xbar) ** 2 for x in xs)
        slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom if denom else 0.0
        slope = max(-2.0, min(0.0, slope))
        intercept = ybar - slope * xbar
        pred = intercept + slope * math.log1p(max(1, target))
        residual = math.sqrt(sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / len(xs))
        method = "monotone_log_trial_fit"
    elif pooled_slope is not None:
        slope = max(-2.0, min(0.0, pooled_slope))
        pred = observed + slope * (math.log1p(max(1, target)) - math.log1p(max(1, int(ordered[0]["trials_per_side"]))))
        residual = max(1.0, abs(slope) * 2.0)
        method = "pooled_log_slope_prior"
    else:
        pred, residual, method = observed, 2.0, "one_point_hold"
    # Long-horizon extrapolation from sparse randomized evidence is fragile.
    # Bound maximum modeled improvement so estimates do not collapse to zero
    # merely because a short curve had a steep first drop.
    max_drop = observed * (0.50 if len(ordered) >= 2 else 0.25)
    pred = min(observed, max(observed - max_drop, max(1.0, pred)))
    low = max(1.0, pred - max(1.0, residual))
    high = min(observed, pred + max(1.0, residual))
    return {"projected_final_distance": round(pred, 4), "projection_low": round(low, 4), "projection_high": round(high, 4), "projection_method": method, "projection_points": len(ordered), "projection_target_trials_per_side": target}


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field) for field in fields}
            for field, value in out.items():
                if isinstance(value, (dict, list)):
                    out[field] = _json(value)
            writer.writerow(out)


def build_corpus(root: Path = DEFAULT_ROOT, out: Path = DEFAULT_OUT, projection_trials: int = DEFAULT_PROJECTION_TRIALS) -> dict[str, Any]:
    root = root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(root.rglob("*.json"))
    candidates: list[dict[str, Any]] = []
    parse_errors = 0
    for path in files:
        obj = _load_json(path)
        if obj is None:
            parse_errors += 1
            continue
        _walk_candidates(obj, path, root, candidates)

    code_rows: dict[str, dict[str, Any]] = {}
    runs: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    path_to_codes: dict[str, set[str]] = defaultdict(set)
    semantic_to_code: dict[str, str] = {}

    for index, item in enumerate(candidates):
        code_id = item["code_id"]
        obj, ctx, rep = item["object"], item["context"], item["representation"]
        n, k = _int(ctx.get("n")), _int(ctx.get("k"))
        if n is None and _int(ctx.get("m")) is not None:
            n = 2 * int(ctx["m"])
        if n is None:
            continue
        if code_id not in code_rows:
            witnesses = _witnesses(obj)
            code_rows[code_id] = {
                "code_id": code_id,
                "n": n,
                "k": k,
                "rate": (k / n) if k is not None and n else None,
                "family": item["family"],
                "representation_kind": item["representation_kind"],
                "representation": rep,
                "semantic_hash": str(obj.get("semantic_hash")) if obj.get("semantic_hash") else None,
                "group": obj.get("group", ctx.get("group")),
                "group_order": _int(obj.get("group_order", ctx.get("group_order"))),
                "group_index": _int(obj.get("group_index", ctx.get("group_index"))),
                "gcd_degree": _int(obj.get("gcd_degree")),
                "has_witness": bool(witnesses["X"] or witnesses["Z"]),
                "witness_x": witnesses["X"],
                "witness_z": witnesses["Z"],
                "witness_weight_x": len(witnesses["X"]) if witnesses["X"] is not None else None,
                "witness_weight_z": len(witnesses["Z"]) if witnesses["Z"] is not None else None,
                "source_first": item["source_path"],
                "source_count": 0,
                "source_paths": [],
                "observed_distance_x": None,
                "observed_distance_z": None,
                "observed_distance": None,
                "run_count": 0,
                "curve_point_count": 0,
                "early_stop_seen": False,
                **_structural(rep, obj, n),
            }
        row = code_rows[code_id]
        row["source_count"] += 1
        if item["source_path"] not in row["source_paths"]:
            row["source_paths"].append(item["source_path"])
        if obj.get("semantic_hash"):
            semantic_to_code[str(obj["semantic_hash"])] = code_id
        path_to_codes[item["source_path"]].add(code_id)
        run_id = _sha({"code_id": code_id, "source": item["source_path"], "key": item["source_key"], "index": index})[:24]
        obs = _obs_from_candidate(item)
        run = {"run_id": run_id, "code_id": code_id, "source_path": item["source_path"], "source_key": item["source_key"], **{k: v for k, v in obs.items() if k not in ("witness_x", "witness_z")}}
        runs.append(run)
        row["run_count"] += 1
        row["early_stop_seen"] = bool(row["early_stop_seen"] or obs.get("early_stopped"))
        for key, side in (("distance_x", "observed_distance_x"), ("distance_z", "observed_distance_z"), ("distance_observed", "observed_distance")):
            value = obs.get(key)
            if value is not None:
                row[side] = value if row[side] is None else min(row[side], value)
        curves.extend(_curve_from_screen(item, code_id, run_id))

    # Validation/chunk reports and screening ladders contain actual staged
    # trial counts. Link them through candidate_path/refined_candidate_path.
    for path in files:
        obj = _load_json(path)
        if not isinstance(obj, dict):
            continue
        link = _resolve_link(obj.get("refined_candidate_path") or obj.get("candidate_path") or obj.get("source"), root)
        linked = path_to_codes.get(_rel(link, root), set()) if link else set()
        if not linked and obj.get("semantic_hash") in semantic_to_code:
            linked = {semantic_to_code[obj["semantic_hash"]]}
        if not linked:
            continue
        report_id = _rel(path, root)
        for code_id in sorted(linked):
            run_id = _sha({"code_id": code_id, "source": report_id, "kind": "staged"})[:24]
            chunks = obj.get("chunks")
            if isinstance(chunks, list):
                cum_x = cum_z = 0
                best_x = best_z = None
                for chunk in chunks:
                    if not isinstance(chunk, dict):
                        continue
                    xtr, ztr = _int(chunk.get("x_trials")) or _int(obj.get("trials_per_side")), _int(chunk.get("z_trials")) or _int(obj.get("trials_per_side"))
                    xtr, ztr = xtr or 0, ztr or 0
                    cum_x += xtr; cum_z += ztr
                    dx, dz = _int(chunk.get("dx_upper")), _int(chunk.get("dz_upper"))
                    if dx is not None: best_x = dx if best_x is None else min(best_x, dx)
                    if dz is not None: best_z = dz if best_z is None else min(best_z, dz)
                    total = cum_x + cum_z
                    for side, value, trial in (("X", best_x, cum_x), ("Z", best_z, cum_z), ("overall", min(x for x in (best_x, best_z) if x is not None) if best_x is not None or best_z is not None else None, max(cum_x, cum_z))):
                        if value is not None and trial:
                            curves.append({"code_id": code_id, "run_id": run_id, "side": side, "trials_per_side": trial, "trials_total": total, "best_distance": value, "early_stopped": bool(chunk.get("stopped_early", False)), "source_kind": "validation_chunk"})
            stages = obj.get("stages") or obj.get("completed_stages")
            if isinstance(stages, list):
                cumulative = 0
                best_by_side = {"X": None, "Z": None}
                for stage in stages:
                    if not isinstance(stage, dict):
                        continue
                    budget = _int(stage.get("trials_per_side")) or _int(stage.get("native_trials"))
                    if not budget:
                        continue
                    cumulative += budget
                    sectors = stage.get("sectors", {})
                    for side in ("X", "Z"):
                        value = _int(sectors.get(side, {}).get("native_best")) if isinstance(sectors, dict) and isinstance(sectors.get(side), dict) else None
                        if value is not None:
                            best_by_side[side] = value if best_by_side[side] is None else min(best_by_side[side], value)
                            curves.append({"code_id": code_id, "run_id": run_id, "side": side, "trials_per_side": cumulative, "trials_total": 2 * cumulative, "best_distance": best_by_side[side], "early_stopped": bool(stage.get("stopped_early", False)), "source_kind": "ladder_stage"})
                    d_after = _int(stage.get("d_after"))
                    if d_after is None and any(v is not None for v in best_by_side.values()):
                        d_after = min(v for v in best_by_side.values() if v is not None)
                    if d_after is not None:
                        curves.append({"code_id": code_id, "run_id": run_id, "side": "overall", "trials_per_side": cumulative, "trials_total": 2 * cumulative, "best_distance": d_after, "early_stopped": bool(stage.get("stopped_early", False)), "source_kind": "ladder_stage"})

    # Normalize curves to monotone best-so-far values at each budget.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for point in curves:
        grouped[(point["code_id"], point["side"])].append(point)
    normalized: list[dict[str, Any]] = []
    for key, values in grouped.items():
        by_budget: dict[int, dict[str, Any]] = {}
        for value in values:
            budget = _int(value.get("trials_per_side"))
            distance = _int(value.get("best_distance"))
            if not budget or distance is None:
                continue
            current = by_budget.get(budget)
            if current is None or distance < current["best_distance"]:
                by_budget[budget] = dict(value, trials_per_side=budget, best_distance=distance)
            else:
                current["early_stopped"] = bool(current.get("early_stopped") or value.get("early_stopped"))
        running = None
        for budget in sorted(by_budget):
            point = by_budget[budget]
            running = point["best_distance"] if running is None else min(running, point["best_distance"])
            point["best_distance"] = running
            normalized.append(point)
    curves = sorted(normalized, key=lambda p: (p["code_id"], p["side"], p["trials_per_side"]))

    slopes: list[float] = []
    normalized_grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for point in curves:
        normalized_grouped[(point["code_id"], point["side"])].append(point)
    for values in normalized_grouped.values():
        ordered = sorted(values, key=lambda p: p["trials_per_side"])
        if len(ordered) >= 2 and ordered[-1]["trials_per_side"] > ordered[0]["trials_per_side"]:
            delta = ordered[-1]["best_distance"] - ordered[0]["best_distance"]
            slope = delta / (math.log1p(ordered[-1]["trials_per_side"]) - math.log1p(ordered[0]["trials_per_side"]))
            slopes.append(max(-2.0, min(0.0, slope)))
    pooled_slope = min(0.0, sorted(slopes)[len(slopes) // 2]) if slopes else None
    curve_by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in curves:
        curve_by_code[point["code_id"]].append(point)
    for code_id, row in code_rows.items():
        overall = curve_by_code.get(code_id, [])
        if not overall:
            row.update({"projected_final_distance": None, "projection_low": None, "projection_high": None, "projection_method": "no_curve", "projection_points": 0, "projection_target_trials_per_side": projection_trials})
            row["projection_status"] = "no_curve"
            continue
        side_projections = {}
        for side in ("X", "Z", "overall"):
            side_points = [p for p in overall if p["side"] == side]
            side_projections[side] = _fit_projection(side_points, projection_trials, pooled_slope)
        primary = side_projections["overall"]
        if primary["projected_final_distance"] is None:
            values = [side_projections[s]["projected_final_distance"] for s in ("X", "Z") if side_projections[s]["projected_final_distance"] is not None]
            primary["projected_final_distance"] = min(values) if values else None
        if not row.get("early_stop_seen"):
            # No extrapolation for complete/non-stopped evidence. Preserve
            # observed terminal point as label; projection fields describe
            # observed endpoint, not a forecast.
            endpoint = [p["best_distance"] for p in overall if p["side"] == "overall"]
            endpoint_value = min(endpoint) if endpoint else row.get("observed_distance")
            primary.update({
                "projected_final_distance": endpoint_value,
                "projection_low": endpoint_value,
                "projection_high": endpoint_value,
                "projection_method": "observed_terminal",
                "projection_points": len([p for p in overall if p["side"] == "overall"]),
            })
            row["projection_status"] = "observed_terminal"
        row.update(primary)
        row["projected_distance_x"] = side_projections["X"]["projected_final_distance"]
        row["projected_distance_z"] = side_projections["Z"]["projected_final_distance"]
        row["projection_x_method"] = side_projections["X"]["projection_method"]
        row["projection_z_method"] = side_projections["Z"]["projection_method"]
        row["projection_status"] = "model_estimate" if row.get("early_stop_seen") else "observed_terminal"
        row["curve_point_count"] = len(overall)

    codes = sorted(code_rows.values(), key=lambda row: row["code_id"])
    split_counts = defaultdict(int)
    for row in codes:
        bucket = int(row["code_id"][:8], 16) % 100
        row["split"] = "test" if bucket < 10 else "validation" if bucket < 20 else "train"
        split_counts[row["split"]] += 1
    for row in runs:
        row["split"] = code_rows[row["code_id"]].get("split")
    for point in curves:
        point["split"] = code_rows[point["code_id"]].get("split")
    codes.sort(key=lambda row: (row["split"], row["family"], row["n"], row["code_id"]))
    runs.sort(key=lambda row: (row["code_id"], row["source_path"], row["run_id"]))

    counts = {
        "json_files_seen": len(files), "json_parse_errors": parse_errors,
        "candidate_occurrences": len(candidates), "unique_codes": len(codes),
        "runs": len(runs), "curve_points": len(curves),
        "codes_with_curves": sum(bool(curve_by_code.get(row["code_id"])) for row in codes),
        "codes_with_early_stop": sum(bool(row.get("early_stop_seen")) for row in codes),
        "codes_with_projection": sum(row.get("projected_final_distance") is not None for row in codes),
        "families": len({row["family"] for row in codes}), "splits": dict(split_counts),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION, "kind": "qldpc_search_corpus",
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_root": str(root),
        "output_dir": str(out), "counts": counts,
        "projection": {
            "target_trials_per_side": projection_trials,
            "model": "monotone log-trial least-squares fit; pooled median slope prior for one-point early stops",
            "clamp": "projected distance constrained to [1, observed best upper bound]",
            "meaning": "model estimate of best witness likely found by target budget; not exact distance and not proof",
            "uncertainty": "residual-based interval; one-point fallback gets wide interval",
        },
        "splits": {"unit": "code_id", "hash_rule": "sha256(code_id) bucket; exact duplicate representations never cross splits", "fractions": {"train": 0.8, "validation": 0.1, "test": 0.1}},
        "files": {"codes": "codes.jsonl", "runs": "runs.jsonl", "curves": "curves.jsonl", "codes_csv": "codes.csv", "runs_csv": "runs.csv", "curves_csv": "curves.csv", "data_dictionary": "DATASET.md"},
        "distance_policy": "Observed distances are witness-backed upper bounds. Randomized no-hit is not promoted to proof. Projected values are estimates only.",
    }
    _write_jsonl(out / "codes.jsonl", codes)
    _write_jsonl(out / "runs.jsonl", runs)
    _write_jsonl(out / "curves.jsonl", curves)
    _write_csv(out / "codes.csv", codes, ["code_id", "split", "n", "k", "rate", "family", "representation_kind", "observed_distance", "observed_distance_x", "observed_distance_z", "witness_weight_x", "witness_weight_z", "projected_final_distance", "projection_low", "projection_high", "projection_method", "projection_status", "projection_points", "projection_target_trials_per_side", "run_count", "curve_point_count", "early_stop_seen", "check_weight_min", "check_weight_max", "check_weight_mean", "check_weight_total", "matrix_density", "rank", "gcd_degree", "semantic_hash", "source_count"])
    _write_csv(out / "runs.csv", runs, ["run_id", "code_id", "split", "source_path", "source_key", "observation_type", "status", "distance_observed", "distance_x", "distance_z", "trials_per_side", "x_trials", "z_trials", "trials_total", "early_stopped", "seed"])
    _write_csv(out / "curves.csv", curves, ["code_id", "run_id", "split", "side", "trials_per_side", "trials_total", "best_distance", "early_stopped", "source_kind"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_dataset_doc(out / "DATASET.md", manifest)
    return manifest


def _write_dataset_doc(path: Path, manifest: dict[str, Any]) -> None:
    counts = manifest["counts"]
    text = f"""# qLDPC Search Corpus

Generated from `{manifest['source_root']}`.

## Contents

`codes.jsonl` is one row per exact code representation. `runs.jsonl` preserves every source observation. `curves.jsonl` stores monotone best-so-far distance against actual randomized trial count, separately for X, Z, and combined CSS distance. CSV mirrors provide tabular loading for pandas/R/SQL.

Counts: {counts['unique_codes']} codes, {counts['runs']} runs, {counts['curve_points']} curve points, {counts['families']} families.

## Key labels

- `observed_distance*`: witness-backed randomized upper bound found in source artifacts.
- `projected_final_distance`: model estimate at `{manifest['projection']['target_trials_per_side']:,}` trials/side; not exact distance, not proof.
- `projection_low/high`: residual-based uncertainty interval.
- `early_stop_seen`: at least one source run stopped before configured budget.
- `split`: deterministic code-level split; no exact code duplicate crosses splits.

## Projection

Curves fit a non-increasing linear model in `log1p(trials_per_side)`. One-point early stops use pooled median slope from multi-point curves. Predictions clamp to `[1, observed best]`. Sparse curves retain explicit method/status fields so downstream ML can filter or weight them.

## Caveats

Search logs use heterogeneous algorithms, seeds, budgets, and stopping rules. Trial counts are retained per sector when available. Population size, pair attempts, and candidate counts are not silently treated as distance trials. Randomized no-hit results remain inconclusive. Use `source_path` for audit/replay.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--projection-trials", type=int, default=DEFAULT_PROJECTION_TRIALS)
    args = parser.parse_args()
    manifest = build_corpus(args.root, args.out, max(1, int(args.projection_trials)))
    print(json.dumps({"out": str(args.out), "counts": manifest["counts"], "projection": manifest["projection"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
