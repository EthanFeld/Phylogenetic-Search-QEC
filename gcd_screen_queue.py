from __future__ import annotations

"""Fresh common-GCD screen for the ML queue.

This is a gate, not a distance proof: every returned ideal witness is an
explicit upper bound and every no-hit result remains inconclusive.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

from generalized_bicycle import build_supports, normalize_code
from hyper_validator import structural_validate
from ideal_x_logical_search import scan_candidate


def _candidate(code: dict) -> dict:
    rep = code.get("representation") or {}
    a_rows, b_rows = rep.get("A"), rep.get("B")
    if not isinstance(a_rows, list) or not isinstance(b_rows, list) or not a_rows or not b_rows:
        raise ValueError("code has no ring A/B representation")
    m = int(code["n"]) // 2
    normalized = normalize_code(m, a_rows[0], b_rows[0])
    hx, hz = build_supports(m, normalized.a, normalized.b)
    wx = list(code.get("witness_x") or [])
    wz = list(code.get("witness_z") or [])
    dx = len(wx)
    dz = len(wz)
    if not wx or not wz:
        raise ValueError("code has no stored X/Z witnesses")
    return {
        "schema_version": "0.1",
        "name": f"ML queue {code['code_id']}",
        "code_type": "CSS", "n": normalized.n, "k": normalized.k,
        "checks": {"X": hx, "Z": hz},
        "distance": {
            "d": min(dx, dz),
            "X": {"value": dx, "confidence": "upper_bound", "witness": wx},
            "Z": {"value": dz, "confidence": "upper_bound", "witness": wz},
        },
        "provenance": {"source_code_id": code["code_id"], "source_first": code.get("source_first")},
        "regulation": {"stage_only": True, "submission_sent": False,
                        "git_commit_performed": False, "distance_is_not_proven": True},
    }


def _scan_task(task: tuple[int, str, str, int, int]) -> tuple[int, dict]:
    index, code_id, candidate_path, trials, pair_depth = task
    record = {"code_id": code_id, "candidate_path": candidate_path}
    try:
        report = scan_candidate(Path(candidate_path), trials=int(trials), seed=20260905 + index * 1009,
                                pair_depth=int(pair_depth), max_seconds=None)
        found = report.get("found_X_logical") or {}
        weight = found.get("weight")
        doc = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
        css_ok = bool((found.get("css_logical_check") or {}).get("ok"))
        score = (doc["k"] * int(weight) ** 2 / doc["n"]) if weight and css_ok else None
        record.update({"status": "gcd_scanned",
                       "g_degree": report.get("algebra", {}).get("g_degree_K"),
                       "ideal_weight": weight if css_ok else None,
                       "ideal_weight_raw": weight,
                       "ideal_score_upper": score,
                       "ideal_css_ok": css_ok,
                       "h_divides": found.get("h_divides_word"),
                       "qualifies_gcd_gate": bool(score is not None and score >= 1542.0),
                       "trials": int(trials), "report": report})
    except (OSError, ValueError, ArithmeticError, KeyError, json.JSONDecodeError) as exc:
        record.update({"status": "rejected_input", "error_type": type(exc).__name__, "error": str(exc)})
    return index, record


def run(queue_path: Path, codes_path: Path, out: Path, top: int, trials: int,
        pair_depth: int, workers: int | None = None) -> dict:
    queue = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    codes = {row["code_id"]: row for row in (json.loads(line) for line in codes_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict | None] = [None] * min(int(top), len(queue))
    tasks = []
    for index, item in enumerate(queue[:int(top)]):
        code = codes[item["code_id"]]
        record = {"code_id": code["code_id"], "family": code.get("family"), "ml": item}
        try:
            candidate = _candidate(code)
            candidate_path = out / f"candidate_{index + 1:03d}_{candidate['n']}_{candidate['k']}.json"
            candidate_path.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
            structural = structural_validate(candidate)
            if not structural["ok"]:
                record.update({"status": "rejected_structural", "errors": structural.get("distance_errors") or structural.get("witnesses")})
                rows[index] = record
            else:
                tasks.append((index, code["code_id"], str(candidate_path.resolve()), int(trials), int(pair_depth)))
        except (OSError, ValueError, ArithmeticError, KeyError, json.JSONDecodeError) as exc:
            record.update({"status": "rejected_input", "error_type": type(exc).__name__, "error": str(exc)})
            rows[index] = record
    max_workers = max(1, int(workers or min(8, os.cpu_count() or 1)))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_scan_task, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            index, result = future.result()
            base = next(item for item in queue[:int(top)] if item["code_id"] == result["code_id"])
            result = {**{"family": codes[result["code_id"]].get("family"), "ml": base}, **result}
            rows[index] = result
            print(json.dumps({"done": done, "total": len(tasks), "code_id": result["code_id"],
                              "status": result["status"], "ideal_weight": result.get("ideal_weight")}), flush=True)
    rows = [row for row in rows if row is not None]
    _write = out / "screen_rows.jsonl"
    with _write.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "1.0", "kind": "ml_queue_common_gcd_gate",
        "target_score": 1542.0, "queue_rows_considered": len(rows),
        "scanned": sum(row["status"] == "gcd_scanned" for row in rows),
        "qualify_gcd_gate": sum(row.get("qualifies_gcd_gate", False) for row in rows),
        "rejected": sum(row["status"] != "gcd_scanned" for row in rows),
        "trials_per_candidate": int(trials), "pair_depth": int(pair_depth), "workers": max_workers,
        "regulation": {"distance_is_not_proven": True, "submission_sent": False, "git_commit_performed": False},
    }
    (out / "screen_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=Path("results/corpus_repaired/search_queue.jsonl"))
    parser.add_argument("--codes", type=Path, default=Path("results/corpus_repaired/codes.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/gcd_screen_ml_queue"))
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--trials", type=int, default=1024)
    parser.add_argument("--pair-depth", type=int, default=12)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.queue, args.codes, args.out, args.top, args.trials, args.pair_depth, args.workers), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
