from __future__ import annotations

"""Successive-halving validation for GB frontier candidates.

The discovery campaign is intentionally broad.  This tournament is narrow:
it keeps spending fresh-seed RIS chunks on the *same* survivors, removes every
independently checked below-target witness, and preserves at least one survivor
per factor branch while slots permit.  Randomized no-hit results remain
evidence only; they are never reported as distance proofs.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

import cpp_fast
from hyper_validator import _logical_check, structural_validate


SEED_STEP = 0x9E3779B9


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _campaign_paths(path: Path) -> list[Path]:
    report = json.loads(path.read_text())
    out = []
    for generation in report.get("generations", []):
        for row in generation.get("final", []):
            candidate = row.get("candidate_path")
            if candidate:
                out.append(Path(candidate).resolve())
    return out


def _candidate_meta(path: Path) -> dict:
    doc = json.loads(path.read_text())
    structural = structural_validate(doc)
    if not structural["ok"]:
        raise ValueError(f"{path} fails structural validation")
    n = int(doc["n"])
    m = n // 2
    first = doc["checks"]["X"][0]
    word_weights = [sum(int(value) < m for value in first),
                    sum(int(value) >= m for value in first)]
    construction = str(doc.get("provenance", {}).get("construction", ""))
    match = re.search(r"factor child \[(\d+)\s*,\s*(\d+)\]", construction)
    branch = [int(match.group(1)), int(match.group(2))] if match else []
    semantic = hashlib.sha256(
        json.dumps(doc["checks"], separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return {
        "candidate_path": str(path.resolve()), "semantic_hash": semantic,
        "n": n, "k": int(doc["k"]), "branch": branch,
        "word_weights": word_weights,
    }


def _install_best(state: dict, side: str, result: dict, chunk: int,
                  seed: int, check: dict) -> None:
    weight = result.get("best_weight")
    witness = result.get("witness")
    if weight is None or witness is None or not check.get("ok"):
        return
    old = state["best_observed"].get(side)
    if old is None or int(weight) < int(old["weight"]):
        state["best_observed"][side] = {
            "weight": int(weight), "witness": list(witness),
            "chunk": int(chunk), "seed": int(seed),
        }


def _run_candidate(task) -> dict:
    state, stage, chunks, trials, target, seed, threads, pair_depth = task
    state = json.loads(json.dumps(state))
    source = json.loads(Path(state["candidate_path"]).read_text())
    structural = structural_validate(source)
    if not structural["ok"]:
        raise ValueError(f"{state['candidate_path']} fails structural validation")
    hx, hz = structural["hx"], structural["hz"]
    started = time.perf_counter()
    base_index = len(state["chunks"])
    for offset in range(int(chunks)):
        index = base_index + offset + 1
        chunk_seed = int(seed) + (index - 1) * SEED_STEP
        result = cpp_fast.css_ris_parallel(
            hx, hz, trials=int(trials), seed=chunk_seed, pair_depth=int(pair_depth),
            target=int(target), stop_on_target=True, threads=int(threads))
        record = {
            "index": index, "stage": int(stage), "seed": chunk_seed,
            "d": result.get("d_upper"), "dx": result.get("dx_upper"),
            "dz": result.get("dz_upper"),
            "x_trials": int(result["x"].get("trials_run") or 0),
            "z_trials": int(result["z"].get("trials_run") or 0),
            "seconds": float(result.get("total_seconds") or 0.0),
        }
        for side, native_side, kernel, rowspace in (
                ("X", "x", hz, hx), ("Z", "z", hx, hz)):
            native = result[native_side]
            witness = native.get("witness")
            check = ({"ok": True, "present": False} if witness is None else
                     _logical_check(witness, kernel, rowspace))
            record[side] = {
                "weight": native.get("best_weight"),
                "check": {
                    key: value for key, value in check.items() if key != "row_rank"
                },
            }
            _install_best(state, side, native, index, chunk_seed, check)
        state["chunks"].append(record)
        below = [(int(item["weight"]), side, item)
                 for side in ("X", "Z")
                 if (item := state["best_observed"].get(side)) is not None
                 and int(item["weight"]) < int(target)]
        if below:
            weight, side, item = min(below)
            state["status"] = "refuted"
            state["refutation"] = {
                "side": side, "weight": weight,
                "witness": list(item["witness"]),
                "chunk": int(item["chunk"]), "seed": int(item["seed"]),
            }
            break
    state["seconds_wall"] = float(state.get("seconds_wall", 0.0)) + (
        time.perf_counter() - started)
    return state


def _hazard(state: dict, target: int) -> dict:
    values = sorted(int(row["d"]) for row in state["chunks"]
                    if row.get("d") is not None)
    if not values:
        return {"minimum": -1, "lower_quartile": -1,
                "target_hits": 10**9, "chunks": 0}
    lower_quartile = values[max(0, (len(values) - 1) // 4)]
    return {
        "minimum": values[0], "lower_quartile": lower_quartile,
        "target_hits": sum(value <= int(target) for value in values),
        "chunks": len(values),
    }


def _rank_key(state: dict, target: int):
    hazard = _hazard(state, target)
    return (int(state.get("status") != "refuted"), hazard["minimum"],
            hazard["lower_quartile"], -hazard["target_hits"],
            sum(state.get("word_weights", [])), hazard["chunks"],
            state["semantic_hash"])


def _diverse_keep(states: list[dict], keep: int, target: int) -> list[dict]:
    ranked = sorted((row for row in states if row.get("status") != "refuted"),
                    key=lambda row: _rank_key(row, target), reverse=True)
    keep = min(max(0, int(keep)), len(ranked))
    chosen, used = [], set()
    for row in ranked:
        branch = tuple(row.get("branch", []))
        if branch and branch not in used and len(chosen) < keep:
            chosen.append(row); used.add(branch)
    chosen_hashes = {row["semantic_hash"] for row in chosen}
    for row in ranked:
        if len(chosen) >= keep:
            break
        if row["semantic_hash"] not in chosen_hashes:
            chosen.append(row); chosen_hashes.add(row["semantic_hash"])
    return chosen


def _parse_stage(value: str) -> tuple[int, int]:
    try:
        chunks, keep = (int(item) for item in value.split(":", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("stage must be CHUNKS:KEEP") from exc
    if chunks < 1 or keep < 1:
        raise argparse.ArgumentTypeError("stage values must be positive")
    return chunks, keep


def run(args) -> dict:
    if not cpp_fast.available():
        raise RuntimeError(cpp_fast.accelerator_info())
    out = Path(args.out)
    if out.exists():
        report = json.loads(out.read_text())
        expected = (int(args.target), int(args.trials), int(args.min_word_weight))
        actual = (int(report.get("target", -1)),
                  int(report.get("trials_per_side_per_chunk", -1)),
                  int(report.get("minimum_word_weight", -1)))
        if actual != expected:
            raise ValueError("checkpoint settings differ from requested tournament")
        first_stage = len(report.get("stages", []))
        stage_plan = list(args.stage)[first_stage:]
    else:
        report = None
        first_stage = 0
        stage_plan = list(args.stage)
    candidate_paths = [Path(value).resolve() for value in args.candidate]
    if args.campaign:
        candidate_paths.extend(_campaign_paths(Path(args.campaign)))
    candidate_paths = list(dict.fromkeys(candidate_paths))
    if report is None:
        states = []
        for path in candidate_paths:
            meta = _candidate_meta(path)
            if min(meta["word_weights"]) < int(args.min_word_weight):
                continue
            states.append({**meta, "status": "active", "chunks": [],
                           "best_observed": {"X": None, "Z": None},
                           "refutation": None, "seconds_wall": 0.0})
        report = {
            "schema_version": "1.0", "kind": "gb_frontier_successive_halving",
            "target": int(args.target), "trials_per_side_per_chunk": int(args.trials),
            "minimum_word_weight": int(args.min_word_weight),
            "submission_sent": False, "git_commit_performed": False,
            "distance_is_exact": False, "stages": [], "candidates": states,
        }
    for stage_index, (chunks, keep) in enumerate(
            stage_plan, first_stage + 1):
        active = [row for row in report["candidates"] if row["status"] == "active"]
        tasks = [(row, stage_index, chunks, args.trials, args.target,
                  args.seed ^ int(row["semantic_hash"][:16], 16),
                  args.threads, args.pair_depth)
                 for row in active]
        completed = []
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = [pool.submit(_run_candidate, task) for task in tasks]
            for done, future in enumerate(as_completed(futures), 1):
                completed.append(future.result())
                print(json.dumps({"stage": stage_index, "done": done,
                                  "total": len(futures)}), flush=True)
        by_hash = {row["semantic_hash"]: row for row in completed}
        report["candidates"] = [by_hash.get(row["semantic_hash"], row)
                                for row in report["candidates"]]
        survivors = _diverse_keep(completed, keep, args.target)
        survivor_hashes = {row["semantic_hash"] for row in survivors}
        for row in report["candidates"]:
            if row["status"] == "active" and row["semantic_hash"] not in survivor_hashes:
                row["status"] = "eliminated"
        report["stages"].append({
            "stage": stage_index, "chunks_added": chunks,
            "entered": len(active), "refuted": sum(row["status"] == "refuted"
                                                     for row in completed),
            "retained": len(survivors),
            "leaders": [{"semantic_hash": row["semantic_hash"],
                         "candidate_path": row["candidate_path"],
                         "branch": row["branch"],
                         "word_weights": row["word_weights"],
                         "hazard": _hazard(row, args.target)} for row in survivors],
        })
        _atomic_json(out, report)
        if not survivors:
            break
    report["leaders"] = sorted(
        ({"semantic_hash": row["semantic_hash"],
          "candidate_path": row["candidate_path"], "branch": row["branch"],
          "word_weights": row["word_weights"], "status": row["status"],
          "hazard": _hazard(row, args.target),
          "best_observed": row["best_observed"]}
         for row in report["candidates"] if row["status"] == "active"),
        key=lambda row: (row["hazard"]["minimum"],
                         row["hazard"]["lower_quartile"]), reverse=True)
    _atomic_json(out, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, default=80)
    parser.add_argument("--min-word-weight", type=int, default=16)
    parser.add_argument("--trials", type=int, default=20_000)
    parser.add_argument("--stage", type=_parse_stage, action="append")
    parser.add_argument("--seed", type=int, default=20269940)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--pair-depth", type=int, default=24)
    args = parser.parse_args()
    if args.stage is None:
        args.stage = [(5, 16), (20, 8), (75, 4), (900, 2)]
    if not args.candidate and not args.campaign:
        parser.error("provide --candidate or --campaign")
    result = run(args)
    print(json.dumps({"out": str(args.out), "leaders": result["leaders"]}, indent=2))


if __name__ == "__main__":
    main()
