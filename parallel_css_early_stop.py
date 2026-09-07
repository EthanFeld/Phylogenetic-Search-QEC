from __future__ import annotations

"""Chunked CSS RIS shards with candidate-wide early cancellation."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path

import cpp_fast
import distance_sketch as ds
from hyper_validator import _logical_check, structural_validate


def _better(best, witness, value, candidate_witness):
    if value is None:
        return best, witness
    value = int(value)
    if best is None or value < int(best):
        return value, candidate_witness
    return best, witness


def _task(args: tuple[str, int, int, int, int, int, object]) -> dict:
    candidate_path, trials, seed, pair_depth, target, chunk_trials, shared_stop = args
    path = Path(candidate_path)
    candidate = json.loads(path.read_text(encoding="utf-8"))
    structural = structural_validate(candidate)
    if not structural.get("ok"):
        return {
            "candidate_path": str(path.resolve()), "seed": int(seed),
            "requested_trials_per_side": int(trials),
            "actual_x_trials": 0, "actual_z_trials": 0,
            "actual_sector_shots": 0, "actual_trials_per_side": 0,
            "chunks": 0, "dx_upper": None, "dz_upper": None,
            "d_upper": None, "target_refuted": False, "css_ok": False,
            "checks": {}, "native_screen_refuted": False,
            "early_stopped": True, "structural_ok": False,
            "rejection_reason": "structural_validation",
            "structural_errors": structural.get("shape", {}).get("errors", []),
        }
    hx = ds.supports_to_binary(candidate["checks"]["X"], candidate["n"])
    hz = ds.supports_to_binary(candidate["checks"]["Z"], candidate["n"])
    best_x = best_z = None
    witness_x = witness_z = None
    actual_x = actual_z = 0
    chunks = 0
    native_early = False
    remaining = int(trials)
    while remaining > 0 and not shared_stop.is_set():
        budget = min(int(chunk_trials), remaining)
        native = cpp_fast.css_ris_parallel(
            hx, hz, trials=budget,
            seed=int(seed) + chunks * 0xD1B54A35,
            pair_depth=int(pair_depth), target=int(target),
            stop_on_target=True, threads=4)
        chunks += 1
        actual_x += int(native.get("x", {}).get("trials_run") or 0)
        actual_z += int(native.get("z", {}).get("trials_run") or 0)
        best_x, witness_x = _better(best_x, witness_x, native.get("dx_upper"),
                                    native.get("x", {}).get("witness"))
        best_z, witness_z = _better(best_z, witness_z, native.get("dz_upper"),
                                    native.get("z", {}).get("witness"))
        native_early = native_early or bool(native.get("screen_refuted"))
        remaining -= budget
        if native_early:
            shared_stop.set()
            break

    checks = {}
    for side, witness, kernel, stabilizer in (
        ("X", witness_x, hz, hx), ("Z", witness_z, hx, hz)
    ):
        check = (_logical_check(witness, kernel, stabilizer)
                 if witness else {"ok": False, "reason": "missing witness"})
        checks[side] = {k: v for k, v in check.items() if k != "row_rank"}
    css_ok = bool(checks["X"].get("ok") and checks["Z"].get("ok"))
    verified = [value for side, value in (("X", best_x), ("Z", best_z))
                if checks[side].get("ok") and value is not None]
    d = min(verified) if verified else None
    target_refuted = bool(any(int(value) < int(target) for value in verified))
    early_stopped = bool(native_early or target_refuted)
    return {
        "candidate_path": str(path.resolve()), "seed": int(seed),
        "requested_trials_per_side": int(trials),
        "actual_x_trials": actual_x, "actual_z_trials": actual_z,
        "actual_sector_shots": actual_x + actual_z,
        "actual_trials_per_side": max(actual_x, actual_z),
        "chunks": chunks, "dx_upper": best_x, "dz_upper": best_z,
        "d_upper": d, "target_refuted": target_refuted,
        "css_ok": css_ok, "checks": checks,
        "native_screen_refuted": native_early, "early_stopped": early_stopped,
        "structural_ok": bool(structural.get("ok")),
    }


def _gpu_scout(candidate_path: Path, hx, hz, n: int, target: int,
               trials: int, seed: int) -> dict:
    """CUDA scout; every returned hit is checked by the CPU CSS verifier."""
    try:
        import gpu_ris_scout
    except Exception as exc:  # pragma: no cover - optional CUDA path
        return {"available": False, "reason": f"GPU scout import failed: {exc}"}
    if not gpu_ris_scout.available():
        return {"available": False, "reason": "CUDA/PyTorch unavailable"}
    sides = {}
    for side, kernel, stabilizer in (("X", hz, hx), ("Z", hx, hz)):
        raw = gpu_ris_scout.scout(kernel, n, trials=trials,
                                  seed=int(seed) + (0 if side == "X" else 1),
                                  keep=64)
        verified = []
        for item in raw.get("candidates", []):
            check = _logical_check(item["support"], kernel, stabilizer)
            if check.get("ok"):
                verified.append({"weight": int(item["weight"]),
                                 "support": item["support"], "check": check})
        sides[side] = {"raw": raw, "verified": sorted(verified, key=lambda x: x["weight"])}
    verified = [item for side in sides.values() for item in side["verified"]]
    best = min(verified, key=lambda x: x["weight"], default=None)
    return {"available": True, "candidate_path": str(candidate_path.resolve()),
            "target_distance": int(target), "sides": sides,
            "best_verified_weight": None if best is None else int(best["weight"]),
            "target_refuted": bool(best is not None and int(best["weight"]) < int(target))}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("candidate", type=Path)
    p.add_argument("--trials-total", type=int, default=20_000_000)
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--target", type=int, default=77)
    p.add_argument("--pair-depth", type=int, default=12)
    p.add_argument("--chunk-trials", type=int, default=65_536)
    p.add_argument("--gpu-scout-trials", type=int, default=0,
                   help="optional CUDA GF(2) scout before CPU RIS")
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    gpu_report = None
    if int(args.gpu_scout_trials) > 0:
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        structural = structural_validate(candidate)
        if not structural.get("ok"):
            raise ValueError(f"candidate fails structural validation: {structural}")
        hx = ds.supports_to_binary(candidate["checks"]["X"], candidate["n"])
        hz = ds.supports_to_binary(candidate["checks"]["Z"], candidate["n"])
        gpu_report = _gpu_scout(args.candidate, hx, hz, int(candidate["n"]),
                                int(args.target), int(args.gpu_scout_trials), int(args.seed))
        gpu_report["structural_ok"] = bool(structural.get("ok"))
        print(json.dumps({"stage": "gpu_scout", "best_verified_weight": gpu_report.get("best_verified_weight"),
                          "target_refuted": gpu_report.get("target_refuted"),
                          "available": gpu_report.get("available")}), flush=True)
        if gpu_report.get("target_refuted"):
            report = {
                "schema_version": "2.0", "kind": "parallel_css_ris_early_stop",
                "out": str(args.out.resolve()), "candidate_path": str(args.candidate.resolve()),
                "target_distance": int(args.target), "requested_trials_per_side_total": 0,
                "actual_trials_per_side_total": {"X": 0, "Z": 0}, "actual_sector_shots_total": 0,
                "shards": 0, "trials_per_shard": 0, "chunk_trials": int(args.chunk_trials),
                "pair_depth": int(args.pair_depth), "gpu_scout": gpu_report,
                "early_stop_condition": f"verified witness weight < {int(args.target)}",
                "best_verified_d_upper": gpu_report.get("best_verified_weight"),
                "target_refuted": True,
                "claim_policy": "CUDA witness independently CSS-verified; no-distance proof.",
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            return 0
    shards = max(1, int(args.shards))
    total_requested = max(0, int(args.trials_total))
    base, remainder = divmod(total_requested, shards)
    manager = mp.Manager()
    shared_stop = manager.Event()
    budgets = [base + int(i < remainder) for i in range(shards)]
    tasks = [(str(args.candidate), budgets[i],
              int(args.seed) + i * 0x9E3779B9, int(args.pair_depth),
              int(args.target), int(args.chunk_trials), shared_stop)
             for i in range(shards) if budgets[i] > 0]
    rows = []
    try:
        with ProcessPoolExecutor(max_workers=shards) as pool:
            futures = [pool.submit(_task, task) for task in tasks]
            for done, future in enumerate(as_completed(futures), 1):
                row = future.result()
                rows.append(row)
                print(json.dumps({"done": done, "total": shards,
                                  "d": row.get("d_upper"),
                                  "actual_x": row.get("actual_x_trials"),
                                  "actual_z": row.get("actual_z_trials"),
                                  "early_stopped": row.get("early_stopped")}), flush=True)
    finally:
        manager.shutdown()
    actual_x = sum(int(row["actual_x_trials"]) for row in rows)
    actual_z = sum(int(row["actual_z_trials"]) for row in rows)
    best = min((row["d_upper"] for row in rows if row.get("d_upper") is not None), default=None)
    report = {
        "schema_version": "2.0", "kind": "parallel_css_ris_early_stop",
        "out": str(args.out.resolve()), "candidate_path": str(args.candidate.resolve()),
        "target_distance": int(args.target),
        "requested_trials_per_side_total": int(sum(budgets)),
        "actual_trials_per_side_total": {"X": actual_x, "Z": actual_z},
        "actual_sector_shots_total": actual_x + actual_z,
        "shards": len(tasks),
        "trials_per_shard": (budgets[0] if len(set(budgets)) == 1 else None),
        "trial_budgets_per_shard": budgets,
        "chunk_trials": int(args.chunk_trials), "pair_depth": int(args.pair_depth),
        "early_stop_condition": f"verified witness weight < {int(args.target)}",
        "gpu_scout": gpu_report,
        "best_verified_d_upper": best,
        "target_refuted": bool(any(row.get("target_refuted") for row in rows)
                             or (best is not None and int(best) < int(args.target))),
        "shard_results": sorted(rows, key=lambda row: row["seed"]),
        "claim_policy": "RIS witness is an upper bound; no-distance proof.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "out", "requested_trials_per_side_total", "actual_trials_per_side_total",
        "actual_sector_shots_total", "best_verified_d_upper", "target_refuted")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
