from __future__ import annotations

"""Matched end-to-end CPU/GPU CSS RIS benchmark."""

import argparse
import json
from pathlib import Path
import statistics
import time

import cpp_fast
import distance_sketch as ds
import gpu_fast
from hyper_validator import _logical_check


def _run_cpu(hx, hz, trials, seed, pair_depth, threads):
    started = time.perf_counter()
    result = cpp_fast.css_ris_parallel(
        hx, hz, trials=trials, seed=seed, pair_depth=pair_depth,
        combo_depth=2, target=None, stop_on_target=False, threads=threads)
    result["wall_seconds"] = time.perf_counter() - started
    return result


def _run_gpu(hx, hz, trials, seed, pair_depth, threads, batch_size, block_width, rank_cap):
    started = time.perf_counter()
    result = gpu_fast.css_ris(
        hx, hz, trials=trials, seed=seed, pair_depth=pair_depth,
        target=None, stop_on_target=False, threads=threads, batch_size=batch_size,
        block_width=block_width, rank_cap=rank_cap)
    result["wall_seconds"] = time.perf_counter() - started
    return result


def _verified(result, hx, hz):
    checks = {}
    for side, witness, kernel, stabilizer in (
        ("X", result.get("x", {}).get("witness"), hz, hx),
        ("Z", result.get("z", {}).get("witness"), hx, hz),
    ):
        check = _logical_check(witness, kernel, stabilizer) if witness else {"ok": False}
        checks[side] = bool(check.get("ok"))
    return checks


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("candidate", type=Path)
    p.add_argument("--trials", type=int, default=2048)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--pair-depth", type=int, default=24)
    p.add_argument("--cpu-threads", type=int, default=16)
    p.add_argument("--gpu-threads", type=int, default=448)
    p.add_argument("--gpu-batch", type=int, default=8192)
    p.add_argument("--gpu-block-width", type=int, default=4)
    p.add_argument("--gpu-rank-cap", type=int, default=0,
                   help="0=full rank; positive values are truncated-RIS experiments")
    p.add_argument("--quality-target", type=int, default=0,
                   help="count verified witnesses <= this weight; 0 disables target metric")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    doc = json.loads(args.candidate.read_text(encoding="utf-8"))
    n = int(doc["n"])
    hx = ds.supports_to_binary(doc["checks"]["X"], n)
    hz = ds.supports_to_binary(doc["checks"]["Z"], n)
    if not gpu_fast.available():
        raise SystemExit(f"CUDA GPU backend unavailable: {gpu_fast.load_error()}")
    # Warm both paths and CUDA context before timed replicates.
    _run_cpu(hx, hz, min(64, args.trials), 90001, args.pair_depth, args.cpu_threads)
    _run_gpu(hx, hz, min(64, args.trials), 90001, args.pair_depth,
             args.gpu_threads, args.gpu_batch, args.gpu_block_width,
             args.gpu_rank_cap)
    cpu, gpu = [], []
    for i in range(max(1, int(args.repeats))):
        c = _run_cpu(hx, hz, args.trials, 91000 + i, args.pair_depth, args.cpu_threads)
        g = _run_gpu(hx, hz, args.trials, 91000 + i, args.pair_depth,
                     args.gpu_threads, args.gpu_batch, args.gpu_block_width,
                     args.gpu_rank_cap)
        cpu.append({"wall_seconds": c["wall_seconds"], "d_upper": c.get("d_upper"),
                    "checks": _verified(c, hx, hz),
                    "target_hit": bool(args.quality_target and c.get("d_upper") is not None and int(c["d_upper"]) <= int(args.quality_target))})
        gpu.append({"wall_seconds": g["wall_seconds"], "d_upper": g.get("d_upper"),
                    "checks": _verified(g, hx, hz),
                    "target_hit": bool(args.quality_target and g.get("d_upper") is not None and int(g["d_upper"]) <= int(args.quality_target))})
        print(json.dumps({"repeat": i, "cpu_seconds": cpu[-1]["wall_seconds"],
                          "gpu_seconds": gpu[-1]["wall_seconds"],
                          "cpu_d": cpu[-1]["d_upper"], "gpu_d": gpu[-1]["d_upper"]}),
              flush=True)
    cpu_median = statistics.median(x["wall_seconds"] for x in cpu)
    gpu_median = statistics.median(x["wall_seconds"] for x in gpu)
    report = {
        "schema_version": "1.0", "kind": "matched_css_ris_backend_benchmark",
        "candidate": str(args.candidate.resolve()), "n": n,
        "trials_per_side": int(args.trials), "pair_depth": int(args.pair_depth),
        "cpu_threads": int(args.cpu_threads), "gpu_threads": int(args.gpu_threads),
        "gpu_batch": int(args.gpu_batch),
        "gpu_block_width": int(args.gpu_block_width),
        "gpu_rank_cap": int(args.gpu_rank_cap),
        "quality_target": int(args.quality_target),
        "repeats": len(cpu),
        "cpu": cpu, "gpu": gpu,
        "quality": {
            "cpu_verified_fraction": sum(all(x["checks"].values()) for x in cpu) / len(cpu),
            "gpu_verified_fraction": sum(all(x["checks"].values()) for x in gpu) / len(gpu),
            "gpu_not_worse_than_cpu_fraction": sum(
                x.get("d_upper") is not None and y.get("d_upper") is not None and int(y["d_upper"]) <= int(x["d_upper"])
                for x, y in zip(cpu, gpu)) / len(cpu),
            "cpu_target_hit_fraction": sum(x["target_hit"] for x in cpu) / len(cpu),
            "gpu_target_hit_fraction": sum(x["target_hit"] for x in gpu) / len(gpu),
        },
        "median_cpu_seconds": cpu_median, "median_gpu_seconds": gpu_median,
        "median_end_to_end_speedup_cpu_over_gpu": cpu_median / gpu_median,
        "gpu_backend": "cuda_randomized_echelon_css_ris",
        "gpu_reduction": "forward_echelon",
        "quality_policy": "all returned witnesses independently CSS-verified",
        "decision_gate": ">=2x median end-to-end speedup required for routing",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "median_cpu_seconds", "median_gpu_seconds",
        "median_end_to_end_speedup_cpu_over_gpu")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
