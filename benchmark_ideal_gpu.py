from __future__ import annotations

"""Matched ideal-RIS GPU benchmark; GPU routing candidate for low-rank stage."""

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

import cpp_fast
import gpu_fast


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=4096)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--threads", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if not gpu_fast.available():
        raise SystemExit(f"CUDA GPU backend unavailable: {gpu_fast.load_error()}")
    rng = np.random.default_rng(args.seed)
    rows = []
    for rank, n in ((66, 341), (85, 337), (91, 331)):
        kernel = rng.integers(0, 2, size=(rank, n), dtype=np.uint8)
        if cpp_fast.gf2_rank(kernel) != rank:
            raise SystemExit(f"random basis rank failure: {(rank, n)}")
        cpp_fast.classical_ris(kernel, trials=64, seed=1, pair_depth=24)
        gpu_fast.classical_ris(kernel, trials=64, seed=1, pair_depth=24,
                               threads=args.threads)
        cpu_times, gpu_times = [], []
        for i in range(max(1, args.repeats)):
            started = time.perf_counter()
            cpu = cpp_fast.classical_ris(kernel, trials=args.trials,
                                         seed=100 + i, pair_depth=24)
            cpu_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            gpu = gpu_fast.classical_ris(kernel, trials=args.trials,
                                         seed=100 + i, pair_depth=24,
                                         threads=args.threads)
            gpu_times.append(time.perf_counter() - started)
            assert gpu["witness"] is None or len(gpu["witness"]) == gpu["best_weight"]
        item = {
            "rank": rank, "n": n, "trials": args.trials,
            "cpu_seconds": cpu_times, "gpu_seconds": gpu_times,
            "cpu_median_seconds": statistics.median(cpu_times),
            "gpu_median_seconds": statistics.median(gpu_times),
        }
        item["gpu_over_cpu_speedup"] = item["cpu_median_seconds"] / item["gpu_median_seconds"]
        rows.append(item)
        print(json.dumps(item), flush=True)
    report = {
        "schema_version": "1.0", "kind": "matched_ideal_ris_gpu_benchmark",
        "seed": args.seed, "repeats": args.repeats, "gpu_threads": args.threads,
        "rows": rows,
        "routing_policy": "GPU eligible for ideal RIS only when workload gate remains >=2x",
        "quality_policy": "GPU witnesses weight-checked; ideal RIS remains upper-bound search",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
