"""Re-screen prior campaign leaders against the projected board-score gate.

Older breakout campaigns used RAW_TARGET=125 everywhere.  That threshold is
appropriate for the d=100 research objective, but can reject a compact
high-rate code before it receives the validation needed to decide whether it
could beat the current score line.  This tool only consumes recorded proxy
leaders and runs the normal pair-24 RIS ladder at the score-aware threshold.
It never writes a submission candidate, commits, or claims a lower bound.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import cpp_fast
from projection_breakout_campaign import (
    CURRENT_SCORE,
    _parallel_screen,
    _passes_score_admission,
    _quality,
    score_admission_distance,
)


def raw_score_distance(row: dict, score_target: float) -> int:
    """Distance needed to exceed a score target with no planning deflation."""
    return math.ceil(math.sqrt(float(score_target) * int(row["n"]) / int(row["k"])))


def admission_rows(document: dict, score_target: float, limit: int,
                   raw_score_gate: bool = False) -> list[dict]:
    """Choose recorded proxy leaders that still clear their own score gate."""
    rows = list(document.get("proxy_top", []))
    rows.sort(key=_quality, reverse=True)
    threshold = raw_score_distance if raw_score_gate else score_admission_distance
    return [row for row in rows
            if int(row.get("screen", {}).get("d_upper") or -1) >= threshold(row, score_target)][:limit]


def run(args):
    if not cpp_fast.available():
        raise RuntimeError(cpp_fast.accelerator_info())
    started = time.perf_counter()
    source = Path(args.source)
    document = json.loads(source.read_text())
    score_target = float(args.score_target)
    raw_score_gate = bool(args.raw_score_gate)
    candidates = admission_rows(document, score_target, int(args.max_candidates),
                                raw_score_gate=raw_score_gate)
    threshold_fn = raw_score_distance if raw_score_gate else score_admission_distance
    threshold = lambda row: threshold_fn(row, score_target)
    passes = lambda row: int(row.get("screen", {}).get("d_upper") or -1) >= threshold(row)
    print(json.dumps({"stage": "select", "source": str(source),
                      "score_target": score_target, "candidates": len(candidates),
                      "thresholds": sorted({threshold(row) for row in candidates})}),
          flush=True)
    deep = _parallel_screen(candidates, trials=int(args.deep_trials),
                            seed=int(args.seed) + 2009, workers=int(args.workers),
                            threads=int(args.threads), label="deep", target=threshold)
    deep.sort(key=_quality, reverse=True)
    survivors = [row for row in deep if passes(row)]
    final = _parallel_screen(survivors[:int(args.final_count)],
                             trials=int(args.final_trials), seed=int(args.seed) + 3009,
                             workers=int(args.workers), threads=int(args.threads),
                             label="final", target=threshold)
    final.sort(key=_quality, reverse=True)
    report = {
        "schema_version": "1.0",
        "kind": "score_admission_revalidation",
        "source": str(source.resolve()),
        "target": {"score": score_target,
                   "projection_factor": 0.8,
                   "threshold_rule": ("ceil(sqrt(score*n/k))" if raw_score_gate else
                                      "ceil(sqrt(score*n/k)/projection_factor)"),
                   "raw_score_gate": raw_score_gate},
        "selection": {"recorded_proxy_leaders": len(document.get("proxy_top", [])),
                      "admitted": len(candidates),
                      "thresholds": sorted({threshold(row) for row in candidates})},
        "validation": {"deep_trials_per_side": int(args.deep_trials),
                       "deep_survivors": len(survivors),
                       "final_trials_per_side": int(args.final_trials),
                       "final_runs": len(final), "seed": int(args.seed)},
        "deep": deep, "final": final,
        "seconds_wall": time.perf_counter() - started,
        "regulation": {"stage_only": True, "submission_sent": False,
                       "git_commit_performed": False,
                       "distance_is_randomized_upper_bound": True,
                       "board_claim_allowed": False},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--score-target", type=float, default=CURRENT_SCORE)
    parser.add_argument("--raw-score-gate", action="store_true",
                        help="validate the actual score line, not the 0.8 planning gate")
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--deep-trials", type=int, default=100_000)
    parser.add_argument("--final-count", type=int, default=4)
    parser.add_argument("--final-trials", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"out": str(args.out), "admitted": report["selection"]["admitted"],
                      "deep_survivors": report["validation"]["deep_survivors"],
                      "final_runs": report["validation"]["final_runs"]}, indent=2))


if __name__ == "__main__":
    main()
