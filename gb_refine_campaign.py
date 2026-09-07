from __future__ import annotations

"""Continue a saved GB campaign without repeating its broad population pass."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import cpp_fast
from generalized_bicycle import build_matrices, build_supports, normalize_code
from gb_aggressive_campaign import (
    M, TARGET_D, _candidate_doc, _json_default, _parallel_map, _task_screen,
)
from hyper_validator import structural_validate


def run(args):
    source = Path(args.source)
    prefinal = json.loads((source / "prefinal.json").read_text())
    rows = list(prefinal.get("deep_top", []))[:int(args.refine_count)]
    started = time.perf_counter()
    refined = _parallel_map(
        _task_screen, rows, seed=args.seed, trials=args.trials,
        workers=args.workers, threads=args.threads, label="refine",
        repeats=args.repeats)
    refined.sort(key=lambda r: ((r.get("d_proxy") or -1), r["semantic_hash"]),
                 reverse=True)

    final = []
    for index, row in enumerate(refined[:max(1, int(args.final_pool))]):
        hx, hz = build_matrices(M, tuple(row["A"][0]), tuple(row["B"][0]))
        seed = int(args.seed) + 900000 + index
        gate = cpp_fast.css_ris_parallel(
            hx, hz, trials=int(args.gate_trials), seed=seed, pair_depth=24,
            target=TARGET_D, stop_on_target=True, threads=0)
        dx = gate["x"].get("best_weight"); dz = gate["z"].get("best_weight")
        stage = "gate_refuted"
        result = gate
        if dx is not None and dz is not None and min(dx, dz) >= TARGET_D:
            result = cpp_fast.css_ris_parallel(
                hx, hz, trials=int(args.full_trials), seed=seed + 0x100000,
                pair_depth=24, target=TARGET_D, stop_on_target=False, threads=0)
            dx = result["x"].get("best_weight"); dz = result["z"].get("best_weight")
            stage = "full_survivor"
        if dx is None or dz is None:
            continue
        candidate = _candidate_doc(
            row, dx, dz, result["x"]["witness"], result["z"]["witness"],
            result.get("mode"))
        path = source / f"refined_candidate_{index}_{min(dx, dz)}.json"
        path.write_text(json.dumps(candidate, indent=2) + "\n")
        validator = Path(args.challenge_root) / "verify" / "validate_candidate.py"
        official = None
        if validator.exists():
            proc = subprocess.run(
                [sys.executable, str(validator), str(path.resolve())],
                cwd=Path(args.challenge_root), capture_output=True, text=True)
            try:
                official = json.loads(proc.stdout) if proc.stdout.strip() else {
                    "returncode": proc.returncode, "stderr": proc.stderr}
            except json.JSONDecodeError:
                official = {"returncode": proc.returncode, "stdout": proc.stdout,
                            "stderr": proc.stderr}
        final.append({"candidate_path": str(path.resolve()), "row": row,
                      "dx": int(dx), "dz": int(dz), "d": min(int(dx), int(dz)),
                      "score_upper": 182 * min(int(dx), int(dz)) ** 2 / 682,
                      "refine_trials_per_side": int(args.trials),
                      "refine_repeats": int(args.repeats),
                      "gate_trials_per_side": int(args.gate_trials),
                      "full_trials_per_side": (int(args.full_trials)
                                                if stage == "full_survivor" else 0),
                      "final_stage": stage,
                      "structural": structural_validate(candidate),
                      "official": official})
        (source / "refined_progress.json").write_text(
            json.dumps(final, indent=2, default=_json_default) + "\n")

    report = {"schema_version": "1.0", "kind": "gb_refinement_campaign",
              "submission_sent": False, "git_commit_performed": False,
              "source": str(source.resolve()),
              "predictor": {"method": "repeated native CSS-RIS distance proxy",
                            "trials_per_side": int(args.trials),
                            "repeats": int(args.repeats),
                            "proof_status": "ranking/refutation signal only"},
              "refined": refined, "final": final,
              "seconds_wall": time.perf_counter() - started}
    (source / "refinement.json").write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--refine-count", type=int, default=64)
    ap.add_argument("--trials", type=int, default=20_000)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--final-pool", type=int, default=8)
    ap.add_argument("--gate-trials", type=int, default=200_000)
    ap.add_argument("--full-trials", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--challenge-root", type=Path,
                    default=Path("challenge_data"))
    return run(ap.parse_args())


if __name__ == "__main__":
    result = main()
    print(json.dumps({"source": result["source"], "final": result["final"],
                      "seconds_wall": round(result["seconds_wall"], 1)},
                     indent=2, default=_json_default))
