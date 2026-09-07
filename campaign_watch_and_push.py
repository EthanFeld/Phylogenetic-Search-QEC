from __future__ import annotations

"""Persistent local campaign watcher.

Waits for the current independent 2M/side cull, records a compact summary,
then starts the next sparse-divisor phylogeny campaign.  All outputs are
stage-only research artifacts; this script never commits, pushes, or submits.
"""

import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
CURRENT = ROOT / "results/gb_m341_k132_sparse_allw15_01/full_2m"
STAGED = ROOT / "results/gb_m341_k132_sparse_allw15_01/staged_20k"
NEXT = ROOT / "results/gb_m341_k132_sparse_w17_auto01"
SUMMARY = CURRENT / "watch_summary.json"
WATCH_LOG = ROOT / "results/gb_m341_k132_campaign_watch.log"


def log(message: str) -> None:
    WATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n"
    WATCH_LOG.open("a", encoding="utf-8").write(line)
    print(line, end="", flush=True)


def terminal_states() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not CURRENT.exists():
        return out
    for path in CURRENT.glob("candidate_*.json"):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if state.get("status") not in {"refuted", "clean_budget_complete"}:
            continue
        candidate = str(state.get("candidate_path", ""))
        if not candidate:
            continue
        out[Path(candidate).name] = state
    return out


def compact(state: dict) -> dict:
    best = state.get("best_observed", {})
    refutation = state.get("refutation")
    return {
        "status": state.get("status"),
        "d_observed": best.get("d"),
        "dx_observed": (best.get("X") or {}).get("weight"),
        "dz_observed": (best.get("Z") or {}).get("weight"),
        "refutation": ({"side": refutation.get("side"),
                        "weight": refutation.get("weight")}
                       if isinstance(refutation, dict) else None),
        "seconds_wall": state.get("seconds_wall"),
    }


def run_next() -> int:
    NEXT.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-X", "utf8", "gb_m341_k132_target.py",
        "--out", str(NEXT),
        "--branch-mode", "sparse_divisor",
        "--divisor-weight-max", "17",
        "--branches", "194",
        "--mine-branches", "96",
        "--detector-runs", "12",
        "--detector-trials", "2048",
        "--child-runs", "3",
        "--child-trials", "2048",
        "--pairs-per-branch", "32",
        "--pair-attempts", "20000",
        "--proxy-trials", "256",
        "--proxy-replicas", "2",
        "--deep-per-branch", "2",
        "--deep-count", "48",
        "--deep-trials", "20000",
        "--deep-replicas", "4",
        "--confirm-count", "4",
        "--confirm-trials", "200000",
        "--confirm-replicas", "1",
        "--workers", "8",
        "--threads", "2",
        "--seed", "2026123101",
    ]
    log("starting w17 campaign: " + " ".join(command))
    log_path = NEXT / "campaign_auto.log"
    with log_path.open("a", encoding="utf-8") as stream:
        completed = subprocess.run(command, cwd=ROOT, stdout=stream,
                                   stderr=subprocess.STDOUT, check=False)
    log(f"w17 campaign exited rc={completed.returncode}")
    return int(completed.returncode)


def main() -> int:
    log("watcher started")
    while True:
        states = terminal_states()
        running = any(
            state.get("status") == "running"
            for state in (json.loads(path.read_text(encoding="utf-8"))
                          for path in CURRENT.glob("candidate_*.json"))
            if isinstance(state, dict)
        ) if CURRENT.exists() else False
        if len(states) >= 10 and not running:
            break
        time.sleep(30)

    summary = {
        "kind": "local_campaign_watch_summary",
        "current_campaign": str(CURRENT.resolve()),
        "expected_candidates": 10,
        "terminal_candidates": len(states),
        "results": {name: compact(state)
                    for name, state in sorted(states.items())},
        "stage_only": True,
        "commit_performed": False,
        "submission_sent": False,
        "next_campaign": str(NEXT.resolve()),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log(f"current campaign terminal: {len(states)} candidates")

    marker = NEXT / "campaign.json"
    if marker.exists():
        log("next campaign already exists; watcher done")
        return 0
    return run_next()


if __name__ == "__main__":
    raise SystemExit(main())
