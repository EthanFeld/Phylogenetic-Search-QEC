from __future__ import annotations

"""Queued continuation for the local stage-only q=66 search."""

import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
NEXT = ROOT / "results/gb_m341_k132_sparse_w17_auto01"
FULL = NEXT / "full_2m"
W19 = ROOT / "results/gb_m341_k132_sparse_w19_auto01"
LOG = ROOT / "results/gb_m341_k132_followup.log"


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n"
    LOG.open("a", encoding="utf-8").write(line)
    print(line, end="", flush=True)


def run_validation(final_paths: list[Path]) -> None:
    FULL.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[Path, Path]] = []
    for index, candidate in enumerate(final_paths):
        if not candidate.exists():
            continue
        out = FULL / f"candidate_{index + 1}_{candidate.stem}.json"
        if out.exists():
            continue
        jobs.append((candidate, out))

    for index, (candidate, out) in enumerate(jobs):
        log_path = FULL / f"{out.stem}.log"
        err_path = FULL / f"{out.stem}.err.log"
        command = [
            sys.executable, "-X", "utf8", "gb_ris_validation_runner.py",
            str(candidate), "--out", str(out), "--chunks", "1",
            "--trials-per-side", "2000000", "--target", "90",
            "--seed", str(2027010107 + index * 300003),
            "--threads", "2", "--pair-depth", "24",
        ]
        log("starting w17 finalist validation: " + candidate.name)
        with log_path.open("a", encoding="utf-8") as stdout, \
                err_path.open("a", encoding="utf-8") as stderr:
            subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)

    expected = len(final_paths)
    states = []
    while expected and True:
        states = []
        for path in FULL.glob("candidate_*.json"):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if state.get("status") in {"refuted", "clean_budget_complete"}:
                states.append(state)
        if len(states) >= expected:
            break
        time.sleep(30)

    rows = []
    for state in states:
        best = state.get("best_observed", {})
        ref = state.get("refutation")
        rows.append({
            "status": state.get("status"),
            "d_observed": best.get("d"),
            "refutation": ({"side": ref.get("side"), "weight": ref.get("weight")}
                           if isinstance(ref, dict) else None),
            "candidate_path": state.get("candidate_path"),
        })
    (FULL / "validation_summary.json").write_text(
        json.dumps({"kind": "w17_finalist_2m_summary", "results": rows,
                    "stage_only": True, "commit_performed": False,
                    "submission_sent": False}, indent=2) + "\n",
        encoding="utf-8")
    log(f"w17 finalist validation terminal: {len(rows)} states")


def run_w19() -> int:
    if (W19 / "campaign.json").exists():
        log("w19 campaign already exists")
        return 0
    W19.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-X", "utf8", "gb_m341_k132_target.py",
        "--out", str(W19), "--branch-mode", "sparse_divisor",
        "--divisor-weight-max", "19", "--branches", "500",
        "--mine-branches", "128", "--detector-runs", "10",
        "--detector-trials", "2048", "--child-runs", "3",
        "--child-trials", "2048", "--pairs-per-branch", "32",
        "--pair-attempts", "20000", "--proxy-trials", "256",
        "--proxy-replicas", "2", "--deep-per-branch", "2",
        "--deep-count", "64", "--deep-trials", "20000",
        "--deep-replicas", "4", "--confirm-count", "4",
        "--confirm-trials", "200000", "--confirm-replicas", "1",
        "--workers", "8", "--threads", "2", "--seed", "2027010201",
    ]
    log("starting w19 campaign: " + " ".join(command))
    with (W19 / "campaign_auto.log").open("a", encoding="utf-8") as stream:
        completed = subprocess.run(command, cwd=ROOT, stdout=stream,
                                   stderr=subprocess.STDOUT, check=False)
    log(f"w19 campaign exited rc={completed.returncode}")
    return int(completed.returncode)


def main() -> int:
    log("follow-up watcher started")
    while not (NEXT / "campaign.json").exists():
        time.sleep(30)
    try:
        report = json.loads((NEXT / "campaign.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log(f"cannot read w17 report: {exc}")
        return 2
    final_paths = [Path(row["candidate_path"])
                   for row in report.get("final", [])
                   if row.get("candidate_path")]
    run_validation(final_paths)
    return run_w19()


if __name__ == "__main__":
    raise SystemExit(main())
