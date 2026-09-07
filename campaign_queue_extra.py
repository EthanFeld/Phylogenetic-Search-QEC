from __future__ import annotations

"""Queued post-w19 campaigns; local evidence only, never git/submission."""

import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
LOG = ROOT / "results/gb_m341_k132_extra_queue.log"
STATUS = ROOT / "results/gb_m341_k132_extra_queue.json"


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n"
    LOG.open("a", encoding="utf-8").write(line)
    print(line, end="", flush=True)


def save_status(tasks: list[dict]) -> None:
    STATUS.write_text(json.dumps({
        "kind": "stage_only_campaign_queue",
        "tasks": tasks,
        "commit_performed": False,
        "submission_sent": False,
    }, indent=2) + "\n", encoding="utf-8")


def wait_for(path: Path, label: str) -> None:
    log(f"waiting for {label}: {path}")
    while not path.exists():
        time.sleep(30)


def run_validation(report_path: Path, out_dir: Path, label: str,
                   seed_base: int) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    final_paths = [Path(row["candidate_path"])
                   for row in report.get("final", [])
                   if row.get("candidate_path")]
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[Path, Path]] = []
    for index, candidate in enumerate(final_paths):
        if candidate.exists():
            jobs.append((candidate,
                         out_dir / f"candidate_{index + 1}_{candidate.stem}.json"))
    for index, (candidate, out) in enumerate(jobs):
        if out.exists():
            continue
        command = [
            sys.executable, "-X", "utf8", "gb_ris_validation_runner.py",
            str(candidate), "--out", str(out), "--chunks", "1",
            "--trials-per-side", "2000000", "--target", "90",
            "--seed", str(seed_base + index * 300003),
            "--threads", "2", "--pair-depth", "24",
        ]
        log(f"launching {label}: {candidate.name}")
        with (out_dir / f"{out.stem}.log").open("a", encoding="utf-8") as stdout, \
                (out_dir / f"{out.stem}.err.log").open("a", encoding="utf-8") as stderr:
            subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)

    terminal = {"refuted", "clean_budget_complete"}
    while len(jobs):
        done = 0
        for _, out in jobs:
            try:
                state = json.loads(out.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if state.get("status") in terminal:
                done += 1
        if done >= len(jobs):
            break
        time.sleep(30)
    rows = []
    for _, out in jobs:
        try:
            state = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        best = state.get("best_observed", {})
        ref = state.get("refutation")
        rows.append({
            "file": str(out.resolve()), "status": state.get("status"),
            "d_observed": best.get("d"),
            "refutation": ({"side": ref.get("side"), "weight": ref.get("weight")}
                           if isinstance(ref, dict) else None),
        })
    (out_dir / "validation_summary.json").write_text(
        json.dumps({"kind": label, "results": rows, "stage_only": True,
                    "commit_performed": False, "submission_sent": False},
                   indent=2) + "\n", encoding="utf-8")
    log(f"{label} terminal: {len(rows)} receipts")


def run_campaign(out_dir: Path, max_weight: int, branches: int,
                 mine_branches: int, seed: int, label: str,
                 branch_mode: str = "sparse_divisor") -> int:
    if (out_dir / "campaign.json").exists():
        log(f"{label} already exists")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-X", "utf8", "gb_m341_k132_target.py",
        "--out", str(out_dir), "--branch-mode", branch_mode,
        "--divisor-weight-max", str(max_weight), "--branches", str(branches),
        "--mine-branches", str(mine_branches), "--detector-runs", "8",
        "--detector-trials", "2048", "--child-runs", "3",
        "--child-trials", "2048", "--pairs-per-branch", "32",
        "--pair-attempts", "20000", "--proxy-trials", "256",
        "--proxy-replicas", "2", "--deep-per-branch", "2",
        "--deep-count", "64", "--deep-trials", "20000",
        "--deep-replicas", "4", "--confirm-count", "4",
        "--confirm-trials", "200000", "--confirm-replicas", "1",
        "--workers", "8", "--threads", "2", "--seed", str(seed),
    ]
    log(f"starting {label}: " + " ".join(command))
    with (out_dir / "campaign_auto.log").open("a", encoding="utf-8") as stream:
        completed = subprocess.run(command, cwd=ROOT, stdout=stream,
                                   stderr=subprocess.STDOUT, check=False)
    log(f"{label} exited rc={completed.returncode}")
    return int(completed.returncode)


def main() -> int:
    tasks = [
        {"name": "w19_finalist_2m", "status": "queued"},
        {"name": "w21_discovery", "status": "queued"},
        {"name": "w21_finalist_2m", "status": "queued"},
        {"name": "random_shape_discovery_and_2m", "status": "queued"},
    ]
    save_status(tasks)
    log("extra queue started")

    w19 = ROOT / "results/gb_m341_k132_sparse_w19_auto01"
    wait_for(w19 / "campaign.json", "w19 campaign")
    tasks[0]["status"] = "running"; save_status(tasks)
    run_validation(w19 / "campaign.json", w19 / "full_2m", "w19_finalist_2m", 2027020307)
    tasks[0]["status"] = "complete"; save_status(tasks)

    w21 = ROOT / "results/gb_m341_k132_sparse_w21_auto01"
    tasks[1]["status"] = "running"; save_status(tasks)
    run_campaign(w21, 21, 900, 192, 2027020401, "w21_discovery")
    tasks[1]["status"] = "complete"; save_status(tasks)

    tasks[2]["status"] = "running"; save_status(tasks)
    wait_for(w21 / "campaign.json", "w21 campaign")
    run_validation(w21 / "campaign.json", w21 / "full_2m", "w21_finalist_2m", 2027020507)
    tasks[2]["status"] = "complete"; save_status(tasks)

    random_shape = ROOT / "results/gb_m341_k132_random_shape_auto01"
    tasks[3]["status"] = "running"; save_status(tasks)
    run_campaign(random_shape, 21, 96, 64, 2027020601,
                 "random_shape_discovery_and_2m", "random_shape")
    if (random_shape / "campaign.json").exists():
        run_validation(random_shape / "campaign.json", random_shape / "full_2m",
                       "random_shape_finalist_2m", 2027020707)
    tasks[3]["status"] = "complete"; save_status(tasks)
    log("extra queue complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
