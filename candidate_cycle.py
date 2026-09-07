from __future__ import annotations

"""Persistent GPU-acquisition -> exact-validation candidate pipeline.

One writer process owns queue state. External enqueue calls use a small
cross-process lock. GPU search is screening only; CPU/native RIS receipts are
the validation authority and use exact per-side budgets.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Iterator


ROOT = Path(__file__).resolve().parent
TERMINAL_RECEIPT = {"refuted", "clean_budget_complete"}
DEFAULT_QUEUE = ROOT / "results/candidate_validation_queue/queue.json"
DEFAULT_WITNESS_CACHE = ROOT / "results/ml_gpu_search_20260911/witness_cache.json"
TARGET_SCORE = 1542.0
VALIDATION_BUDGET = 20_000_000


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def abs_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def required_distance(candidate: dict) -> int:
    return int(math.sqrt(TARGET_SCORE * int(candidate["n"]) / int(candidate["k"]))) + 1


@contextmanager
def queue_lock(path: Path, timeout: float = 60.0) -> Iterator[None]:
    """Windows-safe exclusive lock, avoiding a dependency on portalocker."""
    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    fd = None
    while fd is None:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"queue lock timeout: {lock}")
            time.sleep(0.1)
    try:
        os.write(fd, str(os.getpid()).encode("ascii"))
        yield
    finally:
        os.close(fd)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


class Queue:
    def __init__(self, path: Path):
        self.path = abs_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> dict:
        if not self.path.exists():
            return {"schema_version": "1.0", "created_at": now(),
                    "updated_at": now(), "items": []}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"bad queue manifest {self.path}: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise RuntimeError(f"invalid queue manifest: {self.path}")
        return payload

    def write(self, payload: dict) -> None:
        payload["updated_at"] = now()
        fd, tmp_name = tempfile.mkstemp(prefix=self.path.name + ".",
                                         suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, indent=2)
                stream.write("\n")
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    @contextmanager
    def edit(self) -> Iterator[dict]:
        with queue_lock(self.path):
            payload = self.read()
            yield payload
            self.write(payload)


def candidate_paths_from_campaign(path: Path) -> list[Path]:
    report = json.loads(abs_path(path).read_text(encoding="utf-8"))
    out = []
    for row in report.get("final", []):
        candidate = row.get("candidate_path")
        if candidate:
            item = abs_path(Path(candidate))
            if item.exists():
                out.append(item)
    return out


def add_candidates(queue: Queue, paths: list[Path], *, source: str = "manual",
                   status: str = "queued", receipt_dir: Path | None = None) -> list[str]:
    added = []
    with queue.edit() as payload:
        items = payload["items"]
        known_hashes = {str(row.get("candidate_sha256")) for row in items}
        known_paths = {str(Path(row.get("candidate_path", "")).resolve()) for row in items}
        root = abs_path(receipt_dir or queue.path.parent / "validation")
        root.mkdir(parents=True, exist_ok=True)
        for candidate_path in paths:
            candidate_path = abs_path(candidate_path)
            if not candidate_path.exists():
                continue
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            n, k = int(candidate["n"]), int(candidate["k"])
            digest = sha256(candidate_path)
            if digest in known_hashes or str(candidate_path.resolve()) in known_paths:
                continue
            ident = f"cand_{len(items) + 1:04d}_{digest[:12]}"
            receipt = root / f"{ident}.json"
            requested = candidate.get("distance", {}).get("d")
            item = {
                "id": ident, "candidate_path": str(candidate_path.resolve()),
                "candidate_sha256": digest, "source": source,
                "n": n, "k": k, "target_distance": required_distance(candidate),
                "screen_distance": requested, "screen_score": (
                    float(k * int(requested) ** 2 / n) if requested is not None else None),
                "status": status, "attempts": 0, "created_at": now(),
                "updated_at": now(), "receipt": str(receipt.resolve()),
                "log": str(receipt.with_suffix(".log").resolve()),
                "err_log": str(receipt.with_suffix(".err.log").resolve()),
                "validation_budget_per_side": VALIDATION_BUDGET,
                "pid": None, "seed": 2026092801 + len(items) * 300003,
            }
            items.append(item)
            known_hashes.add(digest)
            known_paths.add(str(candidate_path.resolve()))
            added.append(ident)
    return added


def receipt_state(item: dict) -> dict | None:
    path = Path(item["receipt"])
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return False
    return str(int(pid)) in result.stdout


def reconcile(queue: Queue, protected_pids: set[int] | None = None) -> None:
    protected_pids = protected_pids or set()
    with queue.edit() as payload:
        for item in payload["items"]:
            state = receipt_state(item)
            status = state.get("status") if state else None
            if status in TERMINAL_RECEIPT:
                item["status"] = "completed" if status == "clean_budget_complete" else "refuted"
                item["receipt_status"] = status
                item["finished_at"] = item.get("finished_at") or now()
                item["pid"] = None
            elif item.get("status") == "running":
                pid = item.get("pid")
                if pid and int(pid) in protected_pids:
                    item["dead_pid_checks"] = 0
                elif pid and pid_alive(int(pid)):
                    item["dead_pid_checks"] = 0
                elif pid:
                    # tasklist can miss a just-created process during Windows
                    # PID transition. Never requeue on one negative probe:
                    # duplicate writers can corrupt one resumable receipt.
                    misses = int(item.get("dead_pid_checks", 0)) + 1
                    item["dead_pid_checks"] = misses
                    if misses >= 3:
                        item["status"] = "queued"
                        item["last_error"] = (
                            "validator absent for 3 polls before terminal receipt; requeued")
                        item["pid"] = None
            item["updated_at"] = now()


def queue_status(queue: Queue) -> dict:
    payload = queue.read()
    counts: dict[str, int] = {}
    for item in payload["items"]:
        counts[item.get("status", "unknown")] = counts.get(item.get("status", "unknown"), 0) + 1
    return {"queue": str(queue.path), "counts": counts,
            "items": [{"id": x["id"], "status": x["status"],
                        "candidate": Path(x["candidate_path"]).name,
                        "target": x["target_distance"], "pid": x.get("pid"),
                        "receipt": x["receipt"]} for x in payload["items"]]}


def launch_validation(queue: Queue, item: dict, args: argparse.Namespace) -> subprocess.Popen:
    receipt = Path(item["receipt"])
    receipt.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-X", "utf8", "gb_ris_validation_runner.py",
        item["candidate_path"], "--out", str(receipt), "--chunks", "1000",
        "--trials-per-side", str(args.validation_chunk),
        "--total-trials-per-side", str(VALIDATION_BUDGET),
        "--target", str(item["target_distance"]), "--seed", str(item["seed"]),
        "--threads", str(args.validation_threads), "--pair-depth", "24",
        "--ideal-trials", "0",
    ]
    stdout = Path(item["log"]).open("a", encoding="utf-8")
    stderr = Path(item["err_log"]).open("a", encoding="utf-8")
    proc = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
    stdout.close()
    stderr.close()
    with queue.edit() as payload:
        row = next(x for x in payload["items"] if x["id"] == item["id"])
        row.update(status="running", pid=proc.pid, attempts=int(row.get("attempts", 0)) + 1,
                   dead_pid_checks=0, started_at=now(), updated_at=now(), command=command)
    print(f"VALIDATE {item['id']} pid={proc.pid} target={item['target_distance']}", flush=True)
    return proc


def campaign_command(out: Path, seed: int, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable, "-X", "utf8", "ml_mutation_search_v2.py",
        "--seed-split", args.seed_split, "--seeds", str(args.seeds),
        "--mutations-per-seed", str(args.mutations_per_seed),
        "--model-budget", str(args.model_budget), "--survival-horizon", "65536",
        "--ideal-trials", str(args.ideal_trials), "--proxy-trials", str(args.proxy_trials),
        "--deep-trials", str(args.deep_trials), "--deep-count", str(args.deep_count),
        "--deep-per-parent", "2", "--deep-per-gcd-family", "1",
        "--pair-depth", "24", "--deep-pair-depth", "32", "--workers", "1",
        "--gpu-ideal", "--gpu-css", "--gpu-css-rank-cap", "64",
        "--witness-cache", str(abs_path(args.witness_cache)), "--out", str(out),
        "--seed", str(seed),
    ]


def ingest_campaign(queue: Queue, out: Path) -> list[str]:
    marker = out / "campaign.json"
    if not marker.exists():
        return []
    return add_candidates(queue, candidate_paths_from_campaign(marker),
                          source=f"gpu:{out.name}")


def next_cycle_index(root: Path) -> int:
    root = abs_path(root)
    if not root.exists():
        return 0
    values = []
    for path in root.glob("cycle_*"):
        try:
            values.append(int(path.name.rsplit("_", 1)[1]))
        except (ValueError, IndexError):
            continue
    return max(values, default=0)


def run_loop(args: argparse.Namespace) -> int:
    queue = Queue(args.queue)
    acq_proc = None
    acq_out = None
    validation_proc = None
    # Restart-safe: never reuse an existing campaign directory/seed.
    cycle_index = next_cycle_index(args.acquire_root)
    print(f"CYCLE queue={queue.path} budget={VALIDATION_BUDGET}/side GPU=on", flush=True)
    while True:
        if validation_proc is not None and validation_proc.poll() is not None:
            validation_proc = None
        protected = ({int(validation_proc.pid)}
                     if validation_proc is not None else set())
        reconcile(queue, protected_pids=protected)
        payload = queue.read()
        running = [x for x in payload["items"] if x.get("status") == "running"]
        queued = [x for x in payload["items"] if x.get("status") == "queued"]
        if validation_proc is None and not running and queued:
            queued.sort(key=lambda x: (-float(x.get("screen_score") or 0), x["created_at"]))
            validation_proc = launch_validation(queue, queued[0], args)

        if acq_proc is not None and acq_proc.poll() is not None:
            rc = acq_proc.returncode
            print(f"GPU_CAMPAIGN_DONE out={acq_out} rc={rc}", flush=True)
            if rc == 0:
                ids = ingest_campaign(queue, acq_out)
                print(f"ENQUEUED {len(ids)} from {acq_out.name}", flush=True)
            acq_proc = None
            acq_out = None

        payload = queue.read()
        # Cap waiting work, not active validation. Launching next validator
        # frees one waiting slot immediately, so GPU acquisition keeps
        # cycling instead of deadlocking at ``max_pending + 1`` total rows.
        waiting = sum(x.get("status") == "queued" for x in payload["items"])
        if acq_proc is None and waiting < int(args.max_pending):
            cycle_index += 1
            acq_out = abs_path(args.acquire_root) / f"cycle_{cycle_index:04d}"
            acq_out.mkdir(parents=True, exist_ok=True)
            command = campaign_command(acq_out, int(args.acquire_seed) + cycle_index, args)
            log = (acq_out / "campaign_auto.log").open("a", encoding="utf-8")
            acq_proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            log.close()
            print(f"GPU_CAMPAIGN_START cycle={cycle_index} pid={acq_proc.pid} out={acq_out}", flush=True)

        time.sleep(max(2.0, float(args.poll_seconds)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)

    p = sub.add_parser("enqueue")
    common(p)
    p.add_argument("candidates", nargs="*", type=Path)
    p.add_argument("--campaign", type=Path, action="append", default=[])
    p.add_argument("--source", default="manual")
    p.add_argument("--receipt-dir", type=Path, default=None)

    p = sub.add_parser("status")
    common(p)

    p = sub.add_parser("reconcile")
    common(p)

    p = sub.add_parser("run")
    common(p)
    p.add_argument("--poll-seconds", type=float, default=10)
    p.add_argument("--max-pending", type=int, default=8)
    p.add_argument("--validation-threads", type=int, default=14)
    p.add_argument("--validation-chunk", type=int, default=20_000)
    p.add_argument("--acquire-root", type=Path, default=Path("results/gpu_candidate_cycles"))
    p.add_argument("--acquire-seed", type=int, default=2026092900)
    p.add_argument("--seed-split", choices=("train", "validation", "test"), default="validation")
    p.add_argument("--seeds", type=int, default=24)
    p.add_argument("--mutations-per-seed", type=int, default=48)
    p.add_argument("--model-budget", type=int, default=512)
    p.add_argument("--ideal-trials", type=int, default=512)
    p.add_argument("--proxy-trials", type=int, default=512)
    p.add_argument("--deep-trials", type=int, default=4096)
    p.add_argument("--deep-count", type=int, default=16)
    p.add_argument("--witness-cache", type=Path, default=DEFAULT_WITNESS_CACHE)

    args = parser.parse_args()
    if args.command == "enqueue":
        paths = list(args.candidates)
        for campaign in args.campaign:
            paths.extend(candidate_paths_from_campaign(campaign))
        ids = add_candidates(Queue(args.queue), paths, source=args.source,
                             receipt_dir=args.receipt_dir)
        print(json.dumps({"added": ids, "count": len(ids)}, indent=2))
        return 0
    if args.command == "status":
        print(json.dumps(queue_status(Queue(args.queue)), indent=2))
        return 0
    if args.command == "reconcile":
        reconcile(Queue(args.queue))
        print(json.dumps(queue_status(Queue(args.queue)), indent=2))
        return 0
    return run_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
