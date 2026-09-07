from __future__ import annotations

"""Resumable fresh-seed native RIS validation for one CSS candidate.

Every chunk is independently seeded and checkpointed.  A witness lighter than
the configured target stops the run immediately, after an independent GF(2)
logical check.  This is evidence/refutation only: a clean run never proves a
distance lower bound.  No git or network operation is performed.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import time

import cpp_fast
from hyper_validator import _logical_check, structural_validate
from ideal_x_logical_search import scan_candidate


SEED_STEP = 0x9E3779B9


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _compact_check(check: dict) -> dict:
    return {key: value for key, value in check.items() if key != "row_rank"}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _load_or_start(path: Path, candidate: Path, target: int,
                   trials: int, seed: int, ideal_trials: int = 4096,
                   allow_trial_resize: bool = False,
                   requested_total: int | None = None,
                   candidate_sha256: str | None = None,
                   runner_sha256: str | None = None) -> dict:
    if path.exists():
        state = json.loads(path.read_text())
        if str(candidate.resolve()) != state.get("candidate_path"):
            raise ValueError("checkpoint belongs to a different candidate")
        if int(target) != int(state.get("target")):
            raise ValueError("checkpoint target differs")
        if candidate_sha256 and state.get("candidate_sha256") and candidate_sha256 != state.get("candidate_sha256"):
            raise ValueError("checkpoint candidate content differs")
        if runner_sha256 and state.get("runner_sha256") and runner_sha256 != state.get("runner_sha256"):
            raise ValueError("checkpoint runner source differs")
        recorded_trials = int(state.get("trials_per_side"))
        if int(trials) != recorded_trials and not allow_trial_resize:
            raise ValueError("checkpoint chunk size differs")
        recorded_ideal = state.get("common_gcd_ideal_trials")
        if (recorded_ideal is not None and
                int(ideal_trials) != int(recorded_ideal)):
            raise ValueError("checkpoint common-gcd ideal budget differs")
        if int(trials) != recorded_trials:
            history = state.setdefault("chunk_size_history", [])
            history.append({"from": recorded_trials, "to": int(trials),
                            "after_chunk": len(state.get("chunks", []))})
            state["trials_per_side"] = int(trials)
        if requested_total is not None and state.get("requested_trials_per_side") not in (None, int(requested_total)):
            raise ValueError("checkpoint requested total differs")
        state.setdefault("requested_trials_per_side", int(requested_total or recorded_trials * len(state.get("chunks", []))))
        state.setdefault("candidate_sha256", candidate_sha256)
        state.setdefault("runner_sha256", runner_sha256)
        state.setdefault("x_trials_total", sum(int(x.get("x_trials") or 0) for x in state.get("chunks", [])))
        state.setdefault("z_trials_total", sum(int(x.get("z_trials") or 0) for x in state.get("chunks", [])))
        return state
    return {
        "schema_version": "1.0", "kind": "resumable_native_ris_validation",
        "candidate_path": str(candidate.resolve()), "target": int(target),
        "trials_per_side": int(trials), "base_seed": int(seed),
        "requested_trials_per_side": int(requested_total or trials),
        "candidate_sha256": candidate_sha256, "runner_sha256": runner_sha256,
        "common_gcd_ideal_trials": int(ideal_trials),
        "submission_sent": False, "git_commit_performed": False,
        "distance_is_exact": False, "status": "running", "chunks": [],
        "best_observed": {"X": None, "Z": None, "d": None},
        "x_trials_total": 0, "z_trials_total": 0,
        "refutation": None,
    }


def _lightest_refutation(state: dict) -> dict | None:
    """Recover the lightest independently checked below-target witness.

    A native sector call can expose X and Z refutations in the same chunk.
    Retain the lighter one; receipt order must never accidentally report the
    later, heavier side as the campaign's decisive counterexample.
    """
    target = int(state["target"])
    choices = []
    for side in ("X", "Z"):
        item = state.get("best_observed", {}).get(side)
        if not isinstance(item, dict) or int(item.get("weight", target)) >= target:
            continue
        check = None
        for record in state.get("chunks", []):
            if int(record.get("index", -1)) == int(item.get("chunk", -2)):
                check = record.get("witnesses", {}).get(side, {}).get("check")
                break
        if isinstance(check, dict) and check.get("ok"):
            choices.append((int(item["weight"]), side, item, check))
    if not choices:
        return None
    _, side, item, check = min(choices, key=lambda value: (value[0], value[1]))
    return {"side": side, "weight": int(item["weight"]),
            "witness": list(item["witness"]), "chunk": int(item["chunk"]),
            "seed": int(item["seed"]), "check": _compact_check(check)}


def _install_refined_candidate(source: dict, state: dict, out: Path) -> str | None:
    best = state["best_observed"]
    x, z = best.get("X"), best.get("Z")
    if not isinstance(x, dict) or not isinstance(z, dict):
        return None
    refined = copy.deepcopy(source)
    for side, item in (("X", x), ("Z", z)):
        refined["distance"][side]["value"] = int(item["weight"])
        refined["distance"][side]["witness"] = list(item["witness"])
    refined["distance"]["d"] = min(int(x["weight"]), int(z["weight"]))
    # Keep human-facing parameter label synchronized with the evidence.
    # Older receipts could retain the pre-refinement d claim here.
    if isinstance(refined.get("name"), str):
        refined["name"] = re.sub(
            r"d<=\d+", f"d<={refined['distance']['d']}",
            refined["name"], count=1)
    refined.setdefault("regulation", {}).update(
        stage_only=True, submission_sent=False, git_commit_performed=False,
        official_gate_status="not_run", board_claim_allowed=False,
        distance_is_not_proven=True,
    )
    path = out.with_name(f"{out.stem}_{refined['distance']['d']}.json")
    path.write_text(json.dumps(refined, indent=2) + "\n")
    return str(path.resolve())


def run(candidate_path: Path, *, out: Path, chunks: int, trials: int,
        target: int, seed: int, threads: int, pair_depth: int,
        ideal_trials: int = 4096,
        allow_trial_resize: bool = False,
        total_trials_per_side: int | None = None) -> dict:
    # Validation callers commonly pass a fresh campaign subdirectory.  Make
    # receipt creation transactional from first chunk; native validation must
    # not be lost because refined-candidate output parent was absent.
    out.parent.mkdir(parents=True, exist_ok=True)
    source = json.loads(candidate_path.read_text())
    structural = structural_validate(source)
    if not structural["ok"]:
        raise ValueError("candidate fails structural validation")
    if not cpp_fast.available():
        raise RuntimeError(cpp_fast.accelerator_info())
    hx, hz = structural["hx"], structural["hz"]
    runner_sha256 = _sha256_file(Path(__file__).resolve())
    requested_total = int(total_trials_per_side or int(trials) * int(chunks))
    state = _load_or_start(out, candidate_path, target, trials, seed,
                           ideal_trials, allow_trial_resize,
                           requested_total=requested_total,
                           candidate_sha256=_sha256_file(candidate_path),
                           runner_sha256=runner_sha256)
    state.setdefault("common_gcd_ideal_trials", int(ideal_trials))
    if "common_gcd_ideal_probe" not in state:
        if int(ideal_trials) > 0:
            ideal = scan_candidate(
                candidate_path, trials=int(ideal_trials),
                seed=int(seed) ^ 0x6A09E667, pair_depth=int(pair_depth),
                max_seconds=None)
        else:
            ideal = {"kind": "skipped", "reason": "caller_supplied_external_gcd_screen"}
        state["common_gcd_ideal_probe"] = ideal
        found = ideal.get("found_X_logical") or {}
        if (ideal.get("kind") == "cyclic_ideal_x_logical_search" and
                found.get("css_logical_check", {}).get("ok") and
                int(found.get("weight", target)) < int(target)):
            weight = int(found["weight"])
            state["best_observed"]["X"] = {
                "weight": weight,
                "witness": list(found["block_support"]),
                "chunk": 0,
                "seed": int(seed) ^ 0x6A09E667,
            }
            state["best_observed"]["d"] = weight
            state["refutation"] = {
                "side": "X", "weight": weight,
                "witness": list(found["block_support"]), "chunk": 0,
                "seed": int(seed) ^ 0x6A09E667,
                "check": found["css_logical_check"],
                "source": "common_gcd_ideal",
                "g_degree_K": ideal["algebra"]["g_degree_K"],
                "h_degree": ideal["algebra"]["h_degree"],
            }
            state["status"] = "refuted"
            state["distance_is_exact"] = False
            _atomic_json(out, state)
            return state
        _atomic_json(out, state)
    # Repair receipts made by versions that overwrote an X refutation with a
    # later Z refutation in the same chunk.
    repaired = _lightest_refutation(state)
    if repaired is not None and (not isinstance(state.get("refutation"), dict) or
                                 int(repaired["weight"]) < int(state["refutation"].get("weight", target))):
        state["refutation"] = repaired
        state["status"] = "refuted"
        _atomic_json(out, state)
    if state.get("refutation"):
        return state
    existing = len(state["chunks"])
    prior_seconds = float(state.get("seconds_wall", 0.0))
    started = time.perf_counter()
    for index in range(existing, existing + max(0, int(chunks))):
        completed = max(int(state.get("x_trials_total", 0)), int(state.get("z_trials_total", 0)))
        remaining = int(state["requested_trials_per_side"]) - completed
        if remaining <= 0:
            break
        chunk_trials = min(int(trials), remaining)
        chunk_seed = int(seed) + index * SEED_STEP
        result = cpp_fast.css_ris_parallel(
            hx, hz, trials=chunk_trials, seed=chunk_seed, pair_depth=int(pair_depth),
            target=int(target), stop_on_target=True, threads=int(threads))
        record = {"index": index + 1, "seed": chunk_seed,
                  "requested_trials": chunk_trials,
                  "d_upper": result.get("d_upper"), "dx_upper": result.get("dx_upper"),
                  "dz_upper": result.get("dz_upper"), "seconds": result.get("total_seconds"),
                  "x_trials": result["x"].get("trials_run"),
                  "z_trials": result["z"].get("trials_run"),
                  "stopped_early": bool(result["x"].get("stopped_early") or
                                        result["z"].get("stopped_early")),
                  "witnesses": {}}
        for side, native, kernel, row in (("X", result["x"], hz, hx),
                                          ("Z", result["z"], hx, hz)):
            weight, witness = native.get("best_weight"), native.get("witness")
            if weight is None or witness is None:
                continue
            check = _logical_check(witness, kernel, row)
            record["witnesses"][side] = {"weight": int(weight),
                                          "check": _compact_check(check)}
            if check.get("ok"):
                previous = state["best_observed"].get(side)
                if previous is None or int(weight) < int(previous["weight"]):
                    state["best_observed"][side] = {
                        "weight": int(weight), "witness": list(witness),
                        "chunk": index + 1, "seed": chunk_seed,
                    }
                if int(weight) < int(target):
                    candidate_refutation = {
                        "side": side, "weight": int(weight),
                        "witness": list(witness), "chunk": index + 1,
                        "seed": chunk_seed, "check": _compact_check(check),
                    }
                    previous_refutation = state.get("refutation")
                    if (not isinstance(previous_refutation, dict) or
                            int(candidate_refutation["weight"]) <
                            int(previous_refutation.get("weight", target))):
                        state["refutation"] = candidate_refutation
        values = [item["weight"] for item in state["best_observed"].values()
                  if isinstance(item, dict)]
        state["best_observed"]["d"] = min(values) if values else None
        state["chunks"].append(record)
        state["x_trials_total"] = int(state.get("x_trials_total", 0)) + int(record.get("x_trials") or 0)
        state["z_trials_total"] = int(state.get("z_trials_total", 0)) + int(record.get("z_trials") or 0)
        # ``started`` covers this invocation.  Adding the previously updated
        # state on every chunk produces a triangular overcount (84 chunks were
        # reported as 19.6 h although native chunk time summed to 27 min).
        state["seconds_wall"] = prior_seconds + time.perf_counter() - started
        state["status"] = "refuted" if state["refutation"] else "running"
        state["refined_candidate_path"] = _install_refined_candidate(source, state, out)
        _atomic_json(out, state)
        print(json.dumps({"chunk": index + 1, "d": record["d_upper"],
                          "dx": record["dx_upper"], "dz": record["dz_upper"],
                          "refuted": bool(state["refutation"]),
                          "x_trials": record["x_trials"], "z_trials": record["z_trials"],
                          "x_total": state["x_trials_total"], "z_total": state["z_trials_total"]}),
              flush=True)
        if state["refutation"]:
            break
    if not state["refutation"]:
        complete = (int(state.get("x_trials_total", 0)) == int(state["requested_trials_per_side"])
                    and int(state.get("z_trials_total", 0)) == int(state["requested_trials_per_side"]))
        state["status"] = "clean_budget_complete" if complete else "incomplete"
        _atomic_json(out, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=100)
    parser.add_argument("--trials-per-side", type=int, default=20_000)
    parser.add_argument("--total-trials-per-side", type=int, default=None,
                        help="exact total per side; chunks become checkpoint size")
    parser.add_argument("--target", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2026083107)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--pair-depth", type=int, default=24)
    parser.add_argument("--ideal-trials", type=int, default=4096,
                        help="pre-CSS common-gcd ideal trials; 0=skip after external GCD screen")
    parser.add_argument("--allow-chunk-resize", action="store_true",
                        help="resume checkpoint with a different trials-per-side")
    args = parser.parse_args()
    state = run(args.candidate, out=args.out, chunks=args.chunks,
                trials=args.trials_per_side, target=args.target, seed=args.seed,
                threads=args.threads, pair_depth=args.pair_depth,
                ideal_trials=args.ideal_trials,
                allow_trial_resize=args.allow_chunk_resize,
                total_trials_per_side=args.total_trials_per_side)
    print(json.dumps({"status": state["status"], "chunks": len(state["chunks"]),
                      "best": state["best_observed"], "refutation": state["refutation"]},
                     indent=2))
    return 2 if state["refutation"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
