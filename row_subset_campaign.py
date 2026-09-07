from __future__ import annotations

"""Witness-guided row-subspace search for a row-truncated CSS code.

The 700-206 construction is a useful phylogenetic parent: its X and Z check
sets are independent 247-row subsets of the 350-row lift-350 GB checks.  Any
new independent subsets inherit CSS commutation.  This campaign mutates only
those two row selections, using observed low logicals as cutting planes.

This is a search/ranking tool.  RIS output is randomized evidence, never an
exact distance proof or a submission claim.  It does not commit or submit.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

import cpp_fast
from generalized_bicycle import build_supports
from hyper_validator import structural_validate


def _support_mask(support: list[int] | tuple[int, ...]) -> int:
    value = 0
    for index in support:
        value |= 1 << int(index)
    return value


def _rank_masks(rows: list[int]) -> int:
    basis: dict[int, int] = {}
    rank = 0
    for raw in rows:
        value = int(raw)
        while value:
            pivot = value.bit_length() - 1
            prior = basis.get(pivot)
            if prior is None:
                basis[pivot] = value
                rank += 1
                break
            value ^= prior
    return rank


def _hash_checks(hx: list[list[int]], hz: list[list[int]]) -> str:
    return hashlib.sha256(json.dumps([hx, hz], separators=(",", ":")).encode()).hexdigest()


def _read_parent(candidate: dict) -> tuple[list[list[int]], list[list[int]]]:
    provenance = candidate.get("provenance", {})
    construction = str(provenance.get("construction", ""))
    if "Parent supports:" not in construction:
        raise ValueError("candidate provenance lacks lift-350 parent supports")
    # Keep this parser deliberately strict: it prevents silently optimizing a
    # different parent if the input file changes format.
    import re
    match = re.search(r"a\(x\) exponents \[(.*?)\]; b\(x\) exponents \[(.*?)\]", construction)
    if not match:
        raise ValueError("could not parse parent A/B supports")
    a = [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
    b = [int(x.strip()) for x in match.group(2).split(",") if x.strip()]
    if int(candidate["n"]) != 700:
        raise ValueError("row subset campaign currently requires n=700")
    hx, hz = build_supports(350, a, b)
    return hx, hz


def _map_selected(candidate_rows: list[list[int]], parent_rows: list[list[int]]) -> list[int]:
    lookup = {tuple(row): index for index, row in enumerate(parent_rows)}
    selected = []
    for row in candidate_rows:
        index = lookup.get(tuple(sorted(int(x) for x in row)))
        if index is None:
            raise ValueError("candidate row is not a row of the declared parent")
        selected.append(index)
    if len(selected) != len(set(selected)):
        raise ValueError("candidate repeats a parent row")
    return selected


def _candidate_rows(parent_rows: list[list[int]], selected: list[int]) -> list[list[int]]:
    return [list(parent_rows[index]) for index in sorted(selected)]


def _binary_rows(rows: list[list[int]], n: int = 700) -> np.ndarray:
    matrix = np.zeros((len(rows), int(n)), dtype=np.uint8)
    for row_index, support in enumerate(rows):
        matrix[row_index, [int(x) for x in support]] = 1
    return matrix


def _load_witnesses(path: Path, out: dict[str, list[int]], limit: int) -> None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    def add(side: str, witness):
        if not witness:
            return
        values = [int(x) for x in witness]
        key = tuple(values)
        if key not in {tuple(row) for row in out[side]} and len(out[side]) < limit:
            out[side].append(values)
    distance = data.get("distance", {})
    add("X", distance.get("X", {}).get("witness"))
    add("Z", distance.get("Z", {}).get("witness"))
    for chunk in data.get("chunks", []):
        witnesses = chunk.get("witnesses", {})
        for side in ("X", "Z"):
            add(side, witnesses.get(side, {}).get("support") or
                witnesses.get(side, {}).get("witness"))
    for row in data.get("top", []):
        witnesses = row.get("witnesses", {})
        for side in ("X", "Z"):
            add(side, witnesses.get(side))


def _score_selection(selected_masks: list[int], witness_masks: list[int]) -> tuple:
    if not witness_masks:
        return (0, 0, 0, 0)
    hits = [sum((row & witness).bit_count() & 1 for row in selected_masks)
            for witness in witness_masks]
    blocked = sum(value == 0 for value in hits)
    return (blocked, -sum(hits), -min(hits), max(hits))


def _anneal_rows(parent_rows: list[list[int]], selected: list[int], witness_masks: list[int],
                 *, attempts: int, seed: int, replicas: int) -> tuple[list[int], tuple]:
    parent_masks = [_support_mask(row) for row in parent_rows]
    best = list(selected)
    best_score = _score_selection([parent_masks[i] for i in best], witness_masks)
    rng = random.Random(int(seed))
    count = len(parent_rows)
    capacity = len(selected)

    for replica in range(max(1, int(replicas))):
        current = list(best)
        current_set = set(current)
        # Small restart keeps the search close to a good genome, while larger
        # restarts can escape a witness family saturated at one row pattern.
        for _ in range(replica % 9):
            old = rng.choice(current)
            new = rng.choice([i for i in range(count) if i not in current_set])
            current_set.remove(old); current_set.add(new)
            current[current.index(old)] = new
        current_score = _score_selection([parent_masks[i] for i in current], witness_masks)
        for step in range(max(1, int(attempts))):
            old = current[rng.randrange(capacity)]
            new = rng.randrange(count)
            if new in current_set:
                continue
            proposal = list(current)
            proposal[proposal.index(old)] = new
            proposal_set = set(proposal)
            proposal_score = _score_selection([parent_masks[i] for i in proposal], witness_masks)
            # Lexicographic greedy with a short reheating window.  The first
            # component is the important one: a known logical must be hit by
            # at least one selected opposite-side check row.
            accept = proposal_score < current_score
            if not accept and step % 97 == 0:
                accept = proposal_score[:2] <= current_score[:2]
            if accept:
                # Rank is expensive relative to the bit-mask objective.  Test
                # only proposals that can actually enter the population.
                if _rank_masks([parent_masks[i] for i in proposal]) != capacity:
                    continue
                current, current_set, current_score = proposal, proposal_set, proposal_score
                if current_score < best_score:
                    best, best_score = list(current), current_score
    return sorted(best), best_score


def _screen(hx: np.ndarray, hz: np.ndarray, *, trials: int, seed: int, target: int,
            threads: int) -> dict:
    return cpp_fast.css_ris_parallel(
        hx, hz, trials=int(trials), seed=int(seed), pair_depth=24,
        combo_depth=2, target=int(target), stop_on_target=True,
        threads=int(threads))


def _make_doc(base: dict, hx_rows: list[list[int]], hz_rows: list[list[int]],
              result: dict, lineage: dict) -> dict:
    dx = result["x"].get("best_weight")
    dz = result["z"].get("best_weight")
    d = min(int(value) for value in (dx, dz) if value is not None)
    doc = dict(base)
    doc["name"] = f"[[700,206,d<={d}]] witness-guided row-subspace descendant"
    doc["checks"] = {"X": hx_rows, "Z": hz_rows}
    doc["distance"] = {
        "d": d,
        "X": {"value": int(dx), "confidence": "upper_bound",
               "witness": result["x"].get("witness")},
        "Z": {"value": int(dz), "confidence": "upper_bound",
               "witness": result["z"].get("witness")},
    }
    provenance = dict(base.get("provenance", {}))
    provenance.update({
        "origin": "row_subset_campaign",
        "construction": ("independent rank-247 X/Z row subsets of the declared "
                         "lift-350 generalized-bicycle parent"),
        "notes": "RIS witness-guided mutation; distance remains an upper-bound claim only",
        "lineage": lineage,
    })
    doc["provenance"] = provenance
    doc["search"] = {"semantic_hash": _hash_checks(hx_rows, hz_rows),
                     "lineage": lineage}
    doc["regulation"] = {
        "stage_only": True, "submission_sent": False,
        "git_commit_performed": False, "official_gate_status": "not_run",
        "board_claim_allowed": False, "distance_is_not_proven": True,
        "literature_novelty": "unverified",
    }
    return doc


def _official(path: Path, challenge_root: Path) -> dict | None:
    verifier = challenge_root / "verify" / "qldpc_verify.py"
    if not verifier.exists():
        return None
    proc = subprocess.run([sys.executable, str(verifier), str(path.resolve())],
                          cwd=challenge_root, capture_output=True, text=True)
    try:
        return json.loads(proc.stdout) if proc.stdout.strip() else {
            "returncode": proc.returncode, "stderr": proc.stderr}
    except json.JSONDecodeError:
        return {"returncode": proc.returncode, "stdout": proc.stdout,
                "stderr": proc.stderr}


def run(args):
    if not cpp_fast.available():
        raise RuntimeError(cpp_fast.accelerator_info())
    started = time.perf_counter()
    base = json.loads(Path(args.candidate).read_text())
    parent_x, parent_z = _read_parent(base)
    selected_x = _map_selected(base["checks"]["X"], parent_x)
    selected_z = _map_selected(base["checks"]["Z"], parent_z)
    if len(selected_x) != 247 or len(selected_z) != 247:
        raise ValueError("input must contain 247 X and 247 Z rows")
    parent_masks_x = [_support_mask(row) for row in parent_x]
    parent_masks_z = [_support_mask(row) for row in parent_z]
    if _rank_masks([parent_masks_x[i] for i in selected_x]) != 247:
        raise ValueError("X input rows are not independent")
    if _rank_masks([parent_masks_z[i] for i in selected_z]) != 247:
        raise ValueError("Z input rows are not independent")

    witnesses = {"X": [], "Z": []}
    _load_witnesses(Path(args.candidate), witnesses, args.witness_limit)
    if args.witness_source:
        for source in args.witness_source:
            _load_witnesses(Path(source), witnesses, args.witness_limit)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    archive = []
    current_x, current_z = list(selected_x), list(selected_z)
    for round_index in range(int(args.rounds)):
        hx = _binary_rows(_candidate_rows(parent_x, current_x))
        hz = _binary_rows(_candidate_rows(parent_z, current_z))
        probe = _screen(hx, hz, trials=args.probe_trials,
                        seed=args.seed + round_index * 100003,
                        target=args.target_d, threads=args.threads)
        for side in ("X", "Z"):
            found = probe[side.lower()].get("witness")
            if found and len(witnesses[side]) < args.witness_limit:
                if tuple(found) not in {tuple(row) for row in witnesses[side]}:
                    witnesses[side].append(list(found))
        x_score = _score_selection([parent_masks_z[i] for i in current_z],
                                   [_support_mask(row) for row in witnesses["X"]])
        z_score = _score_selection([parent_masks_x[i] for i in current_x],
                                   [_support_mask(row) for row in witnesses["Z"]])
        next_z, next_z_score = _anneal_rows(
            parent_z, current_z, [_support_mask(row) for row in witnesses["X"]],
            attempts=args.swap_attempts, seed=args.seed + 700001 * round_index,
            replicas=args.replicas)
        next_x, next_x_score = _anneal_rows(
            parent_x, current_x, [_support_mask(row) for row in witnesses["Z"]],
            attempts=args.swap_attempts, seed=args.seed + 900001 * round_index,
            replicas=args.replicas)
        current_x, current_z = next_x, next_z
        hx = _binary_rows(_candidate_rows(parent_x, current_x))
        hz = _binary_rows(_candidate_rows(parent_z, current_z))
        result = _screen(hx, hz, trials=args.screen_trials,
                         seed=args.seed + 3000001 + round_index * 100003,
                         target=args.target_d, threads=args.threads)
        dx = result["x"].get("best_weight")
        dz = result["z"].get("best_weight")
        observed = min(int(value) for value in (dx, dz) if value is not None)
        lineage = {
            "round": round_index + 1, "witness_counts": {k: len(v) for k, v in witnesses.items()},
            "previous_coverage_X": x_score, "previous_coverage_Z": z_score,
            "new_coverage_X": next_z_score, "new_coverage_Z": next_x_score,
            "probe": probe, "screen": result,
        }
        doc = _make_doc(base, _candidate_rows(parent_x, current_x),
                        _candidate_rows(parent_z, current_z), result, lineage)
        structural = structural_validate(doc)
        if not structural["ok"]:
            raise RuntimeError(f"structural validation failed: {structural}")
        path = out / f"candidate_round_{round_index + 1:03d}_d{observed}.json"
        path.write_text(json.dumps(doc, indent=2) + "\n")
        gate = (_official(path, Path(args.challenge_root))
                if args.official and observed >= args.target_d else None)
        entry = {"path": str(path.resolve()), "round": round_index + 1,
                 "dx": dx, "dz": dz, "d_observed": observed,
                 "score_observed": 206 * observed ** 2 / 700,
                 "structural": structural, "official": gate,
                 "witness_counts": {k: len(v) for k, v in witnesses.items()},
                 "semantic_hash": doc["search"]["semantic_hash"]}
        archive.append(entry)
        print(json.dumps({"stage": "row_subset", **entry}, default=str), flush=True)
        if observed >= args.stop_d:
            break

    report = {
        "schema_version": "1.0", "kind": "n700_k206_witness_guided_row_subset_campaign",
        "candidate": str(Path(args.candidate).resolve()), "archive": archive,
        "witness_counts": {k: len(v) for k, v in witnesses.items()},
        "search": {"rounds_requested": int(args.rounds), "rounds_run": len(archive),
                   "probe_trials_per_side": int(args.probe_trials),
                   "screen_trials_per_side": int(args.screen_trials),
                   "swap_attempts": int(args.swap_attempts), "replicas": int(args.replicas),
                   "seconds_wall": time.perf_counter() - started, "seed": int(args.seed)},
        "regulation": {"stage_only": True, "submission_sent": False,
                       "git_commit_performed": False, "board_claim_allowed": False,
                       "distance_is_randomized_upper_bound": True},
    }
    (out / "campaign.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/n700_k206_row_subset_01"))
    ap.add_argument("--witness-source", type=Path, action="append", default=[])
    ap.add_argument("--witness-limit", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--probe-trials", type=int, default=5000)
    ap.add_argument("--screen-trials", type=int, default=20000)
    ap.add_argument("--swap-attempts", type=int, default=2000)
    ap.add_argument("--replicas", type=int, default=8)
    ap.add_argument("--target-d", type=int, default=75)
    ap.add_argument("--stop-d", type=int, default=80)
    ap.add_argument("--seed", type=int, default=2026090307)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--challenge-root", type=Path,
                    default=Path("challenge_data"))
    ap.add_argument("--official", action="store_true",
                    help="run the slow official verifier on every target survivor")
    args = ap.parse_args()
    report = run(args)
    print(json.dumps({"out": str(args.out), "rounds": report["search"]["rounds_run"],
                      "archive": report["archive"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
