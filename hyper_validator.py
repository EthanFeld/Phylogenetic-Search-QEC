from __future__ import annotations

"""High-throughput, fails-closed validator for n<=700 CSS candidates.

Pipeline: exact local structure/witness checks -> parallel native RIS with
adaptive claim refinement -> trusted challenge gate -> optional exact probes.
RIS no-hit is evidence only; exact distance needs solver UNSAT on both sides.
"""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import cpp_fast
from inverse_design_core import gf2_rank


DEFAULT_STAGES = (256, 1024, 4096, 16384)
SEED_STEP = 0x9E3779B9


def _matrix(rows, n: int) -> np.ndarray:
    out = np.zeros((len(rows), int(n)), dtype=np.uint8)
    for i, support in enumerate(rows):
        for q in support:
            out[i, int(q)] ^= 1
    return out


def _shape_checks(doc: dict) -> dict:
    n = int(doc["n"])
    errors = []
    if n < 1 or n > 700:
        errors.append(f"n={n} outside challenge cap 700")
    if int(doc.get("k", -1)) < 0 or int(doc.get("k", -1)) > n:
        errors.append("k outside [0,n]")
    checks = doc.get("checks", {})
    total_support = 0
    for side in ("X", "Z"):
        rows = checks.get(side)
        if not isinstance(rows, list) or not rows:
            errors.append(f"checks.{side} missing/empty")
            continue
        if len(rows) > 10000:
            errors.append(f"checks.{side} exceeds 10000-row budget")
        total = 0
        for r, support in enumerate(rows):
            if not isinstance(support, list):
                errors.append(f"checks.{side}[{r}] not list")
                continue
            vals = [int(q) for q in support]
            if len(vals) != len(set(vals)):
                errors.append(f"checks.{side}[{r}] duplicate support")
            if any(q < 0 or q >= n for q in vals):
                errors.append(f"checks.{side}[{r}] index out of range")
            total += len(vals)
            total_support += len(vals)
        if max((len(row) for row in rows), default=0) > 32:
            errors.append(f"checks.{side} exceeds max row weight 32")
    if total_support > 200000:
        errors.append(f"checks exceed aggregate support budget: {total_support}")
    return {"ok": not errors, "errors": errors}


def _logical_check(witness, H_kernel: np.ndarray, H_row: np.ndarray) -> dict:
    n = H_kernel.shape[1]
    vals = [int(x) for x in witness]
    if len(vals) != len(set(vals)) or any(x < 0 or x >= n for x in vals):
        return {"ok": False, "reason": "witness support invalid"}
    v = np.zeros(n, dtype=np.uint8)
    v[vals] = 1
    if not vals:
        return {"ok": False, "reason": "zero witness"}
    syndrome = int(np.count_nonzero((H_kernel @ v) & 1))
    if syndrome:
        return {"ok": False, "reason": f"syndrome weight={syndrome}"}
    rank_before = gf2_rank(H_row)
    rank_after = gf2_rank(np.vstack([H_row, v]))
    nontrivial = rank_after == rank_before + 1
    return {"ok": bool(nontrivial), "weight": len(vals),
            "syndrome_weight": syndrome, "row_rank": rank_before,
            "nontrivial": bool(nontrivial),
            "reason": None if nontrivial else "witness lies in stabilizer rowspace"}


def block_swap_reverse_support(witness, m: int) -> list[int]:
    """Map j in first block -> m+(-j), second -> (-j), modulo m."""
    m = int(m)
    return sorted(m + (-int(q)) % m if int(q) < m else (-(int(q) - m)) % m
                  for q in witness)


def symmetry_audit(doc: dict, structural: dict) -> dict:
    """Check generalized-bicycle block-swap/index-reversal automorphism."""
    n = int(doc["n"])
    if n % 2:
        return {"present": False, "reason": "n odd"}
    m = n // 2
    hx, hz = structural["hx"], structural["hz"]
    perm = [m + (-j) % m for j in range(m)] + [(-j) % m for j in range(m)]
    mapped_hx = hx[:, perm]
    mapped_hz = hz[:, perm]
    rowspace_hx_to_hz = gf2_rank(np.vstack([hz, mapped_hx])) == gf2_rank(hz)
    rowspace_hz_to_hx = gf2_rank(np.vstack([hx, mapped_hz])) == gf2_rank(hx)
    x = doc["distance"]["X"]["witness"]
    z = doc["distance"]["Z"]["witness"]
    mapped_x = block_swap_reverse_support(x, m)
    mapped_z = block_swap_reverse_support(z, m)
    x_to_z = _logical_check(mapped_x, hx, hz)
    z_to_x = _logical_check(mapped_z, hz, hx)
    return {"present": bool(rowspace_hx_to_hz and rowspace_hz_to_hx),
            "permutation": perm, "rowspace_hx_to_hz": rowspace_hx_to_hz,
            "rowspace_hz_to_hx": rowspace_hz_to_hx,
            "x_to_z": {k: v for k, v in x_to_z.items() if k != "row_rank"},
            "z_to_x": {k: v for k, v in z_to_x.items() if k != "row_rank"},
            "mapped_x_weight": len(mapped_x), "mapped_z_weight": len(mapped_z)}


def structural_validate(doc: dict) -> dict:
    """Exact structure + CSS + witness checks, independent of RIS/native code."""
    shape = _shape_checks(doc)
    if not shape["ok"]:
        return {"ok": False, "shape": shape}
    n = int(doc["n"])
    hx = _matrix(doc["checks"]["X"], n)
    hz = _matrix(doc["checks"]["Z"], n)
    comm = int(np.count_nonzero((hx @ hz.T) & 1))
    rx, rz = gf2_rank(hx), gf2_rank(hz)
    computed_k = n - rx - rz
    distance = doc.get("distance", {})
    side_values = [int(distance[side].get("value", -1)) for side in ("X", "Z")]
    distance_errors = []
    if int(distance.get("d", -1)) != min(side_values):
        distance_errors.append("distance.d != min(distance.X.value, distance.Z.value)")
    witnesses = {}
    for side, kernel, row in (("X", hz, hx), ("Z", hx, hz)):
        item = doc.get("distance", {}).get(side, {})
        wit = item.get("witness")
        witnesses[side] = ({"ok": False, "reason": "missing witness"}
                           if wit is None else _logical_check(wit, kernel, row))
        if witnesses[side].get("ok") and int(item.get("value", -1)) != len(wit):
            witnesses[side] = {"ok": False, "reason": "value != witness weight"}
    ok = (comm == 0 and computed_k == int(doc["k"]) and not distance_errors
          and all(v.get("ok") for v in witnesses.values()))
    return {"ok": bool(ok), "shape": shape, "n": n, "rank_x": rx,
            "rank_z": rz, "computed_k": computed_k,
            "commutation_violations": comm, "distance_errors": distance_errors,
            "witnesses": witnesses,
            "hx": hx, "hz": hz}


def _result_witness(result: dict, side: str, H_kernel: np.ndarray,
                    H_row: np.ndarray) -> tuple[int | None, list[int] | None, dict]:
    item = result.get(side, {})
    weight, witness = item.get("best_weight"), item.get("witness")
    if weight is None or witness is None:
        return None, None, {"ok": True, "present": False}
    check = _logical_check(witness, H_kernel, H_row)
    check["present"] = True
    if not check.get("ok") or int(weight) != len(witness):
        check["ok"] = False
        check["reason"] = "native witness failed independent GF(2) check"
        return None, None, check
    return int(weight), [int(x) for x in witness], check


def refine_candidate(doc: dict, *, stages=DEFAULT_STAGES, seed: int = 0,
                     threads: int = 0, pair_depth: int = 8,
                     max_seconds_per_side: float | None = None) -> dict:
    """Search lower logicals; install only independently verified witnesses."""
    structural = structural_validate(doc)
    if not structural["ok"]:
        return {"status": "invalid", "structural": structural, "candidate": doc,
                "rounds": []}
    if not cpp_fast.available():
        return {"status": "native_unavailable", "structural": structural,
                "candidate": doc, "rounds": []}
    candidate = copy.deepcopy(doc)
    hx, hz = structural["hx"], structural["hz"]
    symmetry = symmetry_audit(doc, structural)
    best_x = int(candidate["distance"]["X"]["value"])
    best_z = int(candidate["distance"]["Z"]["value"])
    rounds = []
    t0 = time.perf_counter()
    for index, trials in enumerate(int(x) for x in stages if int(x) > 0):
        target = min(best_x, best_z)
        result = cpp_fast.css_ris_parallel(
            hx, hz, trials=int(trials), seed=int(seed) + index * SEED_STEP,
            pair_depth=int(pair_depth), max_seconds_per_side=max_seconds_per_side,
            target=target, stop_on_target=True, threads=int(threads))
        round_info = {"stage": index + 1, "trials_per_side": int(trials),
                      "seed": int(seed) + index * SEED_STEP,
                      "target_before": target, "d_found": result.get("d_upper"),
                      "seconds": result.get("total_seconds"),
                      "mode": result.get("mode"), "witness_checks": {}}
        wx, sx, cx = _result_witness(result, "x", hz, hx)
        wz, sz, cz = _result_witness(result, "z", hx, hz)
        round_info["witness_checks"]["X"] = cx
        round_info["witness_checks"]["Z"] = cz
        if wx is not None and wx < best_x:
            best_x = wx
            candidate["distance"]["X"]["value"] = wx
            candidate["distance"]["X"]["witness"] = sx
        if wz is not None and wz < best_z:
            best_z = wz
            candidate["distance"]["Z"]["value"] = wz
            candidate["distance"]["Z"]["witness"] = sz
        candidate["distance"]["d"] = min(best_x, best_z)
        round_info["d_after"] = candidate["distance"]["d"]
        round_info["improved"] = round_info["d_after"] < target
        rounds.append(round_info)
    d = min(best_x, best_z)
    return {"status": "ok", "structural": {k: v for k, v in structural.items()
                                               if k not in ("hx", "hz")},
            "symmetry": symmetry,
            "candidate": candidate, "rounds": rounds,
            "observed": {"dx": best_x, "dz": best_z, "d": d,
                         "seconds": time.perf_counter() - t0,
                         "score_upper": int(candidate["k"]) * d * d / int(candidate["n"])}}


def run_official_gate(candidate: dict, path: Path, challenge_root: Path) -> dict:
    path.write_text(json.dumps(candidate, indent=2) + "\n")
    validator = challenge_root / "verify" / "validate_candidate.py"
    out = {"status": "not_run", "candidate": str(path.resolve()),
           "validator": str(validator.resolve())}
    if not validator.exists():
        return out
    proc = subprocess.run([sys.executable, str(validator), str(path.resolve())],
                          cwd=challenge_root, capture_output=True, text=True)
    receipt = path.with_name(path.stem + "_official.json")
    if proc.stdout.strip():
        receipt.write_text(proc.stdout)
        try:
            verdict = json.loads(proc.stdout)
            passed = bool(verdict.get("passed"))
            advancing = bool(verdict.get("gates", {}).get("novelty", {}).get("board_advancing"))
            out.update({"status": "passed" if verdict.get("passed") else "failed",
                        "passed": passed, "board_advancing": advancing,
                        "receipt": str(receipt.resolve()), "returncode": proc.returncode})
            candidate.setdefault("regulation", {}).update(
                official_gate_status="passed" if passed else "refuted",
                board_claim_allowed=bool(passed and advancing),
                submission_sent=False)
        except json.JSONDecodeError:
            out.update({"status": "unparseable", "receipt": str(receipt.resolve()),
                        "returncode": proc.returncode, "stderr": proc.stderr})
    else:
        out.update({"status": "no_receipt", "returncode": proc.returncode,
                    "stderr": proc.stderr})
    return out


def run_exact_probe(candidate_path: Path, challenge_root: Path, *, solver: str,
                    seconds: float) -> dict:
    """Optional exact probe; timeout/unknown remains inconclusive."""
    if seconds <= 0:
        return {"status": "not_requested", "solver": solver}
    script = challenge_root / "verify" / ("sat_certify.py" if solver == "sat" else "certify.py")
    if not script.exists():
        return {"status": "unavailable", "solver": solver}
    try:
        proc = subprocess.run([sys.executable, str(script), str(candidate_path),
                               "--tlim", str(float(seconds))],
                              cwd=challenge_root, capture_output=True, text=True,
                              timeout=max(1.0, float(seconds) * 2.5 + 5.0))
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "solver": solver}
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"status": "unparseable", "solver": solver,
                "returncode": proc.returncode, "stderr": proc.stderr}
    return {"status": "exact" if result.get("d_exact") else "inconclusive",
            "solver": solver, "result": result, "returncode": proc.returncode}


def validate(path: Path, *, out: Path, challenge_root: Path,
             stages=DEFAULT_STAGES, seed: int = 20260829, threads: int = 0,
             pair_depth: int = 8, exact_solver: str = "none",
             exact_seconds: float = 0.0) -> dict:
    doc = json.loads(path.read_text())
    result = refine_candidate(doc, stages=stages, seed=seed, threads=threads,
                              pair_depth=pair_depth)
    out.parent.mkdir(parents=True, exist_ok=True)
    if result["status"] != "ok":
        report_path = out.with_suffix(".json")
        report_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
        return result
    refined = result["candidate"]
    refined_path = out.with_name(f"{out.stem}_{refined['n']}_{refined['k']}_{refined['distance']['d']}.json")
    gate = run_official_gate(refined, refined_path, challenge_root)
    exact = {}
    for solver in (("milp", "sat") if exact_solver == "both" else (exact_solver,)):
        if solver != "none":
            exact[solver] = run_exact_probe(refined_path, challenge_root,
                                            solver=solver, seconds=exact_seconds)
    result["candidate_path"] = str(refined_path.resolve())
    result["official_gate"] = gate
    result["exact_probes"] = exact
    report_path = out.with_suffix(".json")
    report_path.write_text(json.dumps(result, indent=2, default=lambda x: x.tolist()
                                      if isinstance(x, np.ndarray) else str(x)) + "\n")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/hyper_validation/report"))
    ap.add_argument("--stages", type=int, nargs="+", default=list(DEFAULT_STAGES))
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--pair-depth", type=int, default=8)
    ap.add_argument("--exact", choices=("none", "milp", "sat", "both"), default="none")
    ap.add_argument("--exact-seconds", type=float, default=0.0)
    ap.add_argument("--challenge-root", type=Path,
                    default=Path("challenge_data"))
    args = ap.parse_args()
    result = validate(args.candidate, out=args.out,
                      challenge_root=args.challenge_root, stages=args.stages,
                      seed=args.seed, threads=args.threads,
                      pair_depth=args.pair_depth, exact_solver=args.exact,
                      exact_seconds=args.exact_seconds)
    summary = {k: result.get(k) for k in ("status", "observed", "candidate_path",
                                           "official_gate", "exact_probes") if k in result}
    print(json.dumps(summary, indent=2, default=str))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
