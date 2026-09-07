from __future__ import annotations

"""Reproduce live generalized-bicycle bar, mine nearby contenders, gate them."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from distance_sketch import fast_refute_supports
from generalized_bicycle import (
    REFERENCE_A, REFERENCE_B, REFERENCE_M, REFERENCE_SOURCE,
    REFERENCE_WITNESS_X, REFERENCE_WITNESS_Z, GeneralizedBicycle,
    build_supports, common_gcd_degree, mask_support, mine_low_words,
    normalize_code, reference_code,
)


LIVE_BAR_SCORE = 1501.1


def _score(n: int, k: int, d: int) -> float:
    return float(k * d * d / n)


def make_doc(code: GeneralizedBicycle, dx: int, dz: int,
             witness_x, witness_z, *, origin: str = "submission",
             known: bool = False, screen: dict | None = None) -> dict:
    hx, hz = build_supports(code.m, code.a, code.b)
    d = min(int(dx), int(dz))
    return {
        "schema_version": "0.1",
        "name": f"[[{code.n},{code.k},d<={d}]] generalized bicycle m={code.m}",
        "code_type": "CSS", "n": code.n, "k": code.k,
        "checks": {"X": hx, "Z": hz},
        "distance": {
            "d": d,
            "X": {"value": int(dx), "confidence": "upper_bound",
                  "witness": list(witness_x)},
            "Z": {"value": int(dz), "confidence": "upper_bound",
                  "witness": list(witness_z)},
        },
        "provenance": {
            "authors": ["@natestemen"] if known else ["stage-only autonomous research"],
            "origin": origin,
            "novelty": "known_parameters" if known else "unknown",
            "construction": (
                f"generalized bicycle on Z_{code.m}: H_X=[A|B], H_Z=[B^T|A^T]; "
                f"a={list(code.a)}, b={list(code.b)}, gcd degree={code.gcd_degree}"
            ),
            "references": [REFERENCE_SOURCE] if known else [],
            "notes": (
                "Public live-board reproduction; not claimed as new." if known else
                "Stage-only search candidate; human authorship/novelty review required."
            ),
        },
        "family": "generalized-bicycle",
        "search": screen or {},
        "regulation": {
            "stage_only": True,
            "submission_sent": False,
            "official_gate_status": "not_run",
            "board_claim_allowed": False,
            "distance_is_not_proven": True,
            "provenance_is_self_reported": not known,
            "literature_novelty": "known_parameters" if known else "unverified",
        },
    }


def _candidate_record(code: GeneralizedBicycle, result: dict, screen: dict) -> dict:
    sx, sz = result.get("sectors", {}).get("x", {}), result.get("sectors", {}).get("z", {})
    dx, dz = sx.get("best_weight"), sz.get("best_weight")
    if dx is None or dz is None:
        return {"status": "no_witness", "m": code.m, "a": list(code.a), "b": list(code.b),
                "k": code.k, "check_weight": code.check_weight, "screen": screen}
    d = min(int(dx), int(dz))
    return {
        "status": "ok", "m": code.m, "n": code.n, "k": code.k,
        "d_upper": d, "dx_upper": int(dx), "dz_upper": int(dz),
        "check_weight": code.check_weight, "gcd_degree": code.gcd_degree,
        "score_upper": _score(code.n, code.k, d),
        "a": list(code.a), "b": list(code.b),
        "witness_x": sx.get("witness"), "witness_z": sz.get("witness"),
        "screen": screen,
    }


def _official_gate(doc: dict, path: Path, challenge_root: Path) -> dict:
    """Write candidate + receipt; keep publication claims disabled by default."""
    path.write_text(json.dumps(doc, indent=2) + "\n")
    validator = challenge_root / "verify" / "validate_candidate.py"
    gate = {"status": "not_run", "candidate": str(path.resolve()),
            "validator": str(validator.resolve())}
    if validator.exists():
        proc = subprocess.run([sys.executable, str(validator), str(path.resolve())],
                              cwd=challenge_root, capture_output=True, text=True)
        receipt_path = path.with_name(path.stem + "_official.json")
        if proc.stdout.strip():
            receipt_path.write_text(proc.stdout)
            try:
                verdict = json.loads(proc.stdout)
                passed = bool(verdict.get("passed"))
                advancing = bool(verdict.get("gates", {}).get("novelty", {}).get("board_advancing"))
                gate = {"status": "passed" if passed else "failed", "passed": passed,
                        "board_advancing": advancing,
                        "receipt": str(receipt_path.resolve()),
                        "returncode": proc.returncode}
                doc["regulation"]["official_gate_status"] = "passed" if passed else "refuted"
                doc["regulation"]["board_claim_allowed"] = passed and advancing
            except json.JSONDecodeError:
                gate = {"status": "unparseable", "receipt": str(receipt_path.resolve()),
                        "stderr": proc.stderr, "returncode": proc.returncode}
        else:
            gate = {"status": "no_receipt", "stderr": proc.stderr,
                    "returncode": proc.returncode}
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return gate


def mine_pairs(m: int, words: list[tuple[int, ...]], *, pair_candidates: int, seed: int):
    rng = __import__("numpy").random.default_rng(int(seed))
    unique = {(tuple(REFERENCE_A), tuple(REFERENCE_B))}
    words = list(words)
    if len(words) < 2:
        return list(unique)
    for _ in range(int(pair_candidates) * 8):
        i, j = rng.integers(0, len(words), size=2)
        if i == j:
            continue
        a, b = tuple(words[int(i)]), tuple(words[int(j)])
        if len(a) + len(b) > 32:
            continue
        # Keep high-rate divisor pockets; score target later handles k.
        if common_gcd_degree(m, a, b) >= 85:
            unique.add((a, b))
        if len(unique) >= int(pair_candidates) + 1:
            break
    return list(unique)


def run_campaign(*, out: Path, mine_iterations: int, pair_candidates: int,
                 screen_trials: int, confirm_trials: int, seed: int,
                 challenge_root: Path, max_seconds: float | None = None) -> dict:
    reference = reference_code()
    if reference.n > 700 or reference.check_weight > 32:
        raise ValueError("reference seed violates current challenge resource caps")
    records = [{
        "status": "reference",
        "m": reference.m, "n": reference.n, "k": reference.k,
        "d_upper": 75, "dx_upper": 75, "dz_upper": 75,
        "check_weight": reference.check_weight, "gcd_degree": reference.gcd_degree,
        "score_upper": _score(reference.n, reference.k, 75),
        "a": list(reference.a), "b": list(reference.b),
        "witness_x": list(REFERENCE_WITNESS_X), "witness_z": list(REFERENCE_WITNESS_Z),
        "source": REFERENCE_SOURCE,
    }]
    words = mine_low_words(reference.m, [REFERENCE_A, REFERENCE_B],
                            iterations=mine_iterations, seed=seed, max_words=2048)
    pairs = mine_pairs(reference.m, words, pair_candidates=pair_candidates, seed=seed + 1)
    screened = []
    for index, (a, b) in enumerate(pairs):
        code = normalize_code(reference.m, a, b)
        if code.n > 700 or code.check_weight > 32 or code.k < 160:
            continue
        threshold = max(3, int(math.floor(math.sqrt(LIVE_BAR_SCORE * code.n / code.k))) + 1)
        result = fast_refute_supports(
            *build_supports(code.m, code.a, code.b), code.n, threshold,
            trials=screen_trials, seed=(seed + 17 * index) & 0xffffffff,
            max_seconds=max_seconds, backend="cpp")
        screen = {"threshold": threshold, "trials": screen_trials,
                  "refuted": bool(result.get("refuted")),
                  "d_found": result.get("d_found"), "mode": result.get("mode")}
        record = _candidate_record(code, result, screen)
        if record["status"] == "ok" and record["score_upper"] > LIVE_BAR_SCORE:
            screened.append(record)
    screened.sort(key=lambda row: row["score_upper"], reverse=True)
    confirmed = []
    for row in screened[:8]:
        code = normalize_code(reference.m, row["a"], row["b"])
        result = fast_refute_supports(
            *build_supports(code.m, code.a, code.b), code.n, row["d_upper"],
            trials=confirm_trials, seed=(seed + 0x9E3779B9 + len(confirmed)) & 0xffffffff,
            max_seconds=None, backend="cpp")
        rec = _candidate_record(code, result, {**row["screen"],
                                                "confirm_trials": confirm_trials,
                                                "confirm_refuted": result.get("refuted")})
        if rec["status"] == "ok":
            confirmed.append(rec)
    confirmed.sort(key=lambda row: row["score_upper"], reverse=True)
    all_records = records + screened + confirmed
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "1.0", "kind": "generalized_bicycle_campaign",
        "bar": {"source": REFERENCE_SOURCE, "score": LIVE_BAR_SCORE,
                "n": 682, "k": 182, "d": 75},
        "search": {"m": reference.m, "mine_iterations": mine_iterations,
                   "word_pool": len(words), "pair_candidates": len(pairs),
                   "screen_trials": screen_trials, "confirm_trials": confirm_trials,
                   "seed": seed, "submission_sent": False},
        "reference": records[0], "screened_contenders": screened,
        "confirmed_contenders": confirmed,
        "best_observed": max(all_records, key=lambda row: row.get("score_upper", -1)),
    }
    (out / "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    # Package known bar with public witnesses for independent local official gate.
    reference_doc = make_doc(reference, 75, 75, REFERENCE_WITNESS_X,
                             REFERENCE_WITNESS_Z, origin="baseline", known=True,
                             screen={"source": REFERENCE_SOURCE})
    reference_path = out / "reference_682_182_75.json"
    reference_path.write_text(json.dumps(reference_doc, indent=2) + "\n")
    validator = challenge_root / "verify" / "validate_candidate.py"
    receipt_path = out / "reference_682_182_75_official.json"
    gate = {"status": "not_run", "candidate": str(reference_path.resolve())}
    if validator.exists():
        proc = subprocess.run([sys.executable, str(validator), str(reference_path.resolve())],
                              cwd=challenge_root, capture_output=True, text=True)
        if proc.stdout.strip():
            receipt_path.write_text(proc.stdout)
            try:
                verdict = json.loads(proc.stdout)
                gate = {"status": "passed" if verdict.get("passed") else "failed",
                        "passed": bool(verdict.get("passed")),
                        "receipt": str(receipt_path.resolve())}
            except json.JSONDecodeError:
                gate = {"status": "unparseable", "stderr": proc.stderr}
    report["reference_gate"] = gate
    # Package/gate the strongest confirmed *new* candidate separately. A
    # screen score is never promoted to a claim until this gate runs.
    if confirmed:
        best = confirmed[0]
        best_code = normalize_code(reference.m, best["a"], best["b"])
        best_doc = make_doc(best_code, best["dx_upper"], best["dz_upper"],
                            best["witness_x"], best["witness_z"],
                            origin="submission", known=False, screen=best["screen"])
        best_path = out / f"candidate_{best_code.n}_{best_code.k}_{best['d_upper']}.json"
        best_gate = _official_gate(best_doc, best_path, challenge_root)
        report["best_candidate"] = {"candidate": str(best_path.resolve()),
                                     "official_gate": best_gate,
                                     "score_upper": best["score_upper"],
                                     "d_upper": best["d_upper"]}
    (out / "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/generalized_bicycle_campaign"))
    ap.add_argument("--mine-iterations", type=int, default=100_000)
    ap.add_argument("--pair-candidates", type=int, default=128)
    ap.add_argument("--screen-trials", type=int, default=256)
    ap.add_argument("--confirm-trials", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--max-seconds", type=float, default=3.0)
    ap.add_argument("--challenge-root", type=Path,
                    default=Path("challenge_data"))
    args = ap.parse_args()
    report = run_campaign(out=args.out, mine_iterations=args.mine_iterations,
                          pair_candidates=args.pair_candidates,
                          screen_trials=args.screen_trials,
                          confirm_trials=args.confirm_trials, seed=args.seed,
                          challenge_root=args.challenge_root,
                          max_seconds=args.max_seconds)
    print(json.dumps({"out": str(args.out), "word_pool": report["search"]["word_pool"],
                      "pairs": report["search"]["pair_candidates"],
                      "screened": len(report["screened_contenders"]),
                      "confirmed": len(report["confirmed_contenders"]),
                      "best": report["best_observed"],
                      "reference_gate": report["reference_gate"]}, indent=2))


if __name__ == "__main__":
    main()
