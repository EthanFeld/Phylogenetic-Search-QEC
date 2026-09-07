from __future__ import annotations

"""Build, screen, package, and official-gate serial CSS concatenations."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from concatenated_codes import (css_concatenate, inner_code_catalog, load_npz_code,
                                validate_css_code)
from distance_sketch import css_sqetch


def _supports(matrix):
    return [row.nonzero()[0].astype(int).tolist() for row in matrix]


def _outer_doc(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def run(outer_npz: Path, *, inner_name: str, trials: int, seed: int,
        backend: str, claim_d: int | None = None, max_check_weight: int = 32,
        max_n: int = 700) -> tuple[dict, object]:
    outer = load_npz_code(outer_npz, name=outer_npz.stem)
    catalog = inner_code_catalog()
    if inner_name not in catalog:
        raise ValueError(f"unknown inner code {inner_name}; choose {sorted(catalog)}")
    inner = catalog[inner_name]
    code = css_concatenate(outer, inner)
    specs = validate_css_code(code, expected_k=outer.lx.shape[0], check_rank=False)
    if specs["n"] > int(max_n):
        raise ValueError(
            f"{code.name} blocklength {specs['n']} exceeds challenge cap {max_n}")
    if specs["max_check_weight"] > int(max_check_weight):
        raise ValueError(
            f"{code.name} max check weight {specs['max_check_weight']} exceeds cap {max_check_weight}")
    estimate = css_sqetch(code.hx, code.hz, code.lx, code.lz,
                          num_trials=int(trials), seed=int(seed), backend=backend)
    dx = estimate.get("dx_upper")
    dz = estimate.get("dz_upper")
    d = claim_d if claim_d is not None else estimate.get("d_upper")
    if d is None or dx is None or dz is None:
        raise RuntimeError("distance search returned no sector witness")
    outer_json = _outer_doc(outer_npz.with_suffix(".json"))
    outer_meta = {
        key: outer_json.get(key)
        for key in ("name", "n", "k", "family", "semantic_hash", "distance")
        if key in outer_json
    }
    doc = {
        "schema_version": "0.1",
        "name": f"{code.name} serial concatenation [[{specs['n']},{specs['k']},{d}]]",
        "code_type": "CSS",
        "n": int(specs["n"]),
        "k": int(specs["k"]),
        "checks": {"X": _supports(code.hx), "Z": _supports(code.hz)},
        "distance": {
            "d": int(d),
            "X": {"value": int(dx), "confidence": "upper_bound",
                  "witness": estimate["x"].get("witness")},
            "Z": {"value": int(dz), "confidence": "upper_bound",
                  "witness": estimate["z"].get("witness")},
        },
        "provenance": {
            "authors": ["stage-only autonomous research"],
            "origin": "derived",
            "novelty": "unknown",
            "construction": f"serial CSS concatenation outer={outer.name}, inner={inner.name}",
            "outer": outer_meta,
            "notes": "Derived candidate; authorship and novelty require human review.",
        },
        "family": "serial-concatenated-css",
        "construction": {
            "outer": outer.name,
            "inner": inner.name,
            "inner_k": 1,
            "inner_distance": {"Steane-7": 3, "Repetition-3": 1, "Shor-9": 3}[inner.name],
            "screen": estimate,
        },
        "regulation": {
            "stage_only": True,
            "submission_sent": False,
            "official_gate_status": "not_run",
            "board_claim_allowed": False,
            "distance_is_not_proven": True,
            "provenance_is_self_reported": True,
            "literature_novelty": "unverified",
        },
    }
    return doc, code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outer_npz", type=Path)
    ap.add_argument("--inner", choices=sorted(inner_code_catalog()), default="steane7")
    ap.add_argument("--trials", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--backend", choices=("numpy", "cpp", "auto"), default="cpp")
    ap.add_argument("--claim-d", type=int, default=None)
    ap.add_argument("--max-check-weight", type=int, default=32)
    ap.add_argument("--max-n", type=int, default=700,
                    help="trusted challenge blocklength cap")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--challenge-root", type=Path,
                    default=Path("challenge_data"))
    args = ap.parse_args()
    doc, code = run(args.outer_npz, inner_name=args.inner, trials=args.trials,
                    seed=args.seed, backend=args.backend, claim_d=args.claim_d,
                    max_check_weight=args.max_check_weight, max_n=args.max_n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    import numpy as np
    np.savez_compressed(args.out.with_suffix(".npz"), hx=code.hx, hz=code.hz,
                        lx=code.lx, lz=code.lz)
    validator = args.challenge_root / "verify" / "validate_candidate.py"
    result = {"candidate": str(args.out.resolve()), "validator": str(validator),
              "n": doc["n"], "k": doc["k"], "d": doc["distance"]["d"],
              "score": doc["k"] * doc["distance"]["d"] ** 2 / doc["n"],
              "max_check_weight": max(max(map(len, doc["checks"]["X"])),
                                        max(map(len, doc["checks"]["Z"]))) }
    if validator.exists():
        proc = subprocess.run([sys.executable, str(validator), str(args.out.resolve())],
                              cwd=args.challenge_root, capture_output=True, text=True)
        receipt = args.out.with_name(args.out.stem + "_official.json")
        if proc.stdout.strip():
            receipt.write_text(proc.stdout)
            try:
                official = json.loads(proc.stdout)
                result["official_passed"] = bool(official.get("passed"))
                board_advancing = bool(official.get("gates", {}).get("novelty", {}).get("board_advancing"))
                result["board_advancing"] = board_advancing
                doc["regulation"]["official_gate_status"] = "passed" if official.get("passed") else "refuted"
                doc["regulation"]["board_claim_allowed"] = bool(official.get("passed")) and board_advancing
            except json.JSONDecodeError:
                result["official_parse"] = "failed"
        if proc.stderr:
            result["validator_stderr"] = proc.stderr
        args.out.write_text(json.dumps(doc, indent=2) + "\n")
        result["official_receipt"] = str(receipt.resolve())
        result["returncode"] = proc.returncode
    print(json.dumps(result, indent=2))
    if validator.exists() and result.get("returncode", 0):
        raise SystemExit(result["returncode"])


if __name__ == "__main__":
    main()
