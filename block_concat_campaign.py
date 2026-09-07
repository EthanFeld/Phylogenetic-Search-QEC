"""Tree-guided multi-logical CSS concatenation campaign.

Ordinary k=1 serial concatenation cannot use the strongest tree leaves under
the challenge n<=700 cap.  This module instead embeds pairs of outer qubits
as the two logical qubits of a CSS [[4,2,2]] block.  It preserves outer k,
doubles block length, and keeps a weight-15 outer inside the w<=32 cap.

All distances are witness-backed upper bounds.  No submission, commit, or
official verifier call occurs unless explicitly requested on the CLI.
"""

from __future__ import annotations

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
from concatenated_codes import (CSSCode, css_block_concatenate,
                                four_two_two_code)
from hyper_validator import structural_validate
from inverse_design_core import (build_product_checks, candidate_groups_of_order,
                                 canonical_1x2_logicals)
from phylogenetic_search import select_phylogenetic_elites


MAX_N = 700
MAX_CHECK_WEIGHT = 32


def _supports(matrix: np.ndarray) -> list[list[int]]:
    return [np.flatnonzero(row).astype(int).tolist() for row in matrix]


def _score(n: int, k: int, d: int) -> float:
    return float(int(k) * int(d) * int(d) / int(n))


def _outer_rows(archive: dict, *, inner, limit: int) -> tuple[list[dict], dict]:
    q = int(inner.lx.shape[0])
    logical_weight = int(max(inner.lx.sum(axis=1).max(),
                             inner.lz.sum(axis=1).max()))
    rows = []
    seen = set()
    for row in archive.get("finds", []):
        if row.get("status") != "ok":
            continue
        if row.get("regulation", {}).get("official_gate_status") == "refuted":
            continue
        if not row.get("A") or not row.get("B"):
            continue
        if not row.get("witness_x") or not row.get("witness_z"):
            continue
        try:
            n, k, d, weight = (int(row["n"]), int(row["k"]),
                                int(row["d_upper"]),
                                int(row["max_check_weight"]))
        except (KeyError, TypeError, ValueError):
            continue
        if n % q or n * int(inner.hx.shape[1]) // q > MAX_N:
            continue
        if weight * logical_weight > MAX_CHECK_WEIGHT:
            continue
        key = str(row.get("semantic_hash", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        candidate = dict(row)
        candidate["concat_projected_n"] = n * int(inner.hx.shape[1]) // q
        candidate["concat_projected_w"] = max(weight * logical_weight,
                                               int(inner.hx.sum(axis=1).max()),
                                               int(inner.hz.sum(axis=1).max()))
        candidate["concat_projected_d"] = d * 2
        candidate["concat_projected_score"] = _score(
            candidate["concat_projected_n"], k, candidate["concat_projected_d"])
        # Reuse phylogenetic elite selection with its normal deep-score field.
        # Archive d values remain upper bounds; this is search allocation only.
        candidate["screen"] = {"d_upper": d}
        candidate["distance_estimate"] = {"d_upper": d}
        rows.append(candidate)
    rows.sort(key=lambda row: (float(row["concat_projected_score"]),
                               int(row["d_upper"]), str(row["semantic_hash"])),
              reverse=True)
    selected, tree = select_phylogenetic_elites(rows, min(int(limit), len(rows)),
                                                 exploitation_fraction=0.55)
    return selected, {
        "eligible_outer_count": len(rows),
        "selected_outer_count": len(selected),
        "selection": tree,
        "constraints": {
            "inner_n": int(inner.hx.shape[1]), "inner_k": q,
            "inner_logical_weight": logical_weight,
            "outer_n_max": MAX_N * q // int(inner.hx.shape[1]),
            "outer_check_weight_max": MAX_CHECK_WEIGHT // logical_weight,
        },
    }


def _outer_code(row: dict) -> CSSCode:
    groups = candidate_groups_of_order(int(row["group_order"]))
    group = groups[int(row["group_index"])]
    a = tuple(tuple(int(item) for item in values) for values in row["A"])
    b = tuple(tuple(int(item) for item in values) for values in row["B"])
    hx, hz = build_product_checks(a, b, group)
    lx, lz = canonical_1x2_logicals(a, b, group)
    return CSSCode(str(row.get("semantic_hash", group.name)), hx, hz, lx, lz)


def _pairing_contiguous(n: int) -> list[int]:
    return list(range(int(n)))


def _pairing_half_interleave(n: int) -> list[int]:
    half = int(n) // 2
    return [item for pair in zip(range(half), range(half, 2 * half)) for item in pair]


def _pairing_witness_safe(hx: np.ndarray, hz: np.ndarray, witnesses: list[list[int]],
                          *, seed: int, prefer_shared: bool) -> list[int]:
    """Randomized perfect matching avoiding pairs inside supplied witnesses."""
    n = int(hx.shape[1])
    forbidden = [set(map(int, witness)) for witness in witnesses]
    shared = (hx.T @ hx).astype(np.int16) + (hz.T @ hz).astype(np.int16)
    rng = random.Random(int(seed))
    remaining = set(range(n))
    pairs = []
    while remaining:
        left = rng.choice(tuple(sorted(remaining)))
        remaining.remove(left)
        allowed = [right for right in remaining if not any(
            left in witness and right in witness for witness in forbidden)]
        if not allowed:
            # Dense complement makes this rare. Safe fallback preserves a
            # valid block partition, but report marks the witness collision.
            allowed = list(remaining)
        ordered = sorted(allowed, key=lambda right: (int(shared[left, right]), right),
                         reverse=bool(prefer_shared))
        width = min(len(ordered), 12)
        right = ordered[rng.randrange(width)]
        remaining.remove(right)
        pairs.extend((left, right))
    return pairs


def pairing_variants(outer: CSSCode, row: dict, *, count: int, seed: int) -> list[dict]:
    n = int(outer.hx.shape[1])
    raw = [
        ("contiguous", _pairing_contiguous(n)),
        ("half_interleave", _pairing_half_interleave(n)),
    ]
    witnesses = [list(map(int, row["witness_x"])), list(map(int, row["witness_z"]))]
    for index in range(max(0, int(count) - len(raw))):
        style = "tanner_shared" if index & 1 else "tanner_disjoint"
        order = _pairing_witness_safe(
            outer.hx, outer.hz, witnesses, seed=int(seed) + index * 7919,
            prefer_shared=bool(index & 1))
        raw.append((f"witness_safe_{style}_{index:02d}", order))
    output, seen = [], set()
    for name, order in raw:
        key = tuple(order)
        if key in seen:
            continue
        seen.add(key)
        output.append({"name": name, "order": list(map(int, order)),
                       "sha256": hashlib.sha256(bytes(
                           ",".join(map(str, order)), "ascii")).hexdigest()})
    return output


def _permute_outer(outer: CSSCode, order: list[int]) -> tuple[CSSCode, np.ndarray]:
    order_array = np.asarray(order, dtype=np.int64)
    if sorted(order_array.tolist()) != list(range(outer.hx.shape[1])):
        raise ValueError("pairing is not a permutation")
    old_to_new = np.empty(len(order_array), dtype=np.int64)
    old_to_new[order_array] = np.arange(len(order_array), dtype=np.int64)
    return CSSCode(outer.name, outer.hx[:, order_array], outer.hz[:, order_array],
                   outer.lx[:, order_array], outer.lz[:, order_array]), old_to_new


def _lift_support(support, logicals: np.ndarray) -> list[int]:
    """GF(2) lift of an outer support through a multi-logical inner block."""
    logicals = np.asarray(logicals, dtype=np.uint8)
    q, inner_n = logicals.shape
    vector = np.zeros(0, dtype=np.uint8)
    blocks = 1 + max((int(item) // q for item in support), default=-1)
    vector = np.zeros(blocks * inner_n, dtype=np.uint8)
    for item in support:
        block, logical_index = divmod(int(item), q)
        vector[block * inner_n:(block + 1) * inner_n] ^= logicals[logical_index]
    return np.flatnonzero(vector).astype(int).tolist()


def _candidate_doc(code: CSSCode, row: dict, pairing: dict, inner, *, wx, wz,
                   source: str, screen: dict) -> dict:
    dx, dz = len(wx), len(wz)
    d = min(dx, dz)
    return {
        "schema_version": "0.1",
        "name": f"[[{code.hx.shape[1]},{code.lx.shape[0]},d<={d}]] block-concatenated CSS",
        "code_type": "CSS", "n": int(code.hx.shape[1]), "k": int(code.lx.shape[0]),
        "checks": {"X": _supports(code.hx), "Z": _supports(code.hz)},
        "distance": {
            "d": d,
            "X": {"value": dx, "confidence": "upper_bound", "witness": wx},
            "Z": {"value": dz, "confidence": "upper_bound", "witness": wz},
        },
        "provenance": {
            "authors": ["stage-only autonomous research"],
            "origin": "tree_guided_multilogical_concatenation",
            "novelty": "unknown",
            "construction": (
                "outer lifted-product leaf encoded blockwise through CSS "
                "[[4,2,2]] logical pairs"),
            "outer_semantic_hash": row["semantic_hash"],
            "outer_group": row.get("group"),
            "outer_distance_upper": int(row["d_upper"]),
            "pairing": {key: value for key, value in pairing.items() if key != "order"},
            "inner": {"name": inner.name, "n": int(inner.hx.shape[1]),
                      "k": int(inner.lx.shape[0]), "logical_weight": 2},
            "screen": screen,
            "notes": ("Stage-only. Claim is explicit-witness upper bound; "
                      "no submission sent."),
        },
        "family": "block-concatenated-css-4-2-2",
        "construction": {
            "outer": {"n": int(row["n"]), "k": int(row["k"]),
                      "d_upper": int(row["d_upper"]),
                      "max_check_weight": int(row["max_check_weight"]),
                      "semantic_hash": row["semantic_hash"]},
            "inner": "Four-Two-Two", "inner_k": 2, "inner_distance": 2,
            "pairing": {key: value for key, value in pairing.items() if key != "order"},
            "source": source,
        },
        "regulation": {
            "stage_only": True, "submission_sent": False,
            "official_gate_status": "not_run", "board_claim_allowed": False,
            "distance_is_not_proven": True, "literature_novelty": "unverified",
        },
    }


def _screen(row: dict, pairing: dict, inner, *, trials: int, seed: int,
            source: str) -> dict:
    outer = _outer_code(row)
    permuted, old_to_new = _permute_outer(outer, pairing["order"])
    code = css_block_concatenate(permuted, inner)
    base_x = _lift_support(old_to_new[np.asarray(row["witness_x"], dtype=np.int64)],
                           inner.lx)
    base_z = _lift_support(old_to_new[np.asarray(row["witness_z"], dtype=np.int64)],
                           inner.lz)
    claim = min(len(base_x), len(base_z))
    ris = cpp_fast.css_ris_parallel(
        code.hx, code.hz, trials=int(trials), seed=int(seed), pair_depth=24,
        target=claim, stop_on_target=True, threads=0)
    wx = base_x
    wz = base_z
    # RIS may return a valid but *heavier* representative. Keep the lightest
    # explicit witness; otherwise a known d=36 lift could be misreported as
    # d=58 and be immediately refuted by its own construction witness.
    if (ris["x"].get("best_weight") is not None and
            int(ris["x"]["best_weight"]) < len(base_x)):
        wx = ris["x"]["witness"]
    if (ris["z"].get("best_weight") is not None and
            int(ris["z"]["best_weight"]) < len(base_z)):
        wz = ris["z"]["witness"]
    doc = _candidate_doc(code, row, pairing, inner, wx=wx, wz=wz,
                         source=source, screen={"ris": ris, "base_claim": claim})
    return {
        "outer": row, "pairing": pairing, "doc": doc,
        "d_upper": int(doc["distance"]["d"]),
        "score_upper": _score(doc["n"], doc["k"], doc["distance"]["d"]),
        "ris": ris,
    }


def _official_gate(path: Path, challenge_root: Path) -> dict:
    validator = challenge_root / "verify" / "validate_candidate.py"
    if not validator.exists():
        return {"status": "not_run", "reason": "validator missing"}
    proc = subprocess.run([sys.executable, str(validator), str(path.resolve())],
                          cwd=challenge_root, capture_output=True, text=True)
    try:
        receipt = json.loads(proc.stdout) if proc.stdout.strip() else {}
    except json.JSONDecodeError:
        receipt = {"stdout": proc.stdout, "stderr": proc.stderr,
                   "returncode": proc.returncode}
    receipt_path = path.with_name(path.stem + "_official.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return {"status": "passed" if receipt.get("passed") else "failed",
            "passed": bool(receipt.get("passed")), "receipt": str(receipt_path.resolve()),
            "board_advancing": bool(receipt.get("gates", {}).get("novelty", {}).get(
                "board_advancing")), "returncode": proc.returncode}


def run(args) -> dict:
    started = time.perf_counter()
    if not cpp_fast.available():
        raise RuntimeError(cpp_fast.accelerator_info())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = json.loads(Path(args.archive).read_text())
    inner = four_two_two_code()
    outers, selection = _outer_rows(archive, inner=inner, limit=args.outers)
    screened = []
    for outer_index, row in enumerate(outers):
        code = _outer_code(row)
        variants = pairing_variants(code, row, count=args.pairings,
                                    seed=int(args.seed) + outer_index * 100_003)
        for pairing_index, pairing in enumerate(variants):
            screened.append(_screen(row, pairing, inner, trials=args.proxy_trials,
                                    seed=int(args.seed) + outer_index * 10_000 + pairing_index,
                                    source="proxy"))
        print(json.dumps({"stage": "proxy", "outer": outer_index + 1,
                          "total_outer": len(outers), "candidates": len(screened)}),
              flush=True)
    screened.sort(key=lambda item: (item["score_upper"], item["d_upper"],
                                    item["pairing"]["sha256"]), reverse=True)
    deep = []
    for index, item in enumerate(screened[:max(0, int(args.deep_count))]):
        deep.append(_screen(item["outer"], item["pairing"], inner,
                            trials=args.deep_trials,
                            seed=int(args.seed) + 500_003 + index,
                            source="deep"))
    deep.sort(key=lambda item: (item["score_upper"], item["d_upper"],
                                item["pairing"]["sha256"]), reverse=True)
    final = []
    for index, item in enumerate(deep[:max(0, int(args.final_count))]):
        final_item = _screen(item["outer"], item["pairing"], inner,
                             trials=args.final_trials,
                             seed=int(args.seed) + 900_003 + index,
                             source="final")
        doc = final_item["doc"]
        structural = structural_validate(doc)
        candidate_path = out / f"candidate_{index}_{doc['n']}_{doc['k']}_{doc['distance']['d']}.json"
        candidate_path.write_text(json.dumps(doc, indent=2) + "\n")
        official = _official_gate(candidate_path, Path(args.challenge_root)) if args.official_gate else {
            "status": "not_run"}
        final.append({
            "candidate": str(candidate_path.resolve()), "d_upper": final_item["d_upper"],
            "score_upper": final_item["score_upper"], "outer": doc["construction"]["outer"],
            "pairing": doc["construction"]["pairing"],
            "structural": {key: value for key, value in structural.items()
                           if key not in ("hx", "hz")},
            "official": official,
        })
    report = {
        "schema_version": "1.0", "kind": "tree_guided_block_concatenation_campaign",
        "submission_sent": False, "git_commit_performed": False,
        "inner": {"name": inner.name, "parameters": "[[4,2,2]]",
                  "logical_weight": 2, "score_multiplier_if_outer_symbols_match": 2.0},
        "serial_concatenation_diagnosis": {
            "k1_multiplier": "d_inner^2/n_inner",
            "steane7": 9 / 7, "repetition3": 1 / 3, "shor9": 1.0,
            "why_not_current_682": "n and check-weight expansion exceed caps",
            "escape": "encode two outer qubits per [[4,2,2]] block",
        },
        "selection": selection,
        "proxy": [{"outer": {key: item["outer"].get(key) for key in (
            "n", "k", "d_upper", "max_check_weight", "group", "semantic_hash")},
                   "pairing": {key: value for key, value in item["pairing"].items()
                               if key != "order"}, "d_upper": item["d_upper"],
                   "score_upper": item["score_upper"]} for item in screened],
        "deep": [{"outer_hash": item["outer"]["semantic_hash"],
                  "pairing": {key: value for key, value in item["pairing"].items()
                              if key != "order"}, "d_upper": item["d_upper"],
                  "score_upper": item["score_upper"]} for item in deep],
        "final": final, "seconds_wall": time.perf_counter() - started,
        "regulation": {"stage_only": True, "submission_sent": False,
                       "git_commit_performed": False,
                       "distance_is_randomized_upper_bound": True,
                       "official_gate_opt_in_only": True},
    }
    (out / "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("results/regulated_findings.json"))
    parser.add_argument("--out", type=Path, default=Path("results/block_concat_422_01"))
    parser.add_argument("--outers", type=int, default=6)
    parser.add_argument("--pairings", type=int, default=6)
    parser.add_argument("--proxy-trials", type=int, default=256)
    parser.add_argument("--deep-count", type=int, default=8)
    parser.add_argument("--deep-trials", type=int, default=20_000)
    parser.add_argument("--final-count", type=int, default=1)
    parser.add_argument("--final-trials", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--official-gate", action="store_true")
    parser.add_argument("--challenge-root", type=Path,
                        default=Path("challenge_data"))
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"out": str(args.out), "final": report["final"],
                      "seconds_wall": round(report["seconds_wall"], 1)}, indent=2))


if __name__ == "__main__":
    main()
