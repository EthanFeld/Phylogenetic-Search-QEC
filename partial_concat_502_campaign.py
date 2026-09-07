"""Witness-safe partial [[4,2,2]] concatenation of the [[502,102,50]] leaf.

At least fifty selected outer-coordinate pairs are encoded in [[4,2,2]]
blocks and the remaining coordinates pass through unchanged.  Pairing is
constrained so each supplied X/Z witness sees exactly one coordinate in fifty
selected blocks, giving two explicit weight-100 witnesses.

The native partition kernel minimizes check growth.  A candidate is emitted
only when every lifted outer check has weight at most 32.  Distance remains an
upper bound and no network submission or git action is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

import cpp_fast
from concatenated_codes import (CSSCode, css_heterogeneous_block_concatenate,
                                four_two_two_code, identity_css_code,
                                validate_css_code)
from distance_sketch import gf2_logical_basis_packed, unpack_rows
from hyper_validator import structural_validate


SOURCE_DEFAULT = Path("challenge_data/codes/502-102-50.json")
MAX_CHECK_WEIGHT = 32
MIN_PAIR_COUNT = 50


def _matrix(rows, n: int) -> np.ndarray:
    out = np.zeros((len(rows), int(n)), dtype=np.uint8)
    for index, support in enumerate(rows):
        out[index, np.asarray(support, dtype=np.int64)] = 1
    return out


def _supports(matrix: np.ndarray) -> list[list[int]]:
    return [np.flatnonzero(row).astype(int).tolist() for row in matrix]


def _load_outer(path: Path) -> tuple[dict, CSSCode]:
    doc = json.loads(path.read_text())
    n = int(doc["n"])
    hx = _matrix(doc["checks"]["X"], n)
    hz = _matrix(doc["checks"]["Z"], n)
    lx = unpack_rows(gf2_logical_basis_packed(hz, hx), n)
    lz = unpack_rows(gf2_logical_basis_packed(hx, hz), n)
    code = CSSCode(doc["name"], hx, hz, lx, lz)
    specs = validate_css_code(code)
    if int(specs["k"]) != int(doc["k"]):
        raise ValueError("source JSON k disagrees with reconstructed CSS code")
    return doc, code


def _initial_layout(doc: dict, *, pair_count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make 50 selected pairs, preserving a 2x lift for both witnesses."""
    n = int(doc["n"])
    x = set(map(int, doc["distance"]["X"]["witness"]))
    z = set(map(int, doc["distance"]["Z"]["witness"]))
    x_only, z_only, both = sorted(x - z), sorted(z - x), sorted(x & z)
    if len(x_only) != len(z_only):
        raise ValueError("witness-only support counts must match for safe pairing")
    labels = np.zeros(n, dtype=np.int32)
    tags = np.zeros(n, dtype=np.uint8)
    tags[x_only] |= 1
    tags[z_only] |= 2
    tags[both] |= 3
    outside = iter(sorted(set(range(n)) - x - z))
    label = 1
    for left, right in zip(x_only, z_only):
        labels[left] = labels[right] = label
        label += 1
    for left in both:
        labels[left] = labels[next(outside)] = label
        label += 1
    if label != MIN_PAIR_COUNT + 1 or int(pair_count) < MIN_PAIR_COUNT:
        raise ValueError("unexpected selected-pair count")
    free = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(free)
    for next_label in range(label, int(pair_count) + 1):
        left, right = free[:2]
        labels[left] = labels[right] = next_label
        free = free[2:]
    return labels, (tags != 0).astype(np.uint8), tags


def _cut_count(checks: np.ndarray, labels: np.ndarray, *, pair_count: int) -> np.ndarray:
    return np.asarray([
        sum(int(row[labels == label].sum()) & 1 for label in range(1, int(pair_count) + 1))
        for row in checks
    ], dtype=np.int16)


def _optimize_layout(outer: CSSCode, labels: np.ndarray, active: np.ndarray,
                     tags: np.ndarray, *, attempts: int, steps: int,
                     seed: int, pair_count: int) -> tuple[np.ndarray, dict]:
    checks = np.vstack((outer.hx, outer.hz))
    best = None
    for attempt in range(int(attempts)):
        result = cpp_fast.partition_anneal(
            checks, labels, block_count=int(pair_count) + 1,
            must_active=active, tags=tags, steps=int(steps),
            seed=int(seed) + attempt * 1_000_003, target_odd=MAX_CHECK_WEIGHT - 24)
        cuts = _cut_count(checks, result["labels"], pair_count=pair_count)
        item = {
            "attempt": attempt, "seed": int(seed) + attempt * 1_000_003,
            "native": {key: value for key, value in result.items() if key != "labels"},
            "max_growth": int(cuts.max()), "over_cap_rows": int(np.count_nonzero(cuts > 8)),
            "growth_histogram": np.bincount(cuts).astype(int).tolist(),
        }
        key = (item["over_cap_rows"], item["max_growth"], item["native"]["cost"])
        if best is None or key < best[0]:
            best = (key, result["labels"], item)
        if not item["over_cap_rows"]:
            break
    if best is None:
        raise RuntimeError("no partition attempt executed")
    if best[2]["over_cap_rows"]:
        raise RuntimeError(f"no compliant layout; best={best[2]}")
    return best[1], best[2]


def _permuted_outer(outer: CSSCode, labels: np.ndarray, *, pair_count: int) -> tuple[CSSCode, np.ndarray]:
    pair_parts = [np.flatnonzero(labels == label) for label in range(1, int(pair_count) + 1)]
    if any(len(part) != 2 for part in pair_parts):
        raise ValueError("selected labels must remain two-coordinate blocks")
    order = np.concatenate((*pair_parts, np.flatnonzero(labels == 0))).astype(np.int64)
    if len(order) != outer.hx.shape[1] or len(np.unique(order)) != len(order):
        raise ValueError("layout lost or duplicated an outer coordinate")
    old_to_new = np.empty(len(order), dtype=np.int64)
    old_to_new[order] = np.arange(len(order), dtype=np.int64)
    return CSSCode(outer.name, outer.hx[:, order], outer.hz[:, order],
                   outer.lx[:, order], outer.lz[:, order]), old_to_new


def _lift_witness(support: list[int], old_to_new: np.ndarray, *, x_side: bool,
                  pair_count: int) -> list[int]:
    inner = four_two_two_code()
    logicals = inner.lx if x_side else inner.lz
    out = np.zeros(4 * int(pair_count) + len(old_to_new) - 2 * int(pair_count), dtype=np.uint8)
    for old in support:
        new = int(old_to_new[int(old)])
        if new < 2 * int(pair_count):
            block, logical = divmod(new, 2)
            out[4 * block:4 * block + 4] ^= logicals[logical]
        else:
            out[4 * int(pair_count) + (new - 2 * int(pair_count))] ^= 1
    return np.flatnonzero(out).astype(int).tolist()


def _candidate_doc(code: CSSCode, source: dict, labels: np.ndarray, *, wx, wz,
                   layout: dict, screen: dict, pair_count: int) -> dict:
    d = min(len(wx), len(wz))
    return {
        "schema_version": "0.1",
        "name": f"[[{code.hx.shape[1]},{code.lx.shape[0]},d<={d}]] partial CSS [[4,2,2]] concatenate",
        "code_type": "CSS", "n": int(code.hx.shape[1]), "k": int(code.lx.shape[0]),
        "checks": {"X": _supports(code.hx), "Z": _supports(code.hz)},
        "distance": {
            "d": d,
            "X": {"value": len(wx), "confidence": "upper_bound", "witness": wx},
            "Z": {"value": len(wz), "confidence": "upper_bound", "witness": wz},
        },
        "family": "witness-safe-partial-css-4-2-2-concatenation",
        "provenance": {
            "authors": ["stage-only autonomous research"],
            "origin": "phylogeny-guided_complementary_partial_concatenation",
            "source_code": {"name": source["name"], "n": int(source["n"]),
                            "k": int(source["k"]), "d_upper": int(source["distance"]["d"])},
            "construction": (f"{int(pair_count)} [[4,2,2]] blocks placed by a native Tanner-parity "
                             "partition optimizer; remaining coordinates use identity blocks"),
            "layout": layout, "screen": screen,
            "novelty": "unknown",
            "notes": "Stage-only. Witness-backed distance upper bound; no submission sent.",
        },
        "regulation": {
            "stage_only": True, "submission_sent": False, "git_commit_performed": False,
            "official_gate_status": "not_run", "board_claim_allowed": False,
            "distance_is_not_proven": True, "literature_novelty": "unverified",
        },
    }


def run(args) -> dict:
    started = time.perf_counter()
    if not cpp_fast.available():
        raise RuntimeError(cpp_fast.accelerator_info())
    source, outer = _load_outer(args.source)
    if args.pair_count < MIN_PAIR_COUNT or 2 * args.pair_count > outer.hx.shape[1]:
        raise ValueError(f"pair_count must lie in [{MIN_PAIR_COUNT}, {outer.hx.shape[1] // 2}]")
    labels, active, tags = _initial_layout(source, pair_count=args.pair_count, seed=args.seed)
    labels, layout = _optimize_layout(outer, labels, active, tags, attempts=args.attempts,
                                      steps=args.partition_steps, seed=args.seed,
                                      pair_count=args.pair_count)
    permuted, old_to_new = _permuted_outer(outer, labels, pair_count=args.pair_count)
    identity_count = int(np.count_nonzero(labels == 0))
    code = css_heterogeneous_block_concatenate(
        permuted, [four_two_two_code()] * args.pair_count + [identity_css_code()] * identity_count,
        name="partial-502-through-4-2-2")
    wx = _lift_witness(source["distance"]["X"]["witness"], old_to_new, x_side=True,
                       pair_count=args.pair_count)
    wz = _lift_witness(source["distance"]["Z"]["witness"], old_to_new, x_side=False,
                       pair_count=args.pair_count)
    if (len(wx), len(wz)) != (100, 100):
        raise RuntimeError("witness-safe layout lost its required 2x lift")
    initial_ris = cpp_fast.css_ris_parallel(
        code.hx, code.hz, trials=args.ris_trials, seed=args.seed + 91,
        pair_depth=32, target=None if args.full_screen else 100,
        stop_on_target=not args.full_screen, threads=0)
    if (initial_ris["x"].get("best_weight") is not None and
            int(initial_ris["x"]["best_weight"]) < len(wx)):
        wx = initial_ris["x"]["witness"]
    if (initial_ris["z"].get("best_weight") is not None and
            int(initial_ris["z"]["best_weight"]) < len(wz)):
        wz = initial_ris["z"]["witness"]
    deep_ris = None
    if args.deep_trials:
        target = min(len(wx), len(wz))
        deep_ris = cpp_fast.css_ris_parallel(
            code.hx, code.hz, trials=args.deep_trials, seed=args.seed + 193,
            pair_depth=32, target=target, stop_on_target=True, threads=0)
        if (deep_ris["x"].get("best_weight") is not None and
                int(deep_ris["x"]["best_weight"]) < len(wx)):
            wx = deep_ris["x"]["witness"]
        if (deep_ris["z"].get("best_weight") is not None and
                int(deep_ris["z"]["best_weight"]) < len(wz)):
            wz = deep_ris["z"]["witness"]
    screen = {"base_witness_weight": 100, "initial_ris": initial_ris,
              "deep_ris": deep_ris}
    doc = _candidate_doc(code, source, labels, wx=wx, wz=wz, layout=layout,
                         screen=screen, pair_count=args.pair_count)
    structural = structural_validate(doc)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    candidate = out / f"candidate_502_partial_{doc['n']}_{doc['k']}_{doc['distance']['d']}.json"
    candidate.write_text(json.dumps(doc, indent=2) + "\n")
    np.savez_compressed(out / "candidate_502_partial.npz", hx=code.hx, hz=code.hz,
                        lx=code.lx, lz=code.lz, labels=labels)
    report = {
        "schema_version": "1.0", "kind": "phylogeny_guided_partial_concatenation",
        "candidate": str(candidate.resolve()), "n": doc["n"], "k": doc["k"],
        "d_upper": doc["distance"]["d"],
        "pair_count": int(args.pair_count),
        "score_upper": float(doc["k"] * doc["distance"]["d"] ** 2 / doc["n"]),
        "layout": layout, "structural": {key: value for key, value in structural.items()
                                             if key not in ("hx", "hz")},
        "regulation": doc["regulation"], "submission_sent": False,
        "git_commit_performed": False, "seconds_wall": time.perf_counter() - started,
    }
    (out / "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--out", type=Path, default=Path("results/partial_concat_502_01"))
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--pair-count", type=int, default=MIN_PAIR_COUNT)
    parser.add_argument("--partition-steps", type=int, default=12_000_000)
    parser.add_argument("--ris-trials", type=int, default=200_000)
    parser.add_argument("--full-screen", action="store_true",
                        help="run every RIS trial instead of early-stop screening")
    parser.add_argument("--deep-trials", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
