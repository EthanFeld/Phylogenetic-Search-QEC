"""Local phylogenetic breakout around the strongest Z_339 q86 genome.

This is discovery only.  Mutations stay in the exact degree-86 cyclic ideal,
then undergo fresh RIS screening; no distance claim is treated as exact and
the script never commits or submits.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import time

import cpp_fast
import gb_m339_k172_squarefree_campaign as q339
from generalized_bicycle import build_matrices, common_gcd_degree
from gf2_factor import factor_xm_plus_one, mul
from hyper_validator import _logical_check, structural_validate


CORE = q339.core
M, N, Q, K = 339, 678, 86, 172
CHECK_CAP = 32


def _pair_from_document(document: dict) -> tuple[tuple[int, ...], tuple[int, ...]]:
    row = document["checks"]["X"][0]
    a = tuple(sorted(int(value) for value in row if int(value) < M))
    b = tuple(sorted(int(value) - M for value in row if int(value) >= M))
    if not a or not b:
        raise ValueError("candidate X row 0 does not contain two nonempty blocks")
    return a, b


def _divisor(indices) -> int:
    factors = factor_xm_plus_one(M)
    value = 1
    for index in indices:
        value = mul(value, factors[int(index)])
    return int(value)


def _orbit(word) -> tuple[int, ...]:
    word = tuple(sorted(int(value) % M for value in word))
    if not word:
        return ()
    return min(tuple(sorted((value - anchor) % M for value in word))
               for anchor in word)


def _hash(a, b) -> str:
    return hashlib.sha256(json.dumps([list(a), list(b)],
                                     separators=(",", ":")).encode()).hexdigest()


def _row(a, b, *, lineage: str):
    if not a or not b or len(a) + len(b) > CHECK_CAP:
        return None
    if common_gcd_degree(M, a, b) != Q:
        return None
    key = CORE._affine_pair_key(M, tuple(a), tuple(b))
    a, b = tuple(key[0]), tuple(key[1])
    if len(a) + len(b) > CHECK_CAP or common_gcd_degree(M, a, b) != Q:
        return None
    return {
        "A": [list(a)], "B": [list(b)], "m": M, "n": N, "k": K,
        "gcd_degree": Q, "branch_id": "m339_q86_local_breakout",
        "branch_index": -1, "factor_indices": [1, 6, 9, 11],
        "phylogenetic_arm": "local_exact_ideal_breakout",
        "inherited_q55_ancestors": [], "divisor_multiplier_symmetry": 1,
        "pairing_mode": lineage, "word_weights": [len(a), len(b)],
        "individual_gcd_degrees": [CORE._word_gcd_degree(a),
                                    CORE._word_gcd_degree(b)],
        "cycle_profile": CORE._cycle_profile(a, b),
        "semantic_hash": _hash(a, b),
        "lineage": lineage,
    }


def _mine_bank(a, b, *, iterations: int, max_words: int,
               hill_restarts: int, hill_steps: int, seed: int):
    base = _divisor((1, 6, 9, 11))
    divisor_support = tuple(i for i in range(M) if (base >> i) & 1)
    bases = [divisor_support, tuple(a), tuple(b)]
    words = []
    words.extend(cpp_fast.cyclic_ideal_mine(
        bases, m=M, iterations=int(iterations), min_weight=3,
        max_weight=CHECK_CAP, seed=int(seed), max_words=int(max_words)))
    for index, support in enumerate(bases):
        words.extend(cpp_fast.cyclic_ideal_hillclimb(
            support, m=M, restarts=int(hill_restarts), steps=int(hill_steps),
            min_weight=3, max_weight=CHECK_CAP,
            seed=int(seed) ^ (0xA5A5A5A5 + index * 0x9E3779B9),
            max_words=int(max_words)))
    # Preserve all orbit representatives and the original parents.  The
    # random shift in the mutator supplies the missing translations.
    unique = {_orbit(word) for word in words if word}
    unique.update((_orbit(a), _orbit(b)))
    # Descendants are intentional: the parent pair itself is q86/q114, and
    # XORing a q86 word with a deeper ideal word preserves the q86 parent
    # ideal.  Only the *common* pair gcd is constrained in _row().
    return sorted((word for word in unique
                   if 1 <= len(word) <= CHECK_CAP and
                   CORE._word_gcd_degree(word) >= Q),
                  key=lambda word: (len(word), word))


def _killer_survives(a, b, killer: tuple[int, ...], side: str) -> bool:
    """Cheap exact rejection for a known light logical orbit.

    Cyclic GB codes preserve every cyclic translate of a logical.  Testing
    one representative therefore rejects the whole orbit.  This is a filter,
    not a distance proof: new lighter witnesses still require RIS/solver
    validation.
    """
    hx, hz = build_matrices(M, tuple(a), tuple(b))
    if side == "X":
        check = _logical_check(killer, hz, hx)
    else:
        check = _logical_check(killer, hx, hz)
    return bool(check.get("ok"))


def _mutations(parent_a, parent_b, bank, *, attempts: int,
               max_candidates: int, seed: int,
               killers: list[tuple[tuple[int, ...], str]] | None = None):
    rng = random.Random(int(seed))
    rows = {}
    parents = (tuple(parent_a), tuple(parent_b))

    def add(a, b, lineage):
        item = _row(tuple(sorted(a)), tuple(sorted(b)), lineage=lineage)
        if item is not None:
            if killers and any(_killer_survives(
                    item["A"][0], item["B"][0], witness, side)
                    for witness, side in killers):
                return
            rows[item["semantic_hash"]] = item

    # Parent may contain known killer; keep it only as baseline when no guard.
    add(*parents, "parent")
    # Shell-prioritized stream: preserving 15+17 and 16+16 neighbors is more
    # informative than filling the cap with tiny descendants.
    shell = [word for word in bank if 7 <= len(word) <= 24]
    for _ in range(max(int(attempts), int(max_candidates) * 8)):
        if len(rows) >= int(max_candidates):
            break
        block = rng.randrange(2)
        source = parents[block]
        word = rng.choice(shell or bank)
        shift = rng.randrange(M)
        graft = {((int(value) + shift) % M) for value in word}
        mutated = set(source).symmetric_difference(graft)
        candidate = [set(parents[0]), set(parents[1])]
        candidate[block] = mutated
        if rng.randrange(5) == 0 and bank:
            other = rng.choice(shell or bank)
            other_shift = rng.randrange(M)
            other_graft = {((int(value) + other_shift) % M) for value in other}
            other_block = 1 - block
            candidate[other_block] = candidate[other_block].symmetric_difference(other_graft)
            lineage = "double_exact_ideal_xor"
        else:
            lineage = "single_exact_ideal_xor"
        if len(candidate[0]) + len(candidate[1]) < 12:
            continue
        add(candidate[0], candidate[1], lineage)
    return list(rows.values())


def _screen_task(task):
    row, trials, seed, threads, target, ideal_trials = task
    return CORE._screen_task((row, int(trials), int(seed), int(threads),
                              int(target), int(ideal_trials)))


def _screen(rows, *, trials: int, target: int, workers: int,
            threads: int, seed: int, ideal_trials: int):
    tasks = [(row, int(trials), int(seed) + i * 0x9E3779B9,
              int(threads), int(target), int(ideal_trials))
             for i, row in enumerate(rows)]
    out = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(_screen_task, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            out.append(future.result())
            if done == 1 or done % 64 == 0 or done == len(futures):
                print(json.dumps({"stage": "local_proxy", "done": done,
                                  "total": len(futures),
                                  "seconds": round(time.perf_counter() - started, 1)}),
                      flush=True)
    return out


def run(args):
    started = time.perf_counter()
    parent_path = Path(args.parent)
    parent_doc = json.loads(parent_path.read_text())
    parent_a, parent_b = _pair_from_document(parent_doc)
    killers = []
    for killer_path in args.killer_receipt or []:
        receipt = json.loads(Path(killer_path).read_text())
        refutation = receipt.get("refutation") or {}
        if not refutation.get("witness"):
            raise ValueError("killer receipt has no refutation witness")
        killer_side = str(refutation.get("side", args.killer_side)).upper()
        if killer_side not in {"X", "Z"}:
            raise ValueError("killer side must be X or Z")
        killers.append((tuple(int(value) for value in refutation["witness"]),
                        killer_side))
    bank = _mine_bank(parent_a, parent_b, iterations=args.mine_iterations,
                      max_words=args.max_words, hill_restarts=args.hill_restarts,
                      hill_steps=args.hill_steps, seed=args.seed)
    candidates = _mutations(parent_a, parent_b, bank, attempts=args.attempts,
                            max_candidates=args.max_candidates,
                            seed=args.seed ^ 0xC0FFEE, killers=killers)
    print(json.dumps({"stage": "local_population", "bank": len(bank),
                      "candidates": len(candidates),
                      "parent_weights": [len(parent_a), len(parent_b)],
                      "killer_guard": bool(killers),
                      "killer_sides": [side for _, side in killers],
                      "killer_weights": [len(witness) for witness, _ in killers]}), flush=True)
    screened = _screen(candidates, trials=args.proxy_trials, target=args.target,
                       workers=args.workers, threads=args.threads,
                       seed=args.seed ^ 0xD1B54A32,
                       ideal_trials=args.ideal_trials)
    screened.sort(key=lambda row: (
        int(row.get("screen", {}).get("d_upper") or -1),
        int(row.get("screen", {}).get("dx_upper") or -1),
        int(row.get("screen", {}).get("dz_upper") or -1),
        row["semantic_hash"]), reverse=True)
    survivors = [row for row in screened
                 if int(row.get("screen", {}).get("d_upper") or -1) >= int(args.target)
                 and not row.get("screen", {}).get("stopped_early")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    finals = []
    for index, row in enumerate(survivors[:int(args.final_count)], 1):
        doc = q339._candidate_doc(row, "local_breakout_proxy")
        if doc is None:
            continue
        path = out / f"candidate_{index}_{N}_{K}_{doc['distance']['d']}.json"
        path.write_text(json.dumps(doc, indent=2) + "\n")
        structural = structural_validate(doc)
        finals.append({"candidate_path": str(path.resolve()),
                       "semantic_hash": row["semantic_hash"],
                       "d_upper": doc["distance"]["d"],
                       "dx_upper": doc["distance"]["X"]["value"],
                       "dz_upper": doc["distance"]["Z"]["value"],
                       "structural": {key: value for key, value in structural.items()
                                      if key not in ("hx", "hz")}})
    report = {
        "schema_version": "1.0",
        "kind": "gb_m339_q86_local_phylogenetic_breakout",
        "parent": str(parent_path.resolve()),
        "target": {"n": N, "k": K, "gcd_degree": Q,
                    "target_d": int(args.target), "check_weight_cap": CHECK_CAP},
        "search": {"bank": len(bank), "candidates": len(candidates),
                   "screened": len(screened), "survivors": len(survivors),
                   "proxy_trials_per_side": int(args.proxy_trials),
                   "common_gcd_ideal_trials_per_candidate": int(args.ideal_trials),
                   "seed": int(args.seed),
                   "killer_guard": bool(killers),
                   "killer_sides": [side for _, side in killers],
                   "killer_weights": [len(witness) for witness, _ in killers],
                   "seconds_wall": time.perf_counter() - started},
        "phylogenetic_hypothesis": {
            "parent": "q86 branch [1,6,9,11] d<=80 at 2M, d=78 in fresh 20M stop",
            "mutation": "single/double XOR with exact degree-86 ideal descendants",
            "escape": "retain factor ideal and vary orbit/translation geometry",
            "selection": ("common-gcd <h> X-logical gate, then fresh RIS plus "
                          "exact structural validation"),
        },
        "top": screened[:128], "final": finals,
        "regulation": {"stage_only": True, "submission_sent": False,
                        "git_commit_performed": False,
                        "distance_is_randomized_upper_bound": True,
                        "official_gate_run": False, "board_claim_allowed": False},
    }
    (out / "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=Path(
        "results/gb_m339_k172_squarefree02/candidate2_validation_2m_target79_80.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/gb_m339_q86_local_breakout01"))
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--mine-iterations", type=int, default=1_000_000)
    parser.add_argument("--max-words", type=int, default=4096)
    parser.add_argument("--hill-restarts", type=int, default=512)
    parser.add_argument("--hill-steps", type=int, default=1024)
    parser.add_argument("--attempts", type=int, default=100_000)
    parser.add_argument("--max-candidates", type=int, default=4096)
    parser.add_argument("--proxy-trials", type=int, default=2_000)
    parser.add_argument("--target", type=int, default=79)
    parser.add_argument("--final-count", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--ideal-trials", type=int, default=1024,
                        help="pre-CSS common-gcd ideal trials per candidate")
    parser.add_argument("--killer-receipt", type=Path, action="append",
                        help="resumable RIS receipt; repeat for multiple killers")
    parser.add_argument("--killer-side", choices=("X", "Z"), default="Z")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"out": str(args.out), "search": report["search"],
                      "final": report["final"]}, indent=2))


if __name__ == "__main__":
    main()
