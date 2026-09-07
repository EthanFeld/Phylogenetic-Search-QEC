"""Generic exact-ideal local breakout for a saved generalized-bicycle code.

The parent pair and all generated words stay in one specified cyclic ideal;
only the pair's common gcd and the official structural/RIS gates decide
admission.  Stage-only: no submit and no commit.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import re
import time

import cpp_fast
import gb_m345_k188_phylo_campaign as core
from generalized_bicycle import build_matrices, build_supports, common_gcd_degree
from gf2_factor import factor_xm_plus_one, mul
from hyper_validator import _logical_check, structural_validate


def configure(m: int, k: int):
    core.M, core.N, core.Q, core.K = int(m), 2 * int(m), int(k) // 2, int(k)
    core.CHECK_CAP = 32
    core.MODULUS = (1 << int(m)) | 1
    core.EXACT_EXACT_ONLY = False
    core.KILLER_IDEAL_FACTOR_SETS = ()
    core.ONE_BLOCK_KILLER_IDEAL_FACTOR_SETS = ()
    core._two_block_ideal_span.cache_clear()
    core._one_block_ideal_span.cache_clear()
    core._one_block_divisor_span.cache_clear()


def pair_from_doc(document):
    m = int(document["n"]) // 2
    row = document["checks"]["X"][0]
    a = tuple(sorted(int(v) for v in row if int(v) < m))
    b = tuple(sorted(int(v) - m for v in row if int(v) >= m))
    if not a or not b:
        raise ValueError("X row 0 has an empty block")
    return a, b


def factor_indices(document, supplied):
    if supplied:
        return tuple(sorted(int(v) for v in supplied.split(",") if v.strip()))
    text = str(document.get("provenance", {}).get("construction", ""))
    match = re.search(r"factors?=\[([^]]+)\]", text)
    if not match:
        raise ValueError("pass --factor-indices; provenance has no factor list")
    return tuple(sorted(int(v) for v in match.group(1).split(",") if v.strip()))


def orbit(word, m):
    word = tuple(sorted(int(v) % int(m) for v in word))
    return min(tuple(sorted((v - anchor) % int(m) for v in word))
               for anchor in word) if word else ()


def divisor_support(m, indices):
    factors = factor_xm_plus_one(int(m))
    value = 1
    for index in indices:
        value = mul(value, factors[int(index)])
    return tuple(i for i in range(int(m)) if (value >> i) & 1)


def mine_bank(parent_a, parent_b, *, m, q, indices, iterations,
              max_words, hill_restarts, hill_steps, seed):
    bases = [divisor_support(m, indices), tuple(parent_a), tuple(parent_b)]
    words = []
    words.extend(cpp_fast.cyclic_ideal_mine(
        bases, m=m, iterations=iterations, min_weight=3, max_weight=32,
        seed=seed, max_words=max_words))
    for index, base in enumerate(bases):
        words.extend(cpp_fast.cyclic_ideal_hillclimb(
            base, m=m, restarts=hill_restarts, steps=hill_steps,
            min_weight=3, max_weight=32,
            seed=seed ^ (0xA5A5A5A5 + index * 0x9E3779B9),
            max_words=max_words))
    unique = {orbit(word, m) for word in words if word}
    unique.update((orbit(parent_a, m), orbit(parent_b, m)))
    return sorted((word for word in unique if len(word) <= 32 and
                   core._word_gcd_degree(word) >= int(q)),
                  key=lambda word: (len(word), word))


def pair_row(a, b, *, m, q, indices, lineage):
    if not a or not b or len(a) + len(b) > 32:
        return None
    if common_gcd_degree(m, a, b) != q:
        return None
    key = core._affine_pair_key(m, tuple(a), tuple(b))
    a, b = tuple(key[0]), tuple(key[1])
    if len(a) + len(b) > 32 or common_gcd_degree(m, a, b) != q:
        return None
    return {
        "A": [list(a)], "B": [list(b)], "m": m, "n": 2 * m, "k": 2 * q,
        "gcd_degree": q, "branch_id": "generic_local_breakout",
        "branch_index": -1, "factor_indices": list(indices),
        "phylogenetic_arm": "exact_ideal_local_breakout",
        "inherited_q55_ancestors": [], "divisor_multiplier_symmetry": 1,
        "pairing_mode": lineage, "word_weights": [len(a), len(b)],
        "individual_gcd_degrees": [core._word_gcd_degree(a),
                                    core._word_gcd_degree(b)],
        "cycle_profile": core._cycle_profile(a, b),
        "semantic_hash": hashlib.sha256(json.dumps([list(a), list(b)],
                                                     separators=(",", ":")).encode()).hexdigest(),
        "lineage": lineage,
    }


def _killer_survives(a, b, witness, side, *, m):
    """Reject a mutation retaining a previously verified light logical."""
    # Mutator callers may provide sets or phase-grafted sequences.  Normalize
    # before matrix construction so duplicate exponents cannot abort campaign.
    a = tuple(sorted(set(int(value) % int(m) for value in a)))
    b = tuple(sorted(set(int(value) % int(m) for value in b)))
    hx, hz = build_matrices(int(m), a, b)
    if str(side).upper() == "X":
        check = _logical_check(witness, hz, hx)
    else:
        check = _logical_check(witness, hx, hz)
    return bool(check.get("ok"))


def generate(parent_a, parent_b, bank, *, m, q, indices, attempts,
             max_candidates, seed, killers=(), blocked_hashes=()):
    rng = random.Random(seed)
    rows = {}
    shell = [word for word in bank if 7 <= len(word) <= 24] or bank

    def add(a, b, lineage):
        row = pair_row(tuple(sorted(a)), tuple(sorted(b)), m=m, q=q,
                       indices=indices, lineage=lineage)
        if row:
            if row["semantic_hash"] in blocked_hashes:
                return
            if killers and any(_killer_survives(
                    row["A"][0], row["B"][0], witness, side, m=m)
                    for witness, side in killers):
                return
            rows[row["semantic_hash"]] = row

    add(parent_a, parent_b, "parent")
    # Relative cyclic phase is a genuine two-block degree of freedom.  Shifting
    # one parent block preserves its cyclic ideal membership and exposes every
    # phase class before stochastic grafts consume the candidate cap.  This is
    # especially important for rare high-<h>-weight parents: random mutation
    # otherwise repeatedly lands in the common-gcd descendant that the ideal
    # gate rejects.
    for shift in range(int(m)):
        if len(rows) >= int(max_candidates):
            break
        shifted_b = tuple((int(v) + shift) % int(m) for v in parent_b)
        add(parent_a, shifted_b, "parent_relative_phase_b")
        if len(rows) >= int(max_candidates):
            break
        shifted_a = tuple((int(v) + shift) % int(m) for v in parent_a)
        add(shifted_a, parent_b, "parent_relative_phase_a")
    low_shell = [word for word in bank if 14 <= len(word) <= 18]
    # Relative translation is a genuine GB parameter.  Enumerate it before
    # random mutations; 15+15 and 15+17 pairs are the only dense shells that
    # fit the cap, and their useful phase classes can be rare.
    for a in low_shell:
        for b in low_shell:
            if len(a) + len(b) > 32:
                continue
            for shift in range(m):
                add(a, tuple((int(v) + shift) % m for v in b),
                    "all_shift_low_shell_recombination")
                if len(rows) >= int(max_candidates):
                    break
            if len(rows) >= int(max_candidates):
                break
        if len(rows) >= int(max_candidates):
            break
    for _ in range(max(int(attempts), int(max_candidates) * 8)):
        if len(rows) >= int(max_candidates):
            break
        if rng.randrange(4):
            blocks = [set(parent_a), set(parent_b)]
            block = rng.randrange(2)
            word = rng.choice(shell)
            shift = rng.randrange(m)
            blocks[block].symmetric_difference_update((int(v) + shift) % m for v in word)
            lineage = "single_exact_ideal_xor"
            if rng.randrange(5) == 0:
                other = rng.choice(shell)
                other_block = 1 - block
                other_shift = rng.randrange(m)
                blocks[other_block].symmetric_difference_update(
                    (int(v) + other_shift) % m for v in other)
                lineage = "double_exact_ideal_xor"
            if len(blocks[0]) + len(blocks[1]) < 12:
                continue
            add(blocks[0], blocks[1], lineage)
        else:
            a = rng.choice(bank)
            b = rng.choice(bank)
            b = tuple(sorted((int(v) + rng.randrange(m)) % m for v in b))
            add(a, b, "exact_ideal_recombination")
    return list(rows.values())


def screen_task(task):
    row, trials, seed, threads, target, m, k, ideal_trials = task
    # Windows workers import this module afresh; restore the selected family
    # before constructing matrices or invoking the shared core screen.
    configure(int(m), int(k))
    return core._screen_task((row, trials, seed, threads, target, ideal_trials))


def screen(rows, *, trials, target, workers, threads, seed, m, k,
           ideal_trials=1024):
    tasks = [(row, int(trials), seed + i * 0x9E3779B9, threads, target, m, k,
              ideal_trials)
             for i, row in enumerate(rows)]
    out = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(screen_task, task) for task in tasks]
        started = time.perf_counter()
        for done, future in enumerate(as_completed(futures), 1):
            out.append(future.result())
            if done == 1 or done % 64 == 0 or done == len(futures):
                print(json.dumps({"stage": "generic_proxy", "done": done,
                                  "total": len(futures),
                                  "seconds": round(time.perf_counter() - started, 1)}),
                      flush=True)
    return out


def candidate_doc(row, *, m, k, stage):
    screen_data = row["screen"]
    dx, dz = screen_data.get("dx_upper"), screen_data.get("dz_upper")
    wx = row.get("screen_witnesses", {}).get("X")
    wz = row.get("screen_witnesses", {}).get("Z")
    if dx is None or dz is None or wx is None or wz is None:
        return None
    a, b = row["A"][0], row["B"][0]
    hx, hz = build_supports(m, a, b)
    d = min(int(dx), int(dz))
    return {
        "schema_version": "0.1", "name": f"[[{2*m},{k},d<={d}]] local GB",
        "code_type": "CSS", "n": 2 * m, "k": k,
        "checks": {"X": hx, "Z": hz},
        "distance": {"d": d,
                     "X": {"value": int(dx), "confidence": "upper_bound",
                           "witness": list(wx)},
                     "Z": {"value": int(dz), "confidence": "upper_bound",
                           "witness": list(wz)}},
        "family": "generalized-bicycle",
        "provenance": {"authors": ["stage-only autonomous research"],
                       "origin": "generic_exact_ideal_local_breakout",
                       "novelty": "unknown",
                       "model": "GPT-5 Codex + phylogenetic exact-ideal search",
                       "construction": f"Z_{m} GB factors={row['factor_indices']}; A={tuple(a)}; B={tuple(b)}",
                       "references": [], "notes": f"stage={stage}; randomized upper bound"},
        "search": {"stage": stage, "semantic_hash": row["semantic_hash"],
                   "factor_indices": row["factor_indices"],
                   "cycle_profile": row["cycle_profile"],
                   "common_gcd_ideal_probe": screen_data.get("ideal_x_probe")},
        "regulation": {"stage_only": True, "submission_sent": False,
                       "git_commit_performed": False, "official_gate_status": "not_run",
                       "board_claim_allowed": False, "distance_is_not_proven": True,
                       "literature_novelty": "unverified"},
    }


def run(args):
    started = time.perf_counter()
    parent = json.loads(Path(args.parent).read_text())
    m, k = int(parent["n"]) // 2, int(parent["k"])
    q = k // 2
    configure(m, k)
    indices = factor_indices(parent, args.factor_indices)
    a, b = pair_from_doc(parent)
    killers = []
    blocked_hashes = set()
    parent_hash = parent.get("search", {}).get("semantic_hash")
    if parent_hash:
        blocked_hashes.add(str(parent_hash))
    for receipt_path in args.killer_receipt or []:
        receipt = json.loads(Path(receipt_path).read_text())
        refutation = receipt.get("refutation") or {}
        witness = refutation.get("witness")
        if not witness:
            raise ValueError(f"killer receipt has no refutation witness: {receipt_path}")
        side = str(refutation.get("side", args.killer_side)).upper()
        if side not in {"X", "Z"}:
            raise ValueError("killer side must be X or Z")
        killers.append((tuple(int(value) for value in witness), side))
        candidate_path = receipt.get("candidate_path")
        if candidate_path and Path(candidate_path).exists():
            killed_doc = json.loads(Path(candidate_path).read_text())
            killed_hash = killed_doc.get("search", {}).get("semantic_hash")
            if killed_hash:
                blocked_hashes.add(str(killed_hash))
    bank = mine_bank(a, b, m=m, q=q, indices=indices,
                     iterations=args.mine_iterations, max_words=args.max_words,
                     hill_restarts=args.hill_restarts, hill_steps=args.hill_steps,
                     seed=args.seed)
    rows = generate(a, b, bank, m=m, q=q, indices=indices,
                    attempts=args.attempts, max_candidates=args.max_candidates,
                    seed=args.seed ^ 0xC0FFEE, killers=killers,
                    blocked_hashes=blocked_hashes)
    print(json.dumps({"stage": "generic_population", "m": m, "k": k,
                      "bank": len(bank), "candidates": len(rows),
                      "parent_weights": [len(a), len(b)]}), flush=True)
    screened = screen(rows, trials=args.proxy_trials, target=args.target,
                      workers=args.workers, threads=args.threads,
                      seed=args.seed ^ 0xD1B54A32, m=m, k=k,
                      ideal_trials=args.ideal_trials)
    screened.sort(key=lambda row: (
        int(row.get("screen", {}).get("d_upper") or -1),
        int(row.get("screen", {}).get("dx_upper") or -1),
        int(row.get("screen", {}).get("dz_upper") or -1),
        row["semantic_hash"]), reverse=True)
    survivors = [row for row in screened
                 if int(row.get("screen", {}).get("d_upper") or -1) >= args.target
                 and not row.get("screen", {}).get("stopped_early")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    final = []
    for index, row in enumerate(survivors[:args.final_count], 1):
        doc = candidate_doc(row, m=m, k=k, stage="generic_local_proxy")
        if doc is None:
            continue
        path = out / f"candidate_{index}_{2*m}_{k}_{doc['distance']['d']}.json"
        path.write_text(json.dumps(doc, indent=2) + "\n")
        structural = structural_validate(doc)
        final.append({"candidate_path": str(path.resolve()),
                      "semantic_hash": row["semantic_hash"],
                      "d_upper": doc["distance"]["d"],
                      "dx_upper": doc["distance"]["X"]["value"],
                      "dz_upper": doc["distance"]["Z"]["value"],
                      "structural": {key: value for key, value in structural.items()
                                     if key not in ("hx", "hz")}})
    report = {"schema_version": "1.0", "kind": "generic_exact_ideal_local_breakout",
              "parent": str(Path(args.parent).resolve()),
              "target": {"m": m, "n": 2*m, "k": k, "gcd_degree": q,
                          "target_d": args.target, "check_weight_cap": 32},
              "search": {"bank": len(bank), "candidates": len(rows),
                         "screened": len(screened), "survivors": len(survivors),
                         "proxy_trials_per_side": args.proxy_trials,
                         "common_gcd_ideal_trials_per_candidate": args.ideal_trials,
                         "killer_guard": bool(killers),
                         "killer_weights": [len(witness) for witness, _ in killers],
                         "blocked_semantic_hashes": len(blocked_hashes),
                         "seed": args.seed, "seconds_wall": time.perf_counter() - started},
              "top": screened[:128], "final": final,
              "regulation": {"stage_only": True, "submission_sent": False,
                             "git_commit_performed": False,
                             "distance_is_randomized_upper_bound": True,
                             "official_gate_run": False, "board_claim_allowed": False}}
    out.joinpath("campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--factor-indices")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--mine-iterations", type=int, default=500_000)
    parser.add_argument("--max-words", type=int, default=2048)
    parser.add_argument("--hill-restarts", type=int, default=256)
    parser.add_argument("--hill-steps", type=int, default=512)
    parser.add_argument("--attempts", type=int, default=100_000)
    parser.add_argument("--max-candidates", type=int, default=2048)
    parser.add_argument("--proxy-trials", type=int, default=2_000)
    parser.add_argument("--target", type=int, default=77)
    parser.add_argument("--final-count", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--ideal-trials", type=int, default=1024,
                        help="pre-CSS common-gcd ideal trials per candidate")
    parser.add_argument("--killer-receipt", type=Path, action="append",
                        help="verified refutation receipt; repeat for multiple killers")
    parser.add_argument("--killer-side", choices=("X", "Z"), default="X")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"out": str(args.out), "search": report["search"],
                      "final": report["final"]}, indent=2))


if __name__ == "__main__":
    main()
