"""Divisor-branch escape campaign for the saturated Z_341 GB pocket.

The prior search exhausts low-weight words in one degree-91 ideal.  This
campaign changes the algebraic branch: factor x**341+1 over GF(2), sample
other degree-91 divisors g, mine low-weight multiples a=g*u and b=g*v, then
apply the same native RIS/official gates.  It is intentionally stage-only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from math import comb
from pathlib import Path
import random
import subprocess
import sys
import time

import cpp_fast
from gb_aggressive_campaign import M, ROOTS, TARGET_D, _candidate_doc
from generalized_bicycle import build_matrices, common_gcd_degree
from gf2_factor import degree, factor_xm_plus_one, gcd, mod, mul
from hyper_validator import structural_validate


CHECK_CAP = 32
TARGET_K = 182
TARGET_GCD_DEGREE = TARGET_K // 2


def _support(poly: int) -> tuple[int, ...]:
    return tuple(i for i in range(M) if (int(poly) >> i) & 1)


def _poly_from_support(values) -> int:
    out = 0
    for value in values:
        out ^= 1 << (int(value) % M)
    return out


def _known_divisor() -> int:
    modulus = (1 << M) | 1
    a = _poly_from_support(ROOTS["published75"][0])
    b = _poly_from_support(ROOTS["published75"][1])
    return gcd(gcd(a, b), modulus)


def factor_catalog(m: int = M) -> list[dict]:
    factors = factor_xm_plus_one(int(m))
    return [{"index": i, "degree": degree(poly), "polynomial": poly,
             "support": list(_support(poly))}
            for i, poly in enumerate(factors)]


def _composition_options(catalog: list[dict], target_degree: int):
    groups = {}
    for item in catalog:
        groups.setdefault(int(item["degree"]), []).append(item)
    degrees = sorted(groups)
    options = []

    def walk(pos, remaining, counts):
        if pos == len(degrees):
            if remaining == 0:
                ways = 1
                for d, count in zip(degrees, counts):
                    ways *= comb(len(groups[d]), count)
                options.append((tuple(counts), ways))
            return
        d = degrees[pos]
        for count in range(min(len(groups[d]), remaining // d) + 1):
            walk(pos + 1, remaining - count * d, counts + [count])

    walk(0, int(target_degree), [])
    return groups, degrees, options


def sample_divisors(catalog: list[dict], *, target_degree=TARGET_GCD_DEGREE,
                    count=16, seed=0, exclude=None) -> list[dict]:
    """Sample exact-degree divisor branches, weighted by combination count."""
    groups, degrees, options = _composition_options(catalog, target_degree)
    if not options:
        raise ValueError("target degree is not represented by modulus factors")
    rng = random.Random(int(seed))
    excluded = int(exclude or 0)
    total_weight = sum(ways for _, ways in options)
    branches = {}
    attempts = 0
    max_attempts = max(100, int(count) * 80)
    while len(branches) < int(count) and attempts < max_attempts:
        attempts += 1
        ticket = rng.randrange(total_weight)
        chosen_counts = options[-1][0]
        for counts, ways in options:
            if ticket < ways:
                chosen_counts = counts
                break
            ticket -= ways
        selected = []
        for d, take in zip(degrees, chosen_counts):
            selected.extend(rng.sample(groups[d], take))
        mask = 0
        poly = 1
        for item in selected:
            mask |= 1 << int(item["index"])
            poly = mul(poly, int(item["polynomial"]))
        if mask == excluded or mask in branches:
            continue
        branches[mask] = {
            "branch_id": f"deg{target_degree}_factors_{mask:x}",
            "factor_indices": [int(item["index"]) for item in selected],
            "degree": int(degree(poly)), "polynomial": int(poly),
            "support": list(_support(poly)),
            "polynomial_weight": int(poly.bit_count()),
            "factor_count": len(selected),
        }
    return sorted(branches.values(), key=lambda item: (
        item["polynomial_weight"], item["branch_id"]))


def structured_bivariate_divisors(catalog: list[dict], *, exclude=0) -> list[dict]:
    """Return sparse degree-91 branches predicted by the breakout tree.

    ``341 = 11*31``.  In the isomorphic group algebra
    ``GF(2)[C_11 x C_31]``, use

        g_j = (1+x)(1+y) f_j(y),

    where ``f_j`` ranges over the degree-five factors of ``y**31+1``.
    The active tensor component has dimension ``10*25=250``; therefore the
    common kernel has dimension ``341-250=91`` and the CSS code has k=182.
    These are deliberately hand-shaped sparse divisor branches.  Random
    divisor sampling almost never reaches them, despite their useful
    generator weight (8 or 12).

    The CRT map sends ``(x_exp,y_exp)`` to the unique cyclic exponent e with
    ``e mod 11=x_exp`` and ``e mod 31=y_exp``.  It preserves group-algebra
    multiplication, so the native cyclic miner can be reused without a new
    dense torus kernel.
    """
    factor31 = factor_xm_plus_one(31)
    degree_five = [poly for poly in factor31 if degree(poly) == 5]
    crt = {(e % 11, e % 31): e for e in range(M)}
    one = 0b11
    modulus = (1 << M) | 1
    excluded = int(exclude or 0)
    out = []
    for family_index, f5 in enumerate(degree_five):
        gy = mul(one, int(f5))
        supports = [crt[(x, y)] for x in range(11) for y in range(31)
                    if ((one >> x) & 1) and ((gy >> y) & 1)]
        poly = _poly_from_support(supports)
        divisor = gcd(poly, modulus)
        if degree(divisor) != TARGET_GCD_DEGREE:
            raise ArithmeticError("structured bivariate branch has wrong gcd degree")
        factor_indices = [int(item["index"]) for item in catalog
                           if mod(divisor, int(item["polynomial"])) == 0]
        mask = sum(1 << index for index in factor_indices)
        if mask == excluded:
            continue
        out.append({
            "branch_id": f"bivariate_C11xC31_yfactor_{family_index}",
            "family": "bivariate_bicycle_structured_escape",
            "factor_indices": factor_indices,
            "degree": TARGET_GCD_DEGREE,
            # Keep the sparse torus generator for mining.  ``divisor`` is the
            # canonical cyclic gcd; its expanded representative is much
            # denser (40--48 bits) although it generates the same ideal.
            "polynomial": int(poly),
            "support": list(_support(poly)),
            "polynomial_weight": int(poly.bit_count()),
            "factor_count": len(factor_indices),
            "coordinate_model": "C11_x_C31 via CRT",
            "escape_reason": "sparse tensor-product divisor branch",
        })
    return out


def _orbit_key(word):
    values = tuple(sorted(int(x) % M for x in word))
    rotations = []
    for shift in range(M):
        rotations.append(tuple(sorted((x + shift) % M for x in values)))
    return min(rotations)


def _pair_key(a, b):
    # Integer rotations avoid constructing 2*M temporary tuples per pair.
    # Pair canonicalization is telemetry/dedup only; it never changes a code.
    ring = (1 << M) - 1
    def mask(word):
        value = 0
        for item in word:
            value |= 1 << (int(item) % M)
        return value
    def rotate(value, shift):
        shift %= M
        if not shift:
            return value & ring
        return ((value << shift) | (value >> (M - shift))) & ring
    left, right = mask(a), mask(b)
    best = None
    for shift in range(M):
        pair = (rotate(left, shift), rotate(right, shift))
        if best is None or pair < best:
            best = pair
    def support(value):
        out = []
        while value:
            bit = value & -value
            out.append(bit.bit_length() - 1)
            value ^= bit
        return tuple(out)
    return support(best[0]), support(best[1])


def mine_branch(branch: dict, *, iterations=25_000, max_words=512,
                min_word=3, max_word=31, seed=0, hill_restarts=64,
                hill_steps=2048) -> list[tuple[int, ...]]:
    words = cpp_fast.cyclic_ideal_mine(
        [branch["support"]], m=M, iterations=int(iterations),
        min_weight=int(min_word), max_weight=int(max_word), seed=int(seed),
        max_words=int(max_words))
    # New divisor branches often have no sparse generator.  Add words found
    # by native coordinate descent over the same cyclic ideal.
    words.extend(cpp_fast.cyclic_ideal_hillclimb(
        branch["support"], m=M, restarts=int(hill_restarts),
        steps=int(hill_steps), min_weight=int(min_word),
        max_weight=int(max_word), seed=int(seed) ^ 0xA5A5A5A5,
        max_words=int(max_words)))
    # The archive's strongest Z_341 lineage sits on a balanced 16+16 shell.
    # Broad mining tends to fill the pair budget with 8+24 / 12+20 words,
    # which are fast to find but collapse to tiny logicals.  Always reserve a
    # second native pass for the balanced shell when it is inside the caller's
    # requested range.
    balanced = CHECK_CAP // 2
    if int(min_word) <= balanced <= int(max_word):
        words.extend(cpp_fast.cyclic_ideal_mine(
            [branch["support"]], m=M, iterations=int(iterations),
            min_weight=balanced, max_weight=balanced,
            seed=int(seed) ^ 0x13579BDF, max_words=int(max_words)))
    return sorted(set(tuple(int(x) for x in word) for word in words),
                  key=lambda word: (len(word), word))


def pair_branch_words(branch: dict, words: list[tuple[int, ...]], *, cap=4096):
    # Keep multiple orbit classes. Pairing only one repeated class recreates
    # the old collapse basin, while cross-orbit pairs are the useful escape.
    def collect(source_words, raw_cap):
        by_orbit = {}
        for word in source_words:
            by_orbit.setdefault(_orbit_key(word), []).append(word)
        for values in by_orbit.values():
            values.sort(key=lambda word: (len(word), word), reverse=True)
        orbits = sorted(by_orbit, key=lambda orbit: (
            max(len(word) for word in by_orbit[orbit]),
            min(len(word) for word in by_orbit[orbit]), orbit), reverse=True)
        candidates = {}
        for left_index, left_orbit in enumerate(orbits):
            for right_index, right_orbit in enumerate(orbits):
                if left_index == right_index:
                    continue
                for a in by_orbit[left_orbit]:
                    for b in by_orbit[right_orbit]:
                        if len(a) + len(b) > CHECK_CAP:
                            continue
                        if common_gcd_degree(M, a, b) != TARGET_GCD_DEGREE:
                            continue
                        key = _pair_key(a, b)
                        candidates.setdefault(key, {
                            "branch_id": branch["branch_id"],
                            "lineage": "divisor_branch_cross_orbit",
                            "divisor_factor_indices": branch["factor_indices"],
                            "divisor_polynomial_weight": branch["polynomial_weight"],
                            "word_weights": [len(a), len(b)],
                        })
                        if len(candidates) >= raw_cap:
                            break
                    if len(candidates) >= raw_cap:
                        break
                if len(candidates) >= raw_cap:
                    break
            if len(candidates) >= raw_cap:
                break
        return [{"A": [list(key[0])], "B": [list(key[1])], **meta}
                for key, meta in candidates.items()]

    cap = max(0, int(cap))
    raw_cap = max(cap, cap * 16)
    # Dedicated balanced pass prevents abundant low-weight words from
    # starving the 16+16 shell before the broad fallback gets a turn.
    balanced = collect([word for word in words if len(word) == CHECK_CAP // 2],
                       raw_cap)
    rows_by_key = {(tuple(row["A"][0]), tuple(row["B"][0])): row
                   for row in balanced}
    if len(rows_by_key) < cap:
        for row in collect(words, raw_cap):
            key = (tuple(row["A"][0]), tuple(row["B"][0]))
            rows_by_key.setdefault(key, row)
            if len(rows_by_key) >= raw_cap:
                break
    rows = list(rows_by_key.values())
    rows.sort(key=lambda row: (
        min(row["word_weights"]),
        -abs(row["word_weights"][0] - row["word_weights"][1]),
        sum(row["word_weights"]),
        tuple(row["A"][0]), tuple(row["B"][0])), reverse=True)
    return rows[:max(0, int(cap))]


def _mine_task(task):
    (branch, iterations, max_words, min_word, max_word, hill_restarts,
     hill_steps, pair_cap, seed) = task
    words = mine_branch(branch, iterations=iterations, max_words=max_words,
                        min_word=min_word, max_word=max_word, seed=seed,
                        hill_restarts=hill_restarts, hill_steps=hill_steps)
    candidates = pair_branch_words(branch, words, cap=int(pair_cap))
    return branch, words, candidates


def _screen_task(task):
    row, seed, trials, threads = task
    a, b = tuple(row["A"][0]), tuple(row["B"][0])
    hx, hz = build_matrices(M, a, b)
    result = cpp_fast.css_ris_parallel(
        hx, hz, trials=int(trials), seed=int(seed), pair_depth=24,
        target=TARGET_D, stop_on_target=True, threads=int(threads))
    return {**row, "semantic_hash": hashlib.sha256(
        hx.tobytes() + hz.tobytes()).hexdigest(),
        "d_proxy": result.get("d_upper"),
        "dx_proxy": result["x"].get("best_weight"),
        "dz_proxy": result["z"].get("best_weight"),
        "proxy_trials_per_side": int(trials),
        "proxy_stopped_early": bool(result["x"].get("stopped_early") or
                                     result["z"].get("stopped_early")),
        "proxy_witness_x": result["x"].get("witness"),
        "proxy_witness_z": result["z"].get("witness")}


def _deep_task(task):
    row, seed, trials, threads = task
    a, b = tuple(row["A"][0]), tuple(row["B"][0])
    hx, hz = build_matrices(M, a, b)
    result = cpp_fast.css_ris_parallel(
        hx, hz, trials=int(trials), seed=int(seed), pair_depth=24,
        target=TARGET_D, stop_on_target=True, threads=int(threads))
    return {**row, "d": result.get("d_upper"),
            "dx": result["x"].get("best_weight"),
            "dz": result["z"].get("best_weight"),
            "deep_trials_per_side": int(trials),
            "deep_stopped_early": bool(result["x"].get("stopped_early") or
                                        result["z"].get("stopped_early")),
            "witness_x": result["x"].get("witness"),
            "witness_z": result["z"].get("witness")}


def _parallel(fn, rows, *, seed, trials, workers, threads, label):
    if not rows:
        return []
    tasks = [(row, int(seed) + i * 0x9E3779B9, int(trials), int(threads))
             for i, row in enumerate(rows)]
    started = time.perf_counter()
    out = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(fn, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            out.append(future.result())
            if done == 1 or done % 64 == 0 or done == len(futures):
                print(json.dumps({"stage": label, "done": done,
                                  "total": len(futures),
                                  "seconds": round(time.perf_counter() - started, 1)}),
                      flush=True)
    return out


def run(args):
    started = time.perf_counter()
    info = cpp_fast.accelerator_info()
    if not info["available"]:
        raise RuntimeError(info)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    catalog = factor_catalog(M)
    known = _known_divisor()
    known_mask = 0
    for item in catalog:
        if mod(known, int(item["polynomial"])) == 0:
            known_mask |= 1 << int(item["index"])
    # Tree-derived sparse branches get guaranteed coverage.  Fill the rest
    # with weighted random exact-degree divisors so the campaign still has
    # broad algebraic exploration.
    structured = structured_bivariate_divisors(catalog, exclude=known_mask)
    random_branches = sample_divisors(
        catalog, count=max(0, int(args.branches) - len(structured)),
        seed=args.seed, exclude=known_mask)
    branches = structured + random_branches
    dedup_branches = {}
    for branch in branches:
        dedup_branches.setdefault(tuple(branch["factor_indices"]), branch)
    branches = list(dedup_branches.values())[:max(0, int(args.branches))]
    (out / "factor_catalog.json").write_text(json.dumps({
        "m": M, "modulus": (1 << M) | 1,
        "factors": [{key: value for key, value in item.items()
                     if key != "polynomial"} for item in catalog],
        "known_divisor_factor_mask": known_mask,
        "sampled_branches": [{key: value for key, value in branch.items()
                              if key != "polynomial"} for branch in branches],
    }, indent=2) + "\n")
    tasks = [(branch, args.mine_iterations, args.max_words, args.min_word,
              args.max_word, args.hill_restarts, args.hill_steps, args.pair_cap,
              args.seed + i * 7919)
             for i, branch in enumerate(branches)]
    mined = []
    candidates = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(_mine_task, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            branch, words, branch_candidates = future.result()
            mined.append({"branch": branch, "word_count": len(words),
                          "word_weight_histogram": {
                              str(weight): sum(len(word) == weight for word in words)
                              for weight in sorted({len(word) for word in words})},
                          "candidate_count": len(branch_candidates)})
            candidates.extend(branch_candidates)
            print(json.dumps({"stage": "mine", "done": done,
                              "total": len(futures), "branch": branch["branch_id"],
                              "words": len(words), "candidates": len(branch_candidates)}),
                  flush=True)
    dedup = {}
    for row in candidates:
        key = (tuple(row["A"][0]), tuple(row["B"][0]))
        dedup.setdefault(key, row)
    candidates = list(dedup.values())
    screened = _parallel(_screen_task, candidates, seed=args.seed + 100_003,
                          trials=args.proxy_trials, workers=args.workers,
                          threads=args.threads, label="proxy")
    screened.sort(key=lambda row: (int(row.get("d_proxy") or -1),
                                   int(row.get("dx_proxy") or -1),
                                   str(row.get("semantic_hash", ""))),
                  reverse=True)
    beam = screened[:min(len(screened), int(args.beam_size))]
    deep = _parallel(_deep_task, beam, seed=args.seed + 200_003,
                     trials=args.deep_trials, workers=args.workers,
                     threads=args.threads, label="deep")
    deep.sort(key=lambda row: (int(row.get("d") or -1),
                               int(row.get("d_proxy") or -1),
                               str(row.get("semantic_hash", ""))), reverse=True)
    final = []
    for index, row in enumerate(deep[:max(0, int(args.final_count))]):
        a, b = tuple(row["A"][0]), tuple(row["B"][0])
        hx, hz = build_matrices(M, a, b)
        result = cpp_fast.css_ris_parallel(
            hx, hz, trials=int(args.final_trials),
            seed=int(args.seed) + 900_003 + index, pair_depth=24,
            target=TARGET_D, stop_on_target=True, threads=0)
        dx, dz = result["x"].get("best_weight"), result["z"].get("best_weight")
        if dx is None or dz is None:
            continue
        candidate = _candidate_doc(row, dx, dz, result["x"]["witness"],
                                   result["z"]["witness"], result.get("mode"))
        candidate["provenance"]["origin"] = "divisor_branch_escape_campaign"
        candidate["provenance"]["branch_id"] = row["branch_id"]
        candidate["provenance"]["divisor_factor_indices"] = row["divisor_factor_indices"]
        candidate["provenance"]["notes"] += (
            " Algebraic branch sampled from a different degree-91 divisor; "
            "stage-only until independent/official validation.")
        structural = structural_validate(candidate)
        candidate_path = out / f"candidate_{index}_{min(int(dx), int(dz))}.json"
        candidate_path.write_text(json.dumps(candidate, indent=2) + "\n")
        official = None
        validator = Path(args.challenge_root) / "verify" / "validate_candidate.py"
        if validator.exists():
            proc = subprocess.run([sys.executable, str(validator),
                                   str(candidate_path.resolve())],
                                  cwd=Path(args.challenge_root),
                                  capture_output=True, text=True)
            try:
                official = json.loads(proc.stdout) if proc.stdout.strip() else {
                    "returncode": proc.returncode, "stderr": proc.stderr}
            except json.JSONDecodeError:
                official = {"returncode": proc.returncode, "stdout": proc.stdout,
                            "stderr": proc.stderr}
        final.append({"candidate_path": str(candidate_path.resolve()),
                      "branch_id": row["branch_id"], "dx": int(dx), "dz": int(dz),
                      "d": min(int(dx), int(dz)),
                      "trials_per_side": int(args.final_trials),
                      "stopped_early": bool(result["x"].get("stopped_early") or
                                             result["z"].get("stopped_early")),
                      "structural": structural, "official": official})
        (out / "final_progress.json").write_text(json.dumps(final, indent=2) + "\n")
    report = {
        "schema_version": "1.0", "kind": "gb_divisor_branch_escape_campaign",
        "submission_sent": False, "git_commit_performed": False,
        "target": {"m": M, "n": 2 * M, "k": TARGET_K,
                    "gcd_degree": TARGET_GCD_DEGREE, "check_weight_cap": CHECK_CAP,
                    "frontier_threshold": TARGET_D},
        "escape_mechanism": {
            "trigger": "one degree-91 ideal saturated at four cyclic word orbits",
            "operator": ("tree-guided branch jump: structured C11xC31 sparse divisor "
                         "shell first, then random exact-degree divisors"),
            "lift_rule": "mine low-weight a=g*u and b=g*v, require gcd degree exactly 91",
            "pair_rule": ("balanced 16+16 shell first; cross-orbit only; "
                          "wt(a)+wt(b)<=32"),
            "structured_branch_count": len(structured),
            "balanced_shell_weight": CHECK_CAP // 2,
            "literature_basis": ["divisor-driven cyclic bicycle search",
                                 "ansatz branch expansion/cross-factoring"],
        },
        "factorization": {"factor_count": len(catalog),
                           "factor_degrees": [item["degree"] for item in catalog],
                           "known_divisor_factor_mask": known_mask},
        "branches": mined,
        "population": {"candidate_count": len(candidates),
                       "screened_count": len(screened), "beam_count": len(beam)},
        "proxy_top": screened[:64], "deep_top": deep[:64], "final": final,
        "accelerator": info, "seconds_wall": time.perf_counter() - started,
        "regulation": {"stage_only": True, "submission_sent": False,
                        "git_commit_performed": False,
                        "distance_is_randomized_upper_bound": True,
                        "official_verifier_used_if_available": True},
    }
    (out / "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "top.json").write_text(json.dumps(final, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("results/gb_divisor_escape_01"))
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--branches", type=int, default=12)
    parser.add_argument("--mine-iterations", type=int, default=50_000)
    parser.add_argument("--max-words", type=int, default=512)
    parser.add_argument("--min-word", type=int, default=3)
    parser.add_argument("--max-word", type=int, default=31)
    parser.add_argument("--hill-restarts", type=int, default=64)
    parser.add_argument("--hill-steps", type=int, default=2048)
    parser.add_argument("--pair-cap", type=int, default=512,
                        help="per-branch cross-orbit candidate cap before RIS")
    parser.add_argument("--proxy-trials", type=int, default=64)
    parser.add_argument("--beam-size", type=int, default=16)
    parser.add_argument("--deep-trials", type=int, default=512)
    parser.add_argument("--final-count", type=int, default=2)
    parser.add_argument("--final-trials", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--challenge-root", type=Path,
                        default=Path("challenge_data"))
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"out": str(args.out), "population": report["population"],
                      "final": report["final"],
                      "seconds_wall": round(report["seconds_wall"], 1)}, indent=2))


if __name__ == "__main__":
    main()
