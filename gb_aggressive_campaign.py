from __future__ import annotations

"""Aggressive mutation campaign for the Z_341 generalized-bicycle frontier.

The cheap predictor is deliberately proof-neutral: a small native CSS-RIS
sample plus a diversity-aware phylogenetic rank.  Only survivors are deepened;
the final file is fully rechecked structurally and with a fresh large RIS run.
No network submission or git mutation is performed.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import cpp_fast
from generalized_bicycle import (
    REFERENCE_A, REFERENCE_B, build_matrices, build_supports,
    common_gcd_degree, normalize_code,
)
from hyper_validator import structural_validate
from phylogenetic_search import select_phylogenetic_elites, upgma


M = 341
TARGET_D = 77
ROOTS = {
    "published75": (
        (37, 55, 71, 102, 159, 169, 172, 182, 190, 244, 246, 265, 268, 272, 290, 301),
        (0, 11, 70, 81, 114, 153, 217, 227, 228, 235, 268, 280, 311, 320, 323, 339),
    ),
    "competitor76": (
        (0, 1, 36, 58, 115, 135, 141, 161, 177, 212, 248, 280, 285, 289, 327, 333),
        (0, 1, 12, 14, 23, 67, 76, 84, 128, 133, 181, 199, 270, 305, 310, 314),
    ),
    "local77_miss": (
        (67, 85, 101, 132, 189, 199, 202, 212, 220, 274, 276, 295, 298, 302, 320, 331),
        (4, 13, 16, 32, 34, 45, 104, 115, 148, 187, 251, 261, 262, 269, 302, 314),
    ),
}


def _shift(word, amount):
    return tuple(sorted((int(x) + int(amount)) % M for x in word))


@lru_cache(maxsize=500_000)
def _orbit_canonical_cached(word: tuple[int, ...]) -> tuple[int, ...]:
    # A lexicographically minimal cyclic translate must start at zero; only
    # shifts anchored at an existing support element can win.  O(weight^2)
    # instead of O(M*weight) per mutation.
    return min(tuple(sorted((value - anchor) % M for value in word))
               for anchor in word)


def orbit_canonical(word):
    values = tuple(sorted(int(x) % M for x in word))
    if not values:
        return ()
    return _orbit_canonical_cached(values)


def pair_canonical(a, b):
    """Quotient simultaneous cyclic translations for deduplication."""
    a = tuple(sorted(int(x) % M for x in a))
    b = tuple(sorted(int(x) % M for x in b))
    if not a or not b:
        return a, b
    # Same anchor argument as orbit_canonical; only |a| shifts can minimize
    # the first tuple's leading coordinate.
    return min((tuple(sorted((value - anchor) % M for value in a)),
                tuple(sorted((value - anchor) % M for value in b)))
               for anchor in a)


def _mutate_word(word, rng, radius):
    values = list(sorted(int(x) % M for x in word))
    occupied = set(values)
    for _ in range(max(1, int(radius))):
        index = rng.randrange(len(values))
        old = values[index]
        new = rng.randrange(M)
        while new in occupied:
            new = rng.randrange(M)
        occupied.remove(old)
        occupied.add(new)
        values[index] = new
    return tuple(sorted(values))


def _pool_from_ideal(iterations, max_words, seed):
    bases = [*ROOTS.values()]
    flat = [support for pair in bases for support in pair]
    words = cpp_fast.cyclic_ideal_mine(
        flat, m=M, iterations=int(iterations), min_weight=16,
        max_weight=16, seed=int(seed), max_words=int(max_words))
    return sorted(set(tuple(word) for word in words))


def _make_candidates(pool, mutation_draws, mutation_radius, seed):
    """Build exact-ideal pairings plus randomized gcd-preserving mutations."""
    word_orbits = {word: orbit_canonical(word) for word in pool}
    by_orbit = {}
    for word in pool:
        by_orbit.setdefault(word_orbits[word], []).append(word)
    orbit_reps = sorted(by_orbit)
    candidates = {}
    for a in orbit_reps:
        for b in pool:
            # Same-orbit pairings are a known low-distance basin; keep the
            # predictor budget for cross-orbit recombinations.
            if word_orbits[b] == a:
                continue
            # A is already fixed to one orbit representative.  Keeping the
            # relative B phase gives one representative per simultaneous
            # translation class; no 341-shift canonicalization is needed here.
            key = (a, b)
            candidates.setdefault(key, {"lineage": "ideal_cross_orbit"})

    rng = random.Random(int(seed))
    root_pairs = list(ROOTS.values())
    # Mutate both halves around every known lineage.  The gcd test is cheap
    # compared with RIS and ensures k=182 before any distance work.
    for root_index, (base_a, base_b) in enumerate(root_pairs):
        for draw in range(max(0, int(mutation_draws))):
            radius = int(mutation_radius)
            if radius > 1 and (draw & 3) == 0:
                radius += 1
            a = _mutate_word(base_a, rng, radius)
            b = _mutate_word(base_b, rng, radius)
            if common_gcd_degree(M, a, b) != 91:
                continue
            key = pair_canonical(a, b)
            candidates.setdefault(key, {
                "lineage": f"joint_mutation_root_{root_index}",
                "mutation_radius": radius,
            })
    out = []
    for (a, b), meta in candidates.items():
        out.append({"A": [list(a)], "B": [list(b)], **meta})
    # Restore the required two support blocks after compact pair keys.
    for row in out:
        row["A"] = [list(row["A"][0])]
        row["B"] = [list(row["B"][0])]
    return out


def _task_screen(task):
    row, seed, trials, threads, repeats = task
    a = tuple(row["A"][0]); b = tuple(row["B"][0])
    hx, hz = build_matrices(M, a, b)
    passes = []
    for repeat in range(max(1, int(repeats))):
        result = cpp_fast.css_ris_parallel(
            hx, hz, trials=int(trials), seed=int(seed) + repeat * 0x9E3779B9,
            pair_depth=24, target=TARGET_D, stop_on_target=True, threads=int(threads))
        passes.append(result)
    best_result = min(passes, key=lambda item: item.get("d_upper") or 10**9)
    dx = best_result["x"].get("best_weight")
    dz = best_result["z"].get("best_weight")
    d = best_result.get("d_upper")
    digest = hashlib.sha256(hx.tobytes() + hz.tobytes()).hexdigest()
    return {
        **row, "semantic_hash": digest, "n": 682, "k": 182,
        "d_proxy": None if d is None else int(d),
        "dx_proxy": None if dx is None else int(dx),
        "dz_proxy": None if dz is None else int(dz),
        "proxy_trials_per_side": int(trials),
        "proxy_repeats": len(passes),
        "proxy_repeat_scores": [item.get("d_upper") for item in passes],
        "proxy_mode": best_result.get("mode"),
        "screen": {"d_upper": None if d is None else int(d)},
    }


def _task_deep(task):
    row, seed, trials, threads, _repeats = task
    a = tuple(row["A"][0]); b = tuple(row["B"][0])
    hx, hz = build_matrices(M, a, b)
    result = cpp_fast.css_ris_parallel(
        hx, hz, trials=int(trials), seed=int(seed), pair_depth=24,
        target=TARGET_D, stop_on_target=True, threads=int(threads))
    return {
        **row,
        "d_deep": result.get("d_upper"),
        "dx_deep": result["x"].get("best_weight"),
        "dz_deep": result["z"].get("best_weight"),
        "deep_trials_per_side": int(trials),
        "deep_mode": result.get("mode"),
        "deep_witness_x": result["x"].get("witness"),
        "deep_witness_z": result["z"].get("witness"),
    }


def _parallel_map(fn, rows, *, seed, trials, workers, threads, label, repeats=1):
    tasks = [(row, int(seed) + i * 7919, int(trials), int(threads), int(repeats))
             for i, row in enumerate(rows)]
    out = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futures = [pool.submit(fn, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            out.append(future.result())
            if i == 1 or i % 100 == 0 or i == len(futures):
                print(json.dumps({"stage": label, "done": i,
                                  "total": len(futures),
                                  "seconds": round(time.perf_counter() - started, 1)}),
                      flush=True)
    return out


def _quality(row, field):
    value = row.get(field)
    return -1 if value is None else int(value)


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _candidate_doc(row, dx, dz, wx, wz, source):
    code = normalize_code(M, row["A"][0], row["B"][0])
    hx, hz = build_supports(M, code.a, code.b)
    d = min(int(dx), int(dz))
    return {
        "schema_version": "0.1",
        "name": f"[[{code.n},{code.k},d<={d}]] generalized bicycle Z_341 mutation",
        "code_type": "CSS", "n": code.n, "k": code.k,
        "checks": {"X": hx, "Z": hz},
        "distance": {"d": d,
                     "X": {"value": int(dx), "confidence": "upper_bound",
                           "witness": list(wx)},
                     "Z": {"value": int(dz), "confidence": "upper_bound",
                           "witness": list(wz)}},
        "provenance": {
            "authors": ["stage-only autonomous research"],
            "origin": "local_campaign", "novelty": "unknown",
            "construction": f"Z_341 generalized bicycle; a={list(code.a)}, b={list(code.b)}, gcd degree={code.gcd_degree}",
            "references": [], "model": "GPT-5 Codex",
            "notes": ("Distance is a witness-backed upper bound. "
                       f"Selected by native {source}; submission_sent=false."),
        },
        "family": "generalized-bicycle",
        "regulation": {"stage_only": True, "submission_sent": False,
                        "official_gate_status": "not_run",
                        "board_claim_allowed": False,
                        "distance_is_not_proven": True,
                        "literature_novelty": "unverified"},
    }


def run(args):
    info = cpp_fast.accelerator_info()
    if not info["available"]:
        raise RuntimeError(info)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pool = _pool_from_ideal(args.miner_iterations, args.max_words, args.seed)
    candidates = _make_candidates(pool, args.mutation_draws,
                                  args.mutation_radius, args.seed + 11)
    print(json.dumps({"stage": "population", "ideal_words": len(pool),
                      "candidates": len(candidates),
                      "cpp": info["library"]}), flush=True)
    screened = _parallel_map(_task_screen, candidates, seed=args.seed + 101,
                             trials=args.proxy_trials, workers=args.workers,
                             threads=args.threads, label="proxy",
                             repeats=args.proxy_repeats)
    screened.sort(key=lambda r: (_quality(r, "d_proxy"), r["semantic_hash"]),
                  reverse=True)

    # Spend repeated medium-depth shots only on a manageable beam.  Keep a
    # deterministic exploration slice so the beam does not become a single
    # local lineage even when the cheap predictor is noisy.
    for row in screened:
        row["cheap_proxy_d"] = row.get("d_proxy")
    beam_size = min(max(1, int(args.beam_size)), len(screened))
    beam = list(screened[:beam_size])
    if beam_size < len(screened):
        stride = max(1, len(screened) // max(1, beam_size // 4))
        seen = {row["semantic_hash"] for row in beam}
        for row in screened[::stride]:
            if row["semantic_hash"] not in seen:
                beam.append(row); seen.add(row["semantic_hash"])
    beam = beam[:min(len(screened), beam_size + max(1, beam_size // 4))]
    beam = _parallel_map(_task_screen, beam, seed=args.seed + 202,
                         trials=args.beam_trials, workers=args.workers,
                         threads=args.threads, label="beam",
                         repeats=args.beam_repeats)
    beam.sort(key=lambda r: (_quality(r, "d_proxy"), r["semantic_hash"]),
              reverse=True)

    # Direct genome distance is used only to diversify the parent beam.  It
    # never substitutes for the proxy or a verifier witness.
    selected, phylo = select_phylogenetic_elites(
        beam, min(args.elite_count, len(beam)), pool_factor=8,
        exploitation_fraction=args.exploitation_fraction)
    deep = _parallel_map(_task_deep, selected, seed=args.seed + 303,
                         trials=args.deep_trials, workers=args.workers,
                         threads=args.threads, label="deep")
    deep.sort(key=lambda r: (_quality(r, "d_deep"), _quality(r, "d_proxy"),
                             r["semantic_hash"]), reverse=True)

    # Checkpoint the cheap and medium-cost work before any multi-million-shot
    # run.  A killed long run must never erase the useful search population.
    prefinal = {"submission_sent": False, "git_commit_performed": False,
                "predictor": {"method": "native CSS-RIS proxy + direct-genome-diversified phylogenetic beam",
                              "proxy_trials_per_side": int(args.proxy_trials),
                              "deep_trials_per_side": int(args.deep_trials),
                              "proof_status": "ranking signal only"},
                "population": {"ideal_words": len(pool),
                               "candidate_count": len(candidates),
                               "beam_count": len(beam)},
                "phylogeny": phylo, "proxy_top": screened[:64],
                "beam_top": beam[:64],
                "deep_top": deep[:64],
                "seconds_wall": time.perf_counter() - started}
    (out / "prefinal.json").write_text(json.dumps(prefinal, indent=2, default=_json_default) + "\n")

    final = []
    for index, row in enumerate(deep[:max(1, int(args.final_candidates))]):
        # Full runs are intentionally sequential per candidate, with all
        # native threads available, to make the final evidence reproducible.
        a = tuple(row["A"][0]); b = tuple(row["B"][0])
        hx, hz = build_matrices(M, a, b)
        final_seed = int(args.seed) + 900000 + index
        # Gate cheaply first.  A lighter witness ends the candidate's path;
        # only a survivor receives the full requested budget.
        gate = cpp_fast.css_ris_parallel(
            hx, hz, trials=int(args.gate_trials), seed=final_seed,
            pair_depth=24, target=TARGET_D, stop_on_target=True, threads=0)
        if gate.get("d_upper") is not None and gate["d_upper"] < TARGET_D:
            result = gate
            final_stage = "gate_refuted"
        else:
            result = cpp_fast.css_ris_parallel(
                hx, hz, trials=int(args.final_trials), seed=final_seed + 0x100000,
                pair_depth=24, target=TARGET_D, stop_on_target=True, threads=0)
            final_stage = "full_survivor"
        dx, dz = result["x"].get("best_weight"), result["z"].get("best_weight")
        if dx is None or dz is None:
            continue
        candidate = _candidate_doc(row, dx, dz,
                                   result["x"]["witness"], result["z"]["witness"],
                                   result.get("mode"))
        structural = structural_validate(candidate)
        official = None
        validator = Path(args.challenge_root) / "verify" / "validate_candidate.py"
        candidate_path = out / f"candidate_{index}_{min(dx, dz)}.json"
        candidate_path.write_text(json.dumps(candidate, indent=2) + "\n")
        if validator.exists():
            proc = subprocess.run([sys.executable, str(validator), str(candidate_path.resolve())],
                                  cwd=Path(args.challenge_root), capture_output=True, text=True)
            try:
                official = json.loads(proc.stdout) if proc.stdout.strip() else {
                    "returncode": proc.returncode, "stderr": proc.stderr}
            except json.JSONDecodeError:
                official = {"returncode": proc.returncode, "stdout": proc.stdout,
                            "stderr": proc.stderr}
        final.append({"row": row, "candidate_path": str(candidate_path.resolve()),
                      "dx": int(dx), "dz": int(dz), "d": min(int(dx), int(dz)),
                      "score_upper": 182 * min(int(dx), int(dz)) ** 2 / 682,
                      "gate_trials_per_side": int(args.gate_trials),
                      "full_trials_per_side": (int(args.final_trials)
                                                if final_stage == "full_survivor" else 0),
                      "full_early_stop_target": int(TARGET_D - 1),
                      "full_stopped_early": bool(
                          result.get("x", {}).get("stopped_early") or
                          result.get("z", {}).get("stopped_early")),
                      "final_stage": final_stage,
                      "full_mode": result.get("mode"), "structural": structural,
                      "official": official})
        (out / "final_progress.json").write_text(
            json.dumps(final, indent=2, default=_json_default) + "\n")

    report = {
        "schema_version": "1.0", "kind": "gb_aggressive_phylogenetic_campaign",
        "submission_sent": False, "git_commit_performed": False,
        "target": {"m": M, "n": 682, "k": 182, "check_weight": 32,
                    "frontier_threshold": TARGET_D},
        "predictor": {
            "method": "native CSS-RIS proxy + direct-genome-diversified phylogenetic beam",
            "proxy_trials_per_side": int(args.proxy_trials),
            "deep_trials_per_side": int(args.deep_trials),
            "proof_status": "ranking signal only; no randomized no-hit is a lower bound",
        },
        "population": {"ideal_words": len(pool), "candidate_count": len(candidates),
                       "root_lineages": sorted(ROOTS),
                       "mutation_draws_per_root": int(args.mutation_draws),
                       "mutation_radius": int(args.mutation_radius)},
        "phylogeny": phylo,
        "proxy_top": screened[:min(64, len(screened))],
        "deep_top": deep[:min(64, len(deep))],
        "final": final,
        "accelerator": info,
        "seconds_wall": time.perf_counter() - started,
    }
    (out / "campaign.json").write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n")
    (out / "top.json").write_text(
        json.dumps(final, indent=2, default=_json_default) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/gb_aggressive_campaign"))
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--miner-iterations", type=int, default=50_000_000)
    ap.add_argument("--max-words", type=int, default=20_000)
    ap.add_argument("--mutation-draws", type=int, default=50_000)
    ap.add_argument("--mutation-radius", type=int, default=1)
    ap.add_argument("--proxy-trials", type=int, default=8)
    ap.add_argument("--proxy-repeats", type=int, default=4,
                    help="independent proxy seeds per candidate")
    ap.add_argument("--beam-size", type=int, default=512,
                    help="top candidates entering repeated medium screening")
    ap.add_argument("--beam-trials", type=int, default=512)
    ap.add_argument("--beam-repeats", type=int, default=4)
    ap.add_argument("--elite-count", type=int, default=96)
    ap.add_argument("--deep-trials", type=int, default=512)
    ap.add_argument("--final-candidates", type=int, default=1)
    ap.add_argument("--gate-trials", type=int, default=200_000,
                    help="cheap final gate; full run only for survivors")
    ap.add_argument("--final-trials", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--exploitation-fraction", type=float, default=0.55)
    ap.add_argument("--challenge-root", type=Path,
                    default=Path("challenge_data"))
    return run(ap.parse_args())


if __name__ == "__main__":
    result = main()
    print(json.dumps({"out": "results/gb_aggressive_campaign",
                      "final": result["final"],
                      "seconds_wall": round(result["seconds_wall"], 1)},
                     indent=2, default=_json_default))
