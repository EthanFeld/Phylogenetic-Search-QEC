from __future__ import annotations

"""Collapse-aware discovery and confirmation campaign for the 682/182 GB pocket.

The old campaign ranked a candidate by the *best* randomized pilot.  That is
the wrong statistic for this problem: a sparse low logical can be missed by a
short RIS run, so a lucky candidate is promoted and later collapses.  This
module adds three independent safeguards:

* identify the cyclic orbit pair and relative phase of every genome;
* refuse exact signatures already refuted by a deeper witness; and
* promote candidates by the minimum/median of fresh holdout pilots while
  reserving slots for every orbit pair and phase bin.

All distance values remain randomized upper bounds.  The guard is a search
policy, not a proof or a submission gate.  No git or network mutation occurs.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import cpp_fast
from generalized_bicycle import build_matrices
from gb_aggressive_campaign import (
    M, _candidate_doc, _json_default, _pool_from_ideal,
    _make_candidates,
)
from hyper_validator import structural_validate


# ``[[682,182,77]]`` is the current score line (1582.226).  A d=77
# randomized upper bound is therefore a tie, not a high-score candidate.
# Keep this campaign's target independent from the older d=77 exploration
# scripts so existing historical replays retain their original meaning.
TARGET_D = 78


_FULL_MASK = (1 << M) - 1
_CONSTRUCTION_RE = re.compile(
    r"(?:^|\s)(?:a|A)=\[([^\]]*)\].*?(?:^|\s)(?:b|B)=\[([^\]]*)\]",
    re.DOTALL,
)


def _mask(word: tuple[int, ...] | list[int]) -> int:
    out = 0
    for value in word:
        out |= 1 << (int(value) % M)
    return out


def _rotate(mask: int, amount: int) -> int:
    amount %= M
    if not amount:
        return int(mask) & _FULL_MASK
    return ((int(mask) << amount) | (int(mask) >> (M - amount))) & _FULL_MASK


def _canonical_mask(mask: int) -> int:
    return min(_rotate(mask, shift) for shift in range(M))


def _orbit_phase(mask: int, canonical: int) -> int:
    for shift in range(M):
        if _rotate(canonical, shift) == mask:
            return shift
    raise ValueError("mask is not in its claimed orbit")


def _support_from_mask(mask: int) -> tuple[int, ...]:
    return tuple(i for i in range(M) if (mask >> i) & 1)


def classify_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """Annotate rows with orbit-pair/phase features and return class metadata."""
    canonical_masks: dict[int, int] = {}
    for row in rows:
        for side in ("A", "B"):
            word = tuple(row[side][0])
            canonical_masks.setdefault(_canonical_mask(_mask(word)), 0)
    ordered = sorted(canonical_masks)
    ids = {mask: index for index, mask in enumerate(ordered)}
    annotated = []
    pair_counts: dict[str, int] = {}
    for row in rows:
        a = tuple(row["A"][0]); b = tuple(row["B"][0])
        am = _mask(a); bm = _mask(b)
        ac = _canonical_mask(am); bc = _canonical_mask(bm)
        aid, bid = ids[ac], ids[bc]
        phase = (_orbit_phase(bm, bc) - _orbit_phase(am, ac)) % M
        pair = f"{aid}>{bid}"
        signature = f"{pair}@{phase}"
        digest = row.get("semantic_hash") or hashlib.sha256(
            json.dumps([list(a), list(b)], separators=(",", ":")).encode()
        ).hexdigest()
        out = dict(row)
        out.update({
            "semantic_hash": digest,
            "orbit_a": aid,
            "orbit_b": bid,
            "orbit_pair": pair,
            "relative_phase": phase,
            "phase_bin": phase // 17,
            "collapse_signature": signature,
        })
        annotated.append(out)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    metadata = {
        "orbit_count": len(ordered),
        "orbit_representatives": {
            str(ids[mask]): list(_support_from_mask(mask)) for mask in ordered
        },
        "ordered_orbit_pair_counts": pair_counts,
        "candidate_count": len(annotated),
    }
    return annotated, metadata


def _row_key(row: dict) -> str:
    return str(row.get("collapse_signature") or row.get("semantic_hash") or "")


def _pilot_values(row: dict) -> list[int]:
    values = row.get("guard_scores")
    if values:
        return [int(v) for v in values if v is not None]
    values = row.get("proxy_repeat_scores")
    if values:
        return [int(v) for v in values if v is not None]
    value = row.get("d_proxy")
    return [] if value is None else [int(value)]


def annotate_risk(row: dict, known_bad: set[str], target: int = TARGET_D,
                  rejected_pairs: set[str] | None = None) -> dict:
    """Add conservative risk fields; low values are deliberately penalized."""
    out = dict(row)
    values = _pilot_values(out)
    minimum = min(values) if values else None
    median = statistics.median(values) if values else None
    low_hits = sum(value < target for value in values)
    rejected_pairs = rejected_pairs or set()
    pair = str(out.get("orbit_pair", ""))
    out.update({
        "guard_min": minimum,
        "guard_median": median,
        "guard_low_hits": low_hits,
        "known_bad_signature": _row_key(out) in known_bad,
        "known_bad_pair": pair in rejected_pairs,
        "guard_survives": bool(
            values and minimum >= int(target) and
            _row_key(out) not in known_bad and pair not in rejected_pairs
        ),
    })
    return out


def robust_sort_key(row: dict):
    minimum = row.get("guard_min")
    if minimum is None:
        values = _pilot_values(row)
        minimum = min(values) if values else -1
    median = row.get("guard_median")
    if median is None:
        values = _pilot_values(row)
        median = statistics.median(values) if values else -1
    return (
        int(minimum), float(median), -int(row.get("guard_low_hits", 0)),
        int(row.get("dz_proxy") or -1), int(row.get("dx_proxy") or -1),
        str(row.get("semantic_hash", "")),
    )


def select_guard_beam(rows: list[dict], count: int,
                      per_pair: int | None = None) -> list[dict]:
    """Select a robust beam with coverage across pair and phase bins.

    A short RIS proxy can make many phases of one orbit pair look good at
    once.  A pair cap prevents that correlated pocket from consuming the
    independent holdout budget.
    """
    if not rows:
        return []
    ordered = sorted(rows, key=robust_sort_key, reverse=True)
    count = min(max(1, int(count)), len(ordered))
    pair_cap = None if per_pair is None or int(per_pair) <= 0 else int(per_pair)
    selected: list[dict] = []
    seen_bins: set[tuple[str, int]] = set()
    pair_counts: dict[str, int] = {}
    # First pass: cover distinct pair/phase regions, but still take them in
    # robust order so weak regions do not consume the entire beam.
    for row in ordered:
        key = (str(row.get("orbit_pair")), int(row.get("phase_bin", -1)))
        if key in seen_bins:
            continue
        pair = key[0]
        if pair_cap is not None and pair_counts.get(pair, 0) >= pair_cap:
            continue
        seen_bins.add(key)
        selected.append(row)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        if len(selected) >= count:
            return selected
    selected_hashes = {row.get("semantic_hash") for row in selected}
    for row in ordered:
        if row.get("semantic_hash") in selected_hashes:
            continue
        pair = str(row.get("orbit_pair"))
        if pair_cap is not None and pair_counts.get(pair, 0) >= pair_cap:
            continue
        selected.append(row)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        if len(selected) >= count:
            break
    return selected


def _parse_ab_from_construction(text: str):
    match = _CONSTRUCTION_RE.search(str(text))
    if not match:
        return None
    try:
        a = tuple(int(x.strip()) for x in match.group(1).split(",") if x.strip())
        b = tuple(int(x.strip()) for x in match.group(2).split(",") if x.strip())
    except ValueError:
        return None
    return a, b


def _history_walk(value):
    """Yield (candidate, observed d) from old result/candidate JSON shapes."""
    if isinstance(value, dict):
        candidate = value.get("candidate")
        if isinstance(candidate, dict):
            run = value.get("run") or value.get("full") or value.get("gate") or {}
            observed = run.get("d_upper") if isinstance(run, dict) else None
            if observed is None:
                distance = candidate.get("distance", {})
                observed = distance.get("d") if isinstance(distance, dict) else None
            yield candidate, observed
        elif "distance" in value and "provenance" in value:
            distance = value.get("distance", {})
            observed = distance.get("d") if isinstance(distance, dict) else None
            yield value, observed
        for key, child in value.items():
            # The candidate was already emitted above; descending into it
            # would duplicate every historical observation.
            if key != "candidate":
                yield from _history_walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _history_walk(child)


def load_known_bad(paths: list[Path], rows: list[dict]) -> tuple[set[str], dict]:
    """Map deeper witnessed collapses to exact phase signatures when possible."""
    # Build the same orbit map used by the current candidate pool.
    annotated, _ = classify_rows(rows)
    lookup = {_row_key(row): row for row in annotated}
    bad: set[str] = set()
    pair_min: dict[str, int] = {}
    records = []
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        # Campaign-local feedback file.  Its entries are intentionally keyed
        # by the exact cyclic orbit/phase signature, never by an entire orbit
        # pair, so an untested phase is not thrown away.
        if isinstance(payload, dict) and isinstance(payload.get("entries"), dict):
            for signature, record in payload["entries"].items():
                if not isinstance(record, dict):
                    continue
                observed = record.get("observed_d")
                if observed is None:
                    continue
                observed = int(observed)
                pair = str(signature).split("@", 1)[0]
                pair_min[pair] = min(pair_min.get(pair, observed), observed)
                records.append({"path": str(path), "signature": str(signature),
                                "pair": pair, "observed_d": observed})
                if observed < TARGET_D:
                    bad.add(str(signature))
        for candidate, observed in _history_walk(payload):
            construction = candidate.get("provenance", {}).get("construction", "")
            ab = _parse_ab_from_construction(construction)
            if ab is None or observed is None:
                continue
            a, b = ab
            am, bm = _mask(a), _mask(b)
            ac, bc = _canonical_mask(am), _canonical_mask(bm)
            ids = sorted({_canonical_mask(_mask(tuple(r[side][0])))
                          for r in rows for side in ("A", "B")})
            if ac not in ids or bc not in ids:
                continue
            phase = (_orbit_phase(bm, bc) - _orbit_phase(am, ac)) % M
            pair = f"{ids.index(ac)}>{ids.index(bc)}"
            signature = f"{pair}@{phase}"
            observed = int(observed)
            pair_min[pair] = min(pair_min.get(pair, observed), observed)
            records.append({"path": str(path), "signature": signature,
                            "pair": pair, "observed_d": observed})
            if observed < TARGET_D:
                bad.add(signature)
    # Only exact signatures are hard-rejected.  Pair-level evidence is
    # reported as a prior, not treated as proof that every phase is bad.
    return bad, {"records": records, "pair_min_observed_d": pair_min,
                 "known_bad_signature_count": len(bad)}


def _guard_task(task):
    row, seed, trials, threads = task
    hx, hz = build_matrices(M, tuple(row["A"][0]), tuple(row["B"][0]))
    result = cpp_fast.css_ris_parallel(
        hx, hz, trials=int(trials), seed=int(seed), pair_depth=24,
        target=TARGET_D, stop_on_target=True, threads=int(threads))
    score = result.get("d_upper")
    out = dict(row)
    out.setdefault("guard_scores", [])
    out["guard_scores"] = list(out["guard_scores"]) + [score]
    out["guard_last"] = {
        "d_upper": score,
        "dx_upper": result.get("dx_upper"),
        "dz_upper": result.get("dz_upper"),
        "x_trials": result["x"].get("trials_run"),
        "z_trials": result["z"].get("trials_run"),
        "stopped_early": bool(
            result["x"].get("stopped_early") or
            result["z"].get("stopped_early")
        ),
    }
    return out


def run_guard_round(rows: list[dict], *, seed: int, trials: int,
                    workers: int, threads: int, label: str) -> list[dict]:
    tasks = [(row, int(seed) + index * 0x9E3779B9, int(trials), int(threads))
             for index, row in enumerate(rows)]
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futures = [pool.submit(_guard_task, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if done == 1 or done % 16 == 0 or done == len(futures):
                print(json.dumps({"stage": label, "done": done,
                                  "total": len(futures),
                                  "seconds": round(time.perf_counter() - started, 1)}),
                      flush=True)
    return results


def _rows_from_source(source: Path) -> list[dict]:
    payload = json.loads(source.read_text())
    rows = payload.get("rows", payload if isinstance(payload, list) else [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"no candidate rows in {source}")
    return [dict(row) for row in rows]


def _official_gate(candidate: dict, path: Path, challenge_root: Path):
    validator = challenge_root / "verify" / "validate_candidate.py"
    if not validator.exists():
        return {"status": "not_available", "validator": str(validator)}
    proc = subprocess.run(
        [sys.executable, str(validator), str(path.resolve())],
        cwd=challenge_root, capture_output=True, text=True,
    )
    try:
        verdict = json.loads(proc.stdout) if proc.stdout.strip() else {}
    except json.JSONDecodeError:
        verdict = {"stdout": proc.stdout, "stderr": proc.stderr}
    passed = bool(verdict.get("passed"))
    dedup = verdict.get("gates", {}).get("dedup", {})
    wl_equivalent = dedup.get("wl_equivalent_of")
    candidate["regulation"]["official_gate_status"] = "passed" if passed else "refuted"
    candidate["regulation"]["board_claim_allowed"] = bool(
        passed and not dedup.get("exact_duplicate_of") and not wl_equivalent
    )
    if wl_equivalent:
        candidate["regulation"]["wl_equivalent_of"] = wl_equivalent
    return {"status": "passed" if passed else "failed",
            "passed": passed, "returncode": proc.returncode,
            "verdict": verdict}


def _histogram(values: list[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda item: int(item[0])))


def _compact_structural(value: dict) -> dict:
    """Keep validation verdicts, omit repeated dense H_X/H_Z matrices."""
    if not isinstance(value, dict):
        return value
    return {key: val for key, val in value.items() if key not in ("hx", "hz")}


def _update_blacklist(path: Path, final: list[dict]) -> dict:
    entries = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            if isinstance(payload, dict) and isinstance(payload.get("entries"), dict):
                entries.update(payload["entries"])
        except (OSError, json.JSONDecodeError):
            pass
    for row in final:
        if int(row.get("d", TARGET_D)) >= TARGET_D:
            continue
        signature = row.get("collapse_signature")
        if not signature:
            continue
        old = entries.get(signature, {})
        old_d = old.get("observed_d") if isinstance(old, dict) else None
        observed = int(row["d"])
        entries[signature] = {
            "observed_d": observed if old_d is None else min(int(old_d), observed),
            "evidence": row.get("candidate_path"),
            "source": "deep_guard_campaign",
        }
    payload = {"schema_version": "1.0", "kind": "gb_collapse_blacklist",
               "target": TARGET_D, "entries": dict(sorted(entries.items()))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def run(args):
    started = time.perf_counter()
    if args.source:
        rows = _rows_from_source(Path(args.source))
        discovery = {"mode": "reused_ranked_population",
                     "source": str(Path(args.source).resolve())}
    else:
        pool = _pool_from_ideal(args.miner_iterations, args.max_words, args.seed)
        rows = _make_candidates(pool, 0, 1, args.seed + 17)
        discovery = {"mode": "fresh_native_ideal_discovery",
                     "iterations": int(args.miner_iterations),
                     "ideal_words": len(pool),
                     "max_words": int(args.max_words)}
    rows, topology = classify_rows(rows)
    known_bad, history = load_known_bad(
        [Path(item) for item in args.history], rows)
    pair_bad_counts: dict[str, int] = {}
    for record in history["records"]:
        if int(record.get("observed_d", TARGET_D)) < TARGET_D:
            pair = str(record.get("pair", ""))
            pair_bad_counts[pair] = pair_bad_counts.get(pair, 0) + 1
    rejected_pairs = {
        pair for pair, count in pair_bad_counts.items()
        if count >= int(args.pair_bad_min_samples)
    }
    rows = [annotate_risk(row, known_bad, rejected_pairs=rejected_pairs)
            for row in rows]

    # The robust pre-rank uses the old repeated proxy if available.  If it is
    # absent, every row gets one fresh guard pilot in the first round.
    unknown = [row for row in rows if not _pilot_values(row)]
    if unknown:
        unknown = run_guard_round(
            unknown, seed=args.seed + 1001, trials=args.initial_trials,
            workers=args.workers, threads=args.threads, label="initial_guard")
        updated = {row["semantic_hash"]: row for row in unknown}
        rows = [updated.get(row["semantic_hash"], row) for row in rows]
        rows = [annotate_risk(row, known_bad, rejected_pairs=rejected_pairs)
                for row in rows]

    # Exact historical collapses are removed before spending any holdout
    # budget.  Other phases in the same orbit pair remain eligible.
    eligible = [row for row in rows if not row.get("known_bad_signature")
                and not row.get("known_bad_pair")]
    beam = select_guard_beam(eligible, args.guard_count,
                             per_pair=args.pair_cap)
    round_summaries = []
    trials_schedule = [args.round1_trials, args.round2_trials, args.round3_trials]
    for round_index in range(min(max(1, int(args.rounds)), len(trials_schedule))):
        if not beam:
            break
        beam = run_guard_round(
            beam, seed=args.seed + 2001 + round_index * 0x100000,
            trials=trials_schedule[round_index], workers=args.workers,
            threads=args.threads, label=f"guard_{round_index + 1}")
        beam = [annotate_risk(row, known_bad, rejected_pairs=rejected_pairs)
                for row in beam]
        survivors = [row for row in beam if row["guard_survives"]]
        round_summaries.append({
            "round": round_index + 1,
            "trials_per_side": int(trials_schedule[round_index]),
            "tested": len(beam), "survivors": len(survivors),
            "min_scores": _histogram([int(row["guard_min"]) for row in beam
                                       if row.get("guard_min") is not None]),
        })
        beam = select_guard_beam(survivors, args.guard_count,
                                 per_pair=args.pair_cap)

    finalists = sorted(beam, key=robust_sort_key, reverse=True)
    final = []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    final_limit = (len(finalists) if int(args.final_count) <= 0
                   else min(len(finalists), int(args.final_count)))
    for index, row in enumerate(finalists[:final_limit]):
        hx, hz = build_matrices(M, tuple(row["A"][0]), tuple(row["B"][0]))
        result = cpp_fast.css_ris_parallel(
            hx, hz, trials=int(args.final_trials),
            seed=int(args.seed) + 900001 + index * 0x9E3779B9,
            pair_depth=24, target=TARGET_D, stop_on_target=True,
            threads=int(args.final_threads),
        )
        dx, dz = result["x"].get("best_weight"), result["z"].get("best_weight")
        if dx is None or dz is None:
            continue
        candidate = _candidate_doc(
            row, dx, dz, result["x"]["witness"], result["z"]["witness"],
            result.get("mode"),
        )
        candidate["provenance"]["notes"] += (
            " Collapse guard: exact orbit/phase signature tracked; promotion "
            "required all holdout pilots to clear target."
        )
        d = min(int(dx), int(dz))
        path = out / f"candidate_guard_{index}_{d}.json"
        path.write_text(json.dumps(candidate, indent=2) + "\n")
        structural = structural_validate(candidate)
        official = _official_gate(candidate, path, Path(args.challenge_root))
        path.write_text(json.dumps(candidate, indent=2) + "\n")
        final.append({
            "rank": index + 1,
            "candidate_path": str(path.resolve()),
            "semantic_hash": row.get("semantic_hash"),
            "collapse_signature": row.get("collapse_signature"),
            "orbit_pair": row.get("orbit_pair"),
            "relative_phase": row.get("relative_phase"),
            "guard_scores": row.get("guard_scores", []),
            "dx": int(dx), "dz": int(dz), "d": d,
            "score_upper": 182 * d * d / 682,
            "final_trials_per_side": int(args.final_trials),
            "stopped_early": bool(
                result["x"].get("stopped_early") or
                result["z"].get("stopped_early")
            ),
            "structural": _compact_structural(structural),
            "official": official,
            "unique_board_candidate": bool(
                official.get("passed") and
                not official.get("verdict", {}).get("gates", {})
                    .get("dedup", {}).get("exact_duplicate_of") and
                not official.get("verdict", {}).get("gates", {})
                    .get("dedup", {}).get("wl_equivalent_of")
            ),
        })
    blacklist = _update_blacklist(out / "collapse_blacklist.json", final)
    report = {
        "schema_version": "1.0",
        "kind": "gb_collapse_guard_campaign",
        "submission_sent": False,
        "git_commit_performed": False,
        "discovery": discovery,
        "target": {"m": M, "n": 682, "k": 182, "target_distance": TARGET_D,
                    "check_weight": 32},
        "diagnosis": {
            **topology,
            "saturated_four_orbit_pocket": topology["orbit_count"] == 4,
            "proxy_d_histogram": _histogram(
                [int(row["d_proxy"]) for row in rows
                 if row.get("d_proxy") is not None]),
            "proxy_min_repeat_histogram": _histogram(
                [int(row["guard_min"]) for row in rows
                 if row.get("guard_min") is not None]),
            "known_history": history,
            "known_bad_signatures": sorted(known_bad),
            "rejected_pairs": sorted(rejected_pairs),
            "bad_pair_counts": pair_bad_counts,
            "blacklist_after_campaign": blacklist,
        },
        "guard": {
            "policy": "reject exact known-bad signatures; rank by minimum holdout score; preserve orbit-pair/phase-bin coverage",
            "target": TARGET_D,
            "initial_trials_per_side": int(args.initial_trials),
            "rounds": round_summaries,
            "guard_count": int(args.guard_count),
            "pair_cap": int(args.pair_cap),
        },
        "final": final,
        "accelerator": cpp_fast.accelerator_info(),
        "seconds_wall": time.perf_counter() - started,
    }
    (out / "collapse_guard_campaign.json").write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n"
    )
    (out / "guarded_beam.json").write_text(
        json.dumps({"rows": finalists, "submission_sent": False,
                    "git_commit_performed": False}, indent=2,
                   default=_json_default) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=Path("results/gb_asymmetric_campaign_01/z_ranked.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/gb_collapse_guard_01"))
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--miner-iterations", type=int, default=50_000_000)
    parser.add_argument("--max-words", type=int, default=200_000)
    parser.add_argument("--initial-trials", type=int, default=1024)
    parser.add_argument("--round1-trials", type=int, default=4096)
    parser.add_argument("--round2-trials", type=int, default=16384)
    parser.add_argument("--round3-trials", type=int, default=65536)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--guard-count", type=int, default=48)
    parser.add_argument("--pair-cap", type=int, default=4,
                        help="max orbit-pair members in each holdout beam; 0 disables")
    parser.add_argument("--pair-bad-min-samples", type=int, default=2,
                        help="exclude an orbit pair after this many deep collapses")
    parser.add_argument("--final-count", type=int, default=0,
                        help="20M-validate every ladder survivor; positive caps the count")
    parser.add_argument("--final-trials", type=int, default=20_000_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--final-threads", type=int, default=0)
    parser.add_argument("--history", type=Path, action="append", default=[
        Path("results/gb_aggressive_campaign_03/single_verify_75.json"),
        Path("results/gb_asymmetric_campaign_01/candidate_zfull_1_76.json"),
        Path("results/gb_asymmetric_campaign_01/candidate_zfull_0_78_20m.json"),
        Path("results/gb_asymmetric_campaign_01/candidate_zfull_2_77.json"),
        Path("results/gb_collapse_guard_01/collapse_blacklist.json"),
    ])
    parser.add_argument("--challenge-root", type=Path,
                        default=Path("challenge_data"))
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"out": str(args.out),
                      "orbit_count": report["diagnosis"]["orbit_count"],
                      "rounds": report["guard"]["rounds"],
                      "final": report["final"],
                      "seconds_wall": round(report["seconds_wall"], 1)},
                     indent=2, default=_json_default))


if __name__ == "__main__":
    main()
