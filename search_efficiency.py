from __future__ import annotations
"""Proof-neutral scheduling and evidence caching for code searches.

The expensive operations in the search pipeline are logically monotone:
- a low-weight logical witness permanently refutes a target distance;
- an exhaustive ``certified_none`` result permanently proves a lower bound;
- repeating the same randomized seed/trial budget adds no evidence.

This module stores those facts by exact semantic hash and exposes a frontier gate
that spends randomized shots only until they are useful, then hands survivors to
the exact threshold certifier when requested.
"""

from pathlib import Path
import hashlib
import json
import os
import tempfile
import time
from typing import Any

import numpy as np

from distance_sketch import CSSFullRISRace, fast_refute_supports
from sparse_tanner_certifier import exact_low_weight_logical

CACHE_VERSION = 1


def css_semantic_hash(hx: np.ndarray, hz: np.ndarray) -> str:
    hx = np.ascontiguousarray(np.asarray(hx, dtype=np.uint8))
    hz = np.ascontiguousarray(np.asarray(hz, dtype=np.uint8))
    h = hashlib.sha256()
    h.update(str(hx.shape).encode()); h.update(hx.tobytes())
    h.update(str(hz.shape).encode()); h.update(hz.tobytes())
    return h.hexdigest()


class EvidenceCache:
    """Tiny JSON evidence cache keyed by semantic hash and verifier settings.

    Only reusable mathematical/search evidence is cached.  An exact lower-bound
    proof is safe to reuse indefinitely for identical matrices.  A randomized
    result is reused only for the exact same algorithm/seed/budget tuple, so a
    deliberately fresh-seed challenge run is never mistaken for old evidence.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.exists():
            try:
                obj = json.loads(self.path.read_text())
            except Exception:
                obj = {}
        else:
            obj = {}
        if obj.get("version") != CACHE_VERSION:
            obj = {"version": CACHE_VERSION, "entries": {}}
        obj.setdefault("entries", {})
        self.data = obj
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(parts: tuple[Any, ...]) -> str:
        return json.dumps(parts, separators=(",", ":"), sort_keys=False)

    def get(self, *parts):
        k = self._key(tuple(parts))
        if k in self.data["entries"]:
            self.hits += 1
            return self.data["entries"][k]
        self.misses += 1
        return None

    def put(self, value, *parts):
        self.data["entries"][self._key(tuple(parts))] = value
        return value

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.data, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def stats(self):
        return {"hits": self.hits, "misses": self.misses,
                "entries": len(self.data.get("entries", {})), "path": str(self.path)}


def _run_exact_cached(cache: EvidenceCache | None, sem: str, side: str,
                      H: np.ndarray, D: np.ndarray, cap: int,
                      time_limit: float | None, variable_order: str):
    # Timed-out proofs are intentionally not reused across a larger future budget.
    budget_key = None if time_limit is None else float(time_limit)
    key = ("exact_low_weight", sem, side, int(cap), variable_order, budget_key)
    if cache is not None:
        hit = cache.get(*key)
        if hit is not None:
            out = dict(hit); out["cache_hit"] = True
            return out
    out = exact_low_weight_logical(H, D, int(cap), time_limit=time_limit,
                                   variable_order=variable_order)
    out["cache_hit"] = False
    if cache is not None and out.get("status") != "timeout":
        cache.put({k: v for k, v in out.items() if k != "cache_hit"}, *key)
    return out


def fast_refute_submission(doc: dict, *, seed: int = 0, trials: int = 8000,
                           pair_depth: int = 8,
                           max_seconds: float | None = 10.0,
                           backend: str = "auto") -> dict:
    """Run packed RIS directly on challenge-format support lists.

    Useful for local screening before invoking the challenge CLI.  It never
    upgrades a distance claim: a lighter returned witness refutes, while a
    clean run is only corroboration.  The result shape is intentionally close
    to ``heuristic_distance.refute_check`` plus per-sector diagnostics.
    """
    if not isinstance(doc, dict) or "n" not in doc or "checks" not in doc:
        raise ValueError("submission needs n and checks")
    claimed = int(doc.get("distance", {}).get("d", 0))
    if claimed < 1:
        raise ValueError("submission distance must be positive")
    return fast_refute_supports(doc["checks"]["X"], doc["checks"]["Z"],
                                int(doc["n"]), claimed, seed=int(seed),
                                trials=int(trials), pair_depth=int(pair_depth),
                                max_seconds=max_seconds, backend=backend)


def efficient_frontier_gate(
    hx: np.ndarray,
    hz: np.ndarray,
    lx: np.ndarray,
    lz: np.ndarray,
    target_d: int,
    *,
    seed: int = 0,
    ris_trials_per_side: int = 100,
    pair_depth: int = 8,
    exact: bool = False,
    exact_cap: int | None = None,
    exact_time_limit_per_side: float | None = None,
    cache: EvidenceCache | None = None,
    first: str = "z",
) -> dict:
    """Gate a CSS candidate with minimum useful effort.

    1. Full-kernel RIS races the two sectors and stops after the first witness
       below ``target_d``.
    2. Only a survivor reaches exact enumeration when ``exact=True``.
       By default the exact query is precisely the frontier question
       ``exists logical <= target_d-1?`` rather than a full distance solve.
    3. Exact proof results and identical randomized runs can be cached.

    No randomized null result is promoted to a proof.  ``frontier_certified`` is
    true only after exhaustive exact exclusion on both sectors.
    """
    hx = np.asarray(hx, dtype=np.uint8); hz = np.asarray(hz, dtype=np.uint8)
    lx = np.asarray(lx, dtype=np.uint8); lz = np.asarray(lz, dtype=np.uint8)
    sem = css_semantic_hash(hx, hz)
    ris_key = ("full_ris", sem, int(seed), int(ris_trials_per_side), int(pair_depth),
               int(target_d), str(first).lower())
    ris = cache.get(*ris_key) if cache is not None else None
    if ris is None:
        race = CSSFullRISRace(hx, hz, lx, lz, seed=int(seed), pair_depth=int(pair_depth))
        ris = race.run_stage(int(ris_trials_per_side), int(target_d), first=first)
        ris = race.snapshot(int(target_d))
        ris["cache_hit"] = False
        if cache is not None:
            cache.put({k: v for k, v in ris.items() if k != "cache_hit"}, *ris_key)
    else:
        ris = dict(ris); ris["cache_hit"] = True

    out = {
        "semantic_hash": sem,
        "target_d": int(target_d),
        "ris": ris,
        "refuted": bool(ris.get("screen_refuted")),
        "frontier_certified": False,
        "exact": None,
    }
    if out["refuted"] or not exact:
        if cache is not None: cache.flush()
        return out

    cap = int(target_d) - 1 if exact_cap is None else int(exact_cap)
    order = ("x", "z") if str(first).lower() == "x" else ("z", "x")
    params = {"x": (hz, lz), "z": (hx, lx)}
    exact_results = {}
    t0 = time.perf_counter()
    for side in order:
        H, D = params[side]
        r = _run_exact_cached(cache, sem, side, H, D, cap,
                              exact_time_limit_per_side, "degree_desc")
        exact_results[side] = r
        if r.get("status") == "found":
            out["refuted"] = True
            break
        if r.get("status") != "certified_none":
            break
    out["exact"] = {
        "cap": cap,
        "sectors": exact_results,
        "seconds_wall": time.perf_counter() - t0,
    }
    out["frontier_certified"] = (
        len(exact_results) == 2 and
        all(r.get("status") == "certified_none" for r in exact_results.values())
    )
    if cache is not None: cache.flush()
    return out
