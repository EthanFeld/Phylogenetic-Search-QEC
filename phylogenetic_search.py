"""Small, deterministic phylogeny utilities for search-population diversity.

Genomes are support pairs ``(a0, a1), (b0, b1)``.  UPGMA is used as a
population-management heuristic, never as a code-validity or distance claim.
"""
from __future__ import annotations

from collections.abc import Sequence
from math import ceil


def _support_set(values):
    return set(int(x) for x in values)


def genome_distance(a, b) -> float:
    """Normalized edit distance between two candidate support genomes."""
    if (a.get("group_order"), a.get("group_index")) != (b.get("group_order"), b.get("group_index")):
        return 1.0
    total = 0
    differing = 0
    for field in ("A", "B"):
        aa = a.get(field, ())
        bb = b.get(field, ())
        for xa, xb in zip(aa, bb):
            sa, sb = _support_set(xa), _support_set(xb)
            total += max(len(sa | sb), 1)
            differing += len(sa ^ sb)
        if len(aa) != len(bb):
            total += max(len(aa), len(bb), 1)
            differing += abs(len(aa) - len(bb))
    return float(differing / total) if total else 0.0


def _label(row, index):
    return str(row.get("semantic_hash") or f"leaf-{index:04d}")


def upgma(records: Sequence[dict], max_leaves: int = 64) -> dict:
    """Build compact UPGMA tree and cophenetic distances.

    Input is capped to highest-ranked records by caller/order.  Output remains
    JSON-safe and includes Newick plus merge history for audit/replay.
    """
    rows = list(records)[:max(0, int(max_leaves))]
    labels = [_label(row, i) for i, row in enumerate(rows)]
    n = len(rows)
    if not n:
        return {"algorithm": "UPGMA", "leaf_count": 0, "newick": ";",
                "labels": [], "cophenetic": [], "merges": []}

    distances = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i):
            distances[i][j] = distances[j][i] = genome_distance(rows[i], rows[j])

    clusters = {i: [i] for i in range(n)}
    heights = {i: 0.0 for i in range(n)}
    newicks = {i: labels[i] for i in range(n)}
    cophenetic = [[0.0] * n for _ in range(n)]
    merges = []
    next_id = n

    def average_distance(left, right):
        values = [distances[i][j] for i in clusters[left] for j in clusters[right]]
        return sum(values) / len(values)

    while len(clusters) > 1:
        ids = sorted(clusters)
        best = None
        for pos, left in enumerate(ids[:-1]):
            for right in ids[pos + 1:]:
                key = (average_distance(left, right), left, right)
                if best is None or key < best[0]:
                    best = (key, left, right)
        distance, left, right = best[0][0], best[1], best[2]
        merged = clusters[left] + clusters[right]
        height = distance / 2.0
        left_branch = max(height - heights[left], 0.0)
        right_branch = max(height - heights[right], 0.0)
        for i in clusters[left]:
            for j in clusters[right]:
                cophenetic[i][j] = cophenetic[j][i] = distance
        merges.append({"left": left, "right": right, "distance": distance,
                       "size": len(merged)})
        clusters[next_id] = merged
        heights[next_id] = height
        newicks[next_id] = (f"({newicks[left]}:{left_branch:.6f},"
                            f"{newicks[right]}:{right_branch:.6f})")
        del clusters[left], clusters[right]
        del heights[left], heights[right]
        del newicks[left], newicks[right]
        next_id += 1

    root = next(iter(clusters))
    values = [cophenetic[i][j] for i in range(n) for j in range(i)]
    return {
        "algorithm": "UPGMA",
        "leaf_count": n,
        "labels": labels,
        "newick": newicks[root] + ";",
        "merges": merges,
        "cophenetic": cophenetic,
        "pairwise_distance": {
            "min": min(values) if values else 0.0,
            "mean": sum(values) / len(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        },
    }


def select_phylogenetic_elites(records: Sequence[dict], limit: int,
                               pool_factor: int = 8,
                               exploitation_fraction: float = 0.5):
    """Select strong parents, then fill remaining slots with distant genomes.

    Deep distance estimates outrank cheap screening when they exist.  A
    common-gcd ideal probe is the next tie-breaker: high observed weight in
    ``<h>`` is preferred because any verified low word is an immediate X-logical
    refutation.  The first half of the selection is exploitation; remaining
    slots maximize *direct* genome distance.  UPGMA remains in telemetry/audit
    output, but its averaged merge height is not used as a proxy for parent
    separation.
    """
    ranked = list(records)
    limit = max(0, int(limit))
    if not ranked or not limit:
        return [], {"algorithm": "UPGMA-stratified-direct-PD", "leaf_count": 0,
                     "selected": [], "pool_size": 0}
    def quality(row, index):
        deep = row.get("distance_estimate", {})
        screen = row.get("screen", {})
        probe = row.get("seed_probe", {})
        ideal = screen.get("ideal_x_probe", {})
        ideal_score = ideal.get("best_weight") if isinstance(ideal, dict) else None
        probes = [int(x.get("best_weight", -1)) for x in probe.values()
                  if isinstance(x, dict)]
        deep_score = deep.get("d_upper")
        screen_score = screen.get("d_upper")
        return (int(deep_score is not None),
                int(deep_score if deep_score is not None else screen_score or -1),
                int(screen_score or -1), min(probes) if probes else -1,
                int(ideal_score if ideal_score is not None else -1),
                _label(row, index))

    ranked = [row for _, row in sorted(enumerate(ranked),
                                       key=lambda item: quality(item[1], item[0]),
                                       reverse=True)]
    pool = ranked[:min(len(ranked), max(limit, limit * max(1, int(pool_factor))), 64)]
    tree = upgma(pool, max_leaves=len(pool))
    selected_limit = min(limit, len(pool))
    exploitation_slots = min(
        selected_limit,
        max(1, ceil(selected_limit * max(0.0, min(1.0, exploitation_fraction)))),
    )
    chosen = list(range(exploitation_slots))
    remaining = set(range(exploitation_slots, len(pool)))

    while remaining and len(chosen) < selected_limit:
        def key(index):
            separation = min(genome_distance(pool[index], pool[other]) for other in chosen)
            return (separation, quality(pool[index], index))
        best = max(remaining, key=key)
        chosen.append(best)
        remaining.remove(best)

    selected = [pool[i] for i in chosen]
    pairwise = [genome_distance(pool[i], pool[j])
                for pos, i in enumerate(chosen) for j in chosen[:pos]]
    tree_summary = {key: value for key, value in tree.items()
                    if key != "cophenetic"}
    summary = {
        "algorithm": "UPGMA-stratified-direct-PD",
        "pool_size": len(pool),
        "exploitation_slots": exploitation_slots,
        "exploration_slots": selected_limit - exploitation_slots,
        "selection_distance": "direct_genome_distance",
        "selected": [_label(pool[i], i) for i in chosen],
        "selected_pairwise_distance": {
            "min": min(pairwise) if pairwise else 0.0,
            "mean": sum(pairwise) / len(pairwise) if pairwise else 0.0,
            "max": max(pairwise) if pairwise else 0.0,
        },
        "tree": tree_summary,
    }
    return selected, summary


def compact_history(rows: Sequence[dict], n: int, k: int) -> list[dict]:
    """Keep only reusable same-target genomes from prior campaign/archive data."""
    out = []
    seen = set()
    for row in rows:
        if int(row.get("n", -1)) != int(n) or int(row.get("k", -1)) != int(k):
            continue
        official = row.get("evidence", {}).get("official") or {}
        if (row.get("regulation", {}).get("official_gate_status") == "refuted" or
                official.get("refuted", False)):
            continue
        if not row.get("A") or not row.get("B") or row.get("group_order") is None:
            continue
        item = {key: row[key] for key in ("n", "k", "group_order", "group_index",
                                           "A", "B", "d_upper", "semantic_hash")
                if key in row}
        key = (item.get("group_order"), tuple(map(tuple, item["A"])),
               tuple(map(tuple, item["B"])))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    out.sort(key=lambda row: (-int(row.get("d_upper", -1)),
                              str(row.get("semantic_hash", ""))))
    return out
