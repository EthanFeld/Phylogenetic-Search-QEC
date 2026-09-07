from __future__ import annotations

"""Convert a GCD trial ladder into an auditable mutation-gate summary."""

import argparse
import json
from pathlib import Path


def run(ladder_path: Path, out_path: Path) -> dict:
    ladder = json.loads(ladder_path.read_text(encoding="utf-8"))
    rows = []
    monotonicity_violations = 0
    for item in ladder.get("candidates", []):
        series = item.get("series", [])
        weights = [int(row["best_ideal_weight"]) for row in series
                   if row.get("best_ideal_weight") is not None]
        if any(right > left for left, right in zip(weights, weights[1:])):
            monotonicity_violations += 1
        final = series[-1] if series else {}
        weight = final.get("best_ideal_weight")
        score = (int(item["k"]) * int(weight) ** 2 / int(item["n"])) if weight is not None else None
        rows.append({
            "candidate_path": item["candidate_path"],
            "n": int(item["n"]), "k": int(item["k"]),
            "required_distance": int((1542.0 * int(item["n"]) / int(item["k"])) ** 0.5) + 1,
            "gcd_best_weight_final": weight,
            "gcd_score_upper_final": score,
            "target_refuted": bool(weight is not None and score <= 1542.0),
            "final_witness_present": bool(final.get("new_witness")),
            "weight_curve": weights,
        })
    report = {
        "schema_version": "1.0",
        "kind": "mutation_final_gcd_gate",
        "target_score": 1542.0,
        "source_ladder": str(ladder_path.resolve()),
        "budgets": ladder.get("budgets", []),
        "workers": ladder.get("workers"),
        "candidate_count": len(rows),
        "target_refuted_count": sum(row["target_refuted"] for row in rows),
        "survivor_count": sum(not row["target_refuted"] for row in rows),
        "ladder_monotonicity_violations": monotonicity_violations,
        "max_final_score_upper": max((row["gcd_score_upper_final"] for row in rows
                                      if row["gcd_score_upper_final"] is not None), default=None),
        "rows": rows,
        "claim_policy": "GCD witnesses are upper bounds; no candidate is accepted as a proven distance.",
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ladder", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.ladder, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
