from __future__ import annotations

"""Collect serial-concatenation candidates and trusted gate receipts."""

import argparse
import json
from pathlib import Path


def collect(root: Path) -> dict:
    findings = []
    for receipt_path in sorted(root.glob("*_official.json")):
        candidate_path = receipt_path.with_name(receipt_path.name.replace("_official.json", ".json"))
        if not candidate_path.exists():
            continue
        candidate = json.loads(candidate_path.read_text())
        if candidate.get("family") != "serial-concatenated-css":
            continue
        receipt = json.loads(receipt_path.read_text())
        d = int(candidate["distance"]["d"])
        n = int(candidate["n"])
        k = int(candidate["k"])
        verify = receipt.get("gates", {}).get("verify", {})
        refute = receipt.get("gates", {}).get("refute", {})
        novelty = receipt.get("gates", {}).get("novelty", {})
        findings.append({
            "candidate": str(candidate_path),
            "receipt": str(receipt_path),
            "name": candidate.get("name"),
            "n": n, "k": k, "d_claim": d,
            "score_claim": float(k * d * d / n),
            "max_check_weight": max(
                max(map(len, candidate["checks"]["X"])),
                max(map(len, candidate["checks"]["Z"])),
            ),
            "official_passed": bool(receipt.get("passed")),
            "board_advancing": bool(novelty.get("board_advancing")),
            "verify_ok": bool(verify.get("ok")),
            "failed_checks": verify.get("failed_checks", []),
            "refuted": bool(refute.get("refuted", False)),
            "regulation": candidate.get("regulation", {}),
            "inner": candidate.get("construction", {}).get("inner"),
            "outer": candidate.get("construction", {}).get("outer"),
        })
    findings.sort(key=lambda row: (not row["official_passed"], -row["score_claim"]))
    return {
        "schema_version": "1.0",
        "kind": "stage_only_serial_concatenation_findings",
        "regulation": {
            "max_n": 700,
            "max_check_weight": 32,
            "official_pass_required": True,
            "submission_sent": False,
            "distance_policy": "RIS claims are upper bounds; official no-hit is not proof.",
        },
        "counts": {
            "candidates": len(findings),
            "official_passed": sum(row["official_passed"] for row in findings),
            "eligible": sum(row["official_passed"] and row["board_advancing"] for row in findings),
        },
        "findings": findings,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/concatenation"))
    ap.add_argument("--out", type=Path, default=Path("results/concatenation/findings.json"))
    args = ap.parse_args()
    data = collect(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "counts": data["counts"]}, indent=2))


if __name__ == "__main__":
    main()
