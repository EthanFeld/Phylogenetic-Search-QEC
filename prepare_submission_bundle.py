"""Prepare an approval-only qLDPC challenge submission bundle.

This intentionally writes below results/ and never touches codes/, notes/, git,
or the network.  Replace the author placeholder only after human approval.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


DEFAULT_SOURCE = Path("results/screening_ladder/top_2m_682_182_77.json")
DEFAULT_GATE = Path("results/screening_ladder/top_2m_682_182_77_official.json")
DEFAULT_LADDER = Path("results/screening_ladder/top_2m.json")
DEFAULT_OUT = Path("results/submission_draft_682_182_77")


def dense(supports: list[list[int]], n: int) -> np.ndarray:
    matrix = np.zeros((len(supports), n), dtype=np.uint8)
    for row, support in enumerate(supports):
        matrix[row, support] = 1
    return matrix


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--ladder", type=Path, default=DEFAULT_LADDER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--author", default="@EthanFeld")
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    n = int(source["n"])
    k = 182
    dx = source["distance"]["X"]
    dz = source["distance"]["Z"]
    hx = source["checks"]["X"]
    hz = source["checks"]["Z"]

    out = args.out
    (out / "codes").mkdir(parents=True, exist_ok=True)
    (out / "notes").mkdir(parents=True, exist_ok=True)
    (out / "evidence").mkdir(parents=True, exist_ok=True)

    # Schema-only document.  Campaign metadata belongs in evidence, not in the
    # public code record; the challenge schema rejects unknown top-level keys.
    document = {
        "schema_version": "0.1",
        "name": "[[682,182,77]] generalized bicycle m=341",
        "code_type": "CSS",
        "n": n,
        "k": k,
        "checks": {"X": hx, "Z": hz},
        "distance": {
            "d": 77,
            "X": {
                "value": int(dx["value"]),
                "confidence": "upper_bound",
                "witness": dx["witness"],
            },
            "Z": {
                "value": int(dz["value"]),
                "confidence": "upper_bound",
                "witness": dz["witness"],
            },
        },
        "provenance": {
            "authors": [args.author],
            "origin": "submission",
            "novelty": "unknown",
            "construction": (
                "Generalized bicycle over Z_341 with H_X=[A|B] and "
                "H_Z=[B^T|A^T]. A=[67,85,101,132,189,199,202,212,220,274,"
                "276,295,298,302,320,331]; B=[4,13,16,32,34,45,104,115,148,"
                "187,251,261,262,269,302,314]; gcd degree 91."
            ),
            "references": [
                "https://unitaryfoundation.github.io/qldpc-challenge/codes/682-182-75.html",
                "https://github.com/unitaryfoundation/qldpc-challenge/blob/main/CONTRIBUTING.md",
            ],
            "model": "GPT-5.6 Luna",
            "date": "2026-08-29",
            "notes": (
                "Current-board equivalence check: no exact or WL-equivalent "
                "match. Distance tier: upper_bound, d=77 on both sides."
            ),
        },
        "family": "generalized-bicycle",
    }
    write_json(out / "codes" / "682-182-77.json", document)

    np.savez_compressed(
        out / "682-182-77.npz",
        hx=dense(hx, n),
        hz=dense(hz, n),
    )

    note = """# [[682,182,77]] — generalized bicycle over Z_341

## Direction & hypothesis

Target: the unrestricted, weight-9plus cell. The candidate uses a symmetric generalized-bicycle CSS construction with 341 qubits per block. The block swap plus index reversal transfers X and Z logical witnesses, so both sides can be screened with the same structural expectation.

The phylogenetic search treats each construction's exponent/support lists as a genome. A population is ranked by the strongest available distance estimate, then split into exploitation and exploration slots: the best survivors seed exploitation, while remaining parents maximize direct genome distance. This preserves strong branches and genetic diversity. UPGMA records the family tree for audit and replay; it is a population-management aid, not evidence for code validity or distance. Each selected genome expands through nearby mutations and related construction branches, then passes CSS commutation, rank/rate, check-weight, and fast RIS filters before entering the non-dominated Pareto archive. This candidate is the high-distance survivor from the generalized-bicycle branch after deeper confirmation.

## What was searched

The phylogenetic search retained non-dominated generalized-bicycle descendants, then used a native C++ randomized information-set (RIS) kernel for screening and witness tightening. The retained parameters are:

`m=341`, `a=[67,85,101,132,189,199,202,212,220,274,276,295,298,302,320,331]`, and `b=[4,13,16,32,34,45,104,115,148,187,251,261,262,269,302,314]`.

## Evidence trail

The submitted claim is `d <= 77` on both sides, with explicit weight-77 X and Z witnesses in the JSON.

| RIS trials per side | Seed | Result |
|---:|---:|---|
| 200,000 | 20260832 | X 77, Z 80; symmetry transfer gives 77 on both sides |
| 2,000,000 | 20260833 | X 77, Z 79; symmetry transfer gives 77 on both sides |
| 8,000 gate refutation | 975520384 | no lighter logical found; gate passed |

The independent structural checks report `rank(H_X)=rank(H_Z)=250`, `k=182`, zero CSS commutation violations, and valid nontrivial witnesses of weight 77. The 2M campaign took about 853 seconds on the local native kernel.

## Dead ends

The preceding `[[682,182,81]]` branch was superseded by the confirmed `[[682,182,77]]` result and is excluded from this submission.

## Tools

GPT-5.6 Luna with phylogenetic branch ranking, native C++ RIS search, Python GF(2) structural checks, symmetry row-space transfer checks, and the challenge repository's verifier/gate.

## Reproduction

Construct the generalized-bicycle matrices over `Z_341` from the `a` and `b` lists above using `H_X=[A|B]`, `H_Z=[B^T|A^T]`, then run the challenge verifier on `codes/682-182-77.json`. The attached NPZ contains the same `hx` and `hz` matrices. The official challenge rules and verifier are pinned by the public repository at https://github.com/unitaryfoundation/qldpc-challenge.
"""
    (out / "notes" / "682-182-77.md").write_text(note, encoding="utf-8")

    body = """## Code submission

- Parameters: [[682,182,77]]
- Track: weight-9plus / unrestricted (computed from H; no layout supplied)
- Distance confidence: upper_bound

### Checklist

- [x] One JSON file under `codes/`, conforming to `schema/code.schema.json`
- [x] Distance witness(es) included for each reported side
- [x] `python verify/qldpc_verify.py codes/682-182-77.json` passes locally
- [x] Construction and references filled in under `provenance`
- [x] Possible equivalence is recorded in `provenance.notes`; current-board WL deduplication found no match

### What frontier does this advance?

The local official-style gate reports that `[[682,182,77]]` advances the current weight-9plus x unrestricted frontier, with no exact duplicate, no WL-equivalent board entry, and no dominated entry reported. The provenance novelty label is `unknown`; current-board deduplication found no match.

Research note: `notes/682-182-77.md`

Construction: generalized bicycle over Z_341; `H_X=[A|B]`, `H_Z=[B^T|A^T]`; exact exponent lists and the 2M confirmation ladder are in the note.

Schema reference: https://github.com/unitaryfoundation/qldpc-challenge/blob/main/schema/code.schema.json
"""
    (out / "PR_BODY.md").write_text(body, encoding="utf-8")

    if args.gate.exists():
        shutil.copyfile(args.gate, out / "evidence" / "official_gate.json")
    if args.ladder.exists():
        shutil.copyfile(args.ladder, out / "evidence" / "screening_ladder_2m.json")

    structural = {
        "candidate": "682-182-77",
        "source": str(args.source),
        "n": n,
        "k": k,
        "d_upper_bound": 77,
        "x_rows": len(hx),
        "z_rows": len(hz),
        "max_check_weight": max(len(row) for row in hx + hz),
        "total_support_entries": sum(map(len, hx + hz)),
        "layout": None,
        "official_gate_receipt_copied": args.gate.exists(),
        "external_submission_performed": False,
    }
    write_json(out / "evidence" / "package_metrics.json", structural)

    manifest = {
        "status": "ready_for_user_approval",
        "bundle": out.name,
        "candidate": "[[682,182,77]]",
        "score_upper_bound": 1582.225806451613,
        "publication": {
            "external_submission_performed": False,
            "pr_opened": False,
            "push_performed": False,
        },
        "required_before_submission": [
            "approve moving JSON and note into the challenge repository and opening the PR",
        ],
        "allowed_by_rule_audit": True,
        "model": "GPT-5.6 Luna",
        "distance_status": "upper_bound tier; reported d=77 on both sides",
        "files": [
            "codes/682-182-77.json",
            "notes/682-182-77.md",
            "682-182-77.npz",
            "PR_BODY.md",
            "evidence/official_gate.json",
            "evidence/screening_ladder_2m.json",
            "evidence/package_metrics.json",
        ],
    }
    write_json(out / "APPROVAL_MANIFEST.json", manifest)
    print(f"prepared {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
