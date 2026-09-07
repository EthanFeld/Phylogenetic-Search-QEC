from __future__ import annotations

"""Build one auditable, stage-only archive from campaign and validator output."""

import argparse
import json
from pathlib import Path

import numpy as np

from inverse_design_core import build_product_checks, candidate_groups_of_order
from pareto_campaign import frontier


DEFAULT_CAMPAIGNS = (
    Path("results/campaign_01/campaign.json"),
    Path("results/campaign_220_focus/campaign.json"),
    Path("results/campaign_220_mutation/campaign.json"),
    Path("results/campaign_220_coevolution/campaign.json"),
    Path("results/campaign_350_coevolution/campaign.json"),
    Path("results/campaign_250_300_beam_small/campaign.json"),
    Path("results/campaign_300_recovery/campaign.json"),
    Path("results/campaign_300_beam_fast/campaign.json"),
    Path("results/campaign_350_beam_fast/campaign.json"),
    Path("results/campaign_350_beam_corrected/campaign.json"),
    Path("results/campaign_300_wide_beam/campaign.json"),
    Path("results/campaign_350_wide_beam/campaign.json"),
    Path("results/campaign_300_phylo_history/campaign.json"),
    Path("results/campaign_300_phylo_v2/campaign.json"),
    Path("results/campaign_300_phylo_anchor/campaign.json"),
    Path("results/campaign_350_phylo_two_round/campaign.json"),
    Path("results/campaign_350_weight15_phylo/campaign.json"),
    Path("results/campaign_300_weight15_phylo/campaign.json"),
    Path("results/campaign_350_weight15_exploit/campaign.json"),
    Path("results/campaign_350_weight21_pilot/campaign.json"),
    Path("results/campaign_350_weight21_phylo/campaign.json"),
    Path("results/campaign_350_weight21_hunt2/campaign.json"),
    Path("results/campaign_350_weight21_focused3/campaign.json"),
    Path("results/campaign_350_weight27_pilot/campaign.json"),
    Path("results/campaign_350_weight21_pool4_pilot/campaign.json"),
    Path("results/campaign_350_weight21_pool4_scale/campaign.json"),
    Path("results/campaign_350_weight27_pool4_scale/campaign.json"),
    Path("results/campaign_350_weight27_pool8_hunt2/campaign.json"),
    Path("results/campaign_350_weight27_adaptive4/campaign.json"),
    Path("results/campaign_300_weight27_exploit4/campaign.json"),
    Path("results/campaign_300_weight27_hunt8/campaign.json"),
    Path("results/campaign_300_weight27_radius2_4/campaign.json"),
    Path("results/campaign_250_weight27_hunt8/campaign.json"),
    Path("results/campaign_250_weight27_radius2_4/campaign.json"),
    Path("results/campaign_300_weight27_deep3_4/campaign.json"),
    Path("results/campaign_350_weight27_deep3_4/campaign.json"),
    Path("results/campaign_220_weight27_hunt8/campaign.json"),
    Path("results/campaign_150_weight27_hunt8/campaign.json"),
    Path("results/campaign_accel_smoke/campaign.json"),
    Path("results/campaign_20260829_lit_accel/campaign.json"),
    Path("results/campaign_20260829_phylo_highscore1/campaign.json"),
    Path("results/campaign_20260829_phylo_long1/campaign.json"),
    Path("results/campaign_20260829_phylo_radius2_long/campaign.json"),
    Path("results/campaign_20260829_phylo_quality75_long/campaign.json"),
    Path("results/campaign_20260829_ml_branchfill_n350/campaign.json"),
    Path("results/campaign_20260829_ml_branchfill_n300/campaign.json"),
    Path("results/campaign_20260829_scorepush_n350_pilot/campaign.json"),
    Path("results/campaign_20260829_scorepush_n300_pilot/campaign.json"),
    Path("results/campaign_20260829_hardhunt_n350/campaign.json"),
    Path("results/campaign_20260829_oos_ml_branchfill_n300/campaign.json"),
    Path("results/campaign_20260829_scorepush_n350_compact/campaign.json"),
    Path("results/campaign_20260829_concatfollow_n350/campaign.json"),
)


def _supports(matrix):
    return [np.flatnonzero(row).astype(int).tolist() for row in matrix]


def _checks_for_row(row):
    groups = candidate_groups_of_order(int(row["group_order"]))
    group = groups[int(row["group_index"])]
    A = tuple(tuple(int(v) for v in xs) for xs in row["A"])
    B = tuple(tuple(int(v) for v in xs) for xs in row["B"])
    hx, hz = build_product_checks(A, B, group)
    return {"X": _supports(hx), "Z": _supports(hz)}


def _check_key(checks):
    return json.dumps(checks, sort_keys=True, separators=(",", ":"))


def _receipt_index(root: Path):
    index = {}
    full = []
    for path in sorted(root.glob("results/**/**/*_official.json")):
        try:
            receipt = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        candidate_path = path.with_name(path.name.replace("_official.json", ".json"))
        if not candidate_path.exists():
            continue
        try:
            candidate = json.loads(candidate_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        checks = candidate.get("checks")
        if not isinstance(checks, dict):
            continue
        rel_receipt = path.relative_to(root).as_posix()
        item = {"path": rel_receipt, "receipt": receipt}
        index[_check_key(checks)] = item
        full.append(item)
    return index, full


def _official_summary(item):
    if item is None:
        return None
    r = item["receipt"]
    gates = r.get("gates", {})
    verify = gates.get("verify", {})
    refute = gates.get("refute", {})
    dedup = gates.get("dedup", {})
    novelty = gates.get("novelty", {})
    validator = r.get("validator", {})
    return {
        "receipt": item["path"],
        "passed": bool(r.get("passed", False)),
        "verify_ok": bool(verify.get("ok", False)),
        "failed_checks": verify.get("failed_checks", []),
        "refuted": bool(refute.get("refuted", False)),
        "refutation_seed": refute.get("seed"),
        "exact_duplicate_of": dedup.get("exact_duplicate_of"),
        "wl_equivalent_of": dedup.get("wl_equivalent_of"),
        "board_advancing": novelty.get("board_advancing"),
        "dominated_by": novelty.get("dominated_by", []),
        "weight_class": verify.get("weight_class"),
        "locality_class": verify.get("locality_class"),
        "validator_source_sha256": validator.get("source_sha256"),
    }


def _summary(row, source, official):
    item = dict(row)
    item.pop("target", None)
    item["observed_in"] = [source]
    official_summary = _official_summary(official)
    item["evidence"] = {
        "distance_confidence": "upper_bound",
        "explicit_x_witness": bool(row.get("witness_x")),
        "explicit_z_witness": bool(row.get("witness_z")),
        "confirm_mode": row.get("confirm_mode"),
        "confirm_trials_per_sector": row.get("confirm_trials_per_sector"),
        "official": official_summary,
    }
    item["regulation"] = {
        "stage_only": True,
        "submission_sent": False,
        "official_gate_status": (
            "refuted" if official_summary and official_summary["refuted"] else
            "passed" if official_summary and official_summary["passed"] else
            "failed" if official_summary else "not_run"
        ),
        "board_claim_allowed": bool(official_summary and official_summary["passed"] and
                                      official_summary["board_advancing"]),
        "distance_is_not_proven": True,
        "provenance_is_self_reported": True,
        "literature_novelty": "unverified",
    }
    return item


def build_archive(campaign_paths, out: Path):
    root = Path.cwd()
    receipt_index, receipts = _receipt_index(root)
    unique = {}
    campaigns = []
    successful_rows = 0
    for path in campaign_paths:
        data = json.loads(path.read_text())
        rel = path.relative_to(root).as_posix() if path.is_absolute() else path.as_posix()
        campaigns.append({"path": rel, "campaign": data.get("campaign", {})})
        for row in data.get("results", []):
            if row.get("status") != "ok":
                continue
            successful_rows += 1
            key = row["semantic_hash"]
            existing = unique.get(key)
            if existing is None:
                unique[key] = {"row": dict(row), "sources": [rel]}
            elif rel not in existing["sources"]:
                existing["sources"].append(rel)

    finds = []
    for key, bundle in unique.items():
        row = bundle["row"]
        official = receipt_index.get(_check_key(_checks_for_row(row)))
        record = _summary(row, bundle["sources"][0], official)
        record["observed_in"] = bundle["sources"]
        finds.append(record)
    finds.sort(key=lambda r: (r["n"], -r["k"], -r["d_upper"], r["semantic_hash"]))

    frontier_rows = frontier([x["row"] for x in unique.values()])
    frontier_keys = {x["semantic_hash"] for x in frontier_rows}
    frontier_findings = [x for x in finds if x["semantic_hash"] in frontier_keys]
    board_ready = [x for x in finds if x["regulation"]["board_claim_allowed"]]
    official_passed = sum(x["regulation"]["official_gate_status"] == "passed" for x in finds)
    official_refuted = sum(x["regulation"]["official_gate_status"] == "refuted" for x in finds)
    archive = {
        "schema_version": "1.0",
        "kind": "stage_only_qldpc_research_archive",
        "regulation": {
            "challenge_rules": "https://github.com/unitaryfoundation/qldpc-challenge/blob/main/TRACKS.md",
            "official_validator": "https://github.com/unitaryfoundation/qldpc-challenge/blob/main/verify/validate_candidate.py",
            "stage_only_workflow": "https://github.com/unitaryfoundation/qldpc-challenge/blob/main/research/AUTORESEARCH.md",
            "distance_policy": "RIS values are upper bounds with witnesses; no-hit is not proof.",
            "board_policy": "Only passed official receipts with board_advancing=true are board-ready.",
            "submission_sent": False,
            "provenance_note": "Self-reported stage-only provenance; literature novelty remains unverified.",
        },
        "campaigns": campaigns,
        "counts": {
            "campaigns": len(campaigns),
            "successful_rows": successful_rows,
            "unique_semantic_candidates": len(finds),
            "observed_frontier": len(frontier_findings),
            "official_receipts": len(receipts),
            "official_passed_unique": official_passed,
            "official_refuted_unique": official_refuted,
            "board_ready_receipted": len(board_ready),
        },
        "finds": finds,
        "observed_frontier": frontier_findings,
        "board_ready": board_ready,
        "official_receipts": receipts,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(archive, indent=2) + "\n")
    return archive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/regulated_findings.json"))
    ap.add_argument("campaigns", type=Path, nargs="*", default=list(DEFAULT_CAMPAIGNS))
    args = ap.parse_args()
    archive = build_archive(args.campaigns, args.out)
    print(json.dumps({"out": str(args.out), "counts": archive["counts"],
                      "board_ready": [{"n": x["n"], "k": x["k"], "d": x["d_upper"],
                                       "semantic_hash": x["semantic_hash"]}
                                      for x in archive["board_ready"]]}, indent=2))


if __name__ == "__main__":
    main()
