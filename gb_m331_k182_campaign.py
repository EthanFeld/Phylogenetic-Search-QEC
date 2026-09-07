from __future__ import annotations

"""Immediate x+1 child: square-free Z_331 [[662,182,*]] campaign."""

import json
from pathlib import Path

import gb_m331_k180_campaign as parent


parent.Q = 91
parent.K = 182
parent._configure()
parent.core.CHECK_CAP = 48
parent.core._candidate_doc = parent._candidate_doc


def run(args):
    report = parent.core.run(args)
    report["kind"] = "gb_m331_k182_squarefree_factor_phylogeny_campaign"
    report["algebra"] = {
        "factor_spectrum": {"1": 1, "30": 11},
        "degree91_factor_subsets": 165,
        "square_free": True,
        "lineage": "x+1 immediate child of q90/k180 branches",
        "exact_parent_parity": "even; q91 includes x+1",
    }
    report["phylogenetic_hypothesis"] = {
        "exploitation": "add x+1 to d76 q90 plateau, buying two logical qubits",
        "score_threshold": "d76 beats 1582.226 at n662,k182",
        "exploration": "all 165 x+1 plus three degree-30 factor branches",
        "selection": "generic immediate-child escape, annihilator gate, replicated RIS",
    }
    Path(args.out, "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


parent.run = run


if __name__ == "__main__":
    parent.main()
