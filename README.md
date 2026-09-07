# Phylogenetic Search for qLDPC Codes

Core ML and search code for prioritizing quantum LDPC code candidates. The ML layer ranks search branches and mutation seeds; validators remain the source of truth for code structure and distance claims.

## Core modules

- `branch_ml.py` — branch-priority model.
- `ml_mutation_search.py` — ML-seeded mutation search.
- `ml_mutation_search_v2.py` — fixed-budget classifier and mutation pipeline.
- `repair_and_rank.py` — corpus repair, feature extraction, and ranking.
- `inverse_design_core.py`, `generalized_bicycle.py`, `gf2_factor.py` — core candidate generation and algebra.
- `hyper_validator.py`, `sparse_tanner_certifier.py` — structural and witness validation.

Supporting campaign, benchmark, and verification scripts remain available in the repository. Generated results, corpora, candidate matrices, logs, native binaries, and local challenge data are intentionally excluded from version control.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

NumPy and scikit-learn cover the core ML path. PyTorch is optional for the neural branch model; native C++/CUDA accelerators are optional and built locally by the existing build scripts.

## Running with local data

Bring your own corpus/results and pass paths explicitly. Example:

```powershell
python branch_ml.py `
  --archive path\to\regulated_findings.json `
  --catalog literature_code_families.json `
  --out results\ml_branch_plan.json
```

Scripts that need an external qLDPC challenge checkout accept `--challenge-root`. The default is the local relative path `challenge_data`; pass the flag to use another checkout.

No experiment data is required or included for source review. Do not commit `results/`, corpora, private challenge files, credentials, or generated binaries.

## Agentic search loop

Run research as a bounded, auditable loop. Keep inputs and outputs under local `results/` only.

```text
local corpus
    ↓
repair_and_rank.py       repair rows, build features, split holdouts
    ↓
branch_ml.py             rank under-covered search branches
    ↓
ml_mutation_search_v2.py generate candidates within fixed budget
    ↓
hyper_validator.py       structural/GF(2)/CSS checks
    ↓
gcd_ladder.py + RIS       cheap gate, then randomized distance witnesses
    ↓
holdout_deep_verify.py    fresh-lineage validation
    ↓
new validated summaries ──→ next local loop
```

Each loop must use fixed budgets, deterministic seeds where practical, fresh holdouts, and explicit output paths. ML ranks candidates; validators decide admissibility. Never feed generated children back into training without preserving lineage and split boundaries.

## Scope

This repository contains research code, not a proof system. Randomized distance searches produce witness upper bounds; final claims require independent structural, algebraic, and official validation.
