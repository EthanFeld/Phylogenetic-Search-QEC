from __future__ import annotations

"""Write held-out metrics for v2 fixed-budget survival classifier."""

import argparse
import json
from pathlib import Path

from ml_mutation_search_v2 import _train_fixed_model


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--codes", type=Path, default=Path("results/corpus_repaired/codes.jsonl"))
    p.add_argument("--runs", type=Path, default=Path("results/corpus_repaired/runs.jsonl"))
    p.add_argument("--budget", type=int, default=256)
    p.add_argument("--out", type=Path, default=Path("results/ml_mutation_search_v2_model_eval.json"))
    args = p.parse_args()
    codes = [json.loads(x) for x in args.codes.read_text(encoding="utf-8").splitlines() if x.strip()]
    runs = [json.loads(x) for x in args.runs.read_text(encoding="utf-8").splitlines() if x.strip()]
    _model, meta, _names = _train_fixed_model(codes, runs, int(args.budget))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
