# Search v2

`ml_mutation_search_v2.py` addresses six search failures:

1. Canonical cyclic representatives remove independent block translations before RIS.
2. Candidates are grouped by `(m, gcd(A,B,x^m-1))`; one ideal RIS run is reused, with each child independently checked as a CSS logical.
3. Mutation uses sparse ideal combinations, divisor-factor products, and support rewiring; rotations alone are excluded.
4. ExtraTrees classifier trains on fixed-256-trial survival labels. Train-only fit; validation/test metrics reported.
5. Seed split controls ancestry: test campaign seeds never enter model fit or train campaign.
6. RIS records requested, actual, and cumulative per-side trials. CSS weights count only `css_ok` witnesses.

All scores remain provisional upper bounds. GCD/CSS witnesses require independent verification before admission.

## v3 audit fixes

- Persistent `(m, gcd(A,B,x^m+1))` witness cache; every cached witness rechecked against each child CSS.
- Native CSS target stop now shares one atomic flag across X/Z sectors. Chunked Python shards share manager event; reports track X, Z, and sector-shot totals separately.
- Fixed-budget labels require exact budget and use minimum observed witness; shorter or missing runs become censored and are excluded.
- Corpus regrouping assigns train/validation/test by algebraic family, with split-specific queues and isolated cache for holdout evaluation.
- Mutation uses XOR for GF(2) sparse masks and samples alternate same-degree modulus divisors through `gcd_escape`.
- Child mutations retain parent ML probability/priority, enabling measured ML selection.

## v4 search corrections

- Default corpus now uses `corpus_repaired_algebraic`; campaign aborts when one algebraic family crosses train/validation/test.
- Curve labels can target a long per-side horizon (`--survival-horizon`, default 65,536); missing horizon remains right-censored.
- Mutation children receive fresh model scores from their own supports; parent score retained only as provenance.
- Deep selection caps GCD-family concentration (`--deep-per-gcd-family`) before filling remaining slots.
- Family witness cache stores trial depth/backend and extends non-refuting prefixes with fresh seeds.
- Validation receipts store candidate/source hashes, exact X/Z totals, requested total, remainder chunk, and clean-completion status.
- GPU benchmark records witness quality and target-hit parity alongside speed; truncated rank-cap GPU remains screening-only.

## Persistent candidate cycle

`candidate_cycle.py` now owns a durable GPU-screen -> exact-validation loop:

- `enqueue` deduplicates candidates by SHA-256 and records per-candidate receipt/log paths.
- `run` launches one exact native RIS validator at a time, target `d=77` for `[[682,182]]`, exact 20M trials per side, `--ideal-trials 0` after external GCD screening.
- Concurrent acquisition launches `ml_mutation_search_v2.py` with `--gpu-ideal --gpu-css --gpu-css-rank-cap 64`, ingests campaign finalists, then repeats while pending depth stays below the cap.
- GPU results remain acquisition rankings only; queue terminal states come from independently checked CSS witnesses in validation receipts.

Example:

```powershell
python candidate_cycle.py enqueue results\campaign\candidate.json
python candidate_cycle.py run --queue results\candidate_validation_queue\queue.json
```
