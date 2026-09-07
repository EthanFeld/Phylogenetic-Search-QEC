# Search improvement and GPU RIS plan

Reviewed 2026-09-06. Scope: review, diagnostic benchmarks, implementation and validation.

Recommendation: repair divisor generation and evidence accounting first. Prototype packed CUDA RIS next; integrate only after measured end-to-end gains. Current GPU scout remains a scout, not full RIS.

## Implementation status (2026-09-06)

Implemented and tested in this pass:

- divisor DP now uses immutable per-factor snapshots and final degree/divisibility checks;
- invalid candidates are rejected before CPU RIS, and final exported documents pass an independent structural gate;
- non-divisible shard budgets preserve the exact requested per-side total;
- Torch is lazy-loaded only inside the optional GPU scout path;
- single-worker native RIS now receives the shared cancellation flag;
- GPU scout retains equal-weight masks and reports `combination_samples`, explicitly distinct from RIS trials.

Checks passed: four divisor cases had zero invalid proposals; Python compilation passed; native rebuild and regression test passed; 10 requested trials split across 3 shards produced exactly 10 X and 10 Z trials; 10,000 CUDA combination samples per side returned independently verified candidates. Full CUDA randomized-RREF RIS prototype now builds, but routing gate remains unmet.

CUDA continuation (same date): CUDA 13.3 installed through winget; `sm_86` DLL prototype now exists in `qldpc_gpu.cu` with Python bridge `gpu_fast.py`. It performs packed randomized-column elimination, pair shell, detector filtering, exact completed-trial counts, and batch early stop. Twenty-four random differential cases passed span/logical verification; 682-qubit CSS smoke witnesses passed independent checks. Matched 2,048-trial medians: CSS GPU 2.934 s vs 16-thread CPU 0.421 s (0.144x); ideal rank 66/85/91 GPU speed ratios 0.788x/1.064x/1.185x. GPU routing fails the >=2x gate; CPU remains default. Benchmark artifact: `results/gpu_ris_benchmark_cuda.json`.

GPU optimization pass: detector signatures cached in shared memory; pivot/flag scans parallelized; one global witness publish per trial; CSS bases cached on host; 512 threads/block selected by sweep. Optimized CSS median: 0.379 s GPU vs 0.373 s CPU at 2,048 trials/side, pair depth 24; pair depths 12 and 1 gave GPU/CPU ratios 1.11x and 1.24x. Ideal RIS now passes routing gate over five repeats: rank 66/85/91 speedups 4.38x/5.46x/5.89x. `ml_mutation_search_v2.py --gpu-ideal --workers 1` routes only ideal-family RIS to one CUDA owner; CSS remains CPU. Artifacts: `results/gpu_ris_benchmark_cuda_optimized.json`, `results/gpu_ideal_benchmark_cuda_optimized.json`.

Latest GPU pass: forward-echelon reduction is now explicit (safe for upper-bound witnesses; it skips backward Gauss-Jordan traffic), shared rows are transposed for coalesced row XORs, hot XOR loops are partially unrolled, and the tuned block size is 448. CSS witnesses from random and candidate smoke tests remain independently valid. Full-rank sustained rate on the RTX 3050 is about 16.5k sector shots/s at pair depth 1 and about 9.5k at pair depth 24: 20M/side therefore remains about 20.2–35.1 minutes, so the full-RIS <15-minute gate is still false. Artifacts: `results/gpu_ris_benchmark_transposed_bitfast.json` and the sustained run recorded in the turn ledger.

An explicit `rank_cap` experiment is available in `gpu_fast.py` and `benchmark_ris_backends.py`. It is labeled truncated RIS, not full validation: cap 64 reached about 35.0k sector shots/s at pair depth 24, projecting 9.5 minutes/side, while witnesses stayed CSS-valid. Use it only as a high-throughput screening stage; any survivor still requires full-rank fresh CSS validation. Cap 128/96 did not reliably meet the 15-minute gate under laptop thermal throttling.

## Evidence from past results

| Evidence | Finding | Implication |
|---|---|---|
| 23 `ml_mutation_search_v3*/*css*20m*.json` reports | All report target refutation; 81,972,332 actual X+Z sector trials | Short-budget survival poorly predicts the required validation outcome. These are runs, not necessarily distinct code equivalence classes. |
| v3l candidate ledger | 1,450/1,639 candidates exceed row weight 32 | Historical candidate volume substantially overstates admissible exploration. |
| v3m structural recheck | All 18 exported candidates fail row-weight limits | Its larger survivor count is not progress. Preserve artifacts as invalid examples, exclude from positive labels. |
| v3n candidate ledger | 252 generated, **246 unique**, 45 GCD families; zero rows above weight 32 | Corrects previous summaries calling all 252 unique. |
| v3n vs saved v3f–v3m ledgers | 110/246 already equivalent under existing independent-block translation canonicalization | At least 44.7% repeated exploration; persistent dedup needed. Stronger equivalences may reveal more. |
| v3n final candidates | Six structurally valid; one GCD refutation, five CSS refutations at d=86–89, required d=90 | No admitted >1542 specification. |
| v3n model report | Test AUC 0.59575; AP 0.82523 vs positive prevalence 0.79882 | Weak discrimination; high AP partly reflects easy positive base rate. |
| v3n GPU reports | 10M total fixed-basis samples, best verified d=101–104, zero target refutations | Sampling throughput is not RIS throughput or useful acceleration. |

Sources: [v3n campaign](results/ml_mutation_search_v3n_test/campaign.json), [GCD ladder](results/ml_mutation_search_v3n_test/gcd_ladder.json), [v3m campaign](results/ml_mutation_search_v3m_test/campaign.json), candidate/proxy ledgers and CSS reports in the named directories.

The CPU CSS reports record successful logical checks but omit the discovered CPU witness supports. Their historical refutations cannot be independently replayed from these reports alone. Recover supports from matching artifacts where available; otherwise schedule narrow reproductions. Do not treat stored `checks.ok` as a replacement for an auditable witness.

## Largest bug: invalid divisor proposals

`ml_mutation_search_v2.py::_degree_divisor_candidates` snapshots dictionary entries but retains references to mutable value lists. Updates contaminate later source buckets during the same factor pass, effectively allowing excess multiplicity.

Exact diagnostic: test `mod(x**m + 1, proposed_g) == 0` over GF(2).

| m | Requested degree | Current pool | Non-divisors | Pool using immutable per-pass snapshots | Non-divisors after snapshot |
|---:|---:|---:|---:|---:|---:|
| 341 | 66 | 256 | 250 | 256 | 0 |
| 331 | 91 | 256 | 96 | 165 | 0 |
| 337 | 85 | 256 | 186 | 256 | 0 |
| 341 | 91 | 256 | 255 | 256 | 0 |

The snapshot variant was tested in memory only. Existing exact common-GCD gates still apply; these invalid proposals establish wasted generation, not unsoundness of independently verified witnesses.

## Six implementation priorities

| Order | Change | Acceptance check |
|---:|---|---|
| 1 | Fix divisor DP; snapshot every source list, respect irreducible multiplicities, validate degree/divisibility; remove prefix-biased pool truncation through randomized or stratified sampling | Zero invalid proposals; small-modulus exhaustive reference agreement; coverage across distinct factors and repeated-factor cases |
| 2 | Consolidate validation into one resumable runner with complete witness/provenance records and immediate structural rejection | Replay every emitted witness; exact 20,000,000 trials **on each side** for a clean result; verified subtarget witness permits early rejection |
| 3 | Generate sparse, novel codes within admissible ideals; persist dedup and rejection history | All exported codes pass structure/rank/commutation; no repeat compute without a declared extra-budget experiment; higher novel valid codes/second |
| 4 | Rebuild labels and child ranking around long-budget survival; reserve a fresh evaluation split | Beat simple geometry/diversity baselines on frozen families at equal wall time; calibrated survival curves; no adaptive test reuse |
| 5 | Improve CPU RIS scheduling, cancellation, allocation and measurement | Lower complete-candidate wall time and cancellation latency without changing per-trial work or witness validity |
| 6 | Prototype, benchmark, then conditionally integrate CUDA RIS | Exact differential checks plus >=2x median end-to-end speedup on intended workloads; no material detection-quality loss |

### 1–3: generation and trustworthy validation

- Apply full structural checks before expensive work and again before export. `parallel_css_early_stop.py` currently computes `structural_ok` but continues on failure.
- Reuse `gb_ris_validation_runner.py`'s atomic checkpoints and saved best supports. Strengthen resume identity: code-content hash, binary/source hash, seed namespace, trial range, pair/triple depths, backend version. Its current `clean_budget_complete` means requested chunks completed, not necessarily 20M on each side.
- Partition budgets with quotient **and remainder**. Current parallel runner drops eight trials for 20M/12. Record exact X/Z counts, verified supports, monotone curves, wall time, and cancellation reason. A missing witness or cancelled run is not successful validation.
- Preserve original artifacts; annotate invalid or superseded runs in a new ledger. `PLAN_PROGRESS.json`'s 100% must refer only to completed tasks, never achievement of the >1542 objective.
- Use existing `cpp_fast.cyclic_ideal_mine` / `cyclic_ideal_hillclimb` to find sparse words in valid divisor ideals. Canonicalize shift orbits **before** filling sparse pools; existing pools waste slots on shifts of the same words.
- Admit A/B pairs only with nonempty supports, required exact k and total row weight <=32; preserve applicable n/support-budget limits. Profile the current lower support cutoff of 20 rather than treating it as a challenge rule.
- Build mate pools by compatible geometry and algebraic constraints. Use a diverse seed set for family exploration and several representatives within selected families for mating. Record **both** parents and inherited evidence.
- `affine_combo`, quotient `support_rewire` and quotient `cross_swap` currently implement the same algebraic transformation: multiplication by g distributes over XOR and cyclic rotation. Consolidate these aliases; quota counts are not independent search diversity.
- Add `(m, representation, canonical A/B)` to persistent keys; current support hash/global dedup omit m. Extend proven equivalences to block swap and common invertible exponent multipliers, with differential equivalence tests.
- Track rejection reasons and accepted novel children by operator. Allocate mutation budget by valid novelty and later survival per second, not attempted children. Recheck transferred family witnesses on each candidate before pruning; one CSS refutation does not automatically refute its entire GCD family.

### 4: ML and distance-vs-trials evidence

- Keep each stochastic run/replicate distinct. Do not merge minima at a nominal budget across unequal replicate counts and call that equal-budget evidence.
- Retain actual X/Z trial counts, kernel/algorithm, RNG ranges and pair/triple depths. GCD trials, GPU combination samples, and full CSS RIS trials require separate counters.
- Model time-to-first verified subtarget witness. A clean incomplete run is right-censored; a verified early refutation is an observed failure. Use checkpoint intervals when exact event time is unavailable. Handle X/Z dependence empirically rather than multiplying independent probabilities without evidence.
- Predict survival through remaining 20M/side budget; retain a calibrated monotone distance-vs-log-trials baseline. Report finite-budget witness forecasts and uncertainty, never estimated exact distances. Fix `build_corpus.py` treating every non-early-stopped run as terminal even when below forecast budget.
- Retrain from newly attached refutations. Current exact-512 labels omit most runs and do not distinguish 20M survivors. Score actual **children**: current mutations copy parent probability/priority even after changing GCD family.
- Compare random diverse selection, geometry-only selection, survival regression, and a tree model before larger neural models. Include factor pattern, ideal witness history, support overlap, k/n and measured search cost; generate features at the decision-time cutoff.
- Repeated v3 test-driven tuning makes that split development data. Freeze fresh algebraic/equivalence groups and seed streams before comparing methods; use the same historical cache snapshot across arms. The current shared cache is valuable operational evidence but not fresh holdout discovery.
- Run matched exploratory campaigns on the same admissible seed set, initially three RNG seeds per arm and a fixed wall budget. Main metrics: novel valid candidates/hour, GCD/CSS rejection cost, survival at 64K/1M/20M, and verified target survivors. Proxy AUC alone does not select the winner.

### 5: CPU baseline and immediate efficiency

New read-only benchmark: candidate `v3n/candidate_0002_682_132_95.json`; 1,024 RIS trials per side, no early stop, three seeds 93010–93012, pair-only mode. Both returned witnesses checked independently on every repeat. One 64-trial warmup used seed 93001, four threads, pair depth 24.

| Native threads | Pair depth | Median seconds | X+Z sector trials/s |
|---:|---:|---:|---:|
| 2 | 1 | 0.7390 | 2,771 |
| 2 | 24 | 0.9027 | 2,269 |
| 4 | 24 | 0.5700 | 3,593 |
| 8 | 24 | 0.3639 | 5,628 |
| 16 | 24 | 0.2574 | 7,958 |

Separate ideal benchmark: 66x341 basis, 16,384 pair-24 trials, seed 93022, 1.0232 s (~16,012 trials/s), returned witness passed CSS check.

This is a short, one-code diagnostic, not a sustained throughput guarantee. The extra pair-24 work accounts for ~18% of two-thread runtime; eliminating only that increment would yield ~1.22x. Remaining work includes elimination, RNG, individual-row scoring and setup; profile these separately before assuming an exact RREF percentage. At measured 16-thread rate, 40M total sector trials extrapolate to ~84 minutes, excluding sustained thermal effects.

CPU work:

- Prefer one candidate process and a bounded native thread pool. Benchmark 8/12/16/20 threads on this hybrid CPU. Previous five-candidate x12-shard x4-thread configuration could request 240 native workers on a 20-logical-CPU laptop.
- Load Torch only inside the optional GPU process; current top-level import loads it in every spawned CPU shard and manager. Machine has ~16 GB host RAM; observed memory exhaustion warrants removing this duplication.
- Pass `external_stop` through the one-worker branch; it currently forwards `nullptr`. Use one native stop domain per candidate. Validate stopping witnesses before final rejection.
- Tune batches to measured 0.1–1 s latency; current 262,144-trial chunks delay cross-process cancellation. Record post-hit work explicitly; old aggregate counts cannot determine exactly how much was avoidable.
- Reuse basis/setup and signature buffers. Trial-invariant detector pairing can also be represented as signatures transformed alongside row operations; benchmark versus current per-trial dot products. Keep an untuned CPU reference for differential checks.

## GPU port: concrete prototype and decision gate

Hardware confirmed locally: RTX 3050 Laptop GPU, 4 GB VRAM, compute capability 8.6, 16 SMs; i7-12700H, 14 physical /20 logical cores. Torch 2.13.0+cu130 can run CUDA. CUDA 13.3 toolkit now installed; `nvcc` builds `sm_86` successfully.

Current `gpu_ris_scout.py` samples 1/2/3/4/6/8 rows from one fixed nullspace basis. It performs no randomized RREF. The scout now retains distinct masks, filters zero masks from its returned set, and reports `combination_samples`; these samples remain separate from full RIS trials. Full RREF prototype is separate `gpu_fast.py`; benchmark before routing.

Implemented files: `qldpc_gpu.cu`, `gpu_fast.py`, `build_gpu.py`, `benchmark_ris_backends.py`. C ABI uses packed rows; CPU validation workers remain Torch-free.

1. Establish a compatible CUDA compiler/runtime and MSVC combination; build `sm_86`. First port classical ideal RIS (rank ~66–91, n~331–341), then CSS (rank ~407, n~682). Ideal success alone is not evidence of CSS speedup.
2. Prepare independent full kernel bases and logical detector signatures on CPU once. Keep packed data on device. Each GPU trial uses a fresh column permutation, full GF(2) elimination, single-row plus identical light-row pair/triple scoring, and nonzero logical-signature filtering.
3. Start with one block per trial and 128/256-thread variants; parallelize pivot selection and row XOR within each block. Batch independent trials across blocks. Prototype a warp-per-trial variant for small ideal bases if block synchronization dominates. Use packed integer XOR/popcount operations throughout.
4. Prototype uint32 layouts. For the 407x682 CSS kernel: 407x22x4 = 35,816 bytes; 132 detector bits add 8,140 bytes. About 43 KiB before permutation/scratch/padding. Padding to 32 words raises this footprint; measure occupancy and bank conflicts before choosing layout. Ampere 8.6 has up to 100 KiB shared memory/SM and 99 KiB/block with explicit opt-in above 48 KiB; this makes occupancy a real constraint. [NVIDIA Ampere guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
5. Keep pivot loops within a kernel; avoid a Python-launched kernel for every pivot. Return only completed-trial counters and best verified-candidate masks per batch. Batching and resident data reduce launch/transfer costs; include those costs in measured end-to-end speed. [NVIDIA CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
6. Use counter-indexed RNG keyed by code/side/run/trial, unbiased permutations, deterministic tie rules and no repeated trial ranges on resume. Publish best masks and completed counters without races. Partial/aborted trials never count as completed RIS trials. Batch durations must remain comfortably below Windows display-watchdog limits; do not disable watchdog protection.

### Required tests and go/no-go

- Correctness: supplied identical permutations must produce matching CPU/GPU RREF rank, rowspace, logical signatures and minimum single/pair/triple weight. Exercise dependent rows, padding, zero rows, word-boundary n, no logical qubits, and nontrivial stabilizers. Independently verify all reported CSS witnesses.
- Accounting: zero/one trial, non-divisible batch budgets, early stop on either side, cancellation, restart, and exact 20M per-side clean completion. Results survive process interruption through atomic checkpoints.
- Performance matrix: ideal ranks 66/85/91; CSS n=662/674/682 with low and high k; batch sizes 32/128/512/2048, pair depths 12/24, triples separately. Compare against the best sustained CPU configuration, including startup, transfers, CPU verification, and cancellation tail. Run at least five repeats and a sustained 10-minute finalist test with thermals and peak memory recorded.
- Quality: same-permutation differential correctness first; then multiple fresh RNG runs comparing distance-vs-trials and time-to-refutation curves. Do not compare GPU XOR samples against CPU RIS trials.
- Integrate GPU only where median end-to-end speedup >=2x and detection behavior agrees under matched work, with no >10% p95 slowdown for routed workloads. Compare concurrent CPU+GPU too; shared laptop power/thermal limits may reduce either device's throughput.
- If CUDA benefits only ideal RIS, route only that stage. If it misses the gate after two layout/batching iterations, retain CPU RIS and document the measured result. GPU availability alone does not justify a full rewrite.

## Execution order and completion

### v4 implementation result (2026-09-06)

Implemented split-integrity guard, 65,536-trial curve-censored model labels,
child-level rescoring, GCD-family deep quotas, progressive witness-cache depth,
exact validation receipt accounting, and GPU witness-quality metrics. Fresh
algebraic-isolated validation search produced 3 deep survivors; all remain
provisional. Train-arm family expansion produced 0 proxy survivors. Rank-64
GPU measured 4.50x faster at 512 trials, but GPU returned heavier witnesses in
3/3 matched repeats; retained as screening backend only.

First land priorities 1–2 with focused correctness checks. Then run priority 3/4 ablations while priority 5 establishes sustained CPU timings. Build priority 6 against that baseline; enable by measured workload crossover. Finally validate fresh candidates using **20,000,000 completed RIS trials per side**, stopping on an independently verified witness below `floor(sqrt(1542*n/k))+1`.

Deliverables: repaired proposal/evidence pipeline, auditable run ledger, updated corpus with censoring, matched search report, CPU/GPU benchmark report, and optional GPU routing. Implementation completion and discovery of a >1542 code remain separate outcomes; no percentage of discovery success is currently justified.
