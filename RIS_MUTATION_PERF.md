# RIS / mutation efficiency

Implemented:

- C++ RIS uses reference pivot rows, `nth_element`, and cached detector signatures for pair/triple shells.
- Aggressive GB mutation canonicalization checks only support-anchored shifts, not all 341 shifts.
- M345 mutation caches orbit/GCD results and computes cycle overlap with integer masks.
- K132 mutation removes duplicate common-GCD evaluation.
- GCD queue screens and trial ladders use process workers; output order remains deterministic.

Checks:

- Orbit equivalence: 0 mismatches in 1,000 random cases.
- Cycle-profile equivalence: 0 mismatches in 100 random cases.
- Native RIS witness verification: X/Z both passed.
- Mutation orbit microbenchmark: 1.91 s → 0.096 s for 5,000 words (~20×).
- Parallel screen smoke test: 4 candidates / 4 workers completed successfully.

Distance semantics unchanged: RIS remains randomized witness discovery; no-hit remains inconclusive.
