# V3 experiment

Focused smoke campaign after six fixes.

| Run | Result |
|---|---|
| train, 4 seeds × 16 mutations | 64 unique children; 28 `gcd_escape`; 20 GCD families; 1 deep survivor, d≤92 |
| matched cache rerun | 64/64 family cache hits; same d≤92 survivor |
| algebraic-family test, isolated cache | 62 unique children; 57 `gcd_escape`; 25 GCD families; 1 deep survivor, `[[682,132]]`, d≤98 |
| candidate-wide stop smoke | target80 hit d≤79; 262,144 requested/side → 53,927 X + 53,909 Z trials |

Exact-label model: 4,291 train rows, 539 validation, 76 test; 6,119/5,283/1,538 censored rows excluded. Test AUC 0.733, AP 0.907. Family split has 1,468 groups with zero group overlap.

All distances remain randomized witness upper bounds. No >1542 claim admitted.
