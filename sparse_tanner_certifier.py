from __future__ import annotations
"""Exact low-weight CSS logical search using sparse Tanner parity closure.

For a binary check matrix H, a support S is in ker(H) iff every check node has
even degree into S.  If a kernel support is disconnected in the induced Tanner
subgraph, each connected component is itself a kernel support.  Moreover a
non-trivial logical signature is the XOR of the component signatures, so if the
whole support is non-trivial then at least one connected component is
non-trivial.  Therefore an exact search for any logical of weight <= w may be
restricted to connected supports.

The DFS starts from the minimum-index variable in a prospective component.  At
any partial support with odd-check syndrome, a completion must add at least one
new variable incident on each chosen odd check.  Branching on one deterministic
odd check and every available incident variable is complete.  The search uses
only safe lower bounds for pruning; if it exhausts all starts it certifies that
no non-trivial logical of the requested weight exists.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse, json, time
import os
import numpy as np

try:
    import cpp_fast as _cpp_fast
except Exception:  # pragma: no cover - optional native backend
    _cpp_fast = None


@dataclass
class SearchStats:
    status: str
    cap: int
    nodes: int
    starts_done: int
    seconds: float
    witness_support: list[int] | None = None
    witness_weight: int | None = None
    logical_signature: list[int] | None = None
    timed_out: bool = False


def _bit_positions(x: int):
    while x:
        b = x & -x
        yield b.bit_length() - 1
        x -= b


def exact_low_weight_logical(
    Hcheck: np.ndarray,
    dual_logicals: np.ndarray,
    cap: int,
    *,
    time_limit: float | None = None,
    variable_order: str = "degree_desc",
) -> dict:
    """Find, or exactly rule out, a non-trivial logical of weight <= ``cap``.

    ``Hcheck`` is the opposite-type stabilizer check matrix whose kernel
    contains the logical candidates. ``dual_logicals`` is any complete basis
    whose row-wise inner products detect the quotient class. A candidate is
    non-trivial iff its signature against this basis is non-zero.

    Returns ``status='certified_none'`` only after exhaustive search. A timeout
    is explicitly inconclusive.
    """
    H = np.asarray(Hcheck, dtype=np.uint8)
    D = np.asarray(dual_logicals, dtype=np.uint8)
    if H.ndim != 2 or D.ndim != 2 or H.shape[1] != D.shape[1]:
        raise ValueError("incompatible check/logical matrices")
    # Python's bigint Tanner kernel is faster on the large proof trees in the
    # current benchmark; native exact mode remains available explicitly for
    # small codes/experiments without changing the audited default.
    use_native_exact = os.environ.get("QLDPC_NATIVE_EXACT", "0") == "1"
    if (_cpp_fast is not None and _cpp_fast.available() and
            (H.shape[1] <= 128 or use_native_exact) and H.shape[1] <= 384):
        return _cpp_fast.exact_logical(
            H, D, int(cap), time_limit=time_limit,
            variable_order=variable_order)
    # A qubit permutation is proof-neutral but can drastically change the DFS
    # tree.  High-degree variables first tend to close odd checks earlier and
    # expose stronger packing bounds.  Stable sorting makes the proof fully
    # deterministic.  Any returned witness is mapped back to physical indices.
    n0 = H.shape[1]
    if variable_order == "degree_desc":
        perm = np.argsort(-H.sum(axis=0).astype(np.int64), kind="stable")
    elif variable_order in ("identity", None):
        perm = np.arange(n0, dtype=np.int64)
    else:
        raise ValueError(f"unknown variable_order: {variable_order}")
    if not np.array_equal(perm, np.arange(n0)):
        H = H[:, perm]
        D = D[:, perm]
    if cap < 1:
        return asdict(SearchStats("certified_none", int(cap), 0, H.shape[1], 0.0))

    m, n = H.shape
    col_syn: list[int] = []
    col_sig: list[int] = []
    col_deg: list[int] = []
    check_masks: list[int] = []

    for r in range(m):
        mask = 0
        for v in np.flatnonzero(H[r]):
            mask |= 1 << int(v)
        check_masks.append(mask)

    for v in range(n):
        sm = 0
        for r in np.flatnonzero(H[:, v]):
            sm |= 1 << int(r)
        col_syn.append(sm)
        col_deg.append(sm.bit_count())
        lm = 0
        for i in np.flatnonzero(D[:, v]):
            lm |= 1 << int(i)
        col_sig.append(lm)

    maxdeg = max(col_deg, default=1)
    all_vars = (1 << n) - 1
    start_time = time.perf_counter()
    deadline = None if time_limit is None else start_time + float(time_limit)
    nodes = 0
    starts_done = 0
    memo: set[int] = set()

    def analyze_odd_checks(syn: int, remaining: int):
        """One-pass feasibility, branch choice, and safe lower bound.

        The old implementation built and sorted every odd-check neighborhood,
        then DFS scanned the same odd checks a second time to choose a branch.
        A greedy set of pairwise-disjoint neighborhoods is still a rigorous
        packing lower bound in any order, so sorting is unnecessary.
        """
        best_mask = None
        best_count = n + 1
        used = 0
        packing = 0
        x = syn
        while x:
            rb = x & -x
            r = rb.bit_length() - 1
            x -= rb
            cm = check_masks[r] & remaining
            cnt = cm.bit_count()
            if cnt == 0:
                return None, cap + 1
            if cnt < best_count:
                best_count = cnt
                best_mask = cm
            if cm & used == 0:
                packing += 1
                used |= cm
        degree_lb = (syn.bit_count() + maxdeg - 1) // maxdeg
        return best_mask, max(degree_lb, packing)

    def dfs(start: int, selected: int, syn: int, sig: int, weight: int, avail: int):
        nonlocal nodes
        nodes += 1
        if deadline is not None and (nodes & 4095) == 0 and time.perf_counter() > deadline:
            raise TimeoutError

        if syn == 0:
            if sig != 0:
                return selected, sig
            # A closed trivial component cannot be needed to discover another
            # non-trivial component; that other component will have its own start.
            return None
        if weight >= cap:
            return None
        remaining = avail & ~selected
        best_mask, lb = analyze_odd_checks(syn, remaining)
        if best_mask is None or weight + lb > cap:
            return None

        # Branch on the smallest remaining odd-check neighborhood. Every valid
        # completion must choose at least one variable from it.
        cm = int(best_mask)
        while cm:
            b = cm & -cm
            cm -= b
            v = b.bit_length() - 1
            new_selected = selected | b
            new_syn = syn ^ col_syn[v]
            # Syndrome/signature are deterministic functions of selected; the
            # selected support alone is a complete memo key and halves hashing.
            key = new_selected
            if key in memo:
                continue
            memo.add(key)
            ans = dfs(start, new_selected, new_syn, sig ^ col_sig[v], weight + 1, avail)
            if ans is not None:
                return ans
        return None

    try:
        for start in range(n):
            # The starting variable is by definition the minimum index in this
            # connected support, so only larger variable indices are available.
            avail = all_vars ^ ((1 << (start + 1)) - 1)
            selected = 1 << start
            memo.add(selected)
            ans = dfs(start, selected, col_syn[start], col_sig[start], 1, avail)
            if ans is not None:
                support_mask, sig = ans
                support_internal = list(_bit_positions(support_mask))
                support = sorted(int(perm[i]) for i in support_internal)
                signature = list(_bit_positions(sig))
                return asdict(SearchStats(
                    status="found",
                    cap=int(cap),
                    nodes=int(nodes),
                    starts_done=int(starts_done),
                    seconds=float(time.perf_counter() - start_time),
                    witness_support=support,
                    witness_weight=len(support),
                    logical_signature=signature,
                ))
            starts_done += 1
    except TimeoutError:
        return asdict(SearchStats(
            status="timeout",
            cap=int(cap),
            nodes=int(nodes),
            starts_done=int(starts_done),
            seconds=float(time.perf_counter() - start_time),
            timed_out=True,
        ))

    return asdict(SearchStats(
        status="certified_none",
        cap=int(cap),
        nodes=int(nodes),
        starts_done=int(starts_done),
        seconds=float(time.perf_counter() - start_time),
    ))


def _validate_boundary_witness(support, Hcheck, Hstab, claimed_d):
    if support is None:return None
    sup=sorted(set(map(int,support)))
    if len(sup)!=int(claimed_d):return None
    n=Hcheck.shape[1]
    if any(q<0 or q>=n for q in sup):return None
    v=np.zeros(n,dtype=np.uint8);v[sup]=1
    if np.any((Hcheck@v)&1):return None
    # Quotient nontriviality by rank increase; only two tiny rank calls on finalists.
    from inverse_design_core import gf2_rank
    if gf2_rank(np.vstack([Hstab,v]))==gf2_rank(Hstab):return None
    return {'status':'found','witness_support':sup,'witness_weight':len(sup),'source':'provided_exact_witness'}


def certify_css_distance(
    hx: np.ndarray,
    hz: np.ndarray,
    lx: np.ndarray,
    lz: np.ndarray,
    claimed_d: int,
    *,
    time_limit_per_side: float | None = None,
    boundary_witnesses: dict | None = None,
    first_side: str = "X",
) -> dict:
    """Attempt an exact CSS minimum-distance certificate at ``claimed_d``.

    Efficiency changes are proof-neutral:
    - a found sub-claim logical on the first side refutes the distance, so the
      opposite side is skipped;
    - caller-supplied weight-d witnesses are validated and reused instead of
      repeating the entire exact search at cap d after proving no weight < d.
    """
    cap=int(claimed_d)-1
    order=("Z","X") if str(first_side).upper()=="Z" else ("X","Z")
    params={"X":(hz,lz),"Z":(hx,lx)}
    low={"X":None,"Z":None}
    for side in order:
        H,D=params[side]
        low[side]=exact_low_weight_logical(H,D,cap,time_limit=time_limit_per_side)
        if low[side]["status"]=="found":
            other="Z" if side=="X" else "X"
            low[other]={"status":"skipped_after_refutation","cap":cap,"nodes":0,"starts_done":0,"seconds":0.0}
            return {
                "claimed_d":int(claimed_d),"cap_tested":cap,
                "method":"exact connected-support Tanner parity-closure enumeration",
                "X":low["X"],"Z":low["Z"],"lower_bound_certified":False,"refuted":True,
                "boundary_witness_search":{"X":None,"Z":None},
                "sector_exact_at_claimed_d":{"X":False,"Z":False},
                "exact_distance_certified":False,"status":"refuted",
            }
        if low[side]["status"]!="certified_none":
            # Timeout/incomplete: continuing the other side cannot complete the proof.
            other="Z" if side=="X" else "X"
            low[other]={"status":"skipped_after_incomplete","cap":cap,"nodes":0,"starts_done":0,"seconds":0.0}
            return {
                "claimed_d":int(claimed_d),"cap_tested":cap,
                "method":"exact connected-support Tanner parity-closure enumeration",
                "X":low["X"],"Z":low["Z"],"lower_bound_certified":False,"refuted":False,
                "boundary_witness_search":{"X":None,"Z":None},
                "sector_exact_at_claimed_d":{"X":False,"Z":False},
                "exact_distance_certified":False,"status":"timeout_or_incomplete",
            }
    lower=True
    boundary={"X":None,"Z":None}
    given=boundary_witnesses or {}
    # One weight-d logical on either sector is sufficient once both sectors are
    # proven free of lighter logicals: global d=min(dX,dZ)=d. Validate supplied
    # witnesses first and avoid an unnecessary second boundary search.
    for side,(Hcheck,_) in params.items():
        Hstab=hx if side=="X" else hz
        boundary[side]=_validate_boundary_witness(given.get(side),Hcheck,Hstab,claimed_d)
    if not any(b is not None for b in boundary.values()):
        for side in order:
            Hcheck,D=params[side]
            b=exact_low_weight_logical(Hcheck,D,int(claimed_d),time_limit=time_limit_per_side)
            boundary[side]=b
            if b.get("status")=="found" and b.get("witness_weight")==int(claimed_d):
                other="Z" if side=="X" else "X"
                if boundary[other] is None:
                    boundary[other]={"status":"not_needed_global_boundary","cap":int(claimed_d),"nodes":0,"starts_done":0,"seconds":0.0}
                break
    else:
        for side in ("X","Z"):
            if boundary[side] is None:
                boundary[side]={"status":"not_needed_global_boundary","cap":int(claimed_d),"nodes":0,"starts_done":0,"seconds":0.0}
    exact_sides={side:bool(boundary[side] is not None and boundary[side].get("status")=="found"
                           and boundary[side].get("witness_weight")==int(claimed_d)) for side in ("X","Z")}
    exact_distance=bool(lower and any(exact_sides.values()))
    return {
        "claimed_d":int(claimed_d),"cap_tested":cap,
        "method":"exact connected-support Tanner parity-closure enumeration",
        "X":low["X"],"Z":low["Z"],"lower_bound_certified":True,"refuted":False,
        "boundary_witness_search":boundary,"sector_exact_at_claimed_d":exact_sides,
        "exact_distance_certified":exact_distance,
        "status":"exact" if exact_distance else "lower_bound_certified",
    }


def _load_frozen(path: Path):
    from inverse_design_core import candidate_groups_of_order, build_product_checks, canonical_1x2_logicals
    doc = json.loads(path.read_text())
    c = doc["candidate"]
    g = candidate_groups_of_order(int(c["group_order"]))[int(c["group_index"])]
    A = tuple(tuple(map(int, x)) for x in c["A"])
    B = tuple(tuple(map(int, x)) for x in c["B"])
    hx, hz = build_product_checks(A, B, g)
    lx, lz = canonical_1x2_logicals(A, B, g)
    return hx, hz, lx, lz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--time-limit", type=float)
    ap.add_argument("--out")
    args = ap.parse_args()
    hx, hz, lx, lz = _load_frozen(Path(args.candidate))
    result = certify_css_distance(hx, hz, lx, lz, args.d, time_limit_per_side=args.time_limit)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
