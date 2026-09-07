from __future__ import annotations
import numpy as np, time

try:
    import cpp_fast as _cpp_fast
except Exception:  # pragma: no cover - optional native backend
    _cpp_fast = None


def _native_backend(backend):
    mode = "numpy" if backend is None else str(backend).lower()
    if mode not in {"cpp", "auto"}:
        return None
    if _cpp_fast is not None and _cpp_fast.available():
        return _cpp_fast
    if mode == "cpp":
        detail = "" if _cpp_fast is None else f": {_cpp_fast.load_error()}"
        raise RuntimeError("native qldpc_fast backend unavailable" + detail)
    return None


def _pack_row(row: np.ndarray) -> int:
    """Pack one GF(2) row into a Python integer."""
    row=np.asarray(row,dtype=np.uint8)
    return int.from_bytes(np.packbits(row,bitorder="little").tobytes(),"little")


def pack_rows(A: np.ndarray) -> list[int]:
    """Vectorized ndarray -> Python-int row packing."""
    A=np.asarray(A,dtype=np.uint8)
    if A.ndim!=2:
        raise ValueError("expected a 2-D binary matrix")
    packed=np.packbits(A,axis=1,bitorder="little")
    return [int.from_bytes(row.tobytes(),"little") for row in packed]


def unpack_rows(rows: list[int], n: int) -> np.ndarray:
    out = np.zeros((len(rows), n), dtype=np.uint8)
    for i, x0 in enumerate(rows):
        x = int(x0)
        while x:
            b = x & -x
            out[i, b.bit_length() - 1] = 1
            x -= b
    return out


def _rref_packed_rows(rows: list[int], n: int, column_order=None) -> list[int]:
    """RREF packed rows, optionally scanning pivots in a permuted column order.

    With ``column_order=P`` this is exactly equivalent to
    ``gf2_rref(A[:, P])`` followed by undoing P, but no physical column
    permutation is materialized.
    """
    rows = [int(x) for x in rows]
    m = len(rows)
    r = 0
    columns = range(n) if column_order is None else column_order
    for c0 in columns:
        c = int(c0)
        bit = 1 << c
        p = -1
        for i in range(r, m):
            if rows[i] & bit:
                p = i
                break
        if p < 0:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        pivot = rows[r]
        for i in range(m):
            if i != r and (rows[i] & bit):
                rows[i] ^= pivot
        r += 1
        if r == m:
            break
    return rows[:r]


def gf2_rref(A: np.ndarray) -> np.ndarray:
    """Deterministic GF(2) RREF.

    Retains the original ndarray implementation because several callers rely
    on its exact ndarray behavior. Randomized hot loops use the packed backend
    below instead.
    """
    A=np.asarray(A,dtype=np.uint8).copy(); m,n=A.shape; r=0
    for c in range(n):
        piv=np.flatnonzero(A[r:,c])
        if not len(piv): continue
        p=r+int(piv[0]); A[[r,p]]=A[[p,r]]
        rows=np.flatnonzero(A[:,c]); rows=rows[rows!=r]
        if len(rows): A[rows]^=A[r]
        r+=1
        if r==m: break
    return A[:r]


def gf2_nullspace_packed_basis_rows(rows: list[int], n: int) -> list[int]:
    """Exact nullspace basis as packed Python-int rows."""
    rows=[int(x) for x in rows];m=len(rows);r=0;piv=[]
    for c in range(n):
        bit=1<<c;p=-1
        for i in range(r,m):
            if rows[i]&bit:p=i;break
        if p<0:continue
        rows[r],rows[p]=rows[p],rows[r];pivot=rows[r]
        for i in range(m):
            if i!=r and (rows[i]&bit):rows[i]^=pivot
        piv.append(c);r+=1
        if r==m:break
    pset=set(piv);free=[c for c in range(n) if c not in pset]
    out=[]
    for f in free:
        x=1<<f
        for i in range(len(piv)-1,-1,-1):
            c=piv[i]
            if (rows[i]&x).bit_count()&1:x|=1<<c
        out.append(x)
    return out


def gf2_nullspace_packed_rows(rows: list[int], n: int) -> np.ndarray:
    """Exact nullspace from already-packed parity-check rows."""
    return unpack_rows(gf2_nullspace_packed_basis_rows(rows,n),n)


def gf2_nullspace(A: np.ndarray) -> np.ndarray:
    """Exact GF(2) nullspace using packed elimination.

    The pivot rule and free-variable ordering are identical to the previous
    uint8 implementation, so the returned basis is byte-for-byte identical.
    """
    A=np.asarray(A,dtype=np.uint8)
    return gf2_nullspace_packed_rows(pack_rows(A),A.shape[1])


def gf2_logical_basis_packed(H_check: np.ndarray, H_stab: np.ndarray) -> list[int]:
    """Packed basis of ``ker(H_check) / rowspace(H_stab)``.

    Returned rows are logical representatives, not a distance claim.  This is
    the packed equivalent of the challenge verifier's logical-basis reduction
    and lets the fast refuter avoid dense RREF in its setup path.
    """
    H_check=np.asarray(H_check,dtype=np.uint8)
    H_stab=np.asarray(H_stab,dtype=np.uint8)
    if H_check.ndim!=2 or H_stab.ndim!=2 or H_check.shape[1]!=H_stab.shape[1]:
        raise ValueError("incompatible check/stabilizer matrices")
    n=int(H_check.shape[1])
    reduced=_rref_packed_rows(pack_rows(H_stab),n)
    logical=[]
    for v in gf2_nullspace_packed_basis_rows(pack_rows(H_check),n):
        x=int(v)
        # ``reduced`` is RREF under the physical coordinate order.  Reducing
        # against its pivot rows gives a canonical quotient representative.
        for row in reduced:
            pivot=row & -row
            if x & pivot:
                x ^= row
        if x:
            logical.append(x)
            reduced=_rref_packed_rows(reduced+[x],n)
    return logical


def _logical_nontrivial(row: int, dual_rows: list[int]) -> bool:
    for d in dual_rows:
        if (row & d).bit_count() & 1:
            return True
    return False


def _packed_sqetch_rows(
    packed: list[int],
    n: int,
    dual_rows: list[int],
    num_trials: int,
    k_sub: int,
    seed: int,
    d_target: int | None,
    *,
    require_logical: bool,
) -> dict:
    """Randomized RREF sketch from already-packed independent generator rows."""
    nu=len(packed);k_sub=min(int(k_sub),nu)
    rng=np.random.default_rng(seed)
    best=n+1;best_row=None;t0=time.perf_counter();trials_run=0
    for t in range(int(num_trials)):
        idxs=rng.integers(0,nu,size=k_sub)
        P=rng.permutation(n)
        selected=[packed[int(i)] for i in idxs]
        R=_rref_packed_rows(selected,n,P)
        for row in R:
            w=row.bit_count()
            if not w or w>=best:continue
            if require_logical and not _logical_nontrivial(row,dual_rows):continue
            best=w;best_row=row
            if d_target is not None and best<d_target:
                trials_run=t+1
                return {'best_weight':best,'witness':list(_bit_positions(best_row)),
                        'trials_run':trials_run,'k_sub':k_sub,
                        'seconds':time.perf_counter()-t0,'stopped_early':True}
        trials_run=t+1
    return {'best_weight':None if best==n+1 else best,
            'witness':None if best_row is None else list(_bit_positions(best_row)),
            'trials_run':trials_run,'k_sub':k_sub,
            'seconds':time.perf_counter()-t0,'stopped_early':False}


def _packed_sqetch_from_basis(
    W: np.ndarray,
    L_logical: np.ndarray | None,
    num_trials: int,
    k_sub: int,
    seed: int,
    d_target: int | None,
    *,
    require_logical: bool,
) -> dict:
    W=np.asarray(W,dtype=np.uint8)
    packed=pack_rows(W)
    dual_rows=pack_rows(np.asarray(L_logical,dtype=np.uint8)) if require_logical else []
    return _packed_sqetch_rows(packed,W.shape[1],dual_rows,num_trials,k_sub,seed,d_target,
                               require_logical=require_logical)


def _bit_positions(x: int):
    while x:
        b=x&-x
        yield b.bit_length()-1
        x-=b


def _random_kernel_sketch(W:np.ndarray,k_sub:int,rng:np.random.Generator)->np.ndarray:
    """Draw rows uniformly from the vector space spanned by W.

    Retained as a research primitive. The production sqetch path intentionally
    preserves the historical row-sampling distribution for reproducibility.
    """
    nu=W.shape[0]
    coeff=rng.integers(0,2,size=(k_sub,nu),dtype=np.uint8)
    return (coeff@W)&1


def cpu_sqetch(H_check:np.ndarray,L_logical:np.ndarray,num_trials:int=10000,k_sub:int=32,seed:int=0,d_target:int|None=None)->dict:
    """Randomized logical-distance upper bound with exact witnesses."""
    H_check=np.asarray(H_check,dtype=np.uint8);L_logical=np.asarray(L_logical,dtype=np.uint8)
    n=H_check.shape[1]
    Wp=gf2_nullspace_packed_basis_rows(pack_rows(H_check),n)
    dual_rows=pack_rows(L_logical)
    return _packed_sqetch_rows(Wp,n,dual_rows,num_trials,k_sub,seed,d_target,require_logical=True)


def css_sqetch(hx:np.ndarray,hz:np.ndarray,lx:np.ndarray,lz:np.ndarray,num_trials:int=10000,k_sub:int=32,seed:int=0,d_target:int|None=None,*,short_circuit_target:bool=False,backend:str="numpy")->dict:
    """CSS randomized upper bound.

    If ``short_circuit_target`` is true, the second sector receives zero shots
    once the first sector has already produced a witness below ``d_target``.
    That mode is safe for *screening/rejection* because one bad sector is enough
    to disqualify the code. Final reporting should leave it false.
    """
    native = _native_backend(backend)
    if native is not None:
        # Native RIS uses complete kernels; k_sub is retained in the public
        # API for reproducibility with the Python sketch path.
        return native.css_ris(hx, hz, lx=lx, lz=lz, trials=num_trials, seed=seed,
                              pair_depth=8, target=d_target,
                              stop_on_target=d_target is not None)
    # Preserve historical order/seeds: Z logical search first, then X logical.
    z=cpu_sqetch(hx,lx,num_trials,k_sub,seed,d_target)
    if short_circuit_target and d_target is not None and z['best_weight'] is not None and z['best_weight']<d_target:
        x={'best_weight':None,'witness':None,'trials_run':0,'k_sub':min(int(k_sub),hz.shape[1]),
           'seconds':0.0,'stopped_early':True,'skipped_after_other_sector_refuted':True}
        return {'dx_upper':None,'dz_upper':z['best_weight'],'d_upper':z['best_weight'],'x':x,'z':z,
                'screen_refuted':True}
    x=cpu_sqetch(hz,lz,num_trials,k_sub,seed+1,d_target)
    vals=[v for v in (x['best_weight'],z['best_weight']) if v is not None]
    return {'dx_upper':x['best_weight'],'dz_upper':z['best_weight'],'d_upper':min(vals) if vals else None,
            'x':x,'z':z,'screen_refuted':bool(d_target is not None and vals and min(vals)<d_target)}


def classical_sqetch_packed_generator(rows:list[int],n:int,num_trials:int=1000,k_sub:int=32,seed:int=0,d_target:int|None=None)->dict:
    """Classical randomized upper bound from an already-packed generator basis."""
    return _packed_sqetch_rows([int(x) for x in rows],int(n),[],num_trials,k_sub,seed,d_target,
                               require_logical=False)


def classical_sqetch_generator(W:np.ndarray,num_trials:int=1000,k_sub:int=32,seed:int=0,d_target:int|None=None,*,already_independent:bool=False,backend:str="numpy")->dict:
    """Randomized classical upper bound from a known generator basis."""
    W=np.asarray(W,dtype=np.uint8)
    if not already_independent:
        W=gf2_rref(W)
    native = _native_backend(backend)
    if native is not None:
        return native.classical_ris(W, trials=num_trials, seed=seed,
                                    pair_depth=8, target=d_target,
                                    stop_on_target=d_target is not None)
    return _packed_sqetch_from_basis(W,None,num_trials,k_sub,seed,d_target,require_logical=False)


def classical_sqetch(H:np.ndarray,num_trials:int=1000,k_sub:int=32,seed:int=0,d_target:int|None=None)->dict:
    """Random sketched-ISD upper bound for a classical code ker(H)."""
    H=np.asarray(H,dtype=np.uint8);n=H.shape[1]
    Wp=gf2_nullspace_packed_basis_rows(pack_rows(H),n)
    return _packed_sqetch_rows(Wp,n,[],num_trials,k_sub,seed,d_target,require_logical=False)


class PackedSketchState:
    """Persistent randomized sketch state for staged/refutation searches.

    Construction of the kernel and logical detector is paid once. Calls to
    ``run`` continue the same RNG stream and retain the best witness, so a
    24->72->304 ladder costs exactly 400 shots rather than 24+96+400 and does
    not repeat nullspace/RREF setup between stages.
    """
    def __init__(self,H_check:np.ndarray,L_logical:np.ndarray|None,*,seed:int=0,k_sub:int=32,require_logical:bool=True):
        H=np.asarray(H_check,dtype=np.uint8);self.n=int(H.shape[1])
        self.packed=gf2_nullspace_packed_basis_rows(pack_rows(H),self.n)
        self.dual_rows=pack_rows(np.asarray(L_logical,dtype=np.uint8)) if require_logical else []
        self.require_logical=bool(require_logical)
        self.k_sub=min(int(k_sub),len(self.packed))
        self.rng=np.random.default_rng(seed)
        self.best=self.n+1;self.best_row=None;self.trials_run=0;self.seconds=0.0

    def run(self,additional_trials:int,d_target:int|None=None)->dict:
        additional_trials=max(0,int(additional_trials));t0=time.perf_counter();ran=0
        if d_target is not None and self.best<d_target:
            return self.snapshot(stopped_early=True,delta_trials=0,delta_seconds=0.0)
        for _ in range(additional_trials):
            idxs=self.rng.integers(0,len(self.packed),size=self.k_sub)
            P=self.rng.permutation(self.n)
            R=_rref_packed_rows([self.packed[int(i)] for i in idxs],self.n,P)
            for row in R:
                w=row.bit_count()
                if not w or w>=self.best:continue
                if self.require_logical and not _logical_nontrivial(row,self.dual_rows):continue
                self.best=w;self.best_row=row
                if d_target is not None and self.best<d_target:
                    ran+=1;self.trials_run+=ran
                    dt=time.perf_counter()-t0;self.seconds+=dt
                    return self.snapshot(stopped_early=True,delta_trials=ran,delta_seconds=dt)
            ran+=1
        self.trials_run+=ran;dt=time.perf_counter()-t0;self.seconds+=dt
        return self.snapshot(stopped_early=False,delta_trials=ran,delta_seconds=dt)

    def snapshot(self,*,stopped_early:bool=False,delta_trials:int=0,delta_seconds:float=0.0)->dict:
        return {'best_weight':None if self.best==self.n+1 else int(self.best),
                'witness':None if self.best_row is None else list(_bit_positions(self.best_row)),
                'trials_run':int(self.trials_run),'k_sub':int(self.k_sub),'seconds':float(self.seconds),
                'stopped_early':bool(stopped_early),'delta_trials':int(delta_trials),'delta_seconds':float(delta_seconds)}


class CSSRefutationRace:
    """Two-sector staged CSS refutation with persistent kernels and RNG state."""
    def __init__(self,hx,hz,lx,lz,*,seed:int=0,k_sub:int=32):
        # Preserve historical seeds: Z-logical uses seed, X-logical seed+1.
        self.z=PackedSketchState(hx,lx,seed=seed,k_sub=k_sub,require_logical=True)
        self.x=PackedSketchState(hz,lz,seed=seed+1,k_sub=k_sub,require_logical=True)
        self.stages=[]

    def run_stage(self,additional_trials:int,d_target:int|None=None,*,first:str='z')->dict:
        order=('x','z') if first=='x' else ('z','x')
        stage={'requested_per_sector':int(additional_trials),'first':order[0],'sectors':{}}
        for side in order:
            state=getattr(self,side)
            before=state.trials_run
            snap=state.run(additional_trials,d_target)
            stage['sectors'][side]={'delta_trials':state.trials_run-before,'best_weight':snap['best_weight'],
                                    'delta_seconds':snap['delta_seconds']}
            if d_target is not None and snap['best_weight'] is not None and snap['best_weight']<d_target:
                stage['refuted_by']=side
                break
        self.stages.append(stage)
        return self.snapshot(d_target)

    def snapshot(self,d_target:int|None=None)->dict:
        x=self.x.snapshot();z=self.z.snapshot();vals=[v for v in (x['best_weight'],z['best_weight']) if v is not None]
        d=min(vals) if vals else None
        return {'dx_upper':x['best_weight'],'dz_upper':z['best_weight'],'d_upper':d,'x':x,'z':z,
                'screen_refuted':bool(d_target is not None and d is not None and d<d_target),
                'stages':list(self.stages),'total_sector_shots':x['trials_run']+z['trials_run'],
                'total_seconds':x['seconds']+z['seconds']}


def _consider_full_ris_rows(R:list[int],dual_rows:list[int],best:int,best_row:int|None,pair_depth:int):
    """Evaluate RREF rows and light-row XORs; return improved (best,row)."""
    require_logical=bool(dual_rows)
    weights=[row.bit_count() for row in R]
    for row,w in zip(R,weights):
        if not w or w>=best:continue
        if require_logical and not _logical_nontrivial(row,dual_rows):continue
        best=w;best_row=row
    if pair_depth>1 and len(R)>=2:
        ids=np.argsort(np.asarray(weights,dtype=np.int64))[:min(int(pair_depth),len(R))]
        for ai in range(len(ids)-1):
            a=R[int(ids[ai])]
            for bi in range(ai+1,len(ids)):
                row=a^R[int(ids[bi])];w=row.bit_count()
                if not w or w>=best:continue
                if require_logical and not _logical_nontrivial(row,dual_rows):continue
                best=w;best_row=row
    return best,best_row


class FullRISState:
    """Persistent full-kernel information-set search.

    Unlike the historical 32-row sketch, every shot row-reduces the complete
    kernel in a random coordinate order.  This is basis-invariant and much more
    informative per shot.  Optional light-row pair combinations match the
    challenge-style RIS strengthening.
    """
    def __init__(self,kernel_rows:list[int],n:int,dual_rows:list[int]|None=None,*,seed:int=0,pair_depth:int=8):
        # Canonicalize once so redundant stabilizer/logical spanning rows do not
        # increase per-shot work. RREF does not change the represented space.
        self.n=int(n)
        self.kernel_rows=_rref_packed_rows([int(x) for x in kernel_rows],self.n)
        self.dual_rows=[] if dual_rows is None else [int(x) for x in dual_rows]
        self.rng=np.random.default_rng(seed);self.pair_depth=int(pair_depth)
        self.best=self.n+1;self.best_row=None;self.trials_run=0;self.seconds=0.0

    def run(self,additional_trials:int,d_target:int|None=None,*,max_seconds:float|None=None)->dict:
        additional_trials=max(0,int(additional_trials));t0=time.perf_counter();ran=0
        deadline=None if max_seconds is None else t0+max(0.0,float(max_seconds))
        timed_out=False
        if d_target is not None and self.best<d_target:
            return self.snapshot(stopped_early=True,delta_trials=0,delta_seconds=0.0)
        for _ in range(additional_trials):
            R=_rref_packed_rows(self.kernel_rows,self.n,self.rng.permutation(self.n))
            self.best,self.best_row=_consider_full_ris_rows(
                R,self.dual_rows,self.best,self.best_row,self.pair_depth)
            ran+=1
            if d_target is not None and self.best<d_target:
                self.trials_run+=ran;dt=time.perf_counter()-t0;self.seconds+=dt
                return self.snapshot(stopped_early=True,delta_trials=ran,delta_seconds=dt)
            if deadline is not None and (ran&63)==0 and time.perf_counter()>deadline:
                timed_out=True
                break
        self.trials_run+=ran;dt=time.perf_counter()-t0;self.seconds+=dt
        return self.snapshot(stopped_early=timed_out,delta_trials=ran,delta_seconds=dt,
                             timed_out=timed_out)

    def snapshot(self,*,stopped_early:bool=False,delta_trials:int=0,delta_seconds:float=0.0,
                 timed_out:bool=False)->dict:
        return {'best_weight':None if self.best==self.n+1 else int(self.best),
                'witness':None if self.best_row is None else list(_bit_positions(self.best_row)),
                'trials_run':int(self.trials_run),'seconds':float(self.seconds),
                'stopped_early':bool(stopped_early),'delta_trials':int(delta_trials),
                'delta_seconds':float(delta_seconds),'kernel_dimension':len(self.kernel_rows),
                'pair_depth':int(self.pair_depth),'mode':'full_kernel_ris',
                'timed_out':bool(timed_out)}


def classical_full_ris_packed_generator(rows:list[int],n:int,num_trials:int=10,seed:int=0,d_target:int|None=None,pair_depth:int=8)->dict:
    state=FullRISState(rows,n,seed=seed,pair_depth=pair_depth)
    return state.run(num_trials,d_target)


def css_full_ris(hx:np.ndarray,hz:np.ndarray,lx:np.ndarray,lz:np.ndarray,num_trials:int=100,seed:int=0,d_target:int|None=None,*,pair_depth:int=8,short_circuit_target:bool=False,first:str='z',max_seconds_per_side:float|None=None,backend:str="numpy")->dict:
    """Full-kernel CSS RIS without computing nullspaces.

    For a CSS code with complete logical bases,
      ker(Hx) = rowspace(Hz) + span(Lz),
      ker(Hz) = rowspace(Hx) + span(Lx).
    We use those spanning sets directly and canonicalize once in FullRISState.
    """
    native = _native_backend(backend)
    if native is not None:
        return native.css_ris(hx, hz, lx=lx, lz=lz, trials=num_trials, seed=seed,
                               pair_depth=pair_depth,
                               max_seconds_per_side=max_seconds_per_side,
                               target=d_target,
                               stop_on_target=short_circuit_target)
    hx=np.asarray(hx,dtype=np.uint8);hz=np.asarray(hz,dtype=np.uint8)
    n=hx.shape[1]
    zstate=FullRISState(pack_rows(hz)+pack_rows(lz),n,pack_rows(lx),seed=seed,pair_depth=pair_depth)
    xstate=FullRISState(pack_rows(hx)+pack_rows(lx),n,pack_rows(lz),seed=seed+1,pair_depth=pair_depth)
    order=('x','z') if first=='x' else ('z','x')
    states={'x':xstate,'z':zstate}
    for side in order:
        snap=states[side].run(num_trials,d_target,max_seconds=max_seconds_per_side)
        if short_circuit_target and d_target is not None and snap['best_weight'] is not None and snap['best_weight']<d_target:
            break
    x=xstate.snapshot();z=zstate.snapshot();vals=[v for v in (x['best_weight'],z['best_weight']) if v is not None]
    d=min(vals) if vals else None
    return {'dx_upper':x['best_weight'],'dz_upper':z['best_weight'],'d_upper':d,'x':x,'z':z,
            'screen_refuted':bool(d_target is not None and d is not None and d<d_target),
            'total_sector_shots':x['trials_run']+z['trials_run'],
            'total_seconds':x['seconds']+z['seconds'],'mode':'full_kernel_ris'}


class CSSFullRISRace:
    """Persistent two-sector full-kernel RIS race."""
    def __init__(self,hx,hz,lx,lz,*,seed:int=0,pair_depth:int=8):
        hx=np.asarray(hx,dtype=np.uint8);hz=np.asarray(hz,dtype=np.uint8);n=hx.shape[1]
        self.z=FullRISState(pack_rows(hz)+pack_rows(lz),n,pack_rows(lx),seed=seed,pair_depth=pair_depth)
        self.x=FullRISState(pack_rows(hx)+pack_rows(lx),n,pack_rows(lz),seed=seed+1,pair_depth=pair_depth)
        self.stages=[]

    def run_stage(self,additional_trials:int,d_target:int|None=None,*,first:str='z',max_seconds_per_side:float|None=None)->dict:
        order=('x','z') if first=='x' else ('z','x');stage={'requested_per_sector':int(additional_trials),'first':order[0],'sectors':{}}
        for side in order:
            st=getattr(self,side);before=st.trials_run
            snap=st.run(additional_trials,d_target,max_seconds=max_seconds_per_side)
            stage['sectors'][side]={'delta_trials':st.trials_run-before,'best_weight':snap['best_weight'],'delta_seconds':snap['delta_seconds']}
            if d_target is not None and snap['best_weight'] is not None and snap['best_weight']<d_target:
                stage['refuted_by']=side;break
        self.stages.append(stage);return self.snapshot(d_target)

    def snapshot(self,d_target:int|None=None)->dict:
        x=self.x.snapshot();z=self.z.snapshot();vals=[v for v in (x['best_weight'],z['best_weight']) if v is not None];d=min(vals) if vals else None
        return {'dx_upper':x['best_weight'],'dz_upper':z['best_weight'],'d_upper':d,'x':x,'z':z,
                'screen_refuted':bool(d_target is not None and d is not None and d<d_target),
                'stages':list(self.stages),'total_sector_shots':x['trials_run']+z['trials_run'],
                'total_seconds':x['seconds']+z['seconds'],'mode':'full_kernel_ris'}


def supports_to_binary(supports:list[list[int]],n:int)->np.ndarray:
    """Build verifier-compatible binary checks once from support lists."""
    out=np.zeros((len(supports),int(n)),dtype=np.uint8)
    for i,support in enumerate(supports):
        if support:
            cols=np.asarray(support,dtype=np.int64)
            if np.any(cols<0) or np.any(cols>=n):
                raise ValueError("support index out of range")
            # Submission schema rejects repeats; XOR preserves its semantics
            # for callers using this helper directly.
            for col in cols:
                out[i,int(col)]^=1
    return out


def fast_refute_supports(checks_x:list[list[int]],checks_z:list[list[int]],n:int,
                         claimed_d:int,*,trials:int=8000,seed:int=0,pair_depth:int=8,
                         max_seconds:float|None=10.0,backend:str="auto")->dict:
    """Fast challenge-style RIS refutation from sparse support lists.

    This is sound in the same sense as the challenge gate: a returned lighter
    word is an explicit checkable witness; no-hit is not a proof.  Setup and
    trials use packed GF(2) rows.  ``trials`` is per sector, while
    ``max_seconds`` is split between sectors and may stop a sector early.
    """
    hx=supports_to_binary(checks_x,n);hz=supports_to_binary(checks_z,n)
    native = _native_backend(backend)
    if native is not None:
        per_side=None if max_seconds is None else max(0.0,float(max_seconds))/2.0
        # CSS sectors are independent.  The native race preserves the X/Z
        # seed streams while running both searches concurrently.
        if hasattr(native, "css_ris_parallel"):
            out=native.css_ris_parallel(
                hx,hz,trials=int(trials),seed=int(seed),
                pair_depth=int(pair_depth),
                max_seconds_per_side=per_side,
                target=int(claimed_d),stop_on_target=True,threads=0)
        else:
            out=native.css_ris(hx,hz,trials=int(trials),seed=int(seed),
                               pair_depth=int(pair_depth),
                               max_seconds_per_side=per_side,
                               target=int(claimed_d),stop_on_target=True)
        bests=[(out[side].get("best_weight"),side) for side in ("x","z")
               if out[side].get("best_weight") is not None]
        best_side=min(bests)[1] if bests else None
        best=None if best_side is None else out[best_side]["best_weight"]
        return {'refuted':bool(best is not None and best<int(claimed_d)),
                'd_found':None if best is None else int(best),
                'witness':None if best_side is None else out[best_side]['witness'],
                'trials_run':int(out.get('total_sector_shots',0)),
                'seconds':float(out.get('total_seconds',0.0)),
                'refuted_by':best_side if best is not None and best<int(claimed_d) else None,
                'sectors':{'x':out['x'],'z':out['z']},
                'mode':out.get('mode','cpp_full_kernel_ris')}
    lx=gf2_logical_basis_packed(hz,hx)  # X logical detector: ker(HZ)/row(HX)
    lz=gf2_logical_basis_packed(hx,hz)  # Z logical detector: ker(HX)/row(HZ)
    # Full kernel spans are stabilizers plus quotient representatives.
    # Match the public challenge convention: X uses ``seed`` and Z uses
    # ``seed + 1``.  Search order below is still Z-first so a likely weak
    # sector can short-circuit the second one without changing either stream.
    x=FullRISState(pack_rows(hx)+lx,n,lz,seed=int(seed),pair_depth=pair_depth)
    z=FullRISState(pack_rows(hz)+lz,n,lx,seed=int(seed)+1,pair_depth=pair_depth)
    side_order=('z','x')
    per_side=None if max_seconds is None else max(0.0,float(max_seconds))/2.0
    snapshots={}
    t0=time.perf_counter();refuted_by=None
    for side in side_order:
        state=z if side=='z' else x
        snap=state.run(int(trials),int(claimed_d),max_seconds=per_side)
        snapshots[side]=snap
        if snap['best_weight'] is not None and snap['best_weight']<int(claimed_d):
            refuted_by=side
            break
    bests=[(s['best_weight'],side) for side,s in snapshots.items()
           if s.get('best_weight') is not None]
    best_side=min(bests)[1] if bests else None
    best=None if best_side is None else snapshots[best_side]['best_weight']
    return {'refuted':bool(best is not None and best<int(claimed_d)),
            'd_found':None if best is None else int(best),
            'witness':None if best_side is None else snapshots[best_side]['witness'],
            'trials_run':sum(int(s['trials_run']) for s in snapshots.values()),
            'seconds':time.perf_counter()-t0,'refuted_by':refuted_by,
            'sectors':snapshots,'mode':'packed_full_kernel_ris'}
