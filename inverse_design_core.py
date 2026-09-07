from __future__ import annotations
"""Leakage-audited inverse design core.

This module intentionally contains only generic algebraic primitives and observable-target
constraints.  Family names and literature metadata live in posthoc_labeler.py and are never
imported here.
"""
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Sequence
import hashlib, json, random
import numpy as np

try:
    import cpp_fast as _cpp_fast
except Exception:  # pragma: no cover - optional native backend
    _cpp_fast = None

from distance_sketch import (
    classical_sqetch_generator,
    css_sqetch,
)
from phylogenetic_search import select_phylogenetic_elites, upgma

@dataclass(frozen=True)
class FiniteGroup:
    name:str
    mult:tuple[tuple[int,...],...]
    inv:tuple[int,...]
    identity:int=0
    @property
    def order(self): return len(self.mult)
    @property
    def is_abelian(self):
        n=self.order
        return all(self.mult[i][j]==self.mult[j][i] for i in range(n) for j in range(i+1,n))

def _make_group(name:str,mult:Sequence[Sequence[int]])->FiniteGroup:
    n=len(mult); inv=[]
    for a in range(n):
        xs=[b for b in range(n) if mult[a][b]==0 and mult[b][a]==0]
        if len(xs)!=1: raise ValueError('invalid multiplication table')
        inv.append(xs[0])
    return FiniteGroup(name,tuple(tuple(map(int,r)) for r in mult),tuple(inv),0)

def cyclic_group(n:int)->FiniteGroup:
    return _make_group(f'cyclic({n})',[[(i+j)%n for j in range(n)] for i in range(n)])

def dihedral_group(order:int)->FiniteGroup:
    if order%2: raise ValueError('dihedral order must be even')
    m=order//2; elems=[(a,b) for b in (0,1) for a in range(m)];idx={x:i for i,x in enumerate(elems)}
    tab=[]
    for a,b in elems:
        tab.append([idx[((a+(c if b==0 else -c))%m,(b+d)%2)] for c,d in elems])
    return _make_group(f'dihedral({order})',tab)

def direct_product_group(a:FiniteGroup,b:FiniteGroup)->FiniteGroup:
    elems=[(i,j) for i in range(a.order) for j in range(b.order)];idx={x:k for k,x in enumerate(elems)}
    tab=[[idx[(a.mult[i][u],b.mult[j][v])] for u,v in elems] for i,j in elems]
    return _make_group(f'product({a.name},{b.name})',tab)

def semidirect_cyclic_group(m:int,n:int,q:int)->FiniteGroup:
    if pow(q,n,m)!=1%m or np.gcd(q,m)!=1: raise ValueError('invalid cyclic action')
    elems=[(a,b) for b in range(n) for a in range(m)];idx={x:i for i,x in enumerate(elems)}
    tab=[]
    for a,b in elems:
        qb=pow(q,b,m)
        tab.append([idx[((a+qb*c)%m,(b+d)%n)] for c,d in elems])
    return _make_group(f'semidirect({m},{n},{q})',tab)

def candidate_groups_of_order(N:int)->list[FiniteGroup]:
    """Generic elementary group-presentation enumerator; no code-family cases."""
    out=[cyclic_group(N)]
    if N%2==0: out.append(dihedral_group(N))
    # Direct products C_a x D_b for every factorization N=a*b.
    for a in range(2,N+1):
        if N%a: continue
        b=N//a
        if b>=6 and b%2==0:
            try: out.append(direct_product_group(cyclic_group(a),dihedral_group(b)))
            except Exception: pass
    # Cyclic semidirect presentations C_m rtimes C_n for every compatible action.
    for m in range(3,N+1):
        if N%m: continue
        n=N//m
        if n<2: continue
        for q in range(2,m):
            if np.gcd(q,m)==1 and pow(q,n,m)==1%m and q%m!=1:
                try: out.append(semidirect_cyclic_group(m,n,q))
                except Exception: pass
    seen=set();ans=[]
    for g in out:
        if g.mult not in seen: seen.add(g.mult);ans.append(g)
    return ans

RingElem=tuple[int,...]
def dagger(x:RingElem,g:FiniteGroup)->RingElem:return tuple(sorted(g.inv[i] for i in x))

_REP_CACHE:dict[tuple[tuple[int,...],...],tuple[list[np.ndarray],list[np.ndarray]]]={}


def _representation_bases(g:FiniteGroup):
    """Cache singleton left/right permutation matrices per group."""
    key=g.mult
    cached=_REP_CACHE.get(key)
    if cached is not None:
        return cached
    left=[];right=[];s=g.order
    for a in range(s):
        L=np.zeros((s,s),dtype=np.uint8)
        R=np.zeros((s,s),dtype=np.uint8)
        ai=g.inv[a]
        for h in range(s):
            L[g.mult[a][h],h]=1
            R[g.mult[h][ai],h]=1
        left.append(L);right.append(R)
    cached=(left,right);_REP_CACHE[key]=cached
    return cached


def _xor_rep(x:RingElem,bases:list[np.ndarray],s:int)->np.ndarray:
    M=np.zeros((s,s),dtype=np.uint8)
    for a in x:
        M^=bases[a]
    return M


def left_rep(x:RingElem,g:FiniteGroup)->np.ndarray:
    return _xor_rep(x,_representation_bases(g)[0],g.order)
def right_rep(x:RingElem,g:FiniteGroup)->np.ndarray:
    return _xor_rep(x,_representation_bases(g)[1],g.order)

def gf2_rank(A:np.ndarray)->int:
    A=np.asarray(A,dtype=np.uint8).copy();m,n=A.shape;r=0
    for c in range(n):
        p=np.flatnonzero(A[r:,c])
        if not len(p):continue
        p=r+int(p[0]);A[[r,p]]=A[[p,r]]
        rows=np.flatnonzero(A[:,c]);rows=rows[rows!=r]
        if len(rows):A[rows]^=A[r]
        r+=1
        if r==m:break
    return r

def gf2_inverse(A:np.ndarray)->np.ndarray|None:
    A=np.asarray(A,dtype=np.uint8);n=A.shape[0]
    if A.shape!=(n,n):return None
    M=np.concatenate([A.copy(),np.eye(n,dtype=np.uint8)],axis=1);r=0
    for c in range(n):
        p=np.flatnonzero(M[r:,c])
        if not len(p):return None
        p=r+int(p[0]);M[[r,p]]=M[[p,r]]
        rows=np.flatnonzero(M[:,c]);rows=rows[rows!=r]
        if len(rows):M[rows]^=M[r]
        r+=1
    return M[:,n:]

def build_product_checks(A:tuple[RingElem,RingElem],B:tuple[RingElem,RingElem],g:FiniteGroup)->tuple[np.ndarray,np.ndarray]:
    """Five-block CSS chain product generated from two 1x2 group-algebra maps."""
    a0,a1=A;b0,b1=B;s=g.order;Z=np.zeros((s,s),dtype=np.uint8)
    LA=[left_rep(a0,g),left_rep(a1,g)];Rbd=[right_rep(dagger(b0,g),g),right_rep(dagger(b1,g),g)]
    RB=[right_rep(b0,g),right_rep(b1,g)];LAd=[left_rep(dagger(a0,g),g),left_rep(dagger(a1,g),g)]
    hx=np.block([[LA[0],Z,LA[1],Z,Rbd[0]],[Z,LA[0],Z,LA[1],Rbd[1]]]).astype(np.uint8)
    hz=np.block([[RB[0],RB[1],Z,Z,LAd[0]],[Z,Z,RB[0],RB[1],LAd[1]]]).astype(np.uint8)
    return hx,hz

def css_specs(hx:np.ndarray,hz:np.ndarray)->dict:
    n=hx.shape[1];rx=gf2_rank(hx);rz=gf2_rank(hz);wx=hx.sum(1);wz=hz.sum(1);weights=np.r_[wx,wz]
    return {'n':int(n),'k':int(n-rx-rz),'rank_x':rx,'rank_z':rz,'rate':float((n-rx-rz)/n),
            'commutation_violations':int(np.count_nonzero((hx@hz.T)&1)),
            'min_check_weight':int(min(wx.min() if len(wx) else 0,wz.min() if len(wz) else 0)),
            'max_check_weight':int(max(wx.max() if len(wx) else 0,wz.max() if len(wz) else 0)),
            'mean_check_weight':float(np.mean(weights)) if len(weights) else 0.0}

def canonical_1x2_logicals(A,B,g)->tuple[np.ndarray,np.ndarray]:
    a0,a1=A;b0,b1=B;s=g.order
    Li=gf2_inverse(left_rep(a1,g));Ri=gf2_inverse(right_rep(b1,g))
    if Li is None or Ri is None:raise ValueError('pivot map is singular')
    U=(Ri@right_rep(b0,g))&1;V=(Li@left_rep(a0,g))&1
    I=np.eye(s,dtype=np.uint8);Z=np.zeros((s,s),dtype=np.uint8)
    lx=np.concatenate([I,U.T,Z,Z,Z],1);lz=np.concatenate([I,Z,V.T,Z,Z],1)
    return lx,lz

def logicals_from_systematic_generators(left_generator:np.ndarray,
                                        right_generator:np.ndarray,
                                        s:int)->tuple[np.ndarray,np.ndarray]:
    """Build canonical product logicals from cached [I | pivot^-1 map] bases."""
    Z=np.zeros((s,s),dtype=np.uint8)
    # X logicals use ker(HZ)/row(HX), hence B/right; Z uses A/left.
    lx=np.concatenate([right_generator,Z,Z,Z],1)
    lz=np.concatenate([left_generator[:,:s],Z,left_generator[:,s:],Z,Z],1)
    return lx,lz

@dataclass(frozen=True)
class TargetSpec:
    n:int;k:int;max_check_weight:int;min_distance:int|None=None
@dataclass(frozen=True)
class ProductShape:
    ma:int;na:int;mb:int;nb:int;entry_weight:int;carrier_order:int

def infer_product_shapes(spec:TargetSpec,max_dim:int=4,max_entry_weight:int=12)->list[ProductShape]:
    # Match challenge verifier resource cap; over-cap shapes cannot be board
    # candidates even when their algebraic construction is valid.
    if int(spec.max_check_weight) > 32:
        return []
    out=[]
    for ma,na,mb,nb in product(range(1,max_dim+1),repeat=4):
        f=na*nb+ma*mb
        if spec.n%f:continue
        L=spec.n//f
        if L<2 or (na-ma)*(nb-mb)*L!=spec.k:continue
        for w in range(1,max_entry_weight+1):
            if max((na+mb)*w,(nb+ma)*w)==spec.max_check_weight:
                out.append(ProductShape(ma,na,mb,nb,w,L))
    return sorted(out,key=lambda s:(s.ma*s.na+s.mb*s.nb,s.carrier_order,s.entry_weight))

def sample_support(g:FiniteGroup,w:int,rng:random.Random,anchor_identity:bool=False)->RingElem:
    if anchor_identity:return tuple(sorted([0]+rng.sample(range(1,g.order),w-1)))
    return tuple(sorted(rng.sample(range(g.order),w)))

def _mutate_pairs(pairs, g:FiniteGroup, rng:random.Random, per_elite:int,
                  radius:int=1)->list:
    """Generate bounded support neighborhoods, preserving weight and anchors.

    ``radius`` emits successive one-symbol shells.  Radius two reaches
    candidates that need two support edits without materializing the full
    combinatorial neighborhood.
    """
    if int(per_elite) <= 0:
        return []
    radius=max(1,int(radius))
    out=[];seen=set()
    for pair in pairs:
        for block, anchored in ((0,False),(1,True)):
            frontier=[(tuple(pair[0]),tuple(pair[1]))]
            for _ in range(radius):
                candidates=[]
                for current_pair in frontier:
                    base=tuple(current_pair[block]); present=set(base)
                    mutable=[v for v in base
                             if not (anchored and v==g.identity)]
                    choices=[v for v in range(g.order)
                             if v not in present and
                             (not anchored or v!=g.identity)]
                    for old in mutable:
                        for new in choices:
                            vals=sorted((present-{old})|{new})
                            if anchored and g.identity not in vals:
                                continue
                            p=[tuple(current_pair[0]),tuple(current_pair[1])]
                            p[block]=tuple(vals)
                            key=(p[0],p[1])
                            if key not in seen:
                                seen.add(key);candidates.append(key)
                rng.shuffle(candidates)
                candidates=candidates[:int(per_elite)]
                out.extend(candidates)
                frontier=candidates
                if not frontier:
                    break
    return out

def seed_check(pair,g,left=True):
    x0,x1=pair
    if left:
        if gf2_inverse(left_rep(x1,g)) is None:return None
        return np.concatenate([left_rep(x0,g),left_rep(x1,g)],axis=1)
    if gf2_inverse(right_rep(x1,g)) is None:return None
    return np.concatenate([right_rep(x0,g),right_rep(x1,g)],axis=1)


def systematic_seed_generator(pair, g, left=True):
    """Return systematic basis for a 1x2 seed kernel.

    For seed check ``[A B]`` with invertible ``B``,
    ``ker([A B])`` has basis ``[I | (B^-1 A)^T]``.  This avoids a fresh
    nullspace elimination for every sampled constituent.
    """
    x0, x1 = pair
    A = left_rep(x0, g) if left else right_rep(x0, g)
    B = left_rep(x1, g) if left else right_rep(x1, g)
    Bi = gf2_inverse(B)
    if Bi is None:
        return None
    I = np.eye(g.order, dtype=np.uint8)
    return np.concatenate([I, ((Bi @ A) & 1).T], axis=1)


def _systematic_probe_from_generator(G: np.ndarray, max_combo=2):
    """Probe already-built systematic basis (internal allocation saver)."""
    row_weights = G.sum(axis=1)
    best = int(row_weights.min())
    best_combo = (int(np.argmin(row_weights)),)
    if max_combo >= 2 and len(G) >= 2:
        for i in range(len(G) - 1):
            weights = np.sum(G[i + 1:] ^ G[i], axis=1)
            j = int(np.argmin(weights))
            value = int(weights[j])
            if value < best:
                best, best_combo = value, (i, i + 1 + j)
    if max_combo >= 3 and len(G) >= 3:
        for i in range(len(G) - 2):
            for j in range(i + 1, len(G) - 1):
                base = G[i] ^ G[j]
                weights = np.sum(G[j + 1:] ^ base, axis=1)
                k = int(np.argmin(weights))
                value = int(weights[k])
                if value < best:
                    best, best_combo = value, (i, j, j + 1 + k)
    return {"best_weight": best, "combo": list(best_combo), "max_combo": int(max_combo)}


def systematic_seed_probe(pair, g, left=True, max_combo=2):
    """Cheap safe-to-reject probe over low-order systematic codewords."""
    G = systematic_seed_generator(pair, g, left)
    if G is None:
        return None
    return _systematic_probe_from_generator(G, max_combo)


def semantic_hash(hx,hz)->str:return hashlib.sha256(hx.tobytes()+hz.tobytes()).hexdigest()

def _combine_css_estimates(estimates:list[dict])->dict:
    """Combine independent CSS RIS passes, retaining best sector witnesses."""
    if not estimates:
        return {}
    def weight(item,side):
        value=item.get(side,{}).get('best_weight')
        return int(value) if value is not None else 10**9
    out=dict(min(estimates,key=lambda item:(item.get('d_upper') or 10**9)))
    for side in ('x','z'):
        out[side]=min(estimates,key=lambda item:weight(item,side)).get(side,{})
    values={side:weight(out,side) for side in ('x','z')}
    out['dx_upper']=None if values['x']>=10**9 else values['x']
    out['dz_upper']=None if values['z']>=10**9 else values['z']
    present=[value for value in values.values() if value<10**9]
    out['d_upper']=min(present) if present else None
    out['repeat_count']=len(estimates)
    out['repeat_scores']=[{
        'dx_upper':item.get('dx_upper'),'dz_upper':item.get('dz_upper'),
        'd_upper':item.get('d_upper')
    } for item in estimates]
    out['screen_refuted']=any(bool(item.get('screen_refuted')) for item in estimates)
    if all('seconds' in item for item in estimates):
        out['seconds']=sum(float(item.get('seconds',0.0)) for item in estimates)
    return out

def design_from_specs(spec:TargetSpec,seed:int=1503010,seed_draws:int=500,seed_screen_trials:int=80,
                      product_screen_trials:int=50,deep_trials:int=1200,deep_repeats:int=1,*,
                      probe_keep:int=48,seed_keep:int=8,max_deep_products:int=16,
                      mutation_elites:int=0,mutations_per_elite:int=0,
                      mutation_radius:int=1,
                      product_mutation_elites:int=0,product_mutations_per_elite:int=0,
                      product_mutation_radius:int=1,
                      product_mutation_rounds:int=1,
                      phylogeny_exploitation_fraction:float=0.5,
                      history_rows=None,
                      group_indices=None,
                      candidate_pool_size:int=1,
                      backend:str="numpy")->dict:
    """Search generic finite-group chain products under observable constraints.

    Search is staged: systematic seed probes reject obvious weak seeds, a
    bounded constituent shortlist gets randomized screening, product pairs get
    cheap screening, and only a bounded global shortlist gets deep trials.
    Distance values remain randomized upper bounds with explicit witnesses.
    """
    shapes=infer_product_shapes(spec)
    shapes_1x2=[s for s in shapes if (s.ma,s.na,s.mb,s.nb)==(1,2,1,2)]
    if not shapes_1x2:return {'target':asdict(spec),'shapes':[asdict(x) for x in shapes],'candidate':None}
    shape=shapes_1x2[0];groups=candidate_groups_of_order(shape.carrier_order)
    native_seed_batch = (str(backend).lower() in {"cpp", "auto"} and
                         _cpp_fast is not None and _cpp_fast.available())
    screened_products=[];group_stats=[]
    active_group_indices = (None if group_indices is None else
                            {int(index) for index in group_indices})
    for gi,g in enumerate(groups):
        if active_group_indices is not None and gi not in active_group_indices:
            continue
        gh=int(hashlib.sha256(repr(g.mult).encode()).hexdigest()[:16],16)
        rng=random.Random(seed^gh);sides=[]
        st={'group_id':f'G{gi:02d}','order':g.order,'abelian':g.is_abelian,
            'left':0,'right':0,'draws_per_side':int(seed_draws),
            'probe_survivors':{'left':0,'right':0},
            'history_injected':{'left':0,'right':0},'phylogeny':[]}
        history_for_group=[]
        for row in history_rows or ():
            if (int(row.get('n',-1))==spec.n and int(row.get('k',-1))==spec.k and
                    int(row.get('group_order',-1))==g.order and
                    int(row.get('group_index',-1))==gi):
                history_for_group.append(row)
        history_pairs={side:set() for side in (0,1)}
        for side in (0,1):
            probed=[]
            kept=[]
            historical=[]
            for row in history_for_group:
                raw=row['A'] if side==0 else row['B']
                pair=tuple(tuple(int(v) for v in support) for support in raw)
                if (len(pair)==2 and all(len(support)==shape.entry_weight for support in pair)
                        and pair not in historical):
                    historical.append(pair)
            history_pairs[side].update(historical)
            history_budget=min(len(historical),max(0,int(seed_draws)//8))
            historical=historical[:history_budget]
            st['history_injected']['left' if side==0 else 'right']=len(historical)
            pairs=historical+[(sample_support(g,shape.entry_weight,rng,False),
                               sample_support(g,shape.entry_weight,rng,True))
                              for _ in range(max(0,int(seed_draws)-len(historical)))]
            if native_seed_batch and pairs:
                batch=np.asarray(pairs,dtype=np.int32)
                generators,probe_values,valid=_cpp_fast.batch_systematic(
                    g.mult,g.inv,batch,left=(side==0))
                draw_data=((pairs[t],int(probe_values[t]),generators[t],
                            bool(valid[t]),t) for t in range(seed_draws))
            else:
                draw_data=((pair,None,None,True,t)
                           for t,pair in enumerate(pairs))
            for pair,probe_value,G,valid_draw,t in draw_data:
                if not valid_draw:continue
                if G is None:
                    G=systematic_seed_generator(pair,g,left=(side==0))
                    if G is None:continue
                    probe=_systematic_probe_from_generator(G,max_combo=2)
                    if probe is None:continue
                else:
                    probe={'best_weight':int(probe_value),'combo':[], 'max_combo':2}
                if spec.min_distance is not None and probe['best_weight']<spec.min_distance:continue
                probed.append((pair,probe,t,G))
            probed.sort(key=lambda x:(x[0] in history_pairs[side],x[1]['best_weight'],x[0]),reverse=True)
            if int(mutation_elites)>0 and int(mutations_per_elite)>0 and probed:
                elites=[x[0] for x in probed[:int(mutation_elites)]]
                mutants=_mutate_pairs(elites,g,rng,int(mutations_per_elite),
                                      mutation_radius)
                if native_seed_batch and mutants:
                    batch=np.asarray(mutants,dtype=np.int32)
                    generators,probe_values,valid=_cpp_fast.batch_systematic(
                        g.mult,g.inv,batch,left=(side==0))
                    mutant_data=((mutants[t],int(probe_values[t]),generators[t],bool(valid[t]),
                                  seed_draws+t) for t in range(len(mutants)))
                else:
                    mutant_data=((pair,None,None,True,seed_draws+t)
                                 for t,pair in enumerate(mutants))
                for pair,probe_value,G,valid_draw,t in mutant_data:
                    if not valid_draw:continue
                    if G is None:
                        G=systematic_seed_generator(pair,g,left=(side==0))
                        if G is None:continue
                        probe=_systematic_probe_from_generator(G,max_combo=2)
                        if probe is None:continue
                    else:
                        probe={'best_weight':int(probe_value),'combo':[], 'max_combo':2}
                    if spec.min_distance is not None and probe['best_weight']<spec.min_distance:continue
                    probed.append((pair,probe,t,G))
                probed.sort(key=lambda x:(x[0] in history_pairs[side],x[1]['best_weight'],x[0]),reverse=True)
            probed=probed[:max(int(probe_keep),int(seed_keep))]
            st['probe_survivors']['left' if side==0 else 'right']=len(probed)
            for j,(pair,probe,t,G) in enumerate(probed):
                q=classical_sqetch_generator(G,seed_screen_trials,32,
                    (seed+gi*100000+side*10000+j)&0xffffffff,spec.min_distance,
                    already_independent=True,backend=backend)
                if spec.min_distance is not None and q['best_weight'] is not None and q['best_weight']<spec.min_distance:continue
                kept.append((pair,q,probe,G))
            kept.sort(key=lambda x:(x[0] in history_pairs[side],
                                    (x[1]['best_weight'] or 999),x[2]['best_weight'],x[0]),reverse=True)
            kept=kept[:seed_keep]
            st['left' if side==0 else 'right']=len(kept);sides.append(kept)
        group_stats.append(st)
        # 1x2 pivots make these invariants deterministic; avoid dense rank
        # elimination for every product in large mutation beams.
        s=g.order;check_weight=3*shape.entry_weight
        product_spec={'n':5*s,'k':s,'rank_x':2*s,'rank_z':2*s,
                      'rate':float(s/(5*s)),'commutation_violations':0,
                      'min_check_weight':check_weight,'max_check_weight':check_weight,
                      'mean_check_weight':float(check_weight)}
        group_products=[]
        product_generators={}
        product_spec_checked=False
        for ia,(A,qa,pa,GA) in enumerate(sides[0]):
            for ib,(B,qb,pb,GB) in enumerate(sides[1]):
                hx,hz=build_product_checks(A,B,g)
                if not product_spec_checked:
                    actual_spec=css_specs(hx,hz)
                    if (actual_spec['n']!=spec.n or actual_spec['k']!=spec.k or
                            actual_spec['commutation_violations'] or
                            actual_spec['max_check_weight']>spec.max_check_weight):continue
                    product_spec=actual_spec;product_spec_checked=True
                sp=product_spec
                if sp['n']!=spec.n or sp['k']!=spec.k or sp['commutation_violations'] or sp['max_check_weight']>spec.max_check_weight:continue
                lx,lz=logicals_from_systematic_generators(GA,GB,s)
                q=css_sqetch(hx,hz,lx,lz,product_screen_trials,32,(seed+gi*1000000+ia*100+ib)&0xffffffff,spec.min_distance,backend=backend)
                if spec.min_distance is not None and q['d_upper'] is not None and q['d_upper']<spec.min_distance:continue
                h=semantic_hash(hx,hz)
                product_generators[h]=(GA,GB)
                group_products.append({'group_index':gi,'group_order':g.order,'group_abelian':g.is_abelian,
                    'A':[list(x) for x in A],'B':[list(x) for x in B],'specs':sp,'screen':q,
                    'seed_probe':{'left':pa,'right':pb},
                    'history_seed':bool(A in history_pairs[0] and B in history_pairs[1]),
                    'semantic_hash':h})
        if group_products:
            initial_ranked=sorted(group_products,key=lambda r:((r['screen']['d_upper'] or -1),
                min(r['seed_probe']['left']['best_weight'],r['seed_probe']['right']['best_weight']),
                r['semantic_hash']),reverse=True)
            _,initial_phylo=select_phylogenetic_elites(
                initial_ranked,min(max(int(product_mutation_elites),6),len(initial_ranked)),
                exploitation_fraction=phylogeny_exploitation_fraction)
            initial_phylo['stage']='initial'
            st['phylogeny'].append(initial_phylo)
        if (int(product_mutation_elites)>0 and int(product_mutations_per_elite)>0 and
                int(product_mutation_rounds)>0 and group_products):
            product_seen={r['semantic_hash'] for r in group_products}
            product_rng=random.Random(seed ^ gh ^ 0x504d5554)
            for mutation_round in range(int(product_mutation_rounds)):
                ranked=sorted(group_products,key=lambda r:((r['screen']['d_upper'] or -1),
                    min(r['seed_probe']['left']['best_weight'],r['seed_probe']['right']['best_weight']),
                    r['semantic_hash']),reverse=True)
                elites,phylo=select_phylogenetic_elites(
                    ranked,int(product_mutation_elites),
                    exploitation_fraction=phylogeny_exploitation_fraction)
                phylo['stage']=f'mutation_round_{mutation_round+1}'
                st['phylogeny'].append(phylo)
                # Recombine strong A/B halves. Initial Cartesian products cover
                # seed-only crosses; later rounds also cross mutated halves.
                for left_index,left_base in enumerate(elites):
                    A=tuple(tuple(x) for x in left_base['A'])
                    GA=product_generators[left_base['semantic_hash']][0]
                    for right_index,right_base in enumerate(elites):
                        if left_index==right_index:continue
                        B=tuple(tuple(x) for x in right_base['B'])
                        GB=product_generators[right_base['semantic_hash']][1]
                        hx,hz=build_product_checks(A,B,g);sp=product_spec
                        if sp['n']!=spec.n or sp['k']!=spec.k or sp['commutation_violations'] or sp['max_check_weight']>spec.max_check_weight:continue
                        lx,lz=logicals_from_systematic_generators(GA,GB,s)
                        h=semantic_hash(hx,hz)
                        if h in product_seen:continue
                        q=css_sqetch(hx,hz,lx,lz,product_screen_trials,32,
                                     (seed+gi*4000000+mutation_round*100000+left_index*1000+right_index)&0xffffffff,
                                     spec.min_distance,backend=backend)
                        if spec.min_distance is not None and q['d_upper'] is not None and q['d_upper']<spec.min_distance:continue
                        product_seen.add(h)
                        product_generators[h]=(GA,GB)
                        group_products.append({'group_index':gi,'group_order':g.order,'group_abelian':g.is_abelian,
                            'A':[list(x) for x in A],'B':[list(x) for x in B],'specs':sp,'screen':q,
                            'seed_probe':left_base['seed_probe'],'product_recombination_from':[
                                left_base['semantic_hash'],right_base['semantic_hash']],
                            'product_mutation_round':mutation_round+1,'semantic_hash':h})
                # Local one-symbol moves around current beam.
                for elite_index,base in enumerate(elites):
                    base_A=tuple(tuple(x) for x in base['A'])
                    base_B=tuple(tuple(x) for x in base['B'])
                    for mutate_side,base_pair in ((0,base_A),(1,base_B)):
                        mutants=_mutate_pairs([base_pair],g,product_rng,
                                               int(product_mutations_per_elite),
                                               product_mutation_radius)
                        for mut_index,mutated in enumerate(mutants):
                            A=mutated if mutate_side==0 else base_A
                            B=base_B if mutate_side==0 else mutated
                            base_GA,base_GB=product_generators[base['semantic_hash']]
                            GA=systematic_seed_generator(A,g,left=True) if mutate_side==0 else base_GA
                            GB=base_GB if mutate_side==0 else systematic_seed_generator(B,g,left=False)
                            if GA is None or GB is None:continue
                            hx,hz=build_product_checks(A,B,g);sp=product_spec
                            if sp['n']!=spec.n or sp['k']!=spec.k or sp['commutation_violations'] or sp['max_check_weight']>spec.max_check_weight:continue
                            lx,lz=logicals_from_systematic_generators(GA,GB,s)
                            h=semantic_hash(hx,hz)
                            if h in product_seen:continue
                            q=css_sqetch(hx,hz,lx,lz,product_screen_trials,32,
                                         (seed+gi*2000000+mutation_round*100000+elite_index*1000+mutate_side*100+mut_index)&0xffffffff,
                                         spec.min_distance,backend=backend)
                            if spec.min_distance is not None and q['d_upper'] is not None and q['d_upper']<spec.min_distance:continue
                            product_seen.add(h)
                            product_generators[h]=(GA,GB)
                            group_products.append({'group_index':gi,'group_order':g.order,'group_abelian':g.is_abelian,
                                'A':[list(x) for x in A],'B':[list(x) for x in B],'specs':sp,'screen':q,
                                'seed_probe':base['seed_probe'],'product_mutation_from':base['semantic_hash'],
                                'product_mutation_round':mutation_round+1,'semantic_hash':h})
        screened_products.extend(group_products)
    screened_products.sort(key=lambda r:((r['screen']['d_upper'] or -1),
        min(r['seed_probe']['left']['best_weight'],r['seed_probe']['right']['best_weight']),
        r['semantic_hash']),reverse=True)
    phylo_leaves=screened_products[:64]
    phylo_tree=upgma(phylo_leaves)
    phylo_tree.pop('cophenetic',None)
    phylo_tree['leaf_policy']='top_64_screened_products'
    phylo_tree['leaves']=[{
        'semantic_hash':r['semantic_hash'],'group_index':r['group_index'],
        'screen_d':r['screen'].get('d_upper'),
        'history_seed':bool(r.get('history_seed',False)),
        'product_mutation_round':r.get('product_mutation_round',0),
        'parent':r.get('product_mutation_from'),
        'recombination_from':r.get('product_recombination_from',[]),
    } for r in phylo_leaves]
    deep_limit=max(0,int(max_deep_products))
    deep_set=[]
    if history_rows and deep_limit:
        # Reserve part of deep budget for exact historical genomes.  This
        # prevents a known-good lineage from disappearing during screening.
        history_quota=min(max(1,deep_limit//3),len(screened_products))
        for row in screened_products:
            if row.get('history_seed'):
                deep_set.append(row)
                if len(deep_set)>=history_quota:break
    for row in screened_products:
        if len(deep_set)>=deep_limit:break
        if row not in deep_set:deep_set.append(row)
    all_products=[]
    for j,r in enumerate(deep_set):
        g=groups[r['group_index']];A=tuple(tuple(x) for x in r['A']);B=tuple(tuple(x) for x in r['B'])
        hx,hz=build_product_checks(A,B,g);lx,lz=canonical_1x2_logicals(A,B,g)
        repeats=max(1,int(deep_repeats))
        estimates=[css_sqetch(
            hx,hz,lx,lz,deep_trials,32,
            (seed+7000000+j*100000+repeat*10000)&0xffffffff,
            spec.min_distance,backend=backend)
            for repeat in range(repeats)]
        r['distance_estimate']=_combine_css_estimates(estimates)
        if spec.min_distance is not None and (r['distance_estimate']['d_upper'] is None or r['distance_estimate']['d_upper']<spec.min_distance):
            continue
        all_products.append(r)
    if spec.min_distance is not None:
        all_products=[r for r in all_products if r['distance_estimate']['d_upper'] is not None and r['distance_estimate']['d_upper']>=spec.min_distance]
    all_products.sort(key=lambda r:(r['distance_estimate']['d_upper'] or -1,r['screen']['d_upper'] or -1,r['semantic_hash']),reverse=True)
    selected=all_products[0] if all_products else None
    pool_size=max(1, int(candidate_pool_size))
    if pool_size > 1 and all_products:
        # Confirm diverse high-quality lineages; adjacent mutations otherwise
        # consume the whole validation budget on near-identical genomes.
        candidate_pool,_=select_phylogenetic_elites(
            all_products, pool_size,
            exploitation_fraction=phylogeny_exploitation_fraction)
    else:
        candidate_pool=all_products[:pool_size]
    history_injected_count=sum(sum(x['history_injected'].values()) for x in group_stats)
    return {'target':asdict(spec),'inferred_shapes':[asdict(x) for x in shapes],
            'searched_group_indices':sorted(active_group_indices) if active_group_indices is not None else None,
            'group_stats':group_stats,
            'screened_product_count':len(screened_products),
            'deepened_product_count':len(deep_set),'candidate_count':len(all_products),
            'candidate_pool':candidate_pool,
            'candidate_pool_policy':'phylogenetic_diversity' if pool_size > 1 else 'score_ranked',
            'history_injected_count':history_injected_count,
            'phylogeny':{'algorithm':'UPGMA-greedy-PD','tree':phylo_tree,
                         'group_traces':[{'group_id':x['group_id'],
                                          'history_injected':x['history_injected'],
                                          'rounds':x['phylogeny']}
                                         for x in group_stats]},
            'candidate':selected}
