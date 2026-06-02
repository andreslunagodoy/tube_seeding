"""Fully length-prune the quadruple-cut [1,3,6,8] lift, corner by corner.

Uncut props {2,4,5,7}; order<=3 closes (the four-fold corner is never needed).
Greedy coordinate descent that re-solves the FULL system at every step, so the
running configuration always closes and the final one is guaranteed valid:
  - singles (k=1): fixed at the nested rule length 11.
  - pairs   (k=2): minimal length, binary-searched in [-1, 10] (L=-1 => dropped).
  - triples (k=3): per-triple, decrement from the rule length 9 one rung at a time
                   until closure fails (L=-1 => the corner is not needed at all).
  - quad    (k=4): dropped.
Pairs are minimized first (triples held at the rule), then triples are minimized
with the pairs at their minima.  Reports each corner's minimal length and the
final pruned seed count (cf. 2985 full nested order<=3, 2725 order-pruned only)."""
import os, sys, re, itertools
os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rank10"))
import path_coverage_rank10 as pc
import pyfeyngym as pfg

ES_SEEDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "es_seeds_maxcut_rank12.txt")
TARGET = (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, -12)
CUT = [1, 3, 6, 8]
S = 1


def es_points():
    pts = []
    for line in open(ES_SEEDS):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        x, y, z = line.split()
        pts.append((int(x), int(y), int(z)))
    return pts


PTS = es_points()
VALS = [1] + list(range(0, -S - 1, -1))   # [1, 0, -1]


def build(uncut, corner_len):
    """corner_len: dict frozenset(absent set) -> max a11 length.  k=0 top length 12,
    k=1 singles length 11, k>=2 looked up (absent or L<0 => corner dropped)."""
    seeds = set()
    for (x, y, z) in PTS:
        for assign in itertools.product(VALS, repeat=len(uncut)):
            if sum(max(0, -v) for v in assign) > S:
                continue
            A = frozenset(uncut[i] for i, v in enumerate(assign) if v <= 0)
            k = len(A)
            if k == 0:
                L = 12
            elif k == 1:
                L = 11
            else:
                L = corner_len.get(A, -1)
            if z > L:
                continue
            s = [1] * 8 + [-x, -y, -z]
            for i, v in enumerate(assign):
                s[uncut[i] - 1] = v
            seeds.add(tuple(s))
    return seeds


def nonempty_eqs(seo):
    # drop empty equations (empty Dict -> SparseVec{Any,Any} -> abstract Vector{SparseVec} -> MethodError)
    return [e for e in (a[-1] for a in seo) if len(list(e)) > 0]


def main():
    eqt = pfg.gen_eq_templates(pc.IBP_FILE, pc.M_VALS)
    uncut = [p for p in range(1, 9) if p not in CUT]
    triv = pfg.get_trivial_sectors(pc.TRIV_FILE, cut=CUT, n_indices=11)
    improved = pc.build_improved_seeds(triv)
    seo0, av0 = pfg.gen_eqs(eqt, triv, pc.M_VALS, improved)
    sol0 = pfg.solve_eqs_modulo(nonempty_eqs(seo0), pfg.sort_integrals_desc(av0), pc.MODULUS)
    M = [t[0] for t in sol0[(1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -2)]]
    Mset = set(M)
    del sol0, seo0, av0
    print(f"quad cut {CUT}: uncut={uncut} masters={len(M)}", flush=True)

    def closes(corner_len):
        seeds = pfg.sort_integrals_desc([s for s in build(uncut, corner_len) if pfg.to_sector(s) not in triv])
        seo, av = pfg.gen_eqs(eqt, triv, pc.M_VALS, seeds)
        present = [m for m in M if m in set(av)]
        sol = pfg.solve_eqs_modulo(nonempty_eqs(seo), pfg.sort_integrals_desc(av), pc.MODULUS,
                                   keep_on_rhs=present, complete_pivoting=True, needed_variables={TARGET})
        red = sol.get(TARGET)
        ok = red is not None and not [v for v, _ in red if v not in Mset] and len(present) == len(M)
        return ok, len(seeds)

    pairs = [frozenset(c) for c in itertools.combinations(uncut, 2)]
    triples = [frozenset(c) for c in itertools.combinations(uncut, 3)]
    corner_len = {A: 10 for A in pairs}
    corner_len.update({A: 9 for A in triples})           # quad corner left out => dropped

    ok, n = closes(corner_len)
    print(f"baseline (pairs 10, triples 9, quad dropped): {n} seeds  {'CLOSES' if ok else 'FAILS'}", flush=True)

    # Phase 1: pairs -> minimal length (binary search), greedy
    for A in sorted(pairs, key=sorted):
        def cl(L):
            cc = dict(corner_len); cc[A] = L
            return closes(cc)[0]
        lo, hi = -1, corner_len[A]
        if not cl(hi):
            print(f"  pair {sorted(A)}: rule length fails?!", flush=True); continue
        while lo < hi:
            mid = (lo + hi) // 2
            if cl(mid):
                hi = mid
            else:
                lo = mid + 1
        corner_len[A] = hi
        print(f"  pair {sorted(A)}: min length = {hi}", flush=True)

    # Phase 2: triples -> check if needed, else decrement from rule (9) one rung at a time
    for A in sorted(triples, key=sorted):
        minL = corner_len[A]
        for trial in range(corner_len[A], -2, -1):
            cc = dict(corner_len); cc[A] = trial
            if closes(cc)[0]:
                minL = trial
            else:
                break
        corner_len[A] = minL
        print(f"  triple {sorted(A)}: {'DROPPED (not needed)' if minL < 0 else f'min length = {minL}'}", flush=True)

    ok, n = closes(corner_len)
    print(f"\nFINAL pruned quad cut: {n} seeds  {'CLOSES' if ok else 'FAILS(!)'}", flush=True)
    lens = {tuple(sorted(A)): L for A, L in sorted(corner_len.items(), key=lambda kv: (len(kv[0]), sorted(kv[0])))}
    print("corner lengths:", lens, flush=True)


if __name__ == "__main__":
    main()
