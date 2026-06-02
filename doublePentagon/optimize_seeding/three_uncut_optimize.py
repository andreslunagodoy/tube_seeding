"""Work up the compositional lift one rung at a time: the THREE-uncut-propagator
case (relax 3 props p,q,r; cut = the other 5).  Corners: 3 singles (k=1), 3 pairs
(k=2), 1 triple (k=3).  For each of the C(8,3)=56 triples, apply the per-corner
length pruning IN ORDER, so we can see whether shortening the pairs makes the
triple corner necessary (es_three_minlen found the triple unneeded only with the
pairs held at full length):

  - singles : fixed at the nested rule length 11.
  - pairs   : minimal length, binary-searched in [-1, 10]  (L=-1 => dropped).
  - triple  : with the pairs now at their minima, first test whether it is needed
              at all (drop it); if it is, decrement the length from the rule 9 one
              rung at a time until closure fails -> minimal length.

Greedy coordinate descent re-solves the full system at every step, so the running
configuration always closes.  Reports per triple: pair minima, triple status, and
the pruned seed count vs the full nested baseline (singles 11, pairs 10, triple 9)."""
import os, sys, re, itertools, json
os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rank10"))
import path_coverage_rank10 as pc
import pyfeyngym as pfg

ES_SEEDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "es_seeds_maxcut_rank12.txt")
TARGET = (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, -12)
S = 1
OUTJSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "three_uncut_optimize.json")


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


def main():
    eqt = pfg.gen_eq_templates(pc.IBP_FILE, pc.M_VALS)
    print("THREE uncut propagators: per-corner length pruning over all 56 triples", flush=True)
    print("(singles fixed 11; pairs binary-min; triple checked-then-decremented from 9)\n", flush=True)
    results = {}
    for uncut in itertools.combinations(range(1, 9), 3):
        uncut = list(uncut)
        cut = [p for p in range(1, 9) if p not in uncut]
        triv = pfg.get_trivial_sectors(pc.TRIV_FILE, cut=cut, n_indices=11)
        improved = pc.build_improved_seeds(triv)
        seo0, av0 = pfg.gen_eqs(eqt, triv, pc.M_VALS, improved)
        sol0 = pfg.solve_eqs_modulo([a[-1] for a in seo0], pfg.sort_integrals_desc(av0), pc.MODULUS)
        M = [t[0] for t in sol0[(1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -2)]]
        Mset = set(M)
        del sol0, seo0, av0

        def closes(corner_len):
            seeds = pfg.sort_integrals_desc([s for s in build(uncut, corner_len) if pfg.to_sector(s) not in triv])
            seo, av = pfg.gen_eqs(eqt, triv, pc.M_VALS, seeds)
            present = [m for m in M if m in set(av)]
            sol = pfg.solve_eqs_modulo([a[-1] for a in seo], pfg.sort_integrals_desc(av), pc.MODULUS,
                                       keep_on_rhs=present, complete_pivoting=True, needed_variables={TARGET})
            red = sol.get(TARGET)
            ok = red is not None and not [v for v, _ in red if v not in Mset] and len(present) == len(M)
            return ok, len(seeds)

        pairs = [frozenset(c) for c in itertools.combinations(uncut, 2)]
        triple = frozenset(uncut)
        corner_len = {A: 10 for A in pairs}
        corner_len[triple] = 9
        ok0, n0 = closes(corner_len)                       # full nested baseline

        # Phase 1: pairs -> minimal length (binary search), greedy, triple held at 9
        for A in sorted(pairs, key=sorted):
            def cl(L):
                cc = dict(corner_len); cc[A] = L
                return closes(cc)[0]
            lo, hi = -1, corner_len[A]
            if cl(hi):
                while lo < hi:
                    mid = (lo + hi) // 2
                    if cl(mid):
                        hi = mid
                    else:
                        lo = mid + 1
                corner_len[A] = hi

        # Phase 2: triple -> needed at all? if so decrement from 9
        cc = dict(corner_len); cc[triple] = -1
        if closes(cc)[0]:
            corner_len[triple] = -1
            tstat = "not needed"
        else:
            minL = 9
            for trial in range(9, -1, -1):
                cc = dict(corner_len); cc[triple] = trial
                if closes(cc)[0]:
                    minL = trial
                else:
                    break
            corner_len[triple] = minL
            tstat = f"needed, len {minL}"

        okf, nf = closes(corner_len)
        pmin = {tuple(sorted(A)): corner_len[A] for A in sorted(pairs, key=sorted)}
        print(f"  uncut {uncut}: masters={len(M):>2}  pairs={ {f'{a}{b}':L for (a,b),L in pmin.items()} }  "
              f"triple {tstat}  seeds {n0}->{nf}  {'OK' if okf else 'FAIL'}", flush=True)
        results[str(uncut)] = {"masters": len(M), "pairs": {f"{a},{b}": L for (a, b), L in pmin.items()},
                               "triple": corner_len[triple], "triple_needed": corner_len[triple] >= 0,
                               "baseline_seeds": n0, "pruned_seeds": nf, "closes": okf}
        json.dump(results, open(OUTJSON, "w"), indent=2)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
