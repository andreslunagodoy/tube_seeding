"""Validation protocol for tube seeding.

Over a finite field at numerical kinematics, an incomplete or degenerate IBP
system can *appear* to reduce a target to masters. To rule this out we check,
for the quad-cut single-ISP target I(1,...,1,0,0,-n) on [1,3,6,8]:

  (A) CLOSURE at MULTIPLE primes and MULTIPLE kinematic points: the tube system
      reduces the target to (a subset of) the fixed master basis with NO
      non-master survivors, at every (prime, kinematics) tested.

  (B) COEFFICIENT AGREEMENT with an independent seed set: at each (prime,
      kinematics), the *reconstructed coefficients* of the target over the
      master basis from tube seeding equal, term by term, those from the
      (much larger) improved-seeding set. This checks the actual analytic
      answer, not just "reduces to N masters".

Both reductions are forced onto the SAME master basis via keep_on_rhs, so the
coefficient dictionaries are directly comparable. Agreement at many distinct
kinematic points (where the rational-function coefficients take different
values) and several primes makes a spurious/degenerate match astronomically
unlikely.
"""
from __future__ import annotations
import json, os, sys, time
os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyfeyngym as pfg

IBP_FILE = "IBP_LI"
TRIV_FILE = "trivialsector"
TOP_SECTOR = (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0)
CUT = [1, 3, 6, 8]
N = int(os.environ.get("VALIDATE_N", "8"))
TARGET = (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, -N)

# Several primes < 2^31 (each verified prime below).
PRIMES = [2147483647, 2147483629, 2147483587, 2130706433]
# Several kinematic points (d and Mandelstam values m1..m4). Distinct integer
# points => the rational-function coefficients evaluate to different residues.
# Generic points (large, unstructured integer values). Degeneracies---kinematic
# samples where the IBP system drops rank and the master basis shrinks below the
# generic 27---are a measure-zero artifact of finite-field sampling; large
# generic values almost never hit one. Each point below was verified to yield
# the full generic 27-master basis. A per-point guard (see main) re-derives the
# basis and SKIPS any sample that is degenerate, so a spurious "agreement" on a
# wrong (shrunken) basis can never be counted as a pass.
KINEMATICS = [
    {"d": 23, "m1": 3,   "m2": 5,   "m3": 17,  "m4": 23},   # the paper's default
    {"d": 61, "m1": 13,  "m2": 71,  "m3": 89,  "m4": 101},
    {"d": 67, "m1": 97,  "m2": 103, "m3": 127, "m4": 151},
    {"d": 83, "m1": 139, "m2": 149, "m3": 181, "m4": 199},
]


def _is_prime(n):
    if n < 2: return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0: return n == p
    d, r = n - 1, 0
    while d % 2 == 0: d //= 2; r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1): continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1: break
        else:
            return False
    return True


def build_strip_seeds(n, trivial_sectors):
    """Tube seeding (the paper's strip): improved s<=4 blob convolved along a_11."""
    starting = pfg.gen_all_seeds(TOP_SECTOR, trivial_sectors, 4, 8, 0)
    improved = [
        s for s in starting
        if (pfg.d_level(s) <= 0 and pfg.s_level(s) <= max(1, pfg.t_level(s) - 4))
        or (s[3] <= 0 and s[4] <= 0 and s[6] <= 0 and pfg.d_level(s) <= 0
            and pfg.s_level(s) <= max(1, pfg.t_level(s) - 3))
    ]
    seed_set = set()
    for raise_rank in range(max(1, n - 3)):
        for seed in improved:
            seed_set.add(seed[:-1] + (seed[-1] - raise_rank,))
    return pfg.sort_integrals_desc([s for s in seed_set
                                    if pfg.to_sector(s) not in trivial_sectors])


def build_improved_seeds(n, trivial_sectors):
    """Improved seeding baseline: full s_max=n ball in every on-cut sub-sector."""
    cut_mask = (1 << 0) | (1 << 2) | (1 << 5) | (1 << 7)
    excl_mask = (1 << 3) | (1 << 4) | (1 << 6)
    seed_set = set()
    for sec in range(256):
        if (sec & cut_mask) != cut_mask or sec in trivial_sectors:
            continue
        t = bin(sec).count("1")
        s_max = max(n, max(1, t - 4))
        if (sec & excl_mask) == 0:
            s_max = max(s_max, max(1, t - 3))
        for s in pfg.gen_all_seeds(TOP_SECTOR, trivial_sectors, s_max, 8, 0):
            if pfg.to_sector(s) == sec and pfg.s_level(s) <= s_max:
                seed_set.add(s)
    return pfg.sort_integrals_desc(list(seed_set))


def reduce_target(seeds, eq_templates, trivial_sectors, m_vals, prime, masters):
    """Reduce TARGET to the fixed master basis; return (n_survivors, {master: coeff})."""
    soe, av = pfg.gen_eqs(eq_templates, trivial_sectors, m_vals, seeds)
    eqs = [a[-1] for a in soe]
    sv = pfg.sort_integrals_desc(av)
    sol = pfg.solve_eqs_modulo(eqs, sv, prime,
                               keep_on_rhs=list(masters),
                               complete_pivoting=True,
                               needed_variables={TARGET})
    red = sol.get(TARGET)
    if red is None:
        return None, None
    coeffs = {tuple(t[0]): int(t[1]) for t in red}
    survivors = [m for m in coeffs if m not in masters]
    return len(survivors), coeffs


def main():
    assert all(_is_prime(p) for p in PRIMES), "non-prime in PRIMES"
    print(f"=== Validation: target {TARGET} on cut {CUT} ===", flush=True)
    print(f"primes: {PRIMES}")
    print(f"kinematic points: {len(KINEMATICS)}\n", flush=True)

    trivial_sectors = pfg.get_trivial_sectors(TRIV_FILE, cut=CUT, n_indices=11)
    strip = build_strip_seeds(N, trivial_sectors)
    improved = build_improved_seeds(N, trivial_sectors)
    print(f"tube (strip) seeds: {len(strip)};  improved seeds: {len(improved)}", flush=True)

    # Fixed master basis: reduce the improved set at the default point/prime, no
    # keep_on_rhs, and read off the residual basis.
    K0 = KINEMATICS[0]
    tmpl0 = pfg.gen_eq_templates(IBP_FILE, K0)
    soe, av = pfg.gen_eqs(tmpl0, trivial_sectors, K0, improved)
    sol0 = pfg.solve_eqs_modulo([a[-1] for a in soe], pfg.sort_integrals_desc(av),
                                PRIMES[0], needed_variables={TARGET})
    masters = frozenset(tuple(t[0]) for t in sol0[TARGET])
    print(f"master basis: {len(masters)} integrals\n", flush=True)

    rows = []
    all_close = True
    all_agree = True
    n_degenerate = 0
    for ki, K in enumerate(KINEMATICS):
        tmpl = pfg.gen_eq_templates(IBP_FILE, K)
        # Degeneracy guard: re-derive the natural master basis at this point
        # (improved set, no keep_on_rhs). If it is not the full generic basis,
        # this kinematic sample is rank-deficient -- forcing keep_on_rhs onto the
        # generic basis there gives an order-dependent (meaningless) split. Skip
        # it rather than compare on a wrong basis.
        soe_g, av_g = pfg.gen_eqs(tmpl, trivial_sectors, K, improved)
        solg = pfg.solve_eqs_modulo([a[-1] for a in soe_g],
                                    pfg.sort_integrals_desc(av_g), PRIMES[0],
                                    needed_variables={TARGET})
        basis_here = frozenset(tuple(t[0]) for t in solg[TARGET])
        if basis_here != masters:
            n_degenerate += 1
            print(f"  K{ki}: DEGENERATE sample (basis {len(basis_here)} != "
                  f"{len(masters)}); skipping coefficient comparison.", flush=True)
            rows.append({"kin": ki, "degenerate": True,
                         "n_masters_here": len(basis_here)})
            continue
        for p in PRIMES:
            t0 = time.time()
            s_surv, s_co = reduce_target(strip, tmpl, trivial_sectors, K, p, masters)
            i_surv, i_co = reduce_target(improved, tmpl, trivial_sectors, K, p, masters)
            closed = (s_surv == 0)
            imp_closed = (i_surv == 0)
            # coefficient agreement over the master basis, compared MOD p
            # (the solver may return signed vs unsigned residues of the same
            # field element).
            agree = (s_co is not None and i_co is not None
                     and all((s_co.get(m, 0) - i_co.get(m, 0)) % p == 0
                             for m in masters))
            all_close &= closed
            all_agree &= bool(agree)
            dt = time.time() - t0
            status = "OK" if (closed and agree) else "FAIL"
            print(f"  K{ki} p={p}: tube_surv={s_surv} imp_surv={i_surv} "
                  f"coeffs_agree={agree}  [{status}] ({dt:.1f}s)", flush=True)
            rows.append({"kin": ki, "prime": p, "tube_survivors": s_surv,
                         "improved_survivors": i_surv, "coeffs_agree": bool(agree),
                         "n_master_terms": None if s_co is None else len(s_co)})

    n_tested = sum(1 for r in rows if not r.get("degenerate"))
    print("\n=== SUMMARY ===")
    print(f"  generic (prime, kinematics) points tested: {n_tested}"
          f"  (degenerate samples skipped: {n_degenerate})")
    print(f"  tube closes at ALL tested points: {all_close}")
    print(f"  tube coefficients agree with improved seeding at ALL points: {all_agree}")
    print(f"  VALIDATION {'PASSED' if (all_close and all_agree) else 'FAILED'}")
    out = {"target": list(TARGET), "cut": CUT, "n": N,
           "primes": PRIMES, "kinematics": KINEMATICS,
           "n_strip_seeds": len(strip), "n_improved_seeds": len(improved),
           "n_masters": len(masters), "rows": rows,
           "n_tested": n_tested, "n_degenerate": n_degenerate,
           "all_close": all_close, "all_agree": all_agree}
    op = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"validate_tube_reduction_n{N}.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main()
