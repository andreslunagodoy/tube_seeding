"""
path_coverage_rank10.py

For each of the 66 zig-zag paths used in Table 6 of the paper, attempt to
reduce ALL 66 rank-10 targets simultaneously (passing them all as
needed_variables to pfg.solve_eqs_modulo). Record which targets each path
can reduce. The result is a 66x66 reducibility matrix used to find a
minimal subset of paths that covers all 66 targets (set cover).

Each invocation handles a SLICE of the 66 paths (env var PATH_INDICES,
comma-separated 0-based indices) so different processes can run distinct
paths in parallel.
"""
from __future__ import annotations
import json
import os
import sys
import time

os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")

import pyfeyngym as pfg

DP_DIR = os.path.dirname(os.path.abspath(__file__))
IBP_FILE = DP_DIR + "/IBP_LI"
TRIV_FILE = DP_DIR + "/trivialsector"
MODULUS = 2**31 - 1
M_VALS = {"d": 23, "m1": 3, "m2": 5, "m3": 17, "m4": 23}
TOP_SECTOR = (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0)
CUT = [int(x) for x in os.environ.get("CUT", "1,3,6,8").split(",")]
RANK = 10

# Path assignments copied from Table 6 of stripseeding.tex (column "Path").
# Each entry is ((l, m, n), [axis_indices_in_last_3]) where axis index 0=a_9,
# 1=a_10, 2=a_11 corresponds to last_3 indices (8, 9, 10) of the 11-tuple.
# An empty axis list means the target IS the origin or single 1-axis with
# no zig-zag width (and the depth comes from the nonzero (l,m,n) component).
def axis_idx(label):
    return {9: 0, 10: 1, 11: 2}[label]


# (l, m, n): list of axis-labels in path order
PATHS_RAW = {
    (0, 0, 10): [11],
    (0, 1, 9):  [11, 10],
    (0, 2, 8):  [11, 10],
    (0, 3, 7):  [11, 10],
    (0, 4, 6):  [11, 10],
    (0, 5, 5):  [10, 11],
    (0, 6, 4):  [10, 11],
    (0, 7, 3):  [10, 11],
    (0, 8, 2):  [10, 11],
    (0, 9, 1):  [10, 11],
    (0, 10, 0): [10],
    (1, 0, 9):  [11, 9],
    (1, 1, 8):  [11, 9, 10],
    (1, 2, 7):  [11, 10, 9],
    (1, 3, 6):  [11, 10, 9],
    (1, 4, 5):  [11, 10, 9],
    (1, 5, 4):  [10, 11, 9],
    (1, 6, 3):  [10, 11, 9],
    (1, 7, 2):  [10, 11, 9],
    (1, 8, 1):  [10, 9, 11],
    (1, 9, 0):  [10, 9],
    (2, 0, 8):  [11, 9],
    (2, 1, 7):  [11, 9, 10],
    (2, 2, 6):  [11, 9, 10],
    (2, 3, 5):  [11, 10, 9],
    (2, 4, 4):  [11, 9, 10],
    (2, 5, 3):  [10, 11, 9],
    (2, 6, 2):  [10, 9, 11],
    (2, 7, 1):  [10, 9, 11],
    (2, 8, 0):  [10, 9],
    (3, 0, 7):  [11, 9],
    (3, 1, 6):  [11, 9, 10],
    (3, 2, 5):  [11, 9, 10],
    (3, 3, 4):  [9, 10, 11],
    (3, 4, 3):  [10, 9, 11],
    (3, 5, 2):  [10, 9, 11],
    (3, 6, 1):  [10, 9, 11],
    (3, 7, 0):  [10, 9],
    (4, 0, 6):  [11, 9],
    (4, 1, 5):  [11, 9, 10],
    (4, 2, 4):  [9, 10, 11],
    (4, 3, 3):  [9, 10, 11],
    (4, 4, 2):  [9, 10, 11],
    (4, 5, 1):  [10, 9, 11],
    (4, 6, 0):  [10, 9],
    (5, 0, 5):  [9, 11],
    (5, 1, 4):  [9, 11, 10],
    (5, 2, 3):  [9, 11, 10],
    (5, 3, 2):  [9, 10, 11],
    (5, 4, 1):  [9, 10, 11],
    (5, 5, 0):  [9, 10],
    (6, 0, 4):  [9, 11],
    (6, 1, 3):  [9, 11, 10],
    (6, 2, 2):  [9, 10, 11],
    (6, 3, 1):  [9, 10, 11],
    (6, 4, 0):  [9, 10],
    (7, 0, 3):  [9, 11],
    (7, 1, 2):  [9, 11, 10],
    (7, 2, 1):  [9, 10, 11],
    (7, 3, 0):  [9, 10],
    (8, 0, 2):  [9, 11],
    (8, 1, 1):  [9, 10, 11],
    (8, 2, 0):  [9, 10],
    (9, 0, 1):  [9, 11],
    (9, 1, 0):  [9, 10],
    (10, 0, 0): [9],
}

# Canonical target list, in the order used by Table 6:
# row-major over l, with m+n = 10-l.
ALL_TARGETS = []
for l in range(RANK + 1):
    for m in range(RANK - l + 1):
        n = RANK - l - m
        ALL_TARGETS.append((l, m, n))
assert len(ALL_TARGETS) == 66
assert all(t in PATHS_RAW for t in ALL_TARGETS)


def build_improved_seeds(trivial_sectors, param=4):
    s_max, r_max, d_max = param, 8, 0
    starting = pfg.gen_all_seeds(TOP_SECTOR, trivial_sectors, s_max, r_max, d_max)
    return [
        s for s in starting
        if (pfg.d_level(s) <= 0
            and pfg.s_level(s) <= max(1, pfg.t_level(s) - param))
        or (s[3] <= 0 and s[4] <= 0 and s[6] <= 0 and pfg.d_level(s) <= 0
            and pfg.s_level(s) <= max(1, pfg.t_level(s) - param + 1))
    ]


def build_path_seeds(target, improved, trivial_sectors, width=2):
    """Build the zig-zag tube seed set for a TARGET, using the EXACT axis
    ordering from PATHS_RAW. Junction width (width of strip k+1 in the
    transverse axes already traversed) is `width`.
    """
    l, m, n = target
    depths = {0: l, 1: m, 2: n}   # axis index -> depth
    path = [axis_idx(a) for a in PATHS_RAW[target]]
    seed_set = set(improved)

    if len(path) == 0:
        pass
    elif len(path) == 1:
        ax = path[0]
        d = depths[ax]
        # 1-axis path follows scan_n_feyngym_metrics.build_strip_seeds:
        # n_shifts = max(1, depth - 3), shifts k = 0..n_shifts-1.
        n_shifts = max(1, d - 3)
        for k in range(n_shifts):
            for s in improved:
                idx = list(s)
                idx[8 + ax] -= k
                seed_set.add(tuple(idx))
    elif len(path) == 2:
        ax1, ax2 = path
        d1, d2 = depths[ax1], depths[ax2]
        for k1 in range(1, d1 + 1):
            for s in improved:
                idx = list(s); idx[8 + ax1] -= k1
                seed_set.add(tuple(idx))
        for k2 in range(1, d2 + 1):
            for da1 in range(max(0, d1 - 1), d1 + 1):
                for s in improved:
                    idx = list(s)
                    idx[8 + ax1] -= da1
                    idx[8 + ax2] -= k2
                    seed_set.add(tuple(idx))
    else:
        ax1, ax2, ax3 = path
        d1, d2, d3 = depths[ax1], depths[ax2], depths[ax3]
        for k1 in range(1, d1 + 1):
            for s in improved:
                idx = list(s); idx[8 + ax1] -= k1
                seed_set.add(tuple(idx))
        for k2 in range(1, d2 + 1):
            for da1 in range(max(0, d1 - 1), d1 + 1):
                for s in improved:
                    idx = list(s)
                    idx[8 + ax1] -= da1
                    idx[8 + ax2] -= k2
                    seed_set.add(tuple(idx))
        for k3 in range(1, d3 + 1):
            for da1 in range(max(0, d1 - 1), d1 + 1):
                for da2 in range(max(0, d2 - 1), d2 + 1):
                    for s in improved:
                        idx = list(s)
                        idx[8 + ax1] -= da1
                        idx[8 + ax2] -= da2
                        idx[8 + ax3] -= k3
                        seed_set.add(tuple(idx))

    valid = [s for s in seed_set if pfg.to_sector(s) not in trivial_sectors]
    return pfg.sort_integrals_desc(valid)


def main():
    indices = [int(x) for x in os.environ.get(
        "PATH_INDICES", ",".join(str(i) for i in range(len(ALL_TARGETS)))
    ).split(",") if x.strip()]
    out_file = os.environ.get(
        "OUT_FILE", "path_coverage_rank10_results.json")

    eq_templates = pfg.gen_eq_templates(IBP_FILE, M_VALS)
    trivial_sectors = pfg.get_trivial_sectors(TRIV_FILE, cut=CUT, n_indices=11)
    improved = build_improved_seeds(trivial_sectors)

    # Fix the master basis once (same as Table 6).
    seo0, av0 = pfg.gen_eqs(eq_templates, trivial_sectors, M_VALS, improved)
    sol0 = pfg.solve_eqs_modulo(
        [a[-1] for a in seo0], pfg.sort_integrals_desc(av0), MODULUS)
    masters = [t[0] for t in sol0[(1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -2)]]
    print(f"masters: {len(masters)}", flush=True)
    del sol0, seo0, av0

    target_tuples = [
        (1, 1, 1, 1, 1, 1, 1, 1, -l, -m, -n)
        for (l, m, n) in ALL_TARGETS
    ]
    target_set = set(target_tuples)

    results = {}
    for i in indices:
        l, m, n = ALL_TARGETS[i]
        target = (1, 1, 1, 1, 1, 1, 1, 1, -l, -m, -n)
        path_label = "->".join(str(a) for a in PATHS_RAW[(l, m, n)])
        print(f"\n=== path {i:>2}/{len(ALL_TARGETS)} "
              f"target=({l},{m},{n}) path={path_label} ===", flush=True)

        seeds = build_path_seeds((l, m, n), improved, trivial_sectors,
                                 width=2)
        t0 = time.time()
        seo, av = pfg.gen_eqs(eq_templates, trivial_sectors, M_VALS, seeds)
        equations = [a[-1] for a in seo]
        svars = pfg.sort_integrals_desc(av)
        tgen = time.time() - t0
        n_eqs = len(seo); n_vars = len(svars)
        del seo, av

        t0 = time.time()
        sol = pfg.solve_eqs_modulo(
            equations, svars, MODULUS,
            keep_on_rhs=masters,
            complete_pivoting=True,
            needed_variables=target_set,
        )
        tsolve = time.time() - t0

        # For each of the 66 targets, did this path reduce it?
        coverage = []
        own_ok = False
        for j, tgt in enumerate(target_tuples):
            if tgt in sol and len(sol[tgt]) == len(masters):
                coverage.append(j)
                if tgt == target:
                    own_ok = True
        print(f"  seeds={len(seeds)} eqs={n_eqs} vars={n_vars} "
              f"tgen={tgen:.1f}s tsolve={tsolve:.1f}s "
              f"covers={len(coverage)}/66 own_ok={own_ok}", flush=True)

        results[i] = {
            "target_index": i,
            "target_lmn": [l, m, n],
            "path": path_label,
            "n_seeds": len(seeds),
            "n_eqs": n_eqs,
            "n_vars": n_vars,
            "tgen_s": round(tgen, 2),
            "tsolve_s": round(tsolve, 2),
            "covers": coverage,
            "n_covers": len(coverage),
            "own_ok": own_ok,
        }
        # incremental save
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

        del sol, equations, svars

    print(f"\nWrote {out_file}", flush=True)


if __name__ == "__main__":
    main()
