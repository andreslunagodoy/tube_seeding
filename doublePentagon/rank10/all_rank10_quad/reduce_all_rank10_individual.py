"""
reduce_all_rank10_individual.py

Reduce each rank-10 integral individually using the shortest zig-zag
tube for that specific target. For (0,0,-n): strip along a_11 only.
For (0,-m,-n): zig-zag along a_10 then a_11. For (-l,-m,-n): full
zig-zag along a_9, a_10, a_11.
"""
from __future__ import annotations
import json
import os
import sys
import time

os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")

import pyfeyngym as pfg

DP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IBP_FILE = DP_DIR + "/IBP_LI"
TRIV_FILE = DP_DIR + "/trivialsector"
MODULUS = 2**31 - 1
M_VALS = {"d": 23, "m1": 3, "m2": 5, "m3": 17, "m4": 23}
TOP_SECTOR = (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0)
CUT = [1, 3, 6, 8]
RANK = int(os.environ.get("RANK", "10"))


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


def build_tube_for_target(l, m, n, improved, trivial_sectors, width=2):
    """Build the shortest zig-zag tube for target (1,...,1,-l,-m,-n).
    Only extends along axes with nonzero ISP powers.
    Uses width 2 at junctions between axes."""
    seed_set = set(improved)  # ball at origin

    # Determine which axes to traverse, ordered by DECREASING depth.
    # The longest axis goes first as the thin strip (width 1);
    # shorter axes follow with junction width w, so the multiplier
    # applies to the smallest depth — minimising total seeds.
    axes = []  # list of (axis_index_in_last_3, depth)
    if l > 0: axes.append((0, l))  # a_9
    if m > 0: axes.append((1, m))  # a_10
    if n > 0: axes.append((2, n))  # a_11
    axes.sort(key=lambda x: -x[1])  # largest depth first

    if len(axes) == 0:
        # Target is the top-sector integral itself, already in ball
        pass
    elif len(axes) == 1:
        # Single strip along one axis
        ax, depth = axes[0]
        n_shifts = max(1, depth - 3)
        for k in range(n_shifts):
            for s in improved:
                idx = list(s)
                idx[8 + ax] -= k
                seed_set.add(tuple(idx))
    elif len(axes) == 2:
        # Zig-zag along two axes
        ax1, d1 = axes[0]
        ax2, d2 = axes[1]
        # Strip 1: along ax1
        for k1 in range(1, d1 + 1):
            for s in improved:
                idx = list(s)
                idx[8 + ax1] -= k1
                seed_set.add(tuple(idx))
        # Strip 2: along ax2, at ax1 = -(d1-1) and -(d1)
        for k2 in range(1, d2 + 1):
            for da1 in range(max(0, d1 - 1), d1 + 1):
                for s in improved:
                    idx = list(s)
                    idx[8 + ax1] -= da1
                    idx[8 + ax2] -= k2
                    seed_set.add(tuple(idx))
    else:
        # Full zig-zag along three axes
        ax1, d1 = axes[0]
        ax2, d2 = axes[1]
        ax3, d3 = axes[2]
        # Strip 1: along ax1
        for k1 in range(1, d1 + 1):
            for s in improved:
                idx = list(s)
                idx[8 + ax1] -= k1
                seed_set.add(tuple(idx))
        # Strip 2: along ax2, width 2 in ax1
        for k2 in range(1, d2 + 1):
            for da1 in range(max(0, d1 - 1), d1 + 1):
                for s in improved:
                    idx = list(s)
                    idx[8 + ax1] -= da1
                    idx[8 + ax2] -= k2
                    seed_set.add(tuple(idx))
        # Strip 3: along ax3, width 2 in ax1 and ax2
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
    # Build path description
    ax_names = {0: 'a9', 1: 'a10', 2: 'a11'}
    if len(axes) == 0:
        path_str = '—'
        width_used = 0
    elif len(axes) == 1:
        path_str = ax_names[axes[0][0]]
        width_used = 0
    else:
        path_str = '->'.join(ax_names[a[0]] for a in axes)
        width_used = width
    return pfg.sort_integrals_desc(valid), path_str, width_used


def main():
    rank = RANK
    print(f"=== Reduce all rank-{rank} integrals individually ===", flush=True)

    eq_templates = pfg.gen_eq_templates(IBP_FILE, M_VALS)
    trivial_sectors = pfg.get_trivial_sectors(TRIV_FILE, cut=CUT, n_indices=11)
    improved = build_improved_seeds(trivial_sectors)

    # Masters
    seo0, av0 = pfg.gen_eqs(eq_templates, trivial_sectors, M_VALS, improved)
    sol0 = pfg.solve_eqs_modulo(
        [a[-1] for a in seo0], pfg.sort_integrals_desc(av0), MODULUS)
    masters = [t[0] for t in sol0[(1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -2)]]
    print(f"{len(masters)} masters", flush=True)
    del sol0, seo0, av0

    # Warmup: do one small solve to JIT-compile Julia code paths
    print("Warming up Julia solver...", flush=True)
    t0 = time.time()
    warmup_seeds = improved[:50]
    seo_w, av_w = pfg.gen_eqs(eq_templates, trivial_sectors, M_VALS, warmup_seeds)
    eqs_w = [a[-1] for a in seo_w]
    sv_w = pfg.sort_integrals_desc(av_w)
    _ = pfg.solve_eqs_modulo(eqs_w, sv_w, MODULUS, complete_pivoting=True)
    print(f"  Warmup done in {time.time()-t0:.1f}s", flush=True)
    del seo_w, av_w, eqs_w, sv_w

    # Enumerate targets
    targets = []
    for l in range(rank + 1):
        for m in range(rank - l + 1):
            n = rank - l - m
            targets.append((l, m, n))
    print(f"{len(targets)} targets with l+m+n={rank}\n", flush=True)

    results = []
    total_tgen = 0
    total_tsolve = 0
    max_rss = 0
    n_ok = 0

    print(f"{'(l,m,n)':>10} {'path':<16} {'seeds':>5} {'eqs':>6} "
          f"{'Tgen':>6} {'Tsolve':>7} {'RSSg':>5} {'RSSs':>5} {'ok':>4}",
          flush=True)
    print("-" * 78, flush=True)

    def get_rss_mib():
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) // 1024
        except:
            pass
        return 0

    for l, m, n in targets:
        target = (1, 1, 1, 1, 1, 1, 1, 1, -l, -m, -n)

        # For 0 or 1 nonzero ISPs, no junction width needed.
        # For 2+ nonzero ISPs, try width=2 first, then 3, then 4.
        n_nonzero = sum(1 for x in [l, m, n] if x > 0)
        widths_to_try = [0] if n_nonzero <= 1 else [2, 3, 4]
        ok = False
        for width in widths_to_try:
            seeds, path_str, _ = build_tube_for_target(
                l, m, n, improved, trivial_sectors, width=width)

            t0 = time.time()
            seo, av = pfg.gen_eqs(eq_templates, trivial_sectors, M_VALS, seeds)
            equations = [a[-1] for a in seo]
            svars = pfg.sort_integrals_desc(av)
            tgen = time.time() - t0
            rss_gen = get_rss_mib()

            t0 = time.time()
            sol = pfg.solve_eqs_modulo(
                equations, svars, MODULUS,
                keep_on_rhs=masters,
                complete_pivoting=True,
                needed_variables={target},
            )
            tsolve = time.time() - t0
            rss_solve = get_rss_mib()

            ok = target in sol and len(sol[target]) == len(masters)
            if ok:
                break
            else:
                del sol, equations, svars, seo, av
                equations = None  # mark as failed for this width

        n_eqs = len(equations) if equations is not None else 0
        print(f"({l},{m},{n}){' '*(7-len(f'({l},{m},{n})'))} "
              f"w={width} {path_str:<14} {len(seeds):>5} {n_eqs:>6} "
              f"{tgen:>5.1f}s {tsolve:>6.1f}s "
              f"{rss_gen:>5} {rss_solve:>5} "
              f"{'OK' if ok else 'FAIL':>4}", flush=True)

        total_tgen += tgen
        total_tsolve += tsolve
        if rss_solve > max_rss:
            max_rss = rss_solve
        if ok:
            n_ok += 1

        results.append({
            "l": l, "m": m, "n": n,
            "path": path_str, "width": width,
            "n_seeds": len(seeds), "n_eqs": n_eqs,
            "tgen_s": round(tgen, 2), "tsolve_s": round(tsolve, 2),
            "rss_gen_mib": rss_gen, "rss_solve_mib": rss_solve,
            "ok": ok,
        })
        # Clean up
        sol = equations = svars = seo = av = None

    print(f"\n=== Summary ===", flush=True)
    print(f"Rank: {rank}", flush=True)
    print(f"Targets: {len(targets)}, OK: {n_ok}, FAIL: {len(targets)-n_ok}",
          flush=True)
    print(f"Total Tgen: {total_tgen:.1f}s", flush=True)
    print(f"Total Tsolve: {total_tsolve:.1f}s", flush=True)
    print(f"Total wall: {total_tgen + total_tsolve:.1f}s", flush=True)
    print(f"Peak RSS (hwm): {max_rss} MiB", flush=True)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rank10_individual_results.json"), "w") as f:
        json.dump({
            "rank": rank, "n_targets": len(targets),
            "n_ok": n_ok, "total_tgen_s": round(total_tgen, 1),
            "total_tsolve_s": round(total_tsolve, 1),
            "peak_rss_mib": max_rss,
            "per_target": results,
        }, f, indent=2)


if __name__ == "__main__":
    main()
