"""per_path_rss_worker_allcuts.py PATH_INDEX OUT_FILE

Same as per_path_rss_worker.py (the code behind Table tab:all_rank10_cover) but
(i) records the cut in the output and (ii) filters empty equations before each
solve (the SparseVec MethodError guard) so it survives the triple spanning cuts.
The cut is taken from the CUT env var (read by path_coverage_rank10). Run one
(cut, path) per process so the per-path RSS stays clean, exactly as the original.
"""
from __future__ import annotations
import json
import os
import sys
import threading
import time

os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from path_coverage_rank10 import (  # noqa: E402
    ALL_TARGETS, PATHS_RAW, build_improved_seeds, build_path_seeds, axis_idx,
    IBP_FILE, TRIV_FILE, MODULUS, M_VALS, CUT,
)

import pyfeyngym as pfg  # noqa: E402


def build_path_seeds_width(target, improved, trivial_sectors, width):
    """Like path_coverage_rank10.build_path_seeds but the junction window in each
    already-traversed axis is genuinely `width` wide (the original hardcodes 2,
    ignoring its width arg). width=2 reproduces the original exactly."""
    l, m, n = target
    depths = {0: l, 1: m, 2: n}
    path = [axis_idx(a) for a in PATHS_RAW[target]]
    W = width
    S = set(improved)
    if len(path) == 1:
        ax = path[0]; d = depths[ax]
        for k in range(max(1, d - 3)):
            for s in improved:
                t = list(s); t[8 + ax] -= k; S.add(tuple(t))
    elif len(path) == 2:
        a1, a2 = path; d1, d2 = depths[a1], depths[a2]
        for k1 in range(1, d1 + 1):
            for s in improved:
                t = list(s); t[8 + a1] -= k1; S.add(tuple(t))
        for k2 in range(1, d2 + 1):
            for da1 in range(max(0, d1 - W + 1), d1 + 1):
                for s in improved:
                    t = list(s); t[8 + a1] -= da1; t[8 + a2] -= k2; S.add(tuple(t))
    elif len(path) == 3:
        a1, a2, a3 = path; d1, d2, d3 = depths[a1], depths[a2], depths[a3]
        for k1 in range(1, d1 + 1):
            for s in improved:
                t = list(s); t[8 + a1] -= k1; S.add(tuple(t))
        for k2 in range(1, d2 + 1):
            for da1 in range(max(0, d1 - W + 1), d1 + 1):
                for s in improved:
                    t = list(s); t[8 + a1] -= da1; t[8 + a2] -= k2; S.add(tuple(t))
        for k3 in range(1, d3 + 1):
            for da1 in range(max(0, d1 - W + 1), d1 + 1):
                for da2 in range(max(0, d2 - W + 1), d2 + 1):
                    for s in improved:
                        t = list(s); t[8 + a1] -= da1; t[8 + a2] -= da2; t[8 + a3] -= k3
                        S.add(tuple(t))
    valid = [s for s in S if pfg.to_sector(s) not in trivial_sectors]
    return pfg.sort_integrals_desc(valid)


def read_rss_kb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        return 0
    return 0


def nonempty(eqs):
    return [e for e in eqs if len(list(e)) > 0]


class PhaseSampler(threading.Thread):
    def __init__(self, interval=0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_evt = threading.Event()
        self.phase = None
        self.peaks = {}

    def run(self):
        while not self.stop_evt.is_set():
            rss = read_rss_kb()
            if self.phase is not None:
                if rss > self.peaks.get(self.phase, 0):
                    self.peaks[self.phase] = rss
            self.stop_evt.wait(self.interval)


def build_path_braced(target, improved, trivial_sectors, width):
    """Width-1 legs + an anti-diagonal (hypotenuse) brace at each ISP bend, instead
    of the filled rectangular junction. Junction cost is O(width) per bend, not
    O(width^2). Reduces the seed count (hence RAM); coverage must be checked."""
    l, m, n = target
    depths = {0: l, 1: m, 2: n}
    path = [axis_idx(a) for a in PATHS_RAW[target]]
    W = width
    S = set(improved)

    def shift(*pairs):
        for s in improved:
            t = list(s)
            for ax, d in pairs:
                t[8 + ax] -= d
            S.add(tuple(t))

    if len(path) >= 1:
        a1 = path[0]; d1 = depths[a1]
        if len(path) == 1:
            for k in range(max(1, d1 - 3)):
                shift((a1, k))
        else:
            for k in range(1, d1 + 1):
                shift((a1, k))
    if len(path) >= 2:
        a2 = path[1]; d2 = depths[a2]
        for k2 in range(1, d2 + 1):                     # leg2 spine (width-1 in a1)
            shift((a1, d1), (a2, k2))
        for i in range(1, W):                           # bend1 hypotenuse i+k2=W
            k2 = W - i; da1 = d1 - i
            if k2 >= 1 and da1 >= 0:
                shift((a1, da1), (a2, k2))
    if len(path) >= 3:
        a3 = path[2]; d3 = depths[a3]
        for k3 in range(1, d3 + 1):                     # leg3 spine (width-1 in a1,a2)
            shift((a1, d1), (a2, d2), (a3, k3))
        for i in range(1, W):                           # bend2 hypotenuse i+k3=W (in a2,a3)
            k3 = W - i; da2 = d2 - i
            if k3 >= 1 and da2 >= 0:
                shift((a1, d1), (a2, da2), (a3, k3))
    valid = [s for s in S if pfg.to_sector(s) not in trivial_sectors]
    return pfg.sort_integrals_desc(valid)


def main():
    idx = int(sys.argv[1])
    out_file = sys.argv[2]
    width = int(sys.argv[3]) if len(sys.argv) > 3 else 2   # junction width (default 2)
    mode = sys.argv[4] if len(sys.argv) > 4 else "rect"    # "rect" or "braced"
    cut_tag = ",".join(str(c) for c in CUT)

    sampler = PhaseSampler()
    sampler.start()

    sampler.phase = "setup"
    eq_templates = pfg.gen_eq_templates(IBP_FILE, M_VALS)
    trivial_sectors = pfg.get_trivial_sectors(TRIV_FILE, cut=CUT, n_indices=11)
    improved = build_improved_seeds(trivial_sectors)

    sampler.phase = "masters"
    seo0, av0 = pfg.gen_eqs(eq_templates, trivial_sectors, M_VALS, improved)
    sol0 = pfg.solve_eqs_modulo(
        nonempty([a[-1] for a in seo0]), pfg.sort_integrals_desc(av0), MODULUS)
    masters = [t[0] for t in sol0[(1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -2)]]
    del sol0, seo0, av0

    target_set = {(1, 1, 1, 1, 1, 1, 1, 1, -l, -m, -n)
                  for (l, m, n) in ALL_TARGETS}

    l, m, n = ALL_TARGETS[idx]
    if mode == "braced":
        seeds = build_path_braced((l, m, n), improved, trivial_sectors, width)
    else:
        seeds = build_path_seeds_width((l, m, n), improved, trivial_sectors, width)

    sampler.phase = "gen"
    t0 = time.time()
    seo, av = pfg.gen_eqs(eq_templates, trivial_sectors, M_VALS, seeds)
    equations = nonempty([a[-1] for a in seo])
    svars = pfg.sort_integrals_desc(av)
    tgen = time.time() - t0
    n_eqs = len(equations)
    n_vars = len(svars)
    del seo, av

    sampler.phase = "solve"
    t0 = time.time()
    sol = pfg.solve_eqs_modulo(
        equations, svars, MODULUS,
        keep_on_rhs=masters,
        complete_pivoting=True,
        needed_variables=target_set,
    )
    tsolve = time.time() - t0

    coverage = []
    for j, (al, am, an) in enumerate(ALL_TARGETS):
        tgt = (1, 1, 1, 1, 1, 1, 1, 1, -al, -am, -an)
        if tgt in sol and len(sol[tgt]) == len(masters):
            coverage.append(j)

    sampler.phase = "done"
    sampler.stop_evt.set()
    sampler.join(timeout=1.0)

    rss_gen_mib = round(sampler.peaks.get("gen", 0) / 1024, 2)
    rss_solve_mib = round(sampler.peaks.get("solve", 0) / 1024, 2)
    rss_peak_mib = round(max(sampler.peaks.values()) / 1024, 2)

    rec = {
        "cut": cut_tag, "width": width, "mode": mode, "target_index": idx, "target_lmn": [l, m, n],
        "path": "->".join(str(a) for a in PATHS_RAW[(l, m, n)]),
        "n_masters": len(masters),
        "n_seeds": len(seeds), "n_eqs": n_eqs, "n_vars": n_vars,
        "tgen_s": round(tgen, 2), "tsolve_s": round(tsolve, 2),
        "rss_gen_mib": rss_gen_mib, "rss_solve_mib": rss_solve_mib,
        "rss_peak_mib": rss_peak_mib,
        "covers": coverage, "n_covers": len(coverage),
    }
    with open(out_file, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"METRICS cut={cut_tag} idx={idx} lmn=({l},{m},{n}) masters={len(masters)} "
          f"seeds={len(seeds)} eqs={n_eqs} tgen_s={tgen:.2f} tsolve_s={tsolve:.2f} "
          f"rss_gen_mib={rss_gen_mib} rss_solve_mib={rss_solve_mib} covers={len(coverage)}/66",
          flush=True)


if __name__ == "__main__":
    main()
