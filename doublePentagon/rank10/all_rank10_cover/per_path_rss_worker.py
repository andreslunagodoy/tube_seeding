"""
per_path_rss_worker.py PATH_INDEX OUT_FILE

Run a single rank-10 path in a fresh subprocess. Samples VmRSS in a
background thread during the gen and solve phases and reports the peak
of each phase plus the overall total peak. Writes JSON with the
measurements.
"""
from __future__ import annotations
import json
import os
import sys
import threading
import time

os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")

# Import the path table and seed builder from the previous scan script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from path_coverage_rank10 import (  # noqa: E402
    ALL_TARGETS, PATHS_RAW, build_improved_seeds, build_path_seeds,
    IBP_FILE, TRIV_FILE, MODULUS, M_VALS, CUT, TOP_SECTOR,
)

import pyfeyngym as pfg  # noqa: E402


def read_rss_kb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        return 0
    return 0


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


def main():
    idx = int(sys.argv[1])
    out_file = sys.argv[2]

    sampler = PhaseSampler()
    sampler.start()

    sampler.phase = "setup"
    eq_templates = pfg.gen_eq_templates(IBP_FILE, M_VALS)
    trivial_sectors = pfg.get_trivial_sectors(TRIV_FILE, cut=CUT, n_indices=11)
    improved = build_improved_seeds(trivial_sectors)

    sampler.phase = "masters"
    seo0, av0 = pfg.gen_eqs(eq_templates, trivial_sectors, M_VALS, improved)
    sol0 = pfg.solve_eqs_modulo(
        [a[-1] for a in seo0], pfg.sort_integrals_desc(av0), MODULUS)
    masters = [t[0] for t in sol0[(1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -2)]]
    del sol0, seo0, av0

    target_set = {(1, 1, 1, 1, 1, 1, 1, 1, -l, -m, -n)
                  for (l, m, n) in ALL_TARGETS}

    l, m, n = ALL_TARGETS[idx]
    seeds = build_path_seeds((l, m, n), improved, trivial_sectors, width=2)

    sampler.phase = "gen"
    t0 = time.time()
    seo, av = pfg.gen_eqs(eq_templates, trivial_sectors, M_VALS, seeds)
    equations = [a[-1] for a in seo]
    svars = pfg.sort_integrals_desc(av)
    tgen = time.time() - t0
    n_eqs = len(seo)
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
        "target_index": idx,
        "target_lmn": [l, m, n],
        "path": "->".join(str(a) for a in PATHS_RAW[(l, m, n)]),
        "n_seeds": len(seeds),
        "n_eqs": n_eqs,
        "n_vars": n_vars,
        "tgen_s": round(tgen, 2),
        "tsolve_s": round(tsolve, 2),
        "rss_gen_mib": rss_gen_mib,
        "rss_solve_mib": rss_solve_mib,
        "rss_peak_mib": rss_peak_mib,
        "covers": coverage,
        "n_covers": len(coverage),
    }
    with open(out_file, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"path {idx} ({l},{m},{n}) "
          f"seeds={len(seeds)} eqs={n_eqs} "
          f"tgen={tgen:.1f}s tsolve={tsolve:.1f}s "
          f"rss_gen={rss_gen_mib} rss_solve={rss_solve_mib} "
          f"rss_peak={rss_peak_mib} MiB covers={len(coverage)}/66",
          flush=True)


if __name__ == "__main__":
    main()
