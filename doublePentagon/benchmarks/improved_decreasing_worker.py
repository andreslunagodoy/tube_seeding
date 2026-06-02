"""improved_decreasing_worker.py  N  USE_167  OUT_FILE

Measure improved seeding with the *decreasing* (Kira-3 Eq. 9) rank rule for
the single-ISP target I(1,...,1,0,0,-N) on the quad cut [1,3,6,8], same
protocol as the original Table-2 improved runs (fresh process, PhaseSampler,
gen/solve phase peaks, complete-pivoting solve keeping the 27 masters).

Decreasing rule: in a sub-sector with t propagators,
    s <= max(1, N - (8 - t))            (= Kira Eq.9 with l = 9 - N)
plus the 167-style per-sector override (+1 when props 4,5,7 absent) if USE_167.
"""
from __future__ import annotations
import json, os, sys, threading, time
os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rank10"))
import path_coverage_rank10 as pc
import pyfeyngym as pfg


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
        self.interval = interval; self.stop_evt = threading.Event()
        self.phase = None; self.peaks = {}
    def run(self):
        while not self.stop_evt.is_set():
            rss = read_rss_kb()
            if self.phase is not None and rss > self.peaks.get(self.phase, 0):
                self.peaks[self.phase] = rss
            self.stop_evt.wait(self.interval)


def build_decreasing(trivial, n, use_167):
    starting = pfg.gen_all_seeds(pc.TOP_SECTOR, trivial, n, 8, 0)   # no dots, s<=n
    out = []
    for s in starting:
        if pfg.d_level(s) > 0:
            continue
        t = pfg.t_level(s)
        smax = max(1, n - (8 - t))
        if use_167 and s[3] <= 0 and s[4] <= 0 and s[6] <= 0:
            smax += 1
        if pfg.s_level(s) <= smax:
            out.append(s)
    return out


def main():
    n = int(sys.argv[1]); use_167 = sys.argv[2] == "1"; out_file = sys.argv[3]
    TARGET = (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, -n)
    sampler = PhaseSampler(); sampler.start()

    sampler.phase = "setup"
    eqt = pfg.gen_eq_templates(pc.IBP_FILE, pc.M_VALS)
    triv = pfg.get_trivial_sectors(pc.TRIV_FILE, cut=[1, 3, 6, 8], n_indices=11)

    sampler.phase = "masters"
    base = pc.build_improved_seeds(triv)
    seo0, av0 = pfg.gen_eqs(eqt, triv, pc.M_VALS, base)
    sol0 = pfg.solve_eqs_modulo([a[-1] for a in seo0], pfg.sort_integrals_desc(av0), pc.MODULUS)
    masters = [t[0] for t in sol0[(1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -2)]]
    del sol0, seo0, av0

    seeds = pfg.sort_integrals_desc([s for s in build_decreasing(triv, n, use_167)
                                     if pfg.to_sector(s) not in triv])
    sampler.phase = "gen"
    t0 = time.time()
    seo, av = pfg.gen_eqs(eqt, triv, pc.M_VALS, seeds)
    eqs = [a[-1] for a in seo]; svars = pfg.sort_integrals_desc(av)
    tgen = time.time() - t0; n_eqs = len(seo); n_vars = len(svars); del seo, av

    sampler.phase = "solve"
    t0 = time.time()
    sol = pfg.solve_eqs_modulo(eqs, svars, pc.MODULUS, keep_on_rhs=masters,
                               complete_pivoting=True, needed_variables={TARGET})
    tsolve = time.time() - t0
    red = sol.get(TARGET)
    complete = red is not None and len(red) == len(masters)

    sampler.phase = "done"; sampler.stop_evt.set(); sampler.join(timeout=1.0)
    g = lambda k: round(sampler.peaks.get(k, 0) / 1024 / 1024, 3)
    rec = {"n": n, "use_167": use_167, "rule": "decreasing s<=max(1,n-(8-t))",
           "n_masters": len(masters), "n_seeds": len(seeds), "n_eqs": n_eqs, "n_vars": n_vars,
           "tgen_s": round(tgen, 2), "tsolve_s": round(tsolve, 2),
           "rss_gen_gib": g("gen"), "rss_solve_gib": g("solve"),
           "reduction_complete": bool(complete)}
    json.dump(rec, open(out_file, "w"), indent=2)
    print(f"n={n} 167={use_167}: seeds={len(seeds)} eqs={n_eqs} complete={complete} "
          f"tgen={tgen:.1f}s tsolve={tsolve:.1f}s rss_gen={rec['rss_gen_gib']} "
          f"rss_solve={rec['rss_solve_gib']} GiB", flush=True)


if __name__ == "__main__":
    main()
