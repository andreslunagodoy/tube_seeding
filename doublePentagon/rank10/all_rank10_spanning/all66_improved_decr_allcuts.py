"""all66_improved_decreasing_worker.py OUT_FILE

Rerun the all-66 rank-10 improved-seeding comparison with the *decreasing*
rule (Kira-3 Eq.9, l=9-n) + 167 per-sector override, on the quad cut.
Reduces all 66 targets I(1,...,1,-l,-m,-n), l+m+n=10, simultaneously
(needed_variables = all 66), same protocol as the flat all-66 run that
produced the Table-7/4.5 numbers.
"""
from __future__ import annotations
import json, os, sys, threading, time
os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path_coverage_rank10 as pc
import pyfeyngym as pfg

N = 10


def nonempty(eqs):
    return [e for e in eqs if len(list(e))>0]


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


def build_decreasing(trivial, n, use_167=True):
    starting = pfg.gen_all_seeds(pc.TOP_SECTOR, trivial, n, 8, 0)
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
    out_file = sys.argv[1]; CUTARG=[int(x) for x in (sys.argv[2].split(",") if len(sys.argv)>2 else ["1","3","6","8"])]
    targets = {(1,1,1,1,1,1,1,1,-l,-m,-n) for (l, m, n) in pc.ALL_TARGETS}
    sampler = PhaseSampler(); sampler.start()

    sampler.phase = "setup"
    eqt = pfg.gen_eq_templates(pc.IBP_FILE, pc.M_VALS)
    triv = pfg.get_trivial_sectors(pc.TRIV_FILE, cut=CUTARG, n_indices=11)

    sampler.phase = "masters"
    base = pc.build_improved_seeds(triv)
    seo0, av0 = pfg.gen_eqs(eqt, triv, pc.M_VALS, base)
    sol0 = pfg.solve_eqs_modulo(nonempty([a[-1] for a in seo0]), pfg.sort_integrals_desc(av0), pc.MODULUS)
    masters = [t[0] for t in sol0[(1,1,1,1,1,1,1,1,-1,-1,-2)]]
    del sol0, seo0, av0

    seeds = pfg.sort_integrals_desc([s for s in build_decreasing(triv, N, True)
                                     if pfg.to_sector(s) not in triv])
    sampler.phase = "gen"
    t0 = time.time()
    seo, av = pfg.gen_eqs(eqt, triv, pc.M_VALS, seeds)
    eqs = nonempty([a[-1] for a in seo]); svars = pfg.sort_integrals_desc(av)
    tgen = time.time() - t0; n_eqs = len(seo); n_vars = len(svars); del seo, av

    sampler.phase = "solve"
    t0 = time.time()
    sol = pfg.solve_eqs_modulo(eqs, svars, pc.MODULUS, keep_on_rhs=masters,
                               complete_pivoting=True, needed_variables=targets)
    tsolve = time.time() - t0

    covered = sum(1 for tg in targets if tg in sol and len(sol[tg]) == len(masters))

    sampler.phase = "done"; sampler.stop_evt.set(); sampler.join(timeout=1.0)
    g = lambda k: round(sampler.peaks.get(k, 0) / 1024 / 1024, 3)
    rec = {"cut": ",".join(map(str,CUTARG)), "rule": "decreasing+167, s_max=10", "n_masters": len(masters),
           "n_seeds": len(seeds), "n_eqs": n_eqs, "n_vars": n_vars,
           "tgen_s": round(tgen, 2), "tsolve_s": round(tsolve, 2),
           "rss_gen_gib": g("gen"), "rss_solve_gib": g("solve"),
           "covered": covered, "of": len(targets)}
    json.dump(rec, open(out_file, "w"), indent=2)
    print(f"METRICS cut={','.join(map(str,CUTARG))} all66 decreasing+167: seeds={len(seeds)} eqs={n_eqs} vars={n_vars} "
          f"covered={covered}/{len(targets)} tgen={tgen:.1f}s tsolve={tsolve:.1f}s "
          f"rss_gen={rec['rss_gen_gib']} rss_solve={rec['rss_solve_gib']} GiB", flush=True)


if __name__ == "__main__":
    main()
