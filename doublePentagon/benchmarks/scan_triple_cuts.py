"""
scan_triple_cuts.py

Run the feyngym strip-seeding scan for each of the 10 triple cuts from
high_rank.py's spanning_cuts list, for n=5..20, single run per n.
Each (cut, n) pair spawns a fresh subprocess worker.

Usage:
  PYTHON_JULIACALL_THREADS=1 python scan_triple_cuts.py
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).parent

N_VALUES = [int(x) for x in os.environ.get("SCAN_N", ",".join(
    str(x) for x in range(5, 21))).split(",")]
RUN_TIMEOUT_S = int(os.environ.get("SCAN_TIMEOUT", "36000"))
RESULTS_FILE = os.environ.get("RESULTS_FILE", "triple_cuts_results.json")

TRIPLE_CUTS = [
    [3, 4, 7],
    [2, 5, 8],
    [2, 5, 7],
    [2, 4, 7],
    [1, 4, 6],
    [2, 4, 6],
    [3, 4, 8],
    [1, 5, 6],
    [3, 5, 8],
    [1, 5, 7],
]
# CUT_INDEX env var (1-based) selects a single cut for parallel runs;
# 0 / unset keeps the full list.
_idx = int(os.environ.get("CUT_INDEX", "0"))
if _idx > 0:
    TRIPLE_CUTS = [TRIPLE_CUTS[_idx - 1]]

WORKER_SRC = r'''
import json
import os
import sys
import threading
import time

os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")

import pyfeyngym as pfg

DP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IBP_FILE = DP_DIR + "/IBP_LI"
TRIV_FILE = DP_DIR + "/trivialsector"
MODULUS = 2**31 - 1
M_VALS = {"d": 23, "m1": 3, "m2": 5, "m3": 17, "m4": 23}
TOP_SECTOR = (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0)


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
                prev = self.peaks.get(self.phase, 0)
                if rss > prev:
                    self.peaks[self.phase] = rss
            self.stop_evt.wait(self.interval)


def build_strip_seeds(n, trivial_sectors):
    s_max, r_max, d_max = 4, 8, 0
    starting = pfg.gen_all_seeds(TOP_SECTOR, trivial_sectors, s_max, r_max, d_max)
    improved = [
        s for s in starting
        if (pfg.d_level(s) <= 0
            and pfg.s_level(s) <= max(1, pfg.t_level(s) - 4))
        or (s[3] <= 0 and s[4] <= 0 and s[6] <= 0 and pfg.d_level(s) <= 0
            and pfg.s_level(s) <= max(1, pfg.t_level(s) - 3))
    ]
    n_shifts = max(1, n - 3)
    seed_set = set()
    for raise_rank in range(n_shifts):
        for seed in improved:
            seed_set.add(seed[:-1] + (seed[-1] - raise_rank,))
    valid = [s for s in seed_set if pfg.to_sector(s) not in trivial_sectors]
    return pfg.sort_integrals_desc(valid)


def main():
    n = int(sys.argv[1])
    out_file = sys.argv[2]
    cut = json.loads(sys.argv[3])
    result = {"n": n, "cut": cut}

    sampler = PhaseSampler()
    sampler.start()
    t_total = time.time()

    sampler.phase = "setup"
    eq_templates = pfg.gen_eq_templates(IBP_FILE, M_VALS)
    trivial_sectors = pfg.get_trivial_sectors(TRIV_FILE, cut=cut, n_indices=11)

    seeds = build_strip_seeds(n, trivial_sectors)
    result["n_seeds"] = len(seeds)

    # Master extraction
    sampler.phase = "masters_setup"
    s_max, r_max, d_max = 4, 8, 0
    starting = pfg.gen_all_seeds(TOP_SECTOR, trivial_sectors, s_max, r_max, d_max)
    improved = [
        s for s in starting
        if (pfg.d_level(s) <= 0
            and pfg.s_level(s) <= max(1, pfg.t_level(s) - 4))
        or (s[3] <= 0 and s[4] <= 0 and s[6] <= 0 and pfg.d_level(s) <= 0
            and pfg.s_level(s) <= max(1, pfg.t_level(s) - 3))
    ]
    seo0, av0 = pfg.gen_eqs(eq_templates, trivial_sectors, M_VALS, improved)
    eqs0 = [a[-1] for a in seo0]
    sv0 = pfg.sort_integrals_desc(av0)
    sol0 = pfg.solve_eqs_modulo(eqs0, sv0, MODULUS)
    probe = (1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -2)
    if probe in sol0:
        masters = [term[0] for term in sol0[probe]]
    else:
        masters = [v for v in sv0 if v not in sol0]
    result["n_masters"] = len(masters)
    del sol0, eqs0, seo0, av0, sv0

    # Generation phase
    sampler.phase = "gen"
    t_gen = time.time()
    seo, av = pfg.gen_eqs(eq_templates, trivial_sectors, M_VALS, seeds)
    equations = [a[-1] for a in seo]
    sorted_vars = pfg.sort_integrals_desc(av)
    result["tgen_s"] = round(time.time() - t_gen, 3)
    result["n_eqs"] = len(seo)
    result["n_vars"] = len(sorted_vars)

    # Solve phase
    sampler.phase = "solve"
    target = (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, -n)
    t_solve = time.time()
    solution = pfg.solve_eqs_modulo(
        equations, sorted_vars, MODULUS,
        keep_on_rhs=masters,
        complete_pivoting=True,
        needed_variables={target},
    )
    result["tsolve_s"] = round(time.time() - t_solve, 3)
    reduced = solution[target]
    result["reduced_terms"] = len(reduced)
    result["reduced_ok"] = (len(reduced) == len(masters))

    sampler.phase = "done"
    result["twall_s"] = round(time.time() - t_total, 3)
    sampler.stop_evt.set()
    sampler.join(timeout=1.0)

    result["peak_rss_gen_mib"] = round(sampler.peaks.get("gen", 0) / 1024, 2)
    result["peak_rss_solve_mib"] = round(sampler.peaks.get("solve", 0) / 1024, 2)
    result["peak_rss_total_mib"] = round(max(sampler.peaks.values()) / 1024, 2)

    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
'''


# ─── parent-side RSS sampling ──────────────────────────────────────────────

def read_rss_kb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return 0


def descendants(root):
    children = {}
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                with open(f"/proc/{pid}/status") as f:
                    for line in f:
                        if line.startswith("PPid:"):
                            children.setdefault(int(line.split()[1]), []).append(pid)
                            break
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
    except FileNotFoundError:
        return [root]
    out, stack = [root], [root]
    while stack:
        p = stack.pop()
        for c in children.get(p, []):
            out.append(c); stack.append(c)
    return out


class TreeRssSampler(threading.Thread):
    def __init__(self, pid, interval=0.1):
        super().__init__(daemon=True)
        self.pid, self.interval = pid, interval
        self.peak_kb = 0
        self.stop_evt = threading.Event()

    def run(self):
        while not self.stop_evt.is_set():
            total = sum(read_rss_kb(p) for p in descendants(self.pid))
            if total > self.peak_kb:
                self.peak_kb = total
            self.stop_evt.wait(self.interval)


def run_one(cut, n, rep=0):
    cut_label = "_".join(str(c) for c in cut)
    suffix = "" if rep == 0 and not os.environ.get("MULTIREP") else f"_r{rep}"
    workdir = HERE / f"triple_{cut_label}_n{n}{suffix}"
    workdir.mkdir(parents=True, exist_ok=True)
    worker_path = workdir / "worker.py"
    worker_path.write_text(WORKER_SRC)
    json_out = workdir / "result.json"
    log_out = workdir / "worker.out"
    # Avoid the merge-stale-result.json bug: a leftover result.json from a
    # prior successful attempt would otherwise be folded into a new failure's
    # record by the json_out.exists() branch below.
    if json_out.exists():
        json_out.unlink()

    print(f"  [cut={cut} n={n}] starting...", flush=True)
    t0 = time.time()
    timed_out = False
    with open(log_out, "wb") as log_f:
        proc = subprocess.Popen(
            [sys.executable, str(worker_path), str(n), str(json_out),
             json.dumps(cut)],
            stdout=log_f, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHON_JULIACALL_THREADS": "1"},
        )
        sampler = TreeRssSampler(proc.pid)
        sampler.start()
        try:
            rc = proc.wait(timeout=RUN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            timed_out = True
            for pid in descendants(proc.pid):
                try: os.kill(pid, 9)
                except ProcessLookupError: pass
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired: pass
            rc = -9
        finally:
            sampler.stop_evt.set()
            sampler.join(timeout=1.0)
    wall = time.time() - t0
    rec = {
        "cut": cut, "n": n,
        "exit_code": rc, "timed_out": timed_out,
        "twall_launcher_s": round(wall, 3),
        "tree_peak_rss_mib": round(sampler.peak_kb / 1024, 2),
    }
    if json_out.exists():
        try:
            rec.update(json.loads(json_out.read_text()))
        except Exception as e:
            rec["parse_error"] = str(e)
    return rec


def main():
    repeats = int(os.environ.get("REPEATS", "1"))
    if repeats > 1:
        os.environ["MULTIREP"] = "1"   # force per-rep workdirs in run_one
    # Resume: keep already-successful records from any existing RESULTS_FILE
    # and skip those (cut, n, rep) triples on this run.
    results_path = HERE / RESULTS_FILE
    records: list = []
    done: set = set()
    if results_path.exists():
        try:
            prior = json.loads(results_path.read_text())
            if isinstance(prior, list):
                for r in prior:
                    if r.get("reduced_ok"):
                        records.append(r)
                        done.add((tuple(r.get("cut", [])), r.get("n"), r.get("rep", 0)))
        except Exception as e:
            print(f"[resume] failed to load {RESULTS_FILE}: {e}; starting fresh",
                  flush=True)
    if done:
        print(f"[resume] keeping {len(done)} prior successful (cut,n,rep) records",
              flush=True)
    for cut in TRIPLE_CUTS:
        cut_label = ",".join(str(c) for c in cut)
        print(f"\n=== Cut [{cut_label}] (×{repeats} reps) ===", flush=True)
        for n in N_VALUES:
            for rep in range(repeats):
                if (tuple(cut), n, rep) in done:
                    print(f"  [cut=[{cut_label}] n={n} rep={rep}] SKIP (already done)",
                          flush=True)
                    continue
                rec = run_one(cut, n, rep=rep)
                rec["rep"] = rep
                records.append(rec)
                (HERE / RESULTS_FILE).write_text(json.dumps(records, indent=2))
                ok = "OK" if rec.get("reduced_ok") else "FAIL"
                to = " TIMEOUT" if rec.get("timed_out") else ""
                print(f"  [cut=[{cut_label}] n={n} rep={rep}] "
                      f"masters={rec.get('n_masters','-')} "
                      f"Tsolve={rec.get('tsolve_s','-')}s "
                      f"RSS={rec.get('peak_rss_total_mib','-')} MiB "
                      f"{ok}{to}", flush=True)

    # Summary table
    print("\n" + "=" * 120)
    print("Triple-cut strip scan summary")
    print("=" * 120)
    hdr = f"{'cut':>12} {'n':>3} {'seeds':>5} {'#eqs':>6} {'Tgen':>6} {'Tsolve':>7} {'wall':>7} {'RSSsolve':>9} {'m':>3} {'ok':>4}"
    print(hdr)
    for r in records:
        cl = ",".join(str(c) for c in r["cut"])
        to = "*" if r.get("timed_out") else ""
        print(f"  [{cl:>7}] {r['n']:>3} {r.get('n_seeds','-'):>5} "
              f"{r.get('n_eqs','-'):>6} {str(r.get('tgen_s','-')):>6} "
              f"{str(r.get('tsolve_s','-')):>7} "
              f"{str(r.get('twall_s','-'))+to:>7} "
              f"{str(r.get('peak_rss_solve_mib','-')):>9} "
              f"{str(r.get('n_masters','-')):>3} "
              f"{'OK' if r.get('reduced_ok') else 'FAIL':>4}")


if __name__ == "__main__":
    sys.exit(main() or 0)
