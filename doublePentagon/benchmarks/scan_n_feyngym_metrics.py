"""
scan_n_feyngym_metrics.py

Analogous to scan_n_paper_metrics.py but uses pyfeyngym's equation generator
and solver end-to-end (no Kira). Runs the same strip-shifted improved-seed
set used by scan_n_paper_metrics (build_strip_seeds) and reduces
doublePentagon[1,...,1,0,0,-n] on the quadruple cut [1,3,6,8] for n in
SCAN_N (default 5..20).

For each n we spawn a fresh subprocess that:
  1. generates the strip seed set,
  2. calls pfg.gen_eqs  (equation generation phase),
  3. calls pfg.solve_eqs_modulo with keep_on_rhs=masters, complete_pivoting,
     needed_variables={target} (equation solving phase),
  4. verifies the target was reduced to masters.

Inside the worker we sample /proc/self/status VmRSS every ~50 ms from a
background thread and record peak RSS during each phase separately. The
parent-side launcher also samples the full process tree for a cross-check
on total peak.
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).parent
EXPERIMENT = HERE.parent

N_VALUES = [int(x) for x in os.environ.get("SCAN_N", ",".join(
    str(x) for x in range(5, 21))).split(",")]
RUN_TIMEOUT_S = int(os.environ.get("SCAN_TIMEOUT", "3600"))
REPEATS = int(os.environ.get("REPEATS", "1"))
RESULTS_FILE = os.environ.get("RESULTS_FILE", "scan_n_feyngym_results.json")
# Seeding strategy passed to the worker via argv: "strip" (default) uses
# the strip-shifted improved-seed set; "kira" uses Kira's refined
# improved seeding (sector 255 at s=max(5,n), sector 167 at s=3) with no
# a_11-axis shifts.
STRATEGY = os.environ.get("FG_STRATEGY", "strip")

# ─── worker ─────────────────────────────────────────────────────────────────


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
CUT = [1, 3, 6, 8]


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
    """Samples /proc/self VmRSS in the background. Caller sets .phase and
    reads .peaks after each phase completes."""
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
    return pfg.sort_integrals_desc(valid), len(improved)


def build_kira_improved_seeds(n, trivial_sectors):
    """Kira's refined seeding applied across every on-cut sub-sector,
    widened enough in the a_11 direction to actually reach the target.
    This is the analogue of what Kira's sub-sector descent produces when
    it walks down from sector 255 at s=max(5,n): every on-cut sub-sector
    gets seeds up to s_max = max(n, t-4), r<=8, d=0 (with sector 167's
    looser s<=3 floor). Unlike strip there are no a_11-axis shifts — it's
    just the "full dense envelope up to s=n in every sub-sector", which
    is what kirafilt built on the Kira side and is the closest flat-
    generator approximation to Kira's descent pipeline."""
    cut_mask = (1 << 0) | (1 << 2) | (1 << 5) | (1 << 7)
    excl_mask = (1 << 3) | (1 << 4) | (1 << 6)
    seed_set: set = set()
    for sec in range(256):
        if (sec & cut_mask) != cut_mask:
            continue
        if sec in trivial_sectors:
            continue
        t = bin(sec).count("1")
        s_max = max(n, max(1, t - 4))
        if (sec & excl_mask) == 0:
            s_max = max(s_max, max(1, t - 3))
        sec_seeds = pfg.gen_all_seeds(TOP_SECTOR, trivial_sectors, s_max, 8, 0)
        for s in sec_seeds:
            if pfg.to_sector(s) == sec and pfg.s_level(s) <= s_max:
                seed_set.add(s)
    # n_improved = how many of the chosen seeds obey Kira's base
    # improved-seed rank rule (s <= t-4). This is a diagnostic.
    n_improved = sum(
        1 for s in seed_set
        if pfg.s_level(s) <= max(1, pfg.t_level(s) - 4))
    return pfg.sort_integrals_desc(list(seed_set)), n_improved


def main():
    n = int(sys.argv[1])
    out_file = sys.argv[2]
    strategy = sys.argv[3] if len(sys.argv) > 3 else "strip"
    result = {"n": n, "strategy": strategy}

    sampler = PhaseSampler()
    sampler.start()

    t_total = time.time()

    sampler.phase = "setup"
    eq_templates = pfg.gen_eq_templates(IBP_FILE, M_VALS)
    trivial_sectors = pfg.get_trivial_sectors(TRIV_FILE, cut=CUT, n_indices=11)

    if strategy == "strip":
        seeds, n_improved = build_strip_seeds(n, trivial_sectors)
    elif strategy == "kira":
        seeds, n_improved = build_kira_improved_seeds(n, trivial_sectors)
    else:
        raise SystemExit(f"unknown strategy: {strategy}")
    result["n_seeds"] = len(seeds)
    result["n_improved"] = n_improved

    # Preliminary master extraction — same as high_rank.py. Full
    # reduction of the improved-seeds system, then read off which
    # masters (1,...,1,-1,-1,-2) reduces to. Both strip and
    # kira-improved now contain the improved-seed set (strip adds
    # a_11 shifts, kira-improved adds sector 255 widening), so this
    # preliminary is safe for either strategy.
    sampler.phase = "masters_setup"
    s_max, r_max, d_max = 4, 8, 0
    starting = pfg.gen_all_seeds(TOP_SECTOR, trivial_sectors,
                                 s_max, r_max, d_max)
    improved = [
        s for s in starting
        if (pfg.d_level(s) <= 0
            and pfg.s_level(s) <= max(1, pfg.t_level(s) - 4))
        or (s[3] <= 0 and s[4] <= 0 and s[6] <= 0 and pfg.d_level(s) <= 0
            and pfg.s_level(s) <= max(1, pfg.t_level(s) - 3))
    ]
    seed_op_eq_list0, all_vars0 = pfg.gen_eqs(
        eq_templates, trivial_sectors, M_VALS, improved)
    equations0 = [a[-1] for a in seed_op_eq_list0]
    sorted_vars0 = pfg.sort_integrals_desc(all_vars0)
    solution0 = pfg.solve_eqs_modulo(equations0, sorted_vars0, MODULUS)
    reduced0 = solution0[(1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -2)]
    masters = [term[0] for term in reduced0]
    result["n_masters"] = len(masters)
    del solution0, equations0, seed_op_eq_list0, all_vars0, sorted_vars0

    # === generation phase ===
    sampler.phase = "gen"
    t_gen = time.time()
    seed_op_eq_list, all_variables = pfg.gen_eqs(
        eq_templates, trivial_sectors, M_VALS, seeds)
    equations = [a[-1] for a in seed_op_eq_list]
    sorted_vars = pfg.sort_integrals_desc(all_variables)
    result["tgen_s"] = round(time.time() - t_gen, 3)
    result["n_eqs"] = len(seed_op_eq_list)
    result["n_vars"] = len(sorted_vars)

    # === solving phase ===
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

    result["peak_rss_setup_mib"] = round(sampler.peaks.get("setup", 0) / 1024, 2)
    result["peak_rss_masters_mib"] = round(sampler.peaks.get("masters_setup", 0) / 1024, 2)
    result["peak_rss_gen_mib"] = round(sampler.peaks.get("gen", 0) / 1024, 2)
    result["peak_rss_solve_mib"] = round(sampler.peaks.get("solve", 0) / 1024, 2)
    result["peak_rss_total_mib"] = round(max(sampler.peaks.values()) / 1024, 2)

    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
'''


# ─── parent-side /proc sampling (whole process tree) ────────────────────────

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
                            children.setdefault(
                                int(line.split()[1]), []).append(pid)
                            break
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
    except FileNotFoundError:
        return [root]
    out, stack = [root], [root]
    while stack:
        p = stack.pop()
        for c in children.get(p, []):
            out.append(c)
            stack.append(c)
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


# ─── main loop ──────────────────────────────────────────────────────────────

def run_single(n: int, rep: int) -> dict:
    """Run one worker subprocess for a single (n, rep) and return its
    metrics. Each repeat gets its own workdir so logs aren't overwritten."""
    workdir = HERE / f"feyngym_{STRATEGY}_n{n}_r{rep}"
    workdir.mkdir(parents=True, exist_ok=True)
    worker_path = workdir / "worker.py"
    worker_path.write_text(WORKER_SRC)
    json_out = workdir / "result.json"
    log_out = workdir / "worker.out"

    print(f"  [feyngym/{STRATEGY} n={n} rep={rep}] starting worker "
          f"(timeout {RUN_TIMEOUT_S}s)", flush=True)
    t0 = time.time()
    timed_out = False
    with open(log_out, "wb") as log_f:
        proc = subprocess.Popen(
            [sys.executable, str(worker_path), str(n), str(json_out), STRATEGY],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHON_JULIACALL_THREADS": "1"},
        )
        sampler = TreeRssSampler(proc.pid)
        sampler.start()
        try:
            rc = proc.wait(timeout=RUN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            timed_out = True
            for pid in descendants(proc.pid):
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            rc = -9
        finally:
            sampler.stop_evt.set()
            sampler.join(timeout=1.0)
    wall = time.time() - t0
    tree_peak_mib = round(sampler.peak_kb / 1024, 2)

    rec: dict = {
        "n": n,
        "rep": rep,
        "exit_code": rc,
        "timed_out": timed_out,
        "twall_launcher_s": round(wall, 3),
        "tree_peak_rss_mib": tree_peak_mib,
    }
    if json_out.exists():
        try:
            rec.update(json.loads(json_out.read_text()))
        except Exception as e:
            rec["parse_error"] = str(e)
    return rec


def aggregate(runs: list[dict]) -> dict:
    """Compute mean / std / min / max / median across repeats for the
    hardware-dependent metrics. Hardware-independent counters
    (n_seeds, n_eqs, n_vars, n_masters) should be identical across
    repeats so we just copy the first observed value. Exit status
    summaries are added to aid debugging."""
    import statistics
    ok = [r for r in runs if not r.get("timed_out") and r.get("exit_code") == 0]
    def stats(key):
        vals = [r[key] for r in ok if key in r and r[key] is not None]
        if not vals:
            return None
        d = {"n": len(vals), "min": min(vals), "max": max(vals),
             "mean": round(sum(vals) / len(vals), 3)}
        if len(vals) >= 2:
            d["std"] = round(statistics.stdev(vals), 3)
            d["median"] = round(statistics.median(vals), 3)
        return d
    agg: dict = {
        "n_runs": len(runs),
        "n_runs_ok": len(ok),
        "n_timed_out": sum(1 for r in runs if r.get("timed_out")),
    }
    for key in ("tgen_s", "tsolve_s", "twall_s", "twall_launcher_s",
                "peak_rss_gen_mib", "peak_rss_solve_mib",
                "peak_rss_total_mib", "tree_peak_rss_mib"):
        agg[key] = stats(key)
    # Copy hardware-independent counters from the first ok run.
    if ok:
        first = ok[0]
        for key in ("n_seeds", "n_improved", "n_eqs", "n_vars",
                    "n_masters", "reduced_ok"):
            if key in first:
                agg[key] = first[key]
    return agg


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def print_summary(records: list[dict]) -> None:
    print()
    print("=" * 140)
    print("feyngym (pyfeyngym) strip-seeding reduction on doublePentagon cut "
          "[1,3,6,8] — (1,...,1,0,0,-n)")
    print("=" * 140)
    cols = [
        ("n",          lambda r: fmt(r.get("n"))),
        ("seeds",      lambda r: fmt(r.get("n_seeds"))),
        ("#eqs",       lambda r: fmt(r.get("n_eqs"))),
        ("#vars",      lambda r: fmt(r.get("n_vars"))),
        ("Tgen[s]",    lambda r: fmt(r.get("tgen_s"))),
        ("Tsolve[s]",  lambda r: fmt(r.get("tsolve_s"))),
        ("wall[s]",    lambda r: fmt(r.get("twall_s") or r.get("twall_launcher_s"))
                                   + ("*" if r.get("timed_out") else "")),
        ("RSSgen[MiB]",   lambda r: fmt(r.get("peak_rss_gen_mib"))),
        ("RSSsolve[MiB]", lambda r: fmt(r.get("peak_rss_solve_mib"))),
        ("RSStot[MiB]",   lambda r: fmt(r.get("peak_rss_total_mib"))),
        ("RSStree[MiB]",  lambda r: fmt(r.get("tree_peak_rss_mib"))),
        ("masters",    lambda r: fmt(r.get("n_masters"))),
        ("ok",         lambda r: "OK" if r.get("reduced_ok")
                                  else ("FAIL" if "reduced_ok" in r else "—")),
    ]
    headers = [h for h, _ in cols]
    rows = [[get(r) for _, get in cols] for r in records]
    widths = [max(len(h), max((len(row[i]) for row in rows), default=0))
              for i, h in enumerate(headers)]
    pad = "  "
    print(pad.join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print(pad.join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print(pad.join(row[i].ljust(widths[i]) for i in range(len(row))))
    print()
    print("(* = launcher timeout)")


def main() -> int:
    records: list[dict] = []
    for n in N_VALUES:
        print(f"\n=== n={n} (×{REPEATS}) ===", flush=True)
        runs: list[dict] = []
        for rep in range(REPEATS):
            rec = run_single(n, rep)
            runs.append(rec)
            tstr = (f"wall={rec.get('twall_launcher_s')}s "
                    f"Tsolve={rec.get('tsolve_s')}s "
                    f"tree_rss={rec.get('tree_peak_rss_mib')} MiB "
                    f"ok={rec.get('reduced_ok')}")
            if rec.get("timed_out"):
                tstr = "TIMEOUT " + tstr
            print(f"  [n={n} rep={rep}] {tstr}", flush=True)
        agg = aggregate(runs)
        entry = {"n": n, "runs": runs, "aggregate": agg}
        records.append(entry)
        (HERE / RESULTS_FILE).write_text(json.dumps(records, indent=2))
        if agg.get("tsolve_s"):
            ts = agg["tsolve_s"]
            rs = agg.get("peak_rss_total_mib") or {}
            print(f"  [n={n}] mean Tsolve={ts.get('mean')}±{ts.get('std',0)}s "
                  f"mean RSS={rs.get('mean')}±{rs.get('std',0)} MiB "
                  f"({agg['n_runs_ok']}/{agg['n_runs']} ok)", flush=True)
    print_aggregate_summary(records)
    return 0


def print_aggregate_summary(records: list[dict]) -> None:
    print()
    print("=" * 120)
    print(f"feyngym {STRATEGY} averaged over {REPEATS} repeats "
          "on doublePentagon cut [1,3,6,8]")
    print("=" * 120)
    hdr = [
        ("n",          lambda e: str(e["n"])),
        ("seeds",      lambda e: str(e["aggregate"].get("n_seeds", "—"))),
        ("#eqs",       lambda e: str(e["aggregate"].get("n_eqs", "—"))),
        ("#vars",      lambda e: str(e["aggregate"].get("n_vars", "—"))),
        ("Tgen mean",  lambda e: agg_fmt(e["aggregate"].get("tgen_s"))),
        ("Tsolve mean",lambda e: agg_fmt(e["aggregate"].get("tsolve_s"))),
        ("wall mean",  lambda e: agg_fmt(e["aggregate"].get("twall_s"))),
        ("RSSgen",     lambda e: agg_fmt(e["aggregate"].get("peak_rss_gen_mib"))),
        ("RSSsolve",   lambda e: agg_fmt(e["aggregate"].get("peak_rss_solve_mib"))),
        ("RSStot",     lambda e: agg_fmt(e["aggregate"].get("peak_rss_total_mib"))),
        ("masters",    lambda e: str(e["aggregate"].get("n_masters", "—"))),
        ("ok/runs",    lambda e: f"{e['aggregate']['n_runs_ok']}/{e['aggregate']['n_runs']}"),
    ]
    rows = [[get(e) for _, get in hdr] for e in records]
    widths = [max(len(h), max((len(row[i]) for row in rows), default=0))
              for i, (h, _) in enumerate(hdr)]
    pad = "  "
    print(pad.join(h.ljust(widths[i]) for i, (h, _) in enumerate(hdr)))
    print(pad.join("-" * widths[i] for i in range(len(hdr))))
    for row in rows:
        print(pad.join(row[i].ljust(widths[i]) for i in range(len(row))))


def agg_fmt(s):
    if not s or s.get("mean") is None:
        return "—"
    mean = s["mean"]
    std = s.get("std", 0)
    if std:
        return f"{mean:.3g}±{std:.2g}"
    return f"{mean:.3g}"


if __name__ == "__main__":
    sys.exit(main())
