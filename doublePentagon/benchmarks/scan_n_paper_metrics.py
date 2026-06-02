"""
scan_n_paper_metrics.py

Same comparison as scan_n.py (strip vs kira-improved on the doublePentagon
quadruple cut [1,3,6,8] reducing (1,...,1,0,0,-n)) but using the metrics the
Kira 3 paper (arXiv:2505.20197) reports plus wall time:

    Tgen, # gen, # indep, # sel, TpyRed, peak RSS, # masters, wall time.

Both strategies are run through the same patched Kira binary (the one with
the `[paper metrics]` log line and the seed-filter / skip-selection
patches, all of which are no-ops unless explicitly enabled). The
strip strategy uses KIRA_SEED_FILTER_FILE to restrict Kira's seeds to
the high_rank.ipynb strip set; the kira-improved strategy uses the
example double_pentagon jobs.yaml shape (sectors 255 at s=max(5,n) and
167 at s=3) with the cut applied.

Env vars:
  SCAN_N         comma-separated n values (default "5,6,7,8,9,10,11,12")
  SCAN_TIMEOUT   per-run timeout in seconds (default 1800 = 30 min)
"""
from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).parent
EXPERIMENT = HERE.parent
KIRA_ENV_FILE = EXPERIMENT / "kira_env.sh"
os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")

import pyfeyngym as pfg

TOPO = "doublePentagon"
TOP_SECTOR = (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0)
CUT = [1, 3, 6, 8]
TRIVIAL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trivialsector")

# Numeric values for the kinematic invariants. Same as the existing
# scan_n.py setup so we can reuse the cached config files.
KIRA_SET_VALUE = [
    "--set_value=d=23",
    "--set_value=s23=3",
    "--set_value=s34=5",
    "--set_value=s45=17",
    "--set_value=s51=23",
]

N_VALUES = [int(x) for x in os.environ.get("SCAN_N", "5,6,7,8,9,10,11,12").split(",")]
RUN_TIMEOUT_S = int(os.environ.get("SCAN_TIMEOUT", "1800"))
STRATEGIES = set(os.environ.get(
    "STRATEGY", "strip,strip2sec,kirafilt,kira").split(","))
RESULTS_FILE = os.environ.get("RESULTS_FILE", "scan_n_paper_metrics_results.json")


# ─── seed builders (identical to scan_n.py) ─────────────────────────────────

def build_improved_seeds(trivial_sectors, improved_seed_param: int = 4) -> list[tuple]:
    s_max, r_max, d_max = improved_seed_param, 8, 0
    starting = pfg.gen_all_seeds(TOP_SECTOR, trivial_sectors, s_max, r_max, d_max)
    return [
        s for s in starting
        if (pfg.d_level(s) <= 0 and pfg.s_level(s) <= max(1, pfg.t_level(s) - improved_seed_param))
        or (s[3] <= 0 and s[4] <= 0 and s[6] <= 0 and pfg.d_level(s) <= 0
            and pfg.s_level(s) <= max(1, pfg.t_level(s) - improved_seed_param + 1))
    ]


def build_strip_seeds(n: int, trivial_sectors) -> list[tuple]:
    improved = build_improved_seeds(trivial_sectors, improved_seed_param=4)
    n_shifts = max(1, n - 3)
    seed_set = set()
    for raise_rank in range(n_shifts):
        for seed in improved:
            seed_set.add(seed[:-1] + (seed[-1] - raise_rank,))
    valid = [s for s in seed_set if pfg.to_sector(s) not in trivial_sectors]
    return pfg.sort_integrals_desc(valid)


def build_strip_2sec_seeds(n: int, trivial_sectors) -> list[tuple]:
    """Strip pattern restricted to the same two sectors kira-improved seeds:
    the strip-shifted improved_seeds in sector 255, plus Kira's s=3 box in
    sector 167. Both strategies then rely on Kira's sector symmetries to
    reach the other 14 on-cut sub-sectors, so they're directly comparable
    on what "seed shape inside the listed sectors" costs."""
    strip_full = build_strip_seeds(n, trivial_sectors)
    in_255 = [s for s in strip_full if pfg.to_sector(s) == 255]
    # Sector 167 with Kira's s=3 box (matches kira-improved exactly).
    s_167 = 3
    all_167 = pfg.gen_all_seeds(TOP_SECTOR, trivial_sectors, s_167, 8, 0)
    in_167 = [s for s in all_167
              if pfg.to_sector(s) == 167 and pfg.s_level(s) <= s_167]
    return pfg.sort_integrals_desc(in_255 + in_167)


def derive_nb_sector_bounds(n: int) -> list[tuple[int, int, int]]:
    """Per-sector wide-envelope bounds covering the strip seed set, used
    for the strip-strategy jobs.yaml. The seed_filter file then restricts
    Kira's outer iteration to the actual strip subset."""
    cut_mask = (1 << 0) | (1 << 2) | (1 << 5) | (1 << 7)
    excl_mask = (1 << 3) | (1 << 4) | (1 << 6)
    bounds = []
    for sec in range(256):
        if (sec & cut_mask) != cut_mask:
            continue
        t = bin(sec).count("1")
        s_max = max(1, t - 4)
        if (sec & excl_mask) == 0:
            s_max = max(s_max, max(1, t - 3))
        if sec == 255:
            s_max = max(s_max, n)
        bounds.append((sec, 8, s_max))
    return bounds


# ─── jobs.yaml writers ──────────────────────────────────────────────────────

def write_targets(workdir: Path, n: int) -> None:
    target_tuple = ", ".join(["1"] * 8 + ["0", "0", str(-n)])
    (workdir / "targets").write_text(f"{TOPO}[{target_tuple}]\n")


def copy_config(workdir: Path, disable_symmetries: bool = False) -> None:
    """Use the kira_strategy/config/ from earlier (it already has
    cut_propagators: [1,3,6,8] applied). If disable_symmetries is True,
    patch integralfamilies.yaml to set permutation_option: 0 so Kira
    does NOT auto-detect sector symmetries."""
    src = EXPERIMENT / "kira_strategy" / "config"
    dst = workdir / "config"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    if disable_symmetries:
        # Kira's permutation_option: 1..4 each rank the automatic
        # sector-symmetry finder; removing the line disables that
        # finder entirely, so sub-sector equivalence classes are never
        # merged and every listed sub-sector is reduced on its own.
        ifam = dst / "integralfamilies.yaml"
        text = ifam.read_text()
        text = re.sub(r"^\s*permutation_option:\s*\d+\s*\n", "", text,
                      flags=re.MULTILINE)
        ifam.write_text(text)


def write_kira_improved_jobs(workdir: Path, n: int) -> None:
    """Kira's example double_pentagon seeding with sector-255 s scaled
    to n: {255, r=8, s=max(5,n)} + {167, r=8, s=3}."""
    s_top = max(5, n)
    (workdir / "jobs.yaml").write_text(f"""\
jobs:
  - reduce_sectors:
      reduce:
        - {{topologies: [{TOPO}], sectors: [255], r: 8, s: {s_top}, d: 0}}
        - {{topologies: [{TOPO}], sectors: [167], r: 8, s: 3, d: 0}}
      select_integrals:
        select_mandatory_list:
          - [targets]
      run_initiate: true
      run_firefly: true
""")


def write_kira_truncate_jobs(workdir: Path, n: int) -> None:
    """Kira 3 IMPROVED seeding via truncate_sp (Eq. 9): sector 255 at s=n,
    sector 167 at s=n-2 (the per-sector s_max,sector override that we found
    is required for completeness), and truncate_sp l=9-n so every subsector
    gets s <= t-(8-n) (decreasing rule). Matches the pyfeyngym decreasing+167
    improved set, i.e. the literature's improved seeding -- unlike
    write_kira_improved_jobs which inherits s=n into all subsectors (the old
    conservative seeding)."""
    s_167 = max(1, n - 2)
    l = 9 - n
    (workdir / "jobs.yaml").write_text(f"""\
jobs:
  - reduce_sectors:
      reduce:
        - {{topologies: [{TOPO}], sectors: [255], r: 8, s: {n}, d: 0}}
        - {{topologies: [{TOPO}], sectors: [167], r: 8, s: {s_167}, d: 0}}
      truncate_sp:
        - {{topologies: [{TOPO}], l: {l}}}
      select_integrals:
        select_mandatory_list:
          - [targets]
      run_initiate: true
      run_firefly: true
""")


def write_strip_jobs(workdir: Path, n: int) -> None:
    """Strip strategy: list every on-cut sub-sector at the wide envelope
    and let KIRA_SEED_FILTER_FILE restrict to the actual strip seeds."""
    bounds = derive_nb_sector_bounds(n)
    reduce_lines = "\n".join(
        f"        - {{topologies: [{TOPO}], sectors: [{sec}], r: {r}, s: {s}, d: 0}}"
        for sec, r, s in bounds
    )
    (workdir / "jobs.yaml").write_text(f"""\
jobs:
  - reduce_sectors:
      reduce:
{reduce_lines}
      select_integrals:
        select_mandatory_list:
          - [targets]
      run_initiate: true
      run_firefly: true
""")


def build_kirafilt_seeds(n: int, trivial_sectors) -> list[tuple]:
    """Kira's default seed envelope distributed across every on-cut
    sub-sector. The seed filter then restricts Kira's sub-sector descent
    to that envelope — unlike the plain kira strategy where descent is
    unbounded. Envelope per sub-sector matches derive_nb_sector_bounds
    (the same bounds strip lists in jobs.yaml): sector 255 at s=max(5,n),
    sector 167 at s=3, every other on-cut sub-sector at s=max(1,t-4)
    (tightened to t-3 for the excl_mask sub-family), all at r<=8, d=0.
    Unlike strip, no strip-pattern filter on top: inside each sub-sector
    the filter lets every integral within those bounds through, so Kira's
    own improved-seeding choice governs what gets enumerated."""
    cut_mask = (1 << 0) | (1 << 2) | (1 << 5) | (1 << 7)
    excl_mask = (1 << 3) | (1 << 4) | (1 << 6)
    seed_set: set = set()
    for sec in range(256):
        if (sec & cut_mask) != cut_mask:
            continue
        if sec in trivial_sectors:
            continue
        t = bin(sec).count("1")
        # Widen every sub-sector to s=n so Kira's descent isn't starved
        # in the a_11 direction (the strip strategy handles the same
        # problem via its shift pattern rather than per-sub-sector
        # widening). This is still far tighter than the unbounded
        # descent Kira runs when no filter is set.
        s_max = max(n, max(1, t - 4))
        if (sec & excl_mask) == 0:
            s_max = max(s_max, max(1, t - 3))
        sec_seeds = pfg.gen_all_seeds(TOP_SECTOR, trivial_sectors, s_max, 8, 0)
        for s in sec_seeds:
            if pfg.to_sector(s) == sec and pfg.s_level(s) <= s_max:
                seed_set.add(s)
    return pfg.sort_integrals_desc(list(seed_set))


def write_kira_nosym_jobs(workdir: Path, n: int) -> None:
    """Kira's improved seed shape extended to every on-cut sub-sector
    explicitly, so the reduction does not rely on Kira's sector
    symmetry pipeline. Per-sector bounds come from derive_nb_sector_bounds
    (same list the strip strategy uses); no seed filter file is passed,
    so Kira enumerates its full improved-seed set inside every listed
    sector."""
    bounds = derive_nb_sector_bounds(n)
    reduce_lines = "\n".join(
        f"        - {{topologies: [{TOPO}], sectors: [{sec}], r: {r}, s: {s}, d: 0}}"
        for sec, r, s in bounds
    )
    (workdir / "jobs.yaml").write_text(f"""\
jobs:
  - reduce_sectors:
      reduce:
{reduce_lines}
      select_integrals:
        select_mandatory_list:
          - [targets]
      run_initiate: true
      run_firefly: true
""")


def write_strip_2sec_jobs(workdir: Path, n: int) -> None:
    """Strip-2sec: list ONLY sectors 255 and 167 (the same as
    kira-improved), at wide enough s for the strip seeds to fall inside.
    The KIRA_SEED_FILTER_FILE then restricts Kira's enumeration to the
    actual strip-shifted seed set inside sector 255 and to Kira's s=3
    box inside sector 167. Both strategies rely on Kira's sector
    symmetry pipeline to reach the other 14 on-cut sub-sectors."""
    s_top = max(5, n)  # wide envelope; the filter restricts to strip
    (workdir / "jobs.yaml").write_text(f"""\
jobs:
  - reduce_sectors:
      reduce:
        - {{topologies: [{TOPO}], sectors: [255], r: 8, s: {s_top}, d: 0}}
        - {{topologies: [{TOPO}], sectors: [167], r: 8, s: 3, d: 0}}
      select_integrals:
        select_mandatory_list:
          - [targets]
      run_initiate: true
      run_firefly: true
""")


MASTERS_26 = [
    (1,0,1,0,0,1,0,1,0,0,0), (1,0,1,1,0,1,0,1,0,0,0),
    (1,1,1,1,0,1,0,1,0,0,0), (1,0,1,1,1,1,0,1,0,0,0),
    (1,1,1,0,1,1,0,1,0,0,0), (1,1,1,1,1,1,0,1,0,0,0),
    (1,1,1,1,1,1,-1,1,0,0,0), (1,0,1,1,0,1,1,1,0,0,0),
    (1,1,1,1,0,1,1,1,0,0,0), (1,1,1,1,-1,1,1,1,0,0,0),
    (1,1,1,1,0,1,1,1,-1,0,0), (1,0,1,0,1,1,1,1,0,0,0),
    (1,0,1,1,1,1,1,1,0,0,0), (1,-1,1,1,1,1,1,1,0,0,0),
    (1,1,1,0,1,1,1,1,0,0,0), (1,1,1,-1,1,1,1,1,0,0,0),
    (1,1,1,0,1,1,1,1,-1,0,0), (1,1,1,1,1,1,1,1,0,0,0),
    (1,1,1,1,1,1,1,1,-1,0,0), (1,1,1,1,1,1,1,1,0,-1,0),
    (1,1,1,1,1,1,1,1,0,0,-1), (1,1,1,1,1,1,1,1,-2,0,0),
    (1,1,1,1,1,1,1,1,-1,-1,0), (1,1,1,1,1,1,1,1,-1,0,-1),
    (1,1,1,1,1,1,1,1,0,-2,0), (1,1,1,1,1,1,1,1,0,-1,-1),
]


def write_strip_jobs_with_masters(workdir: Path, n: int) -> None:
    """Same as write_strip_jobs but adds preferred_masters for skip-selection."""
    bounds = derive_nb_sector_bounds(n)
    reduce_lines = "\n".join(
        f"        - {{topologies: [{TOPO}], sectors: [{sec}], r: {r}, s: {s}, d: 0}}"
        for sec, r, s in bounds
    )
    (workdir / "jobs.yaml").write_text(f"""\
jobs:
  - reduce_sectors:
      reduce:
{reduce_lines}
      select_integrals:
        select_mandatory_list:
          - [targets]
      preferred_masters: preferred_masters.txt
      run_initiate: true
      run_firefly: true
""")


def write_numerical_points(workdir: Path) -> None:
    """Write a numerical_points file for a single-probe numeric solve."""
    (workdir / "numerics").write_text(
        "prime 2147483647\n"
        "d s23 s34 s45 s51\n"
        "23 3 5 17 23\n"
    )


def write_strip_jobs_numerical(workdir: Path, n: int) -> None:
    """Jobs.yaml that uses numerical_points instead of run_firefly."""
    bounds = derive_nb_sector_bounds(n)
    reduce_lines = "\n".join(
        f"        - {{topologies: [{TOPO}], sectors: [{sec}], r: {r}, s: {s}, d: 0}}"
        for sec, r, s in bounds
    )
    (workdir / "jobs.yaml").write_text(f"""\
jobs:
  - reduce_sectors:
      reduce:
{reduce_lines}
      select_integrals:
        select_mandatory_list:
          - [targets]
      preferred_masters: preferred_masters.txt
      run_initiate: true
      numerical_points: numerics
""")


def write_preferred_masters(workdir: Path) -> None:
    with open(workdir / "preferred_masters.txt", "w") as f:
        for m in MASTERS_26:
            f.write(f"{TOPO}[{','.join(str(x) for x in m)}]\n")


def write_seed_filter_file(path: Path, seeds: list[tuple]) -> None:
    with open(path, "w") as f:
        for seed in seeds:
            f.write("[" + ",".join(str(int(a)) for a in seed) + "]\n")


# ─── /proc-based RSS sampling ───────────────────────────────────────────────

def read_rss_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return 0


def descendants(root: int) -> list[int]:
    children: dict[int, list[int]] = {}
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
            out.append(c)
            stack.append(c)
    return out


class PeakRssSampler(threading.Thread):
    def __init__(self, pid: int, interval: float = 0.1):
        super().__init__(daemon=True)
        self.pid, self.interval = pid, interval
        self.peak_kb = 0
        self.stop_evt = threading.Event()

    def run(self) -> None:
        while not self.stop_evt.is_set():
            total = sum(read_rss_kb(p) for p in descendants(self.pid))
            if total > self.peak_kb:
                self.peak_kb = total
            self.stop_evt.wait(self.interval)


# ─── kira launcher ──────────────────────────────────────────────────────────

def load_env() -> dict[str, str]:
    env = os.environ.copy()
    for line in KIRA_ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            k, _, v = line[len("export "):].partition("=")
            env[k] = os.path.expandvars(v)
    return env


def run_kira(workdir: Path, label: str, seed_filter_file: Path | None = None,
             skip_selection: bool = False, skip_forward_elim: bool = False,
             sparse_pivot: bool = False) -> dict:
    env = load_env()
    if seed_filter_file is not None:
        env["KIRA_SEED_FILTER_FILE"] = str(seed_filter_file)
    else:
        env.pop("KIRA_SEED_FILTER_FILE", None)
    if skip_selection:
        env["KIRA_SKIP_SELECTION"] = "1"
    else:
        env.pop("KIRA_SKIP_SELECTION", None)
    if skip_forward_elim:
        env["KIRA_SKIP_FORWARD_ELIMINATION"] = "1"
    else:
        env.pop("KIRA_SKIP_FORWARD_ELIMINATION", None)
    if sparse_pivot:
        env["KIRA_SPARSE_PIVOT"] = "1"
    else:
        env.pop("KIRA_SPARSE_PIVOT", None)
    for sub in ("tmp", "results", "sectormappings", "kira.log", "run.out",
                "ff_save", "firefly_saves", "firefly.log"):
        p = workdir / sub
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()

    print(f"  [{label}] starting kira (timeout {RUN_TIMEOUT_S}s)", flush=True)
    t0 = time.time()
    timed_out = False
    with open(workdir / "run.out", "wb") as out_f:
        proc = subprocess.Popen(
            ["kira", *KIRA_SET_VALUE, "jobs.yaml"],
            cwd=str(workdir),
            env=env,
            stdout=out_f,
            stderr=subprocess.STDOUT,
        )
        sampler = PeakRssSampler(proc.pid)
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
    log_text = (workdir / "kira.log").read_text() if (workdir / "kira.log").exists() else ""
    ff_log = (workdir / "firefly.log").read_text() if (workdir / "firefly.log").exists() else ""
    metrics = parse_log(log_text, ff_log)
    metrics.update({
        "exit_code": rc,
        "wall_s": round(wall, 2),
        "peak_rss_mib": round(sampler.peak_kb / 1024, 2),
        "timed_out": timed_out,
    })
    return metrics


def parse_log(text: str, ff_text: str) -> dict:
    """Extract Kira-3-paper-style metrics from a kira.log + firefly.log."""
    m: dict = {}
    for pat, key in [
        (r"(?:Total number|Number) of master integrals:\s*(\d+)", "n_masters"),
        (r"Non trivial sectors in total:\s*(\d+)", "n_nontrivial_sectors"),
    ]:
        match = re.search(pat, text)
        if match:
            m[key] = int(match.group(1))
    counts = [int(x) for x in re.findall(
        r"Number of selected equations to reduce:\s*(\d+)\s*equations", text)]
    if counts:
        m["n_sel"] = counts[-1]
    pm = re.search(
        r"\[paper metrics\]\s*n_generated=(\d+)\s*n_independent=(\d+)", text)
    if pm:
        m["n_gen"] = int(pm.group(1))
        m["n_indep"] = int(pm.group(2))
    # Tgen: timing "( X s )" between "Kira starts the reduction" and the
    # Loading line.
    started = False
    last_timing = None
    for ln in text.splitlines():
        if "Kira starts the reduction" in ln:
            started = True
            continue
        if not started:
            continue
        if "Loading" in ln and "equations" in ln:
            break
        tm = re.match(r"\(\s*([0-9.]+)\s*s\s*\)", ln.strip())
        if tm:
            last_timing = float(tm.group(1))
    if last_timing is not None:
        m["tgen"] = last_timing
    # TpyRed: median of Kira's "Average times" probe blocks.
    blocks = re.findall(
        r"Average times for probe solution steps:\s*\n"
        r"Coefficient evaluation:\s*([0-9.]+)\s*s\s*\n"
        r"Forward elimination:\s*([0-9.]+)\s*s\s*\n"
        r"Back substitution:\s*([0-9.]+)\s*s",
        text)
    totals = [float(a) + float(b) + float(c)
              for a, b, c in blocks if float(a) + float(b) + float(c) > 0]
    if totals:
        totals.sort()
        m["tpyred"] = totals[len(totals) // 2]
    elif ff_text:
        fb = re.search(r"Time for the first black-box probe:\s*([0-9.]+)\s*s", ff_text)
        if fb:
            m["tpyred"] = float(fb.group(1))
            m["tpyred_source"] = "firefly_first_probe"
    return m


# ─── main ──────────────────────────────────────────────────────────────────

def setup_kira_improved(n: int, trivial_sectors) -> tuple[Path, dict]:
    workdir = HERE / f"papermetrics_kira_n{n}"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    copy_config(workdir)
    write_kira_improved_jobs(workdir, n)
    write_targets(workdir, n)
    # Compute a seed count for reporting.
    s_top = max(5, n)
    all_seeds = pfg.gen_all_seeds(TOP_SECTOR, trivial_sectors, max(s_top, 3), 8, 0)
    keep = []
    for s in all_seeds:
        sec = pfg.to_sector(s)
        if sec == 255 and pfg.s_level(s) <= s_top: keep.append(s)
        elif sec == 167 and pfg.s_level(s) <= 3: keep.append(s)
    return workdir, {"n_seeds": len(keep)}


def setup_kira_truncate(n: int, trivial_sectors) -> tuple[Path, dict]:
    workdir = HERE / f"papermetrics_kiratrunc_n{n}"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    copy_config(workdir)
    write_kira_truncate_jobs(workdir, n)
    write_targets(workdir, n)
    return workdir, {}


def setup_strip(n: int, trivial_sectors) -> tuple[Path, Path, dict]:
    workdir = HERE / f"papermetrics_strip_n{n}"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    copy_config(workdir)
    write_strip_jobs(workdir, n)
    write_targets(workdir, n)
    seeds = build_strip_seeds(n, trivial_sectors)
    seed_file = workdir / "strip_seeds.txt"
    write_seed_filter_file(seed_file, seeds)
    return workdir, seed_file, {"n_seeds": len(seeds)}


def setup_kirafilt(n: int, trivial_sectors) -> tuple[Path, Path, dict]:
    """Kira's improved seed shape across ALL on-cut sub-sectors, delivered
    through a seed filter file so Kira's sub-sector descent is constrained
    the same way the strip strategy constrains it. jobs.yaml lists every
    on-cut sub-sector (same bounds as strip) so the filter entries for
    those sub-sectors can fire during descent."""
    workdir = HERE / f"papermetrics_kirafilt_n{n}"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    copy_config(workdir)
    write_strip_jobs(workdir, n)  # same 16-sub-sector envelope as strip
    write_targets(workdir, n)
    seeds = build_kirafilt_seeds(n, trivial_sectors)
    seed_file = workdir / "kirafilt_seeds.txt"
    write_seed_filter_file(seed_file, seeds)
    return workdir, seed_file, {"n_seeds": len(seeds)}


def setup_strip_2sec(n: int, trivial_sectors) -> tuple[Path, Path, dict]:
    workdir = HERE / f"papermetrics_strip2sec_n{n}"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    copy_config(workdir)
    write_strip_2sec_jobs(workdir, n)
    write_targets(workdir, n)
    seeds = build_strip_2sec_seeds(n, trivial_sectors)
    seed_file = workdir / "strip2sec_seeds.txt"
    write_seed_filter_file(seed_file, seeds)
    return workdir, seed_file, {"n_seeds": len(seeds)}


def main() -> int:
    print("Loading trivial sectors with cut [1,3,6,8]...", flush=True)
    trivial_sectors = pfg.get_trivial_sectors(TRIVIAL_FILE, cut=CUT, n_indices=11)

    records: list[dict] = []
    for n in N_VALUES:
        print(f"\n=== n={n} ===", flush=True)
        rec: dict = {"n": n}

        if "strip" in STRATEGIES:
            strip_dir, seed_file, strip_seed_meta = setup_strip(n, trivial_sectors)
            rec["strip_seeds"] = strip_seed_meta["n_seeds"]
            rec["strip"] = run_kira(strip_dir, f"strip    n={n}",
                                    seed_filter_file=seed_file)

        if "stripfast" in STRATEGIES:
            sf_dir = HERE / f"papermetrics_stripfast_n{n}"
            if sf_dir.exists():
                shutil.rmtree(sf_dir)
            sf_dir.mkdir(parents=True)
            copy_config(sf_dir)
            write_strip_jobs(sf_dir, n)
            write_targets(sf_dir, n)
            sf_seeds = build_strip_seeds(n, trivial_sectors)
            sf_file = sf_dir / "strip_seeds.txt"
            write_seed_filter_file(sf_file, sf_seeds)
            # Write preferred_masters and a jobs.yaml that references it
            write_preferred_masters(sf_dir)
            write_strip_jobs_with_masters(sf_dir, n)
            rec["stripfast_seeds"] = len(sf_seeds)
            rec["stripfast"] = run_kira(sf_dir, f"stripfast n={n}",
                                        seed_filter_file=sf_file,
                                        skip_forward_elim=True,
                                        skip_selection=True)

        if "stripsparse" in STRATEGIES:
            ss_dir = HERE / f"papermetrics_stripsparse_n{n}"
            if ss_dir.exists():
                shutil.rmtree(ss_dir)
            ss_dir.mkdir(parents=True)
            copy_config(ss_dir)
            write_strip_jobs(ss_dir, n)
            write_targets(ss_dir, n)
            ss_seeds = build_strip_seeds(n, trivial_sectors)
            ss_file = ss_dir / "strip_seeds.txt"
            write_seed_filter_file(ss_file, ss_seeds)
            rec["stripsparse_seeds"] = len(ss_seeds)
            rec["stripsparse"] = run_kira(ss_dir, f"stripsparse n={n}",
                                          seed_filter_file=ss_file,
                                          sparse_pivot=True)

        if "stripnumerical" in STRATEGIES:
            sn_dir = HERE / f"papermetrics_stripnumerical_n{n}"
            if sn_dir.exists():
                shutil.rmtree(sn_dir)
            sn_dir.mkdir(parents=True)
            copy_config(sn_dir)
            write_targets(sn_dir, n)
            sn_seeds = build_strip_seeds(n, trivial_sectors)
            sn_file = sn_dir / "strip_seeds.txt"
            write_seed_filter_file(sn_file, sn_seeds)
            write_preferred_masters(sn_dir)
            write_numerical_points(sn_dir)
            write_strip_jobs_numerical(sn_dir, n)
            rec["stripnumerical_seeds"] = len(sn_seeds)
            rec["stripnumerical"] = run_kira(
                sn_dir, f"stripnum  n={n}",
                seed_filter_file=sn_file,
                skip_forward_elim=True,
                skip_selection=True,
                sparse_pivot=True)

        if "stripfullsparse" in STRATEGIES:
            sfs_dir = HERE / f"papermetrics_stripfullsparse_n{n}"
            if sfs_dir.exists():
                shutil.rmtree(sfs_dir)
            sfs_dir.mkdir(parents=True)
            copy_config(sfs_dir)
            write_strip_jobs(sfs_dir, n)
            write_targets(sfs_dir, n)
            sfs_seeds = build_strip_seeds(n, trivial_sectors)
            sfs_file = sfs_dir / "strip_seeds.txt"
            write_seed_filter_file(sfs_file, sfs_seeds)
            rec["stripfullsparse_seeds"] = len(sfs_seeds)
            write_preferred_masters(sfs_dir)
            write_strip_jobs_with_masters(sfs_dir, n)
            rec["stripfullsparse"] = run_kira(
                sfs_dir, f"fullsparse n={n}",
                seed_filter_file=sfs_file,
                skip_forward_elim=True,
                skip_selection=True,
                sparse_pivot=True)

        if "strip2sec" in STRATEGIES:
            s2sec_dir, s2sec_file, s2sec_meta = setup_strip_2sec(n, trivial_sectors)
            rec["strip2sec_seeds"] = s2sec_meta["n_seeds"]
            rec["strip2sec"] = run_kira(s2sec_dir, f"strip2sec n={n}",
                                        seed_filter_file=s2sec_file)

        if "kirafilt" in STRATEGIES:
            kf_dir, kf_file, kf_meta = setup_kirafilt(n, trivial_sectors)
            rec["kirafilt_seeds"] = kf_meta["n_seeds"]
            rec["kirafilt"] = run_kira(kf_dir, f"kirafilt n={n}",
                                       seed_filter_file=kf_file)

        if "kira" in STRATEGIES:
            kira_dir, kira_seed_meta = setup_kira_improved(n, trivial_sectors)
            rec["kira_seeds"] = kira_seed_meta["n_seeds"]
            rec["kira"] = run_kira(kira_dir, f"kira     n={n}")

        if "kira_trunc" in STRATEGIES:
            kt_dir, _ = setup_kira_truncate(n, trivial_sectors)
            rec["kira_trunc"] = run_kira(kt_dir, f"kiratrunc n={n}")

        records.append(rec)
        (HERE / RESULTS_FILE).write_text(json.dumps(records, indent=2))

    print_summary(records)
    return 0


def fmt(v, suffix=""):
    if v is None or (isinstance(v, str) and v == "—"):
        return "—"
    if isinstance(v, float):
        return f"{v:.3g}{suffix}"
    return f"{v}{suffix}"


def print_summary(records: list[dict]) -> None:
    print()
    print("=" * 132)
    print("strip vs kira-improved on doublePentagon cut [1,3,6,8] reducing "
          "(1,...,1,0,0,-n) — Kira-3-paper-style metrics + wall time")
    print("=" * 132)
    cols = [
        ("n",        lambda r: str(r["n"])),
        ("strategy", lambda r: r.get("_label", "")),
        ("seeds",    lambda r: str(r.get("_seeds", "—"))),
        ("Tgen[s]",  lambda r: fmt(r.get("_m", {}).get("tgen"))),
        ("#gen",     lambda r: fmt(r.get("_m", {}).get("n_gen"))),
        ("#indep",   lambda r: fmt(r.get("_m", {}).get("n_indep"))),
        ("#sel",     lambda r: fmt(r.get("_m", {}).get("n_sel"))),
        ("Tpyred[s]",lambda r: fmt(r.get("_m", {}).get("tpyred"))),
        ("RSS[MiB]", lambda r: fmt(r.get("_m", {}).get("peak_rss_mib"))),
        ("masters",  lambda r: fmt(r.get("_m", {}).get("n_masters"))),
        ("wall[s]",  lambda r: fmt(r.get("_m", {}).get("wall_s"))
                                  + ("*" if r.get("_m", {}).get("timed_out") else "")),
    ]
    rows = []
    for rec in records:
        for label, key, seeds_key in [
            ("strip",     "strip",     "strip_seeds"),
            ("stripsparse","stripsparse","stripsparse_seeds"),
            ("stripnumerical","stripnumerical","stripnumerical_seeds"),
            ("stripfullsparse","stripfullsparse","stripfullsparse_seeds"),
            ("stripfast", "stripfast", "stripfast_seeds"),
            ("strip2sec", "strip2sec", "strip2sec_seeds"),
            ("kirafilt",  "kirafilt",  "kirafilt_seeds"),
            ("kira",      "kira",      "kira_seeds"),
        ]:
            if label not in STRATEGIES:
                continue
            r = {**rec, "_label": label, "_seeds": rec.get(seeds_key),
                 "_m": rec.get(key, {})}
            rows.append([get(r) for _, get in cols])
    headers = [h for h, _ in cols]
    widths = [max(len(h), max((len(row[i]) for row in rows), default=0))
              for i, h in enumerate(headers)]
    pad = "  "
    print(pad.join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print(pad.join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print(pad.join(row[i].ljust(widths[i]) for i in range(len(row))))
    print()
    print("(* = run hit timeout — Tgen and equation counts captured if reached "
          "before kill)")


if __name__ == "__main__":
    sys.exit(main())
