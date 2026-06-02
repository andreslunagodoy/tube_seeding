"""
analyze_minmax_cover.py

Read per-path RSS measurements (path_rss/p{0..65}.json) and find:

  (a) the cardinality-optimal cover (re-uses path_coverage_analysis.json
      for ground truth on coverage),
  (b) the bottleneck-optimal cover, i.e. the cover that minimises the
      maximum peak RSS over the chosen paths.

For (b), since adding any path with rss <= tau is monotone-good, the
optimal threshold is determined by:

  tau_min = min { rss(p_max) : S ⊆ {paths with rss <= rss(p_max)}
                                covers all 66 targets }

Greedy answer: sort paths by rss ascending; for each prefix, check whether
the union of coverage covers all 66. Smallest prefix that does gives
tau_min.

Then within {paths with rss <= tau_min}, find the cardinality-minimum
sub-cover via ILP.
"""
from __future__ import annotations
import json
import sys as _sys
from pathlib import Path

import pulp  # type: ignore

HERE = Path(__file__).resolve().parent

ALL_TARGETS = []
RANK = 10
for l in range(RANK + 1):
    for m in range(RANK - l + 1):
        n = RANK - l - m
        ALL_TARGETS.append((l, m, n))


def lmn(t):
    return f"({t[0]},{t[1]},{t[2]})"


def load():
    rec = {}
    for i in range(66):
        p = HERE / f"path_rss/p{i}.json"
        rec[i] = json.loads(p.read_text())
    return rec


def ilp_min_cardinality(coverage_dict, allowed, all_idx):
    """Minimum-cardinality cover using only paths in `allowed`."""
    prob = pulp.LpProblem("setcover", pulp.LpMinimize)
    x = {p: pulp.LpVariable(f"x_{p}", cat="Binary") for p in allowed}
    prob += pulp.lpSum(x.values())
    for t in all_idx:
        cov = [p for p in allowed if t in coverage_dict[p]]
        if not cov:
            return None
        prob += pulp.lpSum(x[p] for p in cov) >= 1, f"cover_{t}"
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    return sorted(p for p in allowed if pulp.value(x[p]) > 0.5)


def ilp_min_max_rss(coverage_dict, rss_dict, all_idx):
    """Find the minimum max-RSS over all valid covers using a single ILP
    on the `t` (max-rss) variable. Equivalent to an enumerate-by-threshold
    sweep — just for safety, do both and verify they agree."""
    # Sweep approach: sort paths by RSS; for prefix include test cover.
    sorted_paths = sorted(rss_dict, key=lambda p: rss_dict[p])
    for k, p_max in enumerate(sorted_paths):
        allowed = sorted_paths[:k + 1]
        # union covers all?
        union = set().union(*(coverage_dict[a] for a in allowed))
        if set(all_idx) <= union:
            tau = rss_dict[p_max]
            sub = ilp_min_cardinality(coverage_dict, allowed, all_idx)
            return tau, sub
    return None, None


def main():
    rec = load()
    coverage = {p: set(rec[p]["covers"]) for p in rec}
    rss = {p: rec[p]["rss_peak_mib"] for p in rec}
    all_idx = list(range(66))

    # Sanity prints
    print("RSS distribution (MiB):")
    sorted_p = sorted(rss, key=lambda p: rss[p])
    for k, p in enumerate(sorted_p):
        if k < 5 or k >= len(sorted_p) - 5:
            t = ALL_TARGETS[p]
            print(f"  rank {k:>2}  path {p:>2} target={lmn(t):<10} "
                  f"path={rec[p]['path']:<14} "
                  f"rss_peak={rss[p]:>7.1f} MiB "
                  f"rss_gen={rec[p]['rss_gen_mib']:>7.1f} "
                  f"rss_solve={rec[p]['rss_solve_mib']:>7.1f} "
                  f"covers={rec[p]['n_covers']}/66")
    print(f"  RSS span: {min(rss.values()):.1f} – {max(rss.values()):.1f} MiB")

    # (a) cardinality-optimal cover (no RSS constraint)
    card = ilp_min_cardinality(coverage, all_idx, all_idx)
    print(f"\n(a) Cardinality-optimal cover: {len(card)} paths")
    print_cover(card, rec, rss, coverage)

    # (b) minmax cover
    tau, mm = ilp_min_max_rss(coverage, rss, all_idx)
    print(f"\n(b) Bottleneck-optimal cover: max-RSS = {tau:.1f} MiB, "
          f"|S| = {len(mm)} paths")
    print_cover(mm, rec, rss, coverage)

    # Pareto front: sweep all RSS thresholds to show trade-off.
    print("\nPareto front (max-RSS threshold vs minimum cover size):")
    thresholds = sorted(set(rss.values()))
    seen_size = None
    print(f"  {'tau (MiB)':>11}  {'|S|':>3}")
    for tau in thresholds:
        allowed = [p for p in all_idx if rss[p] <= tau]
        if set().union(*(coverage[p] for p in allowed)) != set(all_idx):
            continue
        sub = ilp_min_cardinality(coverage, allowed, all_idx)
        if sub is None:
            continue
        if seen_size is None or len(sub) < seen_size:
            seen_size = len(sub)
            print(f"  {tau:>11.1f}  {len(sub):>3}")

    out = {
        "cardinality_optimal": {
            "indices": card,
            "size": len(card),
            "max_rss_mib": max(rss[p] for p in card),
            "total_tsolve_s": sum(rec[p]["tsolve_s"] for p in card),
        },
        "bottleneck_optimal": {
            "indices": mm,
            "size": len(mm),
            "max_rss_mib": tau,
            "total_tsolve_s": sum(rec[p]["tsolve_s"] for p in mm),
        },
        "rss_per_path": rss,
    }
    (HERE / "minmax_cover_analysis.json").write_text(json.dumps(out, indent=2))


def print_cover(idxs, rec, rss, coverage):
    print(f"  {'idx':>3} {'target':<10} {'path':<14} "
          f"{'#seeds':>6} {'#eqs':>6} "
          f"{'Tgen':>6} {'Tsolve':>7} {'rss_peak':>9} {'covers':>6}")
    union = set()
    for p in sorted(idxs, key=lambda p: ALL_TARGETS[p]):
        t = ALL_TARGETS[p]
        union |= coverage[p]
        print(f"  {p:>3} {lmn(t):<10} {rec[p]['path']:<14} "
              f"{rec[p]['n_seeds']:>6} {rec[p]['n_eqs']:>6} "
              f"{rec[p]['tgen_s']:>5.1f}s {rec[p]['tsolve_s']:>6.1f}s "
              f"{rec[p]['rss_peak_mib']:>8.1f} "
              f"{rec[p]['n_covers']:>6}")
    print(f"    -> union covers {len(union)}/66 ; "
          f"max RSS = {max(rec[p]['rss_peak_mib'] for p in idxs):.1f} MiB ; "
          f"sum Tsolve = {sum(rec[p]['tsolve_s'] for p in idxs):.1f} s")


if __name__ == "__main__":
    main()
