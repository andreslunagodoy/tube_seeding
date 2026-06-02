"""
analyze_path_coverage.py

Merge the 6 batch JSONs from path_coverage_rank10.py, build the 66x66
coverage matrix, and find a minimal subset of paths covering all 66
rank-10 targets via greedy set cover. Also try integer programming via
PuLP if available; fall back to greedy + brute-force trim otherwise.
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

ALL_TARGETS = []
RANK = 10
for l in range(RANK + 1):
    for m in range(RANK - l + 1):
        n = RANK - l - m
        ALL_TARGETS.append((l, m, n))


def lmn_str(t):
    return f"({t[0]},{t[1]},{t[2]})"


def load_all():
    merged = {}
    for i in range(1, 7):
        p = HERE / f"path_coverage_b{i}.json"
        d = json.loads(p.read_text())
        for k, v in d.items():
            merged[int(k)] = v
    if len(merged) != 66:
        raise SystemExit(f"merged: {len(merged)} entries, expected 66")
    return merged


def greedy_cover(coverage, all_idx):
    """Greedy set cover: pick path that covers most uncovered targets."""
    uncov = set(all_idx)
    chosen = []
    cov_by = {}
    while uncov:
        best, best_set = None, set()
        for p, s in coverage.items():
            new = s & uncov
            if len(new) > len(best_set):
                best, best_set = p, new
        if best is None or not best_set:
            break
        chosen.append(best)
        for t in best_set:
            cov_by[t] = best
        uncov -= best_set
    return chosen, cov_by, uncov


def trim_redundant(chosen, coverage, all_idx):
    """Remove paths whose covered targets are fully covered by the others."""
    chosen = list(chosen)
    changed = True
    while changed:
        changed = False
        for p in list(chosen):
            others = [q for q in chosen if q != p]
            covered_others = set().union(*(coverage[q] for q in others))
            if set(all_idx) <= covered_others:
                chosen.remove(p)
                changed = True
                break
    return chosen


def ilp_cover(coverage, all_idx):
    """Try optimal set cover via PuLP if available."""
    import sys as _sys
    try:
        import pulp  # type: ignore
    except Exception:
        return None
    prob = pulp.LpProblem("setcover", pulp.LpMinimize)
    x = {p: pulp.LpVariable(f"x_{p}", cat="Binary") for p in coverage}
    prob += pulp.lpSum(x.values())
    for t in all_idx:
        covers_t = [p for p, s in coverage.items() if t in s]
        prob += pulp.lpSum(x[p] for p in covers_t) >= 1, f"cover_{t}"
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    return sorted(p for p, var in x.items() if pulp.value(var) > 0.5)


def main():
    rec = load_all()
    coverage = {p: set(rec[p]["covers"]) for p in rec}
    all_idx = list(range(66))

    # Sanity: how many paths reduce themselves?
    own_ok = sum(1 for p in rec.values() if p["own_ok"])
    print(f"paths that reduce their own target: {own_ok}/66")

    # Symmetric matrix? (probably not; coverage is not symmetric in general)
    # Sizes
    sizes = sorted(((len(coverage[p]), p) for p in coverage), reverse=True)
    print(f"\nTop 10 paths by coverage count:")
    for sz, p in sizes[:10]:
        t = ALL_TARGETS[p]
        print(f"  {p:>2}  target={lmn_str(t):<10} path={rec[p]['path']:<14}"
              f"  covers {sz}/66")

    # Identify paths that nothing else covers (i.e., the unique "extreme" cases)
    # Targets: how many paths reduce each target?
    reducers = {t: [p for p, s in coverage.items() if t in s]
                for t in all_idx}
    sole = [t for t in all_idx if len(reducers[t]) == 1]
    print(f"\nTargets reducible by exactly one path: {len(sole)}")
    for t in sole:
        p = reducers[t][0]
        print(f"  {lmn_str(ALL_TARGETS[t])} -- only path {p} ({rec[p]['path']})")

    # Greedy cover
    greedy, cov_by, uncov = greedy_cover(coverage, all_idx)
    if uncov:
        print(f"\nGreedy uncovered: {[lmn_str(ALL_TARGETS[t]) for t in uncov]}")
    print(f"\nGreedy cover: {len(greedy)} paths")

    # Trim
    trimmed = trim_redundant(greedy, coverage, all_idx)
    print(f"After trim: {len(trimmed)} paths")

    # ILP optimal
    opt = ilp_cover(coverage, all_idx)
    if opt is not None:
        print(f"ILP-optimal cover: {len(opt)} paths")
    else:
        print("ILP not available (no pulp); using greedy+trim")
        opt = trimmed

    # Display the chosen paths and their coverage
    chosen = sorted(opt, key=lambda p: ALL_TARGETS[p])
    print(f"\nChosen paths ({len(chosen)}):")
    print(f"  {'idx':>3}  {'target':<10} {'path':<14} {'#seeds':>6} "
          f"{'#eqs':>6} {'Tgen':>6} {'Tsolve':>7} {'covers':>6}")
    union = set()
    for p in chosen:
        t = ALL_TARGETS[p]
        union |= coverage[p]
        print(f"  {p:>3}  {lmn_str(t):<10} {rec[p]['path']:<14} "
              f"{rec[p]['n_seeds']:>6} {rec[p]['n_eqs']:>6} "
              f"{rec[p]['tgen_s']:>5.1f}s {rec[p]['tsolve_s']:>6.1f}s "
              f"{rec[p]['n_covers']:>6}")
    print(f"\nUnion of chosen coverage: {len(union)}/66")

    # Per-path coverage table (compact)
    print(f"\nFull coverage matrix (path -> set of covered targets):")
    print(f"{'idx':>3} {'target':<10} {'path':<14} {'#cov':>4}  covered targets")
    for p in sorted(coverage):
        t = ALL_TARGETS[p]
        covlist = sorted(coverage[p])
        cov_str = ", ".join(lmn_str(ALL_TARGETS[q]) for q in covlist)
        print(f"{p:>3} {lmn_str(t):<10} {rec[p]['path']:<14} "
              f"{len(covlist):>4}  {cov_str}")

    out = {
        "n_paths": 66,
        "coverage": {p: sorted(coverage[p]) for p in coverage},
        "greedy_cover_indices": greedy,
        "trimmed_cover_indices": trimmed,
        "optimal_cover_indices": opt,
        "all_targets": ALL_TARGETS,
        "n_chosen": len(chosen),
        "totals_chosen": {
            "n_seeds": sum(rec[p]["n_seeds"] for p in chosen),
            "n_eqs": sum(rec[p]["n_eqs"] for p in chosen),
            "tgen_s": sum(rec[p]["tgen_s"] for p in chosen),
            "tsolve_s": sum(rec[p]["tsolve_s"] for p in chosen),
        },
    }
    (HERE / "path_coverage_analysis.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved analysis -> {HERE / 'path_coverage_analysis.json'}")


if __name__ == "__main__":
    main()
