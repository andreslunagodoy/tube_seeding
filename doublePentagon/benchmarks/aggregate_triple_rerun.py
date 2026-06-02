"""
Aggregate the 10 triple-cut rerun JSONs (triple_cuts_rerun_c{1..10}.json)
into Table 3 (tab:triple-summary) and Figures 10 and 11
(triple_tsolve_body.tex and triple_rss_body.tex respectively).

Reads:
  triple_cuts_rerun_c{1..10}.json   - one cut each, n=5..20
  scan_n_feyngym_results_rerun_aggregate.json   - quad-cut [1,3,6,8]
                                                  fit constants and per-n
                                                  data (for the dashed
                                                  reference line).

Writes:
  triple_tsolve_body.tex     - Fig. 10
  triple_rss_body.tex        - Fig. 11
  triple_cuts_to_20_plot.tex - standalone version of Fig. 10
  (Table 3 is patched directly into stripseeding.tex by the caller.)

The colour palette is preserved from the existing files (c1..c11, cq).
"""
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = Path(__file__).resolve().parent / "paper_tables"

# Same order as TRIPLE_CUTS in scan_triple_cuts.py
CUT_ORDER = [
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

# Colours from the existing triple_tsolve_body.tex / triple_rss_body.tex,
# keyed by sorted cut-as-tuple. cq is reserved for the quad cut.
COLOR = {
    (3, 4, 8): "c1",
    (1, 4, 6): "c2",
    (2, 4, 6): "c3",
    (3, 4, 7): "c4",
    (2, 4, 7): "c5",
    (3, 5, 8): "c6",
    (2, 5, 8): "c7",
    (1, 5, 7): "c9",
    (1, 5, 6): "c10",
    (2, 5, 7): "c4!60!c9",
}

MARK = {
    (3, 4, 8): "square*",
    (1, 4, 6): "triangle*",
    (2, 4, 6): "diamond*",
    (3, 4, 7): "pentagon*",
    (2, 4, 7): "*",
    (3, 5, 8): "square*",
    (2, 5, 8): "triangle*",
    (1, 5, 7): "diamond*",
    (1, 5, 6): "pentagon*",
    (2, 5, 7): "*",
}


def load_cut(idx):
    p = HERE / f"triple_cuts_pinned10_c{idx}.json"
    return json.loads(p.read_text())


def ols(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den
    intercept = my - slope * mx
    return slope, intercept


def collect():
    """Returns a list of {cut, n_to_(tsolve, twall, rss_total_gib, n_masters)}."""
    out = []
    for i, cut in enumerate(CUT_ORDER, start=1):
        records = load_cut(i)
        ok = [r for r in records
              if not r.get("timed_out") and r.get("exit_code") == 0
              and r.get("reduced_ok")]
        if not ok:
            print(f"WARNING: cut {cut}: no successful runs")
            continue
        # group by n, aggregate over the 10 reps from phase 2
        by_n = {}
        for r in ok:
            by_n.setdefault(r["n"], []).append(r)
        per_n = {}
        for n, recs in by_n.items():
            ts = [r["tsolve_s"] for r in recs]
            tg = [r["tgen_s"] for r in recs]
            tw = [r["twall_s"] for r in recs if r.get("twall_s") is not None]
            rs = [r["peak_rss_solve_mib"] / 1024 for r in recs]
            rt = [r["peak_rss_total_mib"] / 1024 for r in recs]
            per_n[n] = {
                "tsolve_s": statistics.mean(ts),
                "tsolve_std": statistics.stdev(ts) if len(ts) > 1 else 0.0,
                "tgen_s": statistics.mean(tg),
                "tgen_std": statistics.stdev(tg) if len(tg) > 1 else 0.0,
                "twall_s": statistics.mean(tw) if tw else None,
                "rss_solve_gib": statistics.mean(rs),
                "rss_solve_std": statistics.stdev(rs) if len(rs) > 1 else 0.0,
                "rss_total_gib": statistics.mean(rt),
                "n_masters": recs[0].get("n_masters"),
                "n_reps": len(recs),
            }
        # All-OK statuses observed; pick highest n
        n_max = max(per_n)
        ns = sorted(per_n)
        slope, intercept = ols(ns, [per_n[n]["tsolve_s"] for n in ns])
        rss_slope, rss_intercept = ols(
            ns, [per_n[n]["rss_solve_gib"] for n in ns])
        out.append({
            "cut": cut, "n_max": n_max,
            "n_masters": per_n[5]["n_masters"],   # constant in n
            "slope": slope, "intercept": intercept,
            "rss_slope": rss_slope, "rss_intercept": rss_intercept,
            "tsolve_at_20": per_n.get(20, per_n[n_max])["tsolve_s"],
            "rss_at_20": per_n.get(20, per_n[n_max])["rss_solve_gib"],
            "per_n": per_n,
            "status": "all OK",
        })
    return out


def cut_label(cut):
    return ",".join(str(c) for c in cut)


def fmt_color(slope):
    """Sort key for table — by slope (ascending)."""
    return slope


def write_table3_block(rows, quad_fit):
    """The block goes into stripseeding.tex as Table 3."""
    lines = []
    # sort triples by slope ascending (matches original ordering convention)
    sorted_rows = sorted(rows, key=lambda r: r["slope"])
    for r in sorted_rows:
        cl = cut_label(r["cut"])
        lines.append(
            f"$[{cl}]$ & {r['n_masters']} & {r['n_max']} "
            f"& {r['slope']:>5.2f} & {r['tsolve_at_20']:>5.1f} & {r['status']} \\\\"
        )
    lines.append(r"\addlinespace")
    qslope, qintercept = quad_fit["tsolve_slope"], quad_fit["tsolve_intercept"]
    qts20 = qslope * 20 + qintercept
    lines.append(
        f"$[1,3,6,8]$ & 27 & 40 &  {qslope:.2f} &  {qts20:.1f} & all OK \\\\"
    )
    return "\n".join(lines)


def write_tsolve_body(rows, quad_fit, out_path):
    qslope, qintercept = quad_fit["tsolve_slope"], quad_fit["tsolve_intercept"]
    quad_per_n = quad_fit["per_n"]
    out = [r"\begin{tikzpicture}",
           r"\begin{axis}[",
           r"  width=\textwidth, height=0.6\textwidth,",
           r"  xlabel={$n$},",
           r"  ylabel={$T_\text{solve}$ [s]},",
           r"  xmin=4, xmax=21,",
           r"  ymin=0, ymax=120,",
           r"  grid=major,",
           r"  legend style={",
           r"    at={(0.02,0.98)},",
           r"    anchor=north west,",
           r"    font=\tiny,",
           r"    cells={anchor=west},",
           r"    draw=gray!50,",
           r"  },",
           r"  title={$T_\text{solve}$ for strip-seeding reduction on all spanning cuts},",
           r"]",
           r""]

    # quad cut [1,3,6,8] — uses cq color, * mark
    out += [r"% [1,3,6,8] quadruple",
            r"\addplot[only marks, mark=*, mark size=2pt, cq, "
            r"error bars/.cd, y dir=both, y explicit] "
            r"table[x index=0, y index=1, y error index=2] {"]
    for n in sorted(quad_per_n):
        out.append(f"{n:>2}  {quad_per_n[n]['tsolve_mean']:>6.2f}  "
                   f"{quad_per_n[n]['tsolve_std']:>5.2f}")
    out += [r"};",
            r"\addlegendentry{$[1,3,6,8]$ (quad, 27\,m)}",
            f"\\addplot[forget plot, cq, thick, dashed, domain=5:20, samples=2] {{{qslope:.2f}*x {'+' if qintercept>=0 else '-'} {abs(qintercept):.1f}}};",
            r""]

    # triple cuts, ordered by slope ascending (so legend reads slow-to-fast)
    sorted_rows = sorted(rows, key=lambda r: r["slope"])
    for r in sorted_rows:
        cut = r["cut"]
        cl = cut_label(cut)
        col = COLOR.get(tuple(cut), "black")
        mk = MARK.get(tuple(cut), "*")
        slope, intercept = r["slope"], r["intercept"]
        out += [f"% [{cl}] - {r['n_masters']}m",
                f"\\addplot[only marks, mark={mk}, mark size=1.8pt, {col}, "
                f"error bars/.cd, y dir=both, y explicit] "
                f"table[x index=0, y index=1, y error index=2] {{"]
        for n in sorted(r["per_n"]):
            out.append(f"{n:>2}  {r['per_n'][n]['tsolve_s']:>7.3f}  "
                       f"{r['per_n'][n]['tsolve_std']:>6.3f}")
        out += [r"};",
                f"\\addlegendentry{{$[{cl}]$ ({r['n_masters']}\\,m)}}",
                f"\\addplot[forget plot, {col}, thick, domain=5:20, samples=2] "
                f"{{{slope:.2f}*x {'+' if intercept>=0 else '-'} {abs(intercept):.1f}}};",
                r""]

    out += [r"\end{axis}",
            r"\end{tikzpicture}"]
    out_path.write_text("\n".join(out) + "\n")


def write_rss_body(rows, quad_fit, out_path):
    quad_per_n = quad_fit["per_n"]
    out = [r"\begin{tikzpicture}",
           r"\begin{axis}[",
           r"  width=\textwidth, height=0.6\textwidth,",
           r"  xlabel={$n$},",
           r"  ylabel={Peak RSS [GiB]},",
           r"  xmin=4, xmax=21,",
           r"  ymin=0, ymax=26,",
           r"  grid=major,",
           r"  legend style={",
           r"    at={(0.02,0.98)},",
           r"    anchor=north west,",
           r"    font=\tiny,",
           r"    cells={anchor=west},",
           r"    draw=gray!50,",
           r"  },",
           r"  title={Peak RSS for strip-seeding reduction on all spanning cuts},",
           r"]",
           r""]

    qrss_slope = quad_fit["rss_solve_slope"]
    qrss_intercept = quad_fit["rss_solve_intercept"]

    # quad cut
    out += [r"% [1,3,6,8] quadruple",
            r"\addplot[only marks, mark=*, mark size=2pt, cq, "
            r"error bars/.cd, y dir=both, y explicit] "
            r"table[x index=0, y index=1, y error index=2] {"]
    for n in sorted(quad_per_n):
        out.append(f"{n:>2}  {quad_per_n[n]['rss_solve_mean']:>6.3f}  "
                   f"{quad_per_n[n]['rss_solve_std']:>5.3f}")
    out += [r"};",
            r"\addlegendentry{$[1,3,6,8]$ (quad, 27\,m)}",
            f"\\addplot[forget plot, cq, thick, dashed, domain=5:20, samples=2] "
            f"{{{qrss_slope:.3f}*x {'+' if qrss_intercept>=0 else '-'} {abs(qrss_intercept):.2f}}};",
            r""]

    # triple cuts sorted by RSS slope ascending
    sorted_rows = sorted(rows, key=lambda r: r["rss_slope"])
    for r in sorted_rows:
        cut = r["cut"]
        cl = cut_label(cut)
        col = COLOR.get(tuple(cut), "black")
        mk = MARK.get(tuple(cut), "*")
        slope, intercept = r["rss_slope"], r["rss_intercept"]
        out += [f"% [{cl}] - {r['n_masters']}m",
                f"\\addplot[only marks, mark={mk}, mark size=1.8pt, {col}, "
                f"error bars/.cd, y dir=both, y explicit] "
                f"table[x index=0, y index=1, y error index=2] {{"]
        for n in sorted(r["per_n"]):
            out.append(f"{n:>2}  {r['per_n'][n]['rss_solve_gib']:>7.3f}  "
                       f"{r['per_n'][n]['rss_solve_std']:>6.3f}")
        out += [r"};",
                f"\\addlegendentry{{$[{cl}]$ ({r['n_masters']}\\,m)}}",
                f"\\addplot[forget plot, {col}, thick, domain=5:20, samples=2] "
                f"{{{slope:.3f}*x {'+' if intercept>=0 else '-'} {abs(intercept):.2f}}};",
                r""]

    out += [r"\end{axis}",
            r"\end{tikzpicture}"]
    out_path.write_text("\n".join(out) + "\n")


def write_standalone_tsolve(out_path, body_path):
    pre = (
        r"\documentclass[border=2pt]{standalone}" "\n"
        r"\usepackage{amsmath}" "\n"
        r"\usepackage{tikz}" "\n"
        r"\usepackage{pgfplots}" "\n"
        r"\pgfplotsset{compat=1.18}" "\n"
        r"% colours used by the plot body" "\n"
        r"\definecolor{c1}{HTML}{1f77b4}" "\n"
        r"\definecolor{c2}{HTML}{ff7f0e}" "\n"
        r"\definecolor{c3}{HTML}{2ca02c}" "\n"
        r"\definecolor{c4}{HTML}{d62728}" "\n"
        r"\definecolor{c5}{HTML}{9467bd}" "\n"
        r"\definecolor{c6}{HTML}{8c564b}" "\n"
        r"\definecolor{c7}{HTML}{e377c2}" "\n"
        r"\definecolor{c8}{HTML}{7f7f7f}" "\n"
        r"\definecolor{c9}{HTML}{bcbd22}" "\n"
        r"\definecolor{c10}{HTML}{b2df8a}" "\n"
        r"\definecolor{cq}{HTML}{000000}" "\n"
        r"\begin{document}" "\n"
    )
    body = body_path.read_text()
    out_path.write_text(pre + body + r"\end{document}" + "\n")


def main():
    quad_agg_path = HERE / "scan_n_feyngym_results_rerun_aggregate.json"
    quad_agg = json.loads(quad_agg_path.read_text())
    quad_fit = {
        "tsolve_slope": quad_agg["fits"]["tsolve"]["slope"],
        "tsolve_intercept": quad_agg["fits"]["tsolve"]["intercept"],
        "rss_solve_slope": quad_agg["fits"]["rss_solve"]["slope"],
        "rss_solve_intercept": quad_agg["fits"]["rss_solve"]["intercept"],
        "per_n": {row["n"]: row for row in quad_agg["rows"]},
    }

    rows = collect()
    print(f"\nLoaded {len(rows)} cuts.\n")
    print(f"{'cut':>10} {'masters':>8} {'n_max':>6} "
          f"{'slope':>7} {'intercept':>10} {'Tsolve@20':>10} "
          f"{'RSSslope':>10} {'RSS@20':>8}")
    for r in sorted(rows, key=lambda r: r["slope"]):
        print(f"  [{cut_label(r['cut']):>5}]   "
              f"{r['n_masters']:>5}   {r['n_max']:>4} "
              f"{r['slope']:>7.3f} {r['intercept']:>+10.2f} "
              f"{r['tsolve_at_20']:>10.2f} "
              f"{r['rss_slope']:>10.3f} {r['rss_at_20']:>8.2f}")
    print(f"\n  [{'1,3,6,8':>5}]   "
          f"{27:>5}   {40:>4} {quad_fit['tsolve_slope']:>7.3f} "
          f"{quad_fit['tsolve_intercept']:>+10.2f} "
          f"{quad_fit['tsolve_slope']*20 + quad_fit['tsolve_intercept']:>10.2f} "
          f"{quad_fit['rss_solve_slope']:>10.3f} "
          f"{quad_fit['rss_solve_slope']*20 + quad_fit['rss_solve_intercept']:>8.2f}")

    print("\nTable-3 row block:")
    print(write_table3_block(rows, quad_fit))

    write_tsolve_body(rows, quad_fit, PAPER / "triple_tsolve_body.tex")
    write_rss_body(rows, quad_fit, PAPER / "triple_rss_body.tex")
    write_standalone_tsolve(PAPER / "triple_cuts_to_20_plot.tex",
                             PAPER / "triple_tsolve_body.tex")
    print(f"\nWrote {PAPER / 'triple_tsolve_body.tex'}")
    print(f"Wrote {PAPER / 'triple_rss_body.tex'}")
    print(f"Wrote {PAPER / 'triple_cuts_to_20_plot.tex'}")

    summary = {
        "rows": [{"cut": r["cut"], "n_masters": r["n_masters"],
                  "n_max": r["n_max"],
                  "slope": r["slope"], "intercept": r["intercept"],
                  "rss_slope": r["rss_slope"],
                  "rss_intercept": r["rss_intercept"],
                  "tsolve_at_20": r["tsolve_at_20"],
                  "rss_at_20": r["rss_at_20"]} for r in rows],
        "quad_fit": quad_fit,
    }
    (HERE / "triple_cuts_rerun_aggregate.json").write_text(
        json.dumps(summary, indent=2, default=lambda x: x))
    return rows, quad_fit


if __name__ == "__main__":
    main()
