"""
Aggregate the per-batch rerun JSONs (scan_n_feyngym_results_rerun_b{1..6}.json)
into a unified table, fit OLS lines, and produce updated paper-source files
for Table 1 and Figure 9 of stripseeding.tex.

Outputs are written to a local ``paper_tables/`` directory next to this script
(the LaTeX table/figure fragments that were originally inlined into the paper):
  - strip_to_40_feyngym.tex       (standalone full benchmark table)
  - strip_to_40_feyngym_plot.tex  (standalone scaling figure)
  - strip_plot_body.tex           (the figure body fragment)
"""
import json
import math
import os
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = Path(__file__).resolve().parent / "paper_tables"
_env_inputs = os.environ.get("INPUT_JSONS")
if _env_inputs:
    BATCHES = [HERE / p for p in _env_inputs.split(",")]
else:
    BATCHES = [HERE / f"scan_n_feyngym_results_rerun_b{i}.json"
               for i in range(1, 7)]


def load_records():
    records = {}
    for f in BATCHES:
        if not f.exists():
            raise SystemExit(f"missing {f}")
        for entry in json.loads(f.read_text()):
            n = entry["n"]
            ok = [r for r in entry["runs"]
                  if not r.get("timed_out") and r.get("exit_code") == 0
                  and r.get("reduced_ok")]
            if not ok:
                raise SystemExit(f"no successful run at n={n}")
            records[n] = ok
    return records


def stat(vals):
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def ols(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den
    intercept = my - slope * mx
    return slope, intercept


def fmt_uncert(value, sigma, digits=1):
    """siunitx-style 'value(uncert)' where uncert is in last `digits` of value."""
    if sigma is None or sigma == 0:
        return f"{value:.{digits}f}"
    fac = 10 ** digits
    u_int = int(round(sigma * fac))
    if u_int == 0:
        u_int = 1
    return f"{value:.{digits}f}({u_int})"


def round_half_up(x):
    return int(math.floor(x + 0.5))


def aggregate(records):
    rows = []
    for n in sorted(records):
        ok = records[n]
        tgen = [r["tgen_s"] for r in ok if "tgen_s" in r]
        tsolve = [r["tsolve_s"] for r in ok if "tsolve_s" in r]
        twall = [r["twall_s"] for r in ok if "twall_s" in r]
        rss_gen_gib = [r["peak_rss_gen_mib"] / 1024 for r in ok
                       if "peak_rss_gen_mib" in r]
        rss_solve_gib = [r["peak_rss_solve_mib"] / 1024 for r in ok
                         if "peak_rss_solve_mib" in r]
        seeds = ok[0]["n_seeds"]
        n_eqs = ok[0]["n_eqs"]
        n_vars = ok[0]["n_vars"]
        rows.append({
            "n": n,
            "seeds": seeds,
            "n_eqs": n_eqs,
            "n_vars": n_vars,
            "tgen_mean": stat(tgen)[0], "tgen_std": stat(tgen)[1],
            "tsolve_mean": stat(tsolve)[0], "tsolve_std": stat(tsolve)[1],
            "twall_mean": stat(twall)[0], "twall_std": stat(twall)[1],
            "rss_gen_mean": stat(rss_gen_gib)[0],
            "rss_gen_std": stat(rss_gen_gib)[1],
            "rss_solve_mean": stat(rss_solve_gib)[0],
            "rss_solve_std": stat(rss_solve_gib)[1],
            "n_runs": len(ok),
        })
    return rows


def write_full_table(rows, fits, out_path):
    """Standalone strip_to_40_feyngym.tex."""
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage{amsmath}",
        r"\usepackage{booktabs}",
        r"\usepackage[margin=2cm]{geometry}",
        r"\usepackage{siunitx}",
        r"",
        r"\sisetup{",
        r"  separate-uncertainty = true,",
        r"  multi-part-units = single,",
        r"}",
        r"",
        r"\begin{document}",
        r"",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Tube-seeding reduction of",
        r"  $\texttt{doublePentagon}[1,\ldots,1,0,0,-n]$ on the quadruple cut",
        r"  $\{1,3,6,8\}$ using \texttt{pyfeyngym}, averaged over 5~independent",
        r"  runs per~$n$.  All runs produce 27~master integrals.}",
        r"\label{tab:strip-feyngym}",
        r"\small",
        r"\begin{tabular}{",
        r"  r r r r",
        r"  S[table-format=2.1(2)]",
        r"  S[table-format=3.1(2)]",
        r"  S[table-format=3.1(2)]",
        r"  S[table-format=1.3(3)]",
        r"  S[table-format=2.3(3)]",
        r"}",
        r"\toprule",
        r"{$n$} & {seeds} & {\#eqs} & {\#vars}",
        r"  & {$T_\text{gen}$\,[s]}",
        r"  & {$T_\text{solve}$\,[s]}",
        r"  & {$T_\text{wall}$\,[s]}",
        r"  & {RSS$_\text{gen}$\,[GiB]}",
        r"  & {RSS$_\text{solve}$\,[GiB]} \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['n']:>2} & {r['seeds']:>5} & {r['n_eqs']:>6} & {r['n_vars']:>6}"
            f" & {fmt_uncert(r['tgen_mean'], r['tgen_std'], 1):>9}"
            f" & {fmt_uncert(r['tsolve_mean'], r['tsolve_std'], 1):>10}"
            f" & {fmt_uncert(r['twall_mean'], r['twall_std'], 1):>10}"
            f" & {fmt_uncert(r['rss_gen_mean'], r['rss_gen_std'], 3):>12}"
            f" & {fmt_uncert(r['rss_solve_mean'], r['rss_solve_std'], 3):>14} \\\\")
    tg_s, tg_i = fits["tgen"]
    ts_s, ts_i = fits["tsolve"]
    rg_s, rg_i = fits["rss_gen"]
    rs_s, rs_i = fits["rss_solve"]
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\medskip",
        r"\noindent",
        r"Unweighted linear fits (ordinary least squares on the 5-run means):",
        r"\begin{align*}",
        f"  T_\\text{{gen}}(n)            &= {tg_s:.3f}\\,n {tg_i:+.1f} \\;\\text{{s}}   \\\\",
        f"  T_\\text{{solve}}(n)          &= {ts_s:.3f}\\,n {ts_i:+.1f} \\;\\text{{s}}  \\\\",
        f"  \\text{{RSS}}_\\text{{gen}}(n)   &= {rg_s*1024:.1f}\\,n {rg_i*1024:+.0f} \\;\\text{{MiB}}",
        f"                              = {rg_s:.4f}\\,n {rg_i:+.3f} \\;\\text{{GiB}} \\\\",
        f"  \\text{{RSS}}_\\text{{solve}}(n) &= {rs_s*1024:.0f}\\,n {rs_i*1024:+.0f} \\;\\text{{MiB}}",
        f"                              = {rs_s:.3f}\\,n {rs_i:+.2f} \\;\\text{{GiB}}",
        r"\end{align*}",
        r"\end{table}",
        r"",
        r"\end{document}",
        r"",
    ]
    out_path.write_text("\n".join(lines))


def make_plot_body(rows, fits):
    """Produce the inline pgfplots body used in both strip_plot_body.tex
    and the inline copy embedded in stripseeding.tex / sections_3_4.tex."""
    tg_s, tg_i = fits["tgen"]
    ts_s, ts_i = fits["tsolve"]
    rg_s, rg_i = fits["rss_gen"]
    rs_s, rs_i = fits["rss_solve"]

    def fit_label(slope, intercept, prec_s, prec_i):
        sgn = "+" if intercept >= 0 else "-"
        return f"{slope:.{prec_s}f}\\,n {sgn} {abs(intercept):.{prec_i}f}"

    out = [r"\begin{tikzpicture}",
           r"\pgfplotsset{",
           r"  every axis/.style={",
           r"    width=0.9\textwidth, height=0.58\textwidth,",
           r"    xmin=4, xmax=41,",
           r"    grid=major,",
           r"  },",
           r"}",
           r"",
           r"% Left axis: time [s]",
           r"\begin{axis}[",
           r"  axis y line*=left,",
           r"  xlabel={$n$},",
           r"  ylabel={Time [s]},",
           r"  ymin=-10, ymax=260,",
           r"  ylabel style={color=black},",
           r"  legend style={",
           r"    at={(0.03,0.60)},",
           r"    anchor=west,",
           r"    font=\footnotesize,",
           r"    cells={anchor=west},",
           r"  },",
           r"  title={Tube-seeding reduction via \texttt{pyfeyngym} (10 runs per $n$)},",
           r"]",
           r"",
           r"% T_gen data",
           r"\addplot[",
           r"  only marks,",
           r"  mark=square*,",
           r"  mark size=1.5pt,",
           r"  color=blue!80!black,",
           r"  error bars/.cd, y dir=both, y explicit,",
           r"] table [x index=0, y index=1, y error index=2] {"]
    for r in rows:
        out.append(f"{r['n']:>2} {r['tgen_mean']:>6.2f}  {r['tgen_std']:.2f}")
    out += [r"};",
            r"\addlegendentry{$T_\text{gen}$ data}",
            r"",
            r"% T_gen fit (unweighted OLS)",
            f"\\addplot[blue!80!black, thick, domain=5:40, samples=2] {{{tg_s:.3f}*x {'+' if tg_i>=0 else '-'} {abs(tg_i):.1f}}};",
            f"\\addlegendentry{{$T_\\text{{gen}} = {fit_label(tg_s, tg_i, 2, 1)}$\\,s}}",
            r"",
            r"% T_solve data",
            r"\addplot[",
            r"  only marks,",
            r"  mark=*,",
            r"  mark size=1.5pt,",
            r"  color=red!80!black,",
            r"  error bars/.cd, y dir=both, y explicit,",
            r"] table [x index=0, y index=1, y error index=2] {"]
    for r in rows:
        out.append(f"{r['n']:>2}  {r['tsolve_mean']:>7.2f}  {r['tsolve_std']:.2f}")
    out += [r"};",
            r"\addlegendentry{$T_\text{solve}$ data}",
            r"",
            r"% T_solve fit (unweighted OLS)",
            f"\\addplot[red!80!black, thick, domain=5:40, samples=2] {{{ts_s:.3f}*x {'+' if ts_i>=0 else '-'} {abs(ts_i):.1f}}};",
            f"\\addlegendentry{{$T_\\text{{solve}} = {fit_label(ts_s, ts_i, 2, 1)}$\\,s}}",
            r"",
            r"\end{axis}",
            r"",
            r"% Right axis: RSS [GiB]",
            r"\begin{axis}[",
            r"  axis y line*=right,",
            r"  axis x line=none,",
            r"  ylabel={Peak RSS [GiB]},",
            r"  ymin=-0.7, ymax=18.5,",
            r"  ylabel style={color=black},",
            r"  legend style={",
            r"    at={(0.03,0.97)},",
            r"    anchor=north west,",
            r"    font=\footnotesize,",
            r"    cells={anchor=west},",
            r"  },",
            r"]",
            r"",
            r"% RSS_gen data (GiB)",
            r"\addplot[",
            r"  only marks,",
            r"  mark=triangle*,",
            r"  mark size=2pt,",
            r"  color=green!60!black,",
            r"  error bars/.cd, y dir=both, y explicit,",
            r"] table [x index=0, y index=1, y error index=2] {"]
    for r in rows:
        out.append(f"{r['n']:>2}  {r['rss_gen_mean']:>5.3f}  {r['rss_gen_std']:.3f}")
    out += [r"};",
            r"\addlegendentry{RSS$_\text{gen}$ data}",
            r"",
            f"% RSS_gen fit (unweighted OLS): {rg_s:.4f}*n {'+' if rg_i>=0 else '-'} {abs(rg_i):.3f} GiB",
            f"\\addplot[green!60!black, thick, domain=5:40, samples=2] {{{rg_s:.4f}*x {'+' if rg_i>=0 else '-'} {abs(rg_i):.3f}}};",
            f"\\addlegendentry{{RSS$_\\text{{gen}} = {fit_label(rg_s, rg_i, 3, 2)}$\\,GiB}}",
            r"",
            r"% RSS_solve data (GiB)",
            r"\addplot[",
            r"  only marks,",
            r"  mark=diamond*,",
            r"  mark size=2pt,",
            r"  color=orange!80!black,",
            r"  error bars/.cd, y dir=both, y explicit,",
            r"] table [x index=0, y index=1, y error index=2] {"]
    for r in rows:
        out.append(f"{r['n']:>2}  {r['rss_solve_mean']:>6.3f}  {r['rss_solve_std']:.3f}")
    out += [r"};",
            r"\addlegendentry{RSS$_\text{solve}$ data}",
            r"",
            f"% RSS_solve fit (unweighted OLS): {rs_s:.4f}*n {'+' if rs_i>=0 else '-'} {abs(rs_i):.3f} GiB",
            f"\\addplot[orange!80!black, thick, domain=5:40, samples=2] {{{rs_s:.4f}*x {'+' if rs_i>=0 else '-'} {abs(rs_i):.4f}}};",
            f"\\addlegendentry{{RSS$_\\text{{solve}} = {fit_label(rs_s, rs_i, 2, 1)}$\\,GiB}}",
            r"",
            r"\end{axis}",
            r"\end{tikzpicture}"]
    return "\n".join(out) + "\n"


def write_plot_body(body, out_path):
    out_path.write_text(body)


def write_standalone_plot(body, fits, out_path):
    """Standalone strip_to_40_feyngym_plot.tex."""
    pre = (
        r"\documentclass[border=2pt]{standalone}" "\n"
        r"\usepackage{amsmath}" "\n"
        r"\usepackage{tikz}" "\n"
        r"\usepackage{pgfplots}" "\n"
        r"\pgfplotsset{compat=1.18}" "\n"
        r"\begin{document}" "\n"
    )
    label = (
        r"% standalone plot for Figure 9 (fig:strip-feyngym)" "\n"
    )
    out_path.write_text(pre + label + body + r"\end{document}" + "\n")


def main():
    records = load_records()
    rows = aggregate(records)
    ns = [r["n"] for r in rows]
    if ns != list(range(5, 41)):
        print(f"WARNING: expected n=5..40, got {ns}")

    fits = {
        "tgen": ols([r["n"] for r in rows], [r["tgen_mean"] for r in rows]),
        "tsolve": ols([r["n"] for r in rows], [r["tsolve_mean"] for r in rows]),
        "rss_gen": ols([r["n"] for r in rows], [r["rss_gen_mean"] for r in rows]),
        "rss_solve": ols([r["n"] for r in rows], [r["rss_solve_mean"] for r in rows]),
    }
    print("OLS fits:")
    for k, (s, i) in fits.items():
        print(f"  {k:9s}: slope={s:.4f}  intercept={i:+.3f}")
    print()
    print(f"{'n':>3} {'seeds':>5} {'#eqs':>6} {'#vars':>6} "
          f"{'Tgen':>10} {'Tsolve':>11} {'wall':>11} "
          f"{'RSSgen':>13} {'RSSsolve':>14}")
    for r in rows:
        print(f"{r['n']:>3} {r['seeds']:>5} {r['n_eqs']:>6} {r['n_vars']:>6} "
              f"{fmt_uncert(r['tgen_mean'], r['tgen_std'], 1):>10} "
              f"{fmt_uncert(r['tsolve_mean'], r['tsolve_std'], 1):>11} "
              f"{fmt_uncert(r['twall_mean'], r['twall_std'], 1):>11} "
              f"{fmt_uncert(r['rss_gen_mean'], r['rss_gen_std'], 3):>13} "
              f"{fmt_uncert(r['rss_solve_mean'], r['rss_solve_std'], 3):>14}")

    write_full_table(rows, fits, PAPER / "strip_to_40_feyngym.tex")
    body = make_plot_body(rows, fits)
    write_plot_body(body, PAPER / "strip_plot_body.tex")
    write_standalone_plot(body, fits, PAPER / "strip_to_40_feyngym_plot.tex")

    # save aggregate JSON for reproducibility
    summary = {"rows": rows, "fits": {k: {"slope": s, "intercept": i}
                                       for k, (s, i) in fits.items()}}
    (HERE / "scan_n_feyngym_results_rerun_aggregate.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nWrote {PAPER / 'strip_to_40_feyngym.tex'}")
    print(f"Wrote {PAPER / 'strip_plot_body.tex'}")
    print(f"Wrote {PAPER / 'strip_to_40_feyngym_plot.tex'}")
    print(f"Wrote {HERE / 'scan_n_feyngym_results_rerun_aggregate.json'}")
    return rows, fits


if __name__ == "__main__":
    main()
