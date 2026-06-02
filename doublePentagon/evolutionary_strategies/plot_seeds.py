#!/usr/bin/env python3
"""Plot the maximal-cut seed integrals chosen by the CMA-ES run.

Reads the ``(seed, operator_number)`` list written by ``optimize-ordering-NN``
(see ``run_cma_es.sh``), collects the distinct seed integrals, and draws a 3-D
scatter of them projected onto the three irreducible-scalar-product (ISP) indices
``(a9, a10, a11)``, shown as positive depths ``(-a9, -a10, -a11)``. This is the
tube visualization discussed in Sec. 4.2.1 of the paper.

The script needs only ``numpy`` and ``matplotlib`` (no Julia / pyfeyngym), so it
can be run on the committed ``maxhigh/resorted_seed_op_list.txt`` without the
solver environment. Example (as in ``run_cma_es.sh``):

    python plot_seeds.py --seed-op-file maxhigh/resorted_seed_op_list.txt \\
        --export-figure maxhigh/seeds.png \\
        --title "Maximal-cut seeds for reducing (1,...,1,0,0,-12)"
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402  (registers the 3d projection)


def load_distinct_seeds(path, n_indices=11):
    seeds = set()
    with open(path) as fh:
        for line_no, raw in enumerate(fh, start=1):
            parts = raw.split()
            if not parts:
                continue
            vals = list(map(int, parts))
            if len(vals) < n_indices + 1:
                raise ValueError(
                    f"{path}, line {line_no}: expected at least {n_indices + 1} "
                    f"integers (a seed plus an operator number), got {len(vals)}"
                )
            seeds.add(tuple(vals[:n_indices]))
    return sorted(seeds)


def parse_target(text):
    if text is None:
        return None
    return tuple(int(x) for x in text.split(","))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed-op-file", required=True,
                    help="(seed, operator_number) list to plot")
    ap.add_argument("--export-figure", default=None,
                    help="output image path; if omitted, defaults to seeds.png "
                         "next to the seed-op file")
    ap.add_argument("--title", default="Seed integrals")
    ap.add_argument("--target", default=None,
                    help="optional target integral to mark with a star, e.g. "
                         "1,1,1,1,1,1,1,1,0,0,-12")
    ap.add_argument("--n-indices", type=int, default=11)
    ap.add_argument("--elev", type=float, default=30.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    seeds = load_distinct_seeds(args.seed_op_file, args.n_indices)
    if not seeds:
        raise SystemExit(f"no seeds found in {args.seed_op_file}")

    pts = np.array([(-s[-3], -s[-2], -s[-1]) for s in seeds], dtype=float)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c=z, cmap="viridis", s=22, alpha=0.85, edgecolors="none")
    ax.set_xlabel(r"$-a_9$")
    ax.set_ylabel(r"$-a_{10}$")
    ax.set_zlabel(r"$-a_{11}$")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_title(args.title, fontsize=10)

    target = parse_target(args.target)
    if target is not None:
        tx, ty, tz = -target[-3], -target[-2], -target[-1]
        ax.scatter([tx], [ty], [tz], c="red", marker="*", s=160,
                   depthshade=False, label="target")
        ax.legend(loc="upper left", fontsize=8)

    out = args.export_figure
    if out is None:
        out = os.path.join(os.path.dirname(os.path.abspath(args.seed_op_file)),
                           "seeds.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({len(seeds)} distinct seeds)")


if __name__ == "__main__":
    main()
