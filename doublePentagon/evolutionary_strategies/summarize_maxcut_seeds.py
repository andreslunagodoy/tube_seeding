#!/usr/bin/env python3
"""Summarize the seed integrals chosen by the maximal-cut CMA-ES run.

Reads the ``(seed, operator_number)`` list written by ``optimize-ordering-NN``
(see ``run_cma_es.sh``) and reports the distinct seed integrals that were used to
reduce the single-ISP target on the maximal cut. On the maximal cut the eight
propagator indices are all ``1``, so each seed is written in the compact form
``(1,...,1, a9, a10, a11)``; the three irreducible-scalar-product depths
``(x, y, z) = (-a9, -a10, -a11)`` are what vary across the tube.

By default the script reads ``maxhigh/resorted_seed_op_list.txt`` located next to
this file, so it can be run with no arguments (as in ``run_cma_es.sh``).
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

DEFAULT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "maxhigh", "resorted_seed_op_list.txt",
)


def load_seed_ops(path, n_indices=11):
    """Return a list of ``(seed_tuple, operator_number)`` pairs."""
    pairs = []
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
            pairs.append((tuple(vals[:n_indices]), vals[n_indices]))
    return pairs


def seed_label(seed, n_props=8):
    """``(1,...,1, a9, a10, a11)`` if all propagator indices are 1, else the tuple."""
    isps = seed[n_props:]
    if all(p == 1 for p in seed[:n_props]):
        return "(1,...,1," + ",".join(str(a) for a in isps) + ")"
    return "(" + ",".join(str(a) for a in seed) + ")"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--seed-op-file", default=DEFAULT_FILE,
        help=f"(seed, operator_number) list to summarize. Default: {DEFAULT_FILE}",
    )
    ap.add_argument("--n-indices", type=int, default=11,
                    help="number of integral indices per seed (default: 11)")
    args = ap.parse_args()

    pairs = load_seed_ops(args.seed_op_file, args.n_indices)
    ops_per_seed = Counter(seed for seed, _ in pairs)
    distinct = sorted(ops_per_seed)

    print("Summary of seeds (1,...,1, -x, -y, -z) used")
    print(f"  file:                  {args.seed_op_file}")
    print(f"  (seed, operator) rows: {len(pairs)}")
    print(f"  distinct seeds:        {len(distinct)}")

    if distinct:
        depths = [(-s[-3], -s[-2], -s[-1]) for s in distinct]
        xs, ys, zs = zip(*depths)
        print(f"  ISP reach (max x,y,z): {max(xs)}, {max(ys)}, {max(zs)}")
        print()
        print(f"  {'seed':<30} #operators")
        for seed in distinct:
            print(f"  {seed_label(seed):<30} {ops_per_seed[seed]}")


if __name__ == "__main__":
    main()
