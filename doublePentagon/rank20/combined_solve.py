"""
combined_solve.py

Single configuration point for the two-pass Feynman integral reduction.
Imports the seed-generation logic from both pass1 (high_rank_one_cut_firstpass) and
pass2 (high_rank_one_cut_secondpass), generates equations from the union of
their seed sets, and solves the combined linear system in one go.

Usage:
    python combined_solve.py --cut 2,4,6
    python combined_solve.py --cut 2,4,7 --target 8,6,6
    python combined_solve.py --cut-number 4
"""

import argparse
import sys
import time

import pyfeyngym as pfg
import high_rank_one_cut_firstpass as pass1
import high_rank_one_cut_secondpass as pass2

# ---------------------------------------------------------------------------
#  Known spanning cuts (from high_rank.py)
# ---------------------------------------------------------------------------
spanning_cuts = [
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
    [1, 3, 6, 8],
]


# ===========================================================================
#  Parse command line
# ===========================================================================
def parse_cut(cut_arg):
    return [int(item.strip()) for item in cut_arg.split(",") if item.strip()]


def parse_target(target_arg):
    return tuple(int(item.strip()) for item in target_arg.split(","))


parser = argparse.ArgumentParser(
    description="Combined two-pass Feynman integral reduction")
parser.add_argument("--cut", type=parse_cut, default=[2, 4, 6],
                    help="Comma-separated cut indices, e.g. 2,4,7  (default: %(default)s)")
parser.add_argument("--cut-number", type=int, default=None,
                    help=("Override --cut with 1-based numbering into spanning_cuts "
                          "for positive values; negative values use Python indexing"))
parser.add_argument("--target", type=parse_target, default=(8, 6, 6),
                    help="Comma-separated ISP powers (l,m,n) for target integral "
                         "(1,...,1,-l,-m,-n)  (default: %(default)s)")
parser.add_argument("--no-pass2", action="store_true",
                    help="Skip the second seed set (use pass1 only)")
parser.add_argument("--statistics-only", action="store_true",
                    help="Print seed/equation counts without running the solver")
args = parser.parse_args()

# Resolve cut-number if provided
if args.cut_number is not None:
    if args.cut_number == 0:
        parser.error("--cut-number 0 is invalid; use positive 1-based values "
                     "or negative Python-style indices")
    cut_index = args.cut_number - 1 if args.cut_number > 0 else args.cut_number
    if not (-len(spanning_cuts) <= cut_index < len(spanning_cuts)):
        parser.error(
            f"--cut-number selects invalid spanning_cuts index {cut_index}; "
            f"valid indices are between {-len(spanning_cuts)} and {len(spanning_cuts) - 1}")
    print(f"--cut-number provided, using spanning_cuts[{cut_index}] = "
          f"{spanning_cuts[cut_index]}")
    args.cut = spanning_cuts[cut_index]

CHOSEN_CUT = args.cut
TARGET_L, TARGET_M, TARGET_N = args.target

# ---------------------------------------------------------------------------
# Initialise both modules with the chosen cut
# ---------------------------------------------------------------------------
pass1.init(CHOSEN_CUT)
pass2.init(CHOSEN_CUT)

# Shared state (both modules produce identical eq_templates / trivial_sectors
# for the same IBP_file, so we use pass1's copies).
eq_templates = pass1.eq_templates
trivial_sectors = pass1.trivial_sectors
masters = pass1.masters          # full top-sector master list
MODULUS = pass1.MODULUS
M_VALS = pass1.M_VALS

use_pass2 = (
    not args.no_pass2
    and pass2.improved_seeds is not None
    and len(pass2.improved_seeds) > 0
)

print(f"\n=== Combined Solve ===")
print(f"Cut: {CHOSEN_CUT}")
print(f"Target ISP powers (l,m,n): {TARGET_L, TARGET_M, TARGET_N}")
print(f"Number of masters (from pass1): {len(masters)}")
print(f"Pass2 available: {use_pass2}")

# ---------------------------------------------------------------------------
# Build seed sets from pass1 (always) and pass2 (if available)
# ---------------------------------------------------------------------------
print("\nBuilding seed sets ...")
t0 = time.time()

seed_set_1 = pass1.build_seed_set(pass1.improved_seeds,
                                  TARGET_L, TARGET_M, TARGET_N)
print(f"Pass1 seed_set size: {len(seed_set_1)}  ({time.time() - t0:.1f}s)")

if use_pass2:
    t1 = time.time()
    seed_set_2 = pass2.build_seed_set(pass2.improved_seeds,
                                      TARGET_L - 3, TARGET_M, TARGET_N - 3)
    print(f"Pass2 seed_set size: {len(seed_set_2)}  ({time.time() - t1:.1f}s)")
    combined_seed_set = seed_set_1 | seed_set_2
else:
    seed_set_2 = set()
    combined_seed_set = seed_set_1
print(f"Combined seed_set size: {len(combined_seed_set)}")

# ---------------------------------------------------------------------------
# Statistics: per-zigzag and combined equation counts
# ---------------------------------------------------------------------------
def _count_eqs(seed_set, label):
    seeds = pfg.sort_integrals_desc(list(seed_set))
    _, all_vars = pfg.gen_eqs(eq_templates, trivial_sectors, M_VALS, seeds)
    neq = 18 * len(seeds)  # 18 IBP+LI operators per seed
    print(f"{label}: {len(seed_set)} seeds, {neq} equations, "
          f"{len(all_vars)} variables")

_count_eqs(seed_set_1, "Zig-zag 1 (top sector)")
if use_pass2 and len(seed_set_2) > 0:
    _count_eqs(seed_set_2, "Zig-zag 2 (subsector)")
_count_eqs(combined_seed_set, "Combined")

if args.statistics_only:
    print("\n(Statistics only; solver skipped.)")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Generate equations and solve
# ---------------------------------------------------------------------------
print("\nGenerating equations ...")
t0 = time.time()

final_seeds = pfg.sort_integrals_desc(list(combined_seed_set))
seed_op_eq_list, all_variables = pfg.gen_eqs(
    eq_templates, trivial_sectors, M_VALS, final_seeds)
equations = [a[-1] for a in seed_op_eq_list]
sorted_vars = pfg.sort_integrals_desc(all_variables)

print(f"Time generating equations: {time.time() - t0:.1f}s")
print(f"Number of equations in combined system: {len(equations)}")
print(f"Number of variables: {len(sorted_vars)}")

# ---------------------------------------------------------------------------
# Solve the combined system
# ---------------------------------------------------------------------------
target_integral = (1, 1, 1, 1, 1, 1, 1, 1,
                   -TARGET_L, -TARGET_M, -TARGET_N)
print(f"target_integral: {target_integral}")

print("\nSolving equations ...")
t0 = time.time()

solution = pfg.solve_eqs_modulo(
    equations, sorted_vars, MODULUS,
    keep_on_rhs=masters,
    complete_pivoting=True,
    needed_variables={target_integral},
)

print(f"Time solving: {time.time() - t0:.1f}s")

reduced = solution[target_integral]
print(f"Reduced target integral: {reduced}")
print(f"Number of terms in reduced: {len(reduced)}")

integrals_in_reduced = [a[0] for a in reduced]
non_masters = set(integrals_in_reduced) - set(masters)

print(f"Non-master terms in reduced result: {non_masters}")

if non_masters:
    print("\n*** WARNING: Combined solve still has non-master terms! ***")
    sys.exit(1)
else:
    print("\n*** SUCCESS: Combined solve fully reduced to master integrals. ***")
    sys.exit(0)
