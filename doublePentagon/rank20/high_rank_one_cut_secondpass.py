import pyfeyngym as pfg

# ---------------------------------------------------------------------------
# Module-level constants (cut-independent)
# ---------------------------------------------------------------------------
IBP_file = "IBP_LI"
trivial_sector_file = "trivialsector"
MODULUS = 2**31 - 1
M_VALS = {'d': 23, 'm1': 3, 'm2': 5, 'm3': 17, 'm4': 23}

# These are set by init() — do not use before calling init()
eq_templates = None
trivial_sectors = None
improved_seeds = None
masters = None
CHOSEN_CUT = None


def init(chosen_cut):
    """Initialise all cut-dependent module state.
    Must be called once before accessing eq_templates, trivial_sectors,
    improved_seeds, masters, or build_seed_set."""
    global CHOSEN_CUT, eq_templates, trivial_sectors, improved_seeds, masters
    CHOSEN_CUT = list(chosen_cut)

    eq_templates = pfg.gen_eq_templates(IBP_file, M_VALS)
    print("[pass2] Number of IBP+LI operators:", len(eq_templates))

    trivial_sectors = pfg.get_trivial_sectors(
        trivial_sector_file, cut=CHOSEN_CUT, n_indices=11)

    top_sector = (1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0)
    s_max, r_max, d_max = 4, 8, 0

    starting_seeds = pfg.gen_all_seeds(top_sector, trivial_sectors,
                                       s_max, r_max, d_max)

    improved_seed_param = 2
    improved_seeds = [s for s in starting_seeds if
                      (pfg.d_level(s) <= 0 and
                       pfg.s_level(s) <= max(1, pfg.t_level(s) - improved_seed_param))]
    print("[pass2] Number of improved seeds:", len(improved_seeds))

    if len(improved_seeds) == 0:
        print("[pass2] Sub-sector has no seeds for this cut — skipping pass2.")
        masters = []
        return

    seed_op_eq_list, all_variables = pfg.gen_eqs(
        eq_templates, trivial_sectors, M_VALS, improved_seeds)
    equations = [a[-1] for a in seed_op_eq_list]
    sorted_vars = pfg.sort_integrals_desc(all_variables)
    test_integral = (1, 1, 0, 1, 0, 1, 1, 0, -1, -2, 0)
    solution = pfg.solve_eqs_modulo(equations, sorted_vars, MODULUS,
                                    needed_variables={test_integral})

    reduced = solution[test_integral]
    masters = [term[0] for term in reduced]
    print("[pass2] Number of masters:", len(masters))
    print("[pass2] masters: ", masters)


# Reduction of (1, ... , 1, -l, -m, -n)

def build_seed_set(improved_seeds, l, m, n):
    """Build a zig-zag strip seed set for the subsector.

    Parameters
    ----------
    l, m, n : int
        Positive powers of the three ISPs (the actual integral has -l, -m, -n).
    """
    seed_set = set()

    # Helper: only keep seeds where all ISP powers (last 3 indices) are <= 0.
    def _add(integral_tuple):
        x, y, z = integral_tuple[-3:]
        if x <= 0 and y <= 0 and z <= 0:
            seed_set.add(integral_tuple)

    # Strip 1: (0, 0, 0) to (l-3, 0, 0)
    for raise_rank in range(l-2):
        for seed in improved_seeds:
            last_three_indices = (seed[-3]-raise_rank, seed[-2], seed[-1])
            _add(seed[:-3] + last_three_indices)

    # Strip 2: (l-3, 1, 0) to (l-3, m, 0)
    for raise_rank in range(1, m+1):
        for seed in improved_seeds:
            last_three_indices = (seed[-3] - (l-3), seed[-2] - raise_rank, seed[-1])
            _add(seed[:-3] + last_three_indices)

    # Strip 3: (l-3, m-1 to m, 1) to (l-3, m-1 to m, n)
    for raise_rank in range(1, n+1):
        for seed in improved_seeds:
            last_three_indices = (seed[-3] - (l-3), seed[-2] - m, seed[-1] - raise_rank)
            _add(seed[:-3] + last_three_indices)
            last_three_indices = (seed[-3] - (l-3), seed[-2] - (m-1), seed[-1] - raise_rank)
            _add(seed[:-3] + last_three_indices)
    return seed_set


def _main(l=5, m=6, n=6):
    """Standalone solve using this module's seed set alone."""
    seed_set = build_seed_set(improved_seeds, l, m, n)
    print("Number of seeds in seed_set:", len(seed_set))

    final_seeds = pfg.sort_integrals_desc(list(seed_set))
    seed_op_eq_list, all_variables = pfg.gen_eqs(
        eq_templates, trivial_sectors, M_VALS, final_seeds)
    equations = [a[-1] for a in seed_op_eq_list]
    sorted_vars = pfg.sort_integrals_desc(all_variables)

    target_integral = (1, 1, 0, 1, 0, 1, 1, 0, -l, -m, -n)
    print("target_integral: ", target_integral)
    print("Number of equations:", len(equations))
    solution = pfg.solve_eqs_modulo(equations, sorted_vars, MODULUS,
                                    keep_on_rhs=masters,
                                    complete_pivoting=True,
                                    needed_variables={target_integral})

    reduced = solution[target_integral]
    print("Reduced target integral:", reduced)
    print("Number of terms in reduced:", len(reduced))

    integrals_in_reduced = [a[0] for a in reduced]
    print("Non-master terms in reduced result:",
          set(integrals_in_reduced) - set(masters))


if __name__ == "__main__":
    init([2, 4, 6])
    _main(5, 6, 6)
