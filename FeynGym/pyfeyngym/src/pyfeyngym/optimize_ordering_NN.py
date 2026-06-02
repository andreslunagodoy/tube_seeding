# Using an "eager" IBP solver with a fixed ordering of equations and variables,
# recording the IBP reduction cost in terms of the number of arithmetic operations

import pyfeyngym as pfg
import torch
import torch.nn as nn
import time
import numpy as np
import cma
import os
import pickle
import argparse
import tomllib


class IntegralScoringNetwork(nn.Module):
    def __init__(self, input_dim: int = 17):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                input_dim, 16
            ),  # input_dim = n_indices + (n_ibp_operators+1); or n_indices when --optimize-seeds-only
            nn.GELU(),
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


def encode_integral_operator(
    integral,
    operator_number,
    optimize_seeds_only: bool = False,
    n_ibp_operators: int = None,
):
    """Convert (integral, operator_number) to network input tensor.
    - Default: concatenate len(integral) with a (n_ibp_operators+1)-bit one-hot; index 0 is reserved.
    - With optimize_seeds_only: return only the indices (no operator encoding).
    """
    integral_tensor = torch.tensor(integral, dtype=torch.float32)
    if optimize_seeds_only:
        return integral_tensor
    assert n_ibp_operators is not None, "n_ibp_operators must be provided"
    onehot = torch.zeros(n_ibp_operators + 1, dtype=torch.float32)
    onehot[int(operator_number)] = 1.0
    return torch.cat([integral_tensor, onehot])


def encode_integral_with_special_bit(
    integral, optimize_seeds_only: bool = False, n_ibp_operators: int = None
):
    """Encode an integral with the special one-hot bit (index 0) active in the operator channel.
    - Default: len(integral) + (n_ibp_operators+1) dims, bit 0 set to 1.0.
    - With optimize_seeds_only: not used; returns only the indices if ever called.
    """
    integral_tensor = torch.tensor(integral, dtype=torch.float32)
    if optimize_seeds_only:
        return integral_tensor
    assert n_ibp_operators is not None, "n_ibp_operators must be provided"
    onehot = torch.zeros(n_ibp_operators + 1, dtype=torch.float32)
    onehot[0] = 1.0
    return torch.cat([integral_tensor, onehot])


def reorder_seed_op_eq_list(
    seed_op_eq_list,
    network,
    optimize_seeds_only: bool = False,
    n_ibp_operators: int = None,
):
    """Reorder seed_op_eq_list using neural network scores."""
    network.eval()
    with torch.no_grad():
        inputs = torch.stack(
            [
                encode_integral_operator(
                    integral,
                    operator_number,
                    optimize_seeds_only=optimize_seeds_only,
                    n_ibp_operators=n_ibp_operators,
                )
                for integral, operator_number, _ in seed_op_eq_list
            ]
        )
        scores = network(inputs).squeeze(1)
        sorted_indices = torch.argsort(scores, descending=True).tolist()
        return [seed_op_eq_list[i] for i in sorted_indices]


def reorder_vars(
    vars_list, network, optimize_seeds_only: bool = False, n_ibp_operators: int = None
):
    """Reorder variables (tuples of variable length).
    - With optimize_seeds_only: use pfg.sort_integrals_desc (no NN).
    - Otherwise: score with NN via encode_integral_with_special_bit.
    """
    if optimize_seeds_only:
        return pfg.sort_integrals_desc(vars_list)
    network.eval()
    with torch.no_grad():
        inputs = torch.stack(
            [
                encode_integral_with_special_bit(
                    v, optimize_seeds_only=False, n_ibp_operators=n_ibp_operators
                )
                for v in vars_list
            ]
        )
        scores = network(inputs).squeeze(1)
        sorted_indices = torch.argsort(scores, descending=True).tolist()
        return [vars_list[i] for i in sorted_indices]


def evaluate_cost(
    network,
    seed_op_eq_list,
    vars_sorted_nonmasters,
    vars_sorted_masters,
    target_integral,
    masters,
    normalization_factor=1e6,
    modulus=2**31 - 1,
    cost_cutoff=-1,
    optimize_seeds_only: bool = False,
    freeze_variable_ordering: bool = False,
    return_n_eqs: bool = False,
    n_ibp_operators: int = None,
):
    """Evaluate the IBP reduction cost with the current network ordering."""
    t0 = time.perf_counter()
    seed_op_eq_list = reorder_seed_op_eq_list(
        seed_op_eq_list,
        network,
        optimize_seeds_only=optimize_seeds_only,
        n_ibp_operators=n_ibp_operators,
    )
    equations = [a[2] for a in seed_op_eq_list]
    vars_resorted_nonmasters = reorder_vars(
        vars_sorted_nonmasters,
        network,
        optimize_seeds_only=(optimize_seeds_only or freeze_variable_ordering),
        n_ibp_operators=n_ibp_operators,
    )
    vars_resorted = vars_resorted_nonmasters + vars_sorted_masters
    reduction_complete, cost, n_eqs_used = pfg.run_ibp_no_reordering(
        equations, vars_resorted, target_integral, masters, modulus, cost_cutoff
    )
    if cost_cutoff > 0:
        assert cost <= cost_cutoff, "IBP reduction exceeded cost cutoff"
        if not reduction_complete:
            cost = cost_cutoff
    elapsed = time.perf_counter() - t0
    print(
        f"No. of arithmetic operations: {cost}, equations used: {n_eqs_used}; elapsed: {elapsed:.3f} s"
    )
    if not reduction_complete and cost==cost_cutoff:
        print("(IBP reduction terminated early due to cost cutoff)")
    cost_normalized = cost / normalization_factor
    if return_n_eqs:
        return cost_normalized, n_eqs_used
    return cost_normalized


# --- CMA-ES optimization using pycma ---
def _flatten_params(model) -> np.ndarray:
    return np.concatenate([p.data.view(-1).cpu().numpy() for p in model.parameters()])


def _assign_params_(model, x: np.ndarray):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        w = torch.from_numpy(x[idx : idx + n]).view_as(p).to(p.device)
        p.data.copy_(w)
        idx += n


def save_network_weights(network: nn.Module, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    network.to("cpu")
    network.eval()
    torch.save(network.state_dict(), path)
    print(f"Saved network weights to {path}")


def cma_optimize_network(
    network,
    seed_op_eq_list,
    vars_sorted_nonmasters,
    vars_sorted_masters,
    target_integral,
    masters,
    max_evals=1500,
    cost_cutoff=-1,
    sigma0=None,
    mutation_size=0.02,
    seed=42,
    popsize=None,
    restarts=3,
    incpopsize=2,
    l2_threshold=3.0,
    penalty_weight=1e-4,
    penalty_power=2.0,
    optimize_seeds_only: bool = False,
    freeze_variable_ordering: bool = False,
    n_ibp_operators: int = None,
    checkpoint_dir: str = None,
):
    """
    Optimize network params with CMA-ES using pycma's restart feature.
    Restarts trigger automatically on stagnation; population size increases by incpopsize^k on restart k.
    Adds a small hinge L2 penalty on the parameter vector beyond l2_threshold:
        excess = max(0, ||theta||_2 - l2_threshold)
        penalty = penalty_weight * excess**penalty_power
    Why square it (penalty_power=2)?
      - Grows faster than linear, strongly discouraging very large norms while leaving small excesses mild.
      - Zero penalty (and zero gradient) below the threshold, so uniform rescaling that preserves rankings isn’t incentivized.
      - Smoother objective than a pure linear hinge, which is helpful for CMA-ES stability.
    Set penalty_power=1.0 to use a linear hinge if preferred. The penalty is applied only during CMA-ES, not in evaluate_cost.
    """
    x0 = _flatten_params(network)
    if sigma0 is None:
        m = float(np.mean(np.abs(x0)))
        sigma0 = mutation_size * (m + 1.0)

    best_observed = {
        "objective": np.inf,
        "cost": np.inf,
        "n_eqs": None,
        "x": None,
    }

    def objective(x):
        x_arr = np.asarray(x, dtype=np.float32)
        _assign_params_(network, x_arr)
        base_cost, n_eqs = evaluate_cost(
            network,
            seed_op_eq_list,
            vars_sorted_nonmasters,
            vars_sorted_masters,
            target_integral,
            masters,
            optimize_seeds_only=optimize_seeds_only,
            freeze_variable_ordering=freeze_variable_ordering,
            return_n_eqs=True,
            n_ibp_operators=n_ibp_operators,
            cost_cutoff=cost_cutoff,
        )
        l2 = float(np.linalg.norm(x))
        excess = max(0.0, l2 - float(l2_threshold))
        penalty = float(penalty_weight) * (excess ** float(penalty_power))
        objective_value = base_cost + penalty
        best_n_eqs = best_observed["n_eqs"]
        if objective_value < best_observed["objective"] or (
            objective_value == best_observed["objective"]
            and (best_n_eqs is None or n_eqs < best_n_eqs)
        ):
            best_observed.update(
                {
                    "objective": objective_value,
                    "cost": base_cost,
                    "n_eqs": n_eqs,
                    "x": x_arr.copy(),
                }
            )
        return objective_value

    opts = {"maxfevals": max_evals, "seed": seed, "verb_log": 0, "verb_disp": 1}
    if popsize is not None:
        opts["popsize"] = popsize

    def _write_seed_checkpoint(x, path, n_eqs_used=None):
        _assign_params_(network, np.asarray(x, dtype=np.float32))
        resorted = reorder_seed_op_eq_list(
            seed_op_eq_list,
            network,
            optimize_seeds_only=optimize_seeds_only,
            n_ibp_operators=n_ibp_operators,
        )
        if n_eqs_used is not None:
            resorted = resorted[:n_eqs_used]
        with open(path, "w") as f:
            for integral, operator_number, _ in resorted:
                f.write(" ".join(map(str, integral)) + f" {operator_number}\n")

    def _save_weight_checkpoint(filename):
        if not checkpoint_dir:
            return
        save_network_weights(network, os.path.join(checkpoint_dir, filename))

    # --- checkpoint callback: write best observed resorted_seed_op_list after each iteration ---
    checkpoint_counter = [0]  # mutable closure

    def _checkpoint_callback(es):
        checkpoint_counter[0] += 1
        xbest = best_observed["x"]
        if xbest is None:
            xbest = es.result.xbest
        if xbest is None:
            print(
                f"Checkpoint {checkpoint_counter[0]} skipped: no evaluated solution yet"
            )
            return
        cp_path = os.path.join(
            checkpoint_dir,
            f"resorted_seed_op_list.checkpoint{checkpoint_counter[0]}.txt",
        )
        _write_seed_checkpoint(xbest, cp_path, best_observed["n_eqs"])
        _save_weight_checkpoint("integrals_sorting_network.checkpoint.pt")
        print(
            f"Checkpoint {checkpoint_counter[0]}: wrote {cp_path} "
            f"(iter {es.countiter}, best_objective={best_observed['objective']:.6g}, "
            f"best_cost={best_observed['cost']:.6g}, n_eqs={best_observed['n_eqs']})"
        )

    t0 = time.perf_counter()
    try:
        xbest, es = cma.fmin2(
            objective,
            x0,
            sigma0,
            options=opts,
            restarts=restarts,
            incpopsize=incpopsize,
            callback=_checkpoint_callback if checkpoint_dir else None,
        )
    except KeyboardInterrupt:
        x_interrupt = best_observed["x"]
        if x_interrupt is not None:
            _assign_params_(network, np.asarray(x_interrupt, dtype=np.float32))
        _save_weight_checkpoint("integrals_sorting_network.interrupted.pt")
        _save_weight_checkpoint("integrals_sorting_network.checkpoint.pt")
        if x_interrupt is not None and checkpoint_dir:
            interrupted_cp_path = os.path.join(
                checkpoint_dir, "resorted_seed_op_list.interrupted.txt"
            )
            _write_seed_checkpoint(
                x_interrupt, interrupted_cp_path, best_observed["n_eqs"]
            )
            print(
                f"Interrupted checkpoint: wrote {interrupted_cp_path} "
                f"(best_objective={best_observed['objective']:.6g}, "
                f"best_cost={best_observed['cost']:.6g}, n_eqs={best_observed['n_eqs']})"
            )
        raise

    # Evaluate cost for the best-by-objective parameters (for logging)
    xbest = best_observed["x"] if best_observed["x"] is not None else xbest
    _assign_params_(network, np.asarray(xbest, dtype=np.float32))
    best_cost, n_eqs = evaluate_cost(
        network,
        seed_op_eq_list,
        vars_sorted_nonmasters,
        vars_sorted_masters,
        target_integral,
        masters,
        optimize_seeds_only=optimize_seeds_only,
        freeze_variable_ordering=freeze_variable_ordering,
        return_n_eqs=True,
        n_ibp_operators=n_ibp_operators,
    )
    if checkpoint_dir:
        final_cp_path = os.path.join(
            checkpoint_dir, "resorted_seed_op_list.checkpoint_final.txt"
        )
        _write_seed_checkpoint(xbest, final_cp_path, n_eqs)
        _save_weight_checkpoint("integrals_sorting_network.checkpoint.pt")
        print(
            f"Final checkpoint: wrote {final_cp_path} "
            f"(best_cost={best_cost:.6g}, n_eqs={n_eqs})"
        )

    t1 = time.perf_counter()
    print(
        f"CMA-ES (with restarts) finished in {t1 - t0:.3f} s; best_cost={best_cost:.6g}; min_equations_used={n_eqs}"
    )
    return best_cost, n_eqs


def pretrain_target_function(a, n_indices, n_props_top_level, s_max_top_level):
    """Length-agnostic heuristic target.
    a = list(indices) + [flag], where flag>0 means variable-only sample, flag<=0 means paired-with-operator.
    """
    idx = a[:n_indices]
    normalization = 0.1
    if a[n_indices] > 0:  # variables-only samples
        return normalization * sum(2 * i if i > 0 else -i for i in idx)
    else:  # paired with any operator
        if (
            pfg.d_level(idx) <= 0
            and pfg.s_level(idx) + (n_props_top_level - pfg.t_level(idx))
            <= s_max_top_level
        ):
            return normalization * sum(2 * i if i > 0 else -i for i in idx)
        else:
            penalty = pfg.d_level(idx)
            adjusted_rank = pfg.s_level(idx) + (n_props_top_level - pfg.t_level(idx))
            if adjusted_rank > s_max_top_level:
                penalty += adjusted_rank - s_max_top_level
            return -normalization * penalty


def parse_args():
    # Hardcoded defaults (lowest priority)
    hardcoded_defaults = {
        "ibp_file": "IBP",
        "trivial_sector_file": "trivialsector",
        "cut": "",
        "n_indices": 7,
        "n_ibp_operators": 9,
        "folder": ".",
        "top_sector": "1,1,1,1,1,1,0",
        "s_max": 3,
        "r_max": None,
        "d_max": 0,
        "rectangular": None,
        "target_integral": "",
        "variables": "d->31293,m2->17917,m3->22913",
        "optimize_seeds_only": False,
        "max_evals": 1000,
        "mutation_size": 0.02,
        "seed": 42,
        "restarts": 3,
        "incpopsize": 2,
        "pretrain_epochs": 20,
        "pretrain_batch_size": 256,
        "learning_rate": 1e-3,
        "save_path": "save",
        "load_weights": "",
        "pretrain_loaded_weights": False,
        "gen_eqs_only": False,
        "config": "",
        "freeze_variable_ordering": False,
        "cost_cutoff": -1,
    }

    parser = argparse.ArgumentParser(
        description="Optimize IBP orderings with neural networks."
    )
    # Integral family settings
    parser.add_argument(
        "--ibp-file", type=str, help="Path under --folder to the IBP template file."
    )
    parser.add_argument(
        "--trivial-sector-file",
        type=str,
        help="Path under --folder to the trivial sector file.",
    )
    parser.add_argument(
        "--cut",
        type=str,
        help="List of comma-separated integers indicating which propagators to cut (e.g. '1,3' to cut the first and third propagators).",
    )
    parser.add_argument(
        "--n-indices",
        type=int,
        help="Number of integral indices in the family (length of an integral tuple).",
    )
    parser.add_argument(
        "--n-ibp-operators",
        type=int,
        help="Number of IBP operators; one-hot has size n_operators+1 with index 0 reserved for the special bit.",
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Base folder for all I/O. All paths are resolved as join(--folder, path).",
    )
    # Integrals and seeding settings
    parser.add_argument(
        "--top-sector",
        type=str,
        help="Comma-separated 0/1 list defining the top sector; default for triangle is '1,1,1,1,1,1,0'.",
    )
    parser.add_argument(
        "--s-max",
        type=int,
        help="Max sum of absolute values of negative indices; default 3.",
    )
    parser.add_argument(
        "--r-max",
        type=int,
        help="Max sum of positive indices; default is the number of 1's in --top-sector.",
    )
    parser.add_argument(
        "--d-max",
        type=int,
        help="Max allowed number of 'dots' (d-level) for seed generation; default 0.",
    )
    parser.add_argument(
        "--rectangular",
        type=int,
        nargs=2,
        metavar=("M", "N"),
        default=None,
        help="Post-filter seeds to a rectangular region: a1<=M, a2<=N. "
        "Only valid for 2-index integral families (e.g., bubble). "
        "When not set, no per-index filtering is applied.",
    )
    parser.add_argument(
        "--target-integral",
        type=str,
        help="Comma-separated tuple for the target integral; default is --top-sector with the last entry set to -3.",
    )
    parser.add_argument(
        "--variables",
        type=str,
        help="Required: single string mapping for variables used in IBP templates, e.g. 'd->31293,m2->17917,m3->22913'. "
        "Note: TOML [variables] tables are not supported; use a single 'variables = \"...\"' string.",
    )
    # Optimization settings
    parser.add_argument(
        "--optimize-seeds-only",
        action="store_true",
        help="Treat all IBP operators equally; the network input is only the integral indices for seed/operator pairs; variables are ordered by sort_integrals_desc (no NN scoring).",
    )
    parser.add_argument(
        "--freeze-variable-ordering",
        action="store_true",
        help="When set, variables ordering is frozen: `reorder_vars` is called with optimize_seeds_only=True regardless of other flags.",
    )
    parser.add_argument(
        "--cost-cutoff",
        type=int,
        help="Maximum allowed arithmetic operations (cost cutoff). Use -1 to disable (default).",
    )
    parser.add_argument(
        "--max-evals", type=int, help="Max function evaluations for CMA-ES."
    )
    parser.add_argument(
        "--mutation-size",
        type=float,
        help="Scale factor used to derive the default CMA-ES sigma0 as mutation_size * (mean(|x0|) + 1.0).",
    )
    parser.add_argument("--seed", type=int, help="Random seed for CMA-ES.")
    parser.add_argument("--restarts", type=int, help="Number of CMA-ES restarts.")
    parser.add_argument(
        "--incpopsize", type=int, help="Population size multiplier per restart."
    )
    parser.add_argument(
        "--pretrain-epochs", type=int, help="Number of pretraining epochs."
    )
    parser.add_argument(
        "--pretrain-batch-size", type=int, help="Batch size for pretraining."
    )
    parser.add_argument(
        "--learning-rate", type=float, help="Learning rate for pretraining (Adam)."
    )
    parser.add_argument(
        "--load-weights",
        type=str,
        help="Path to a saved network state_dict to resume from. Relative paths are resolved under --folder.",
    )
    parser.add_argument(
        "--pretrain-loaded-weights",
        action="store_true",
        help="When --load-weights is set, run heuristic pretraining before CMA-ES instead of resuming directly.",
    )
    # Output settings
    parser.add_argument(
        "--save-path",
        type=str,
        help="Directory under --folder where artifacts are saved (saved_eps.pkl, integrals_sorting_network.pt, resorted_seed_op_list.txt, vars_resorted.txt).",
    )
    parser.add_argument(
        "--gen-eqs-only",
        action="store_true",
        help="Generate/load equations (writing cache under save_dir) and exit before training/optimization.",
    )
    # External config (TOML)
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a TOML config file (under --folder). CLI options override config.",
    )

    # Two-stage parse: first get --config and --folder if present
    prelim, _ = parser.parse_known_args()
    effective_defaults = hardcoded_defaults.copy()
    if prelim.config:
        # Resolve config under --folder unless an absolute path was given
        config_path = (
            prelim.config
            if os.path.isabs(prelim.config)
            else os.path.join(prelim.folder or ".", prelim.config)
        )
        with open(config_path, "rb") as f:
            cfg = tomllib.load(f)
        for key, val in cfg.items():
            key_normalized = key.replace("-", "_")
            if key_normalized == "variables":
                if isinstance(val, dict):
                    raise ValueError(
                        "TOML [variables] table is not supported. "
                        'Use a single line: variables = "d->31293,m2->17917,m3->22913"'
                    )
                effective_defaults["variables"] = str(val)
            elif key_normalized == "print_stepinfo":
                # Expect a boolean in TOML; coerce otherwise
                effective_defaults["print_stepinfo"] = bool(val)
            else:
                effective_defaults[key_normalized] = val
    # Apply merged defaults, then final parse
    parser.set_defaults(**effective_defaults)
    return parser.parse_args()


def _parse_variable_mapping(spec) -> dict:
    """Parse a single string 'name->value' or 'name=value' comma-separated mapping into a dict of ints.
    TOML [variables] tables (dicts) are not supported.
    """
    if isinstance(spec, dict):
        raise ValueError(
            "TOML [variables] table is not supported; use a single string like "
            "\"variables = 'd->31293,m2->17917,m3->22913'\""
        )
    mapping = {}
    if not spec:
        return mapping
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "->" in tok:
            k, v = tok.split("->", 1)
        elif "=" in tok:
            k, v = tok.split("=", 1)
        else:
            raise ValueError(f"Invalid variable spec '{tok}', expected name->value")
        k = k.strip()
        v = v.strip()
        if not k or not v:
            raise ValueError(f"Invalid variable spec '{tok}'")
        mapping[k] = int(v)
    return mapping


def _parse_index_tuple(spec: str) -> tuple[int, ...]:
    """Parse a comma/space-separated list of ints, optionally wrapped in () or [], into a tuple[int,...]."""
    if spec is None:
        return tuple()
    s = str(spec).strip()
    if not s:
        return tuple()
    # Strip optional wrappers
    if (s[0] in "([") and (s[-1] in ")]"):
        s = s[1:-1].strip()
    # Split on commas or whitespace
    parts = (
        [p for tok in s.split(",") for p in tok.split() if p.strip()]
        if ("," in s)
        else s.split()
    )
    return tuple(int(p) for p in parts)


def generate_equations(
    eq_templates, trivialsectorlist, m_vals, all_seeds, save_dir, masters
):
    """Load or generate equations and variables, then return sorted variable lists.
    Args:
        eq_templates: list of equation templates
        trivialsectorlist: list of trivial sectors
        m_vals: dict of mass values
        all_seeds: generated seeds
        save_dir: directory to store/load the cache 'saved_eps.pkl'
        masters: list of master integrals
    Returns:
        (seed_op_eq_list, all_variables, vars_sorted_nonmasters, vars_sorted_masters)
    """
    loaded_from_cache = False
    os.makedirs(save_dir, exist_ok=True)
    cache_path = os.path.join(save_dir, "saved_eps.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                saved_all_seeds, saved_seed_op_eq_list, saved_all_variables = (
                    pickle.load(f)
                )
            if saved_all_seeds == all_seeds:
                seed_op_eq_list = saved_seed_op_eq_list
                all_variables = saved_all_variables
                loaded_from_cache = True
                print(f"Loaded seed_op_eq_list and all_variables from {cache_path}")
            else:
                print(
                    "Saved all_seeds does not match current all_seeds; will regenerate equations."
                )
        except Exception as e:
            print(f"Could not load {cache_path} ({e}); will regenerate equations.")
    if not loaded_from_cache:
        print("Starting equation generation...")
        _t0 = time.perf_counter()
        seed_op_eq_list, all_variables = pfg.gen_eqs(
            eq_templates, trivialsectorlist, m_vals, all_seeds
        )
        _t1 = time.perf_counter()
        print(f"Equation generation finished in {_t1 - _t0:.3f} s")
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(
                    (all_seeds, seed_op_eq_list, all_variables),
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            print(f"Saved (all_seeds, seed_op_eq_list, all_variables) to {cache_path}")
        except Exception as e:
            print(f"Failed to save {cache_path}: {e}")

    vars_sorted = pfg.sort_integrals_desc(all_variables)

    # Will move masters to the end regardless of NN sorting
    master_set = set(masters)
    vars_sorted_nonmasters = [v for v in vars_sorted if v not in master_set]
    vars_sorted_masters = [v for v in vars_sorted if v in master_set]

    return seed_op_eq_list, all_variables, vars_sorted_nonmasters, vars_sorted_masters


def pretrain(
    network,
    target_function,
    epochs,
    batch_size,
    all_variables,
    seed_op_eq_list,
    n_props_top_level,
    s_max_top_level,
    learning_rate: float = 1e-3,
    optimize_seeds_only: bool = False,
    n_ibp_operators: int = None,
):
    """Pretrain the scoring network on heuristic targets."""
    inputs, targets = [], []
    if not optimize_seeds_only:
        for v in all_variables:
            x = encode_integral_with_special_bit(
                v, optimize_seeds_only=False, n_ibp_operators=n_ibp_operators
            )
            inputs.append(x)
            targets.append(
                float(
                    target_function(
                        list(v) + [1], len(v), n_props_top_level, s_max_top_level
                    )
                )
            )
    for integral, operator_number, _ in seed_op_eq_list:
        x = encode_integral_operator(
            integral,
            operator_number,
            optimize_seeds_only=optimize_seeds_only,
            n_ibp_operators=n_ibp_operators,
        )
        inputs.append(x)
        targets.append(
            float(
                target_function(
                    list(integral) + [0],
                    len(integral),
                    n_props_top_level,
                    s_max_top_level,
                )
            )
        )

    X = torch.stack(inputs)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(X, y)
    print(f"Dataset size for pretraining: {len(dataset)} samples")

    # Choose device for pretraining
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = device.type == "cuda"
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_mem
    )

    # Move model to device for training
    network.to(device)
    network.train()
    optimizer = torch.optim.Adam(
        network.parameters(), lr=learning_rate
    )
    loss_fn = torch.nn.MSELoss()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = network(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(dataset)
        print(f"Pretrain epoch {epoch+1}: loss={epoch_loss:.6f}")
    network.eval()

    # Post-pretraining sanity checks (on the same device)
    test_integral = list(all_variables[len(all_variables) // 2])
    if not optimize_seeds_only:
        test_input = (
            encode_integral_with_special_bit(
                test_integral,
                optimize_seeds_only=False,
                n_ibp_operators=n_ibp_operators,
            )
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            nn_score = network(test_input).item()
        pretrain_score = target_function(
            test_integral + [1], len(test_integral), n_props_top_level, s_max_top_level
        )
        print(
            f"Test integral: {test_integral}, NN score: {nn_score:.6f}, Pretrain target: {pretrain_score:.6f}"
        )

    test_integral, test_operator, _ = seed_op_eq_list[len(seed_op_eq_list) // 2]
    test_integral = list(test_integral)
    test_input = (
        encode_integral_operator(
            test_integral,
            test_operator,
            optimize_seeds_only=optimize_seeds_only,
            n_ibp_operators=n_ibp_operators,
        )
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        nn_score = network(test_input).item()
    pretrain_score = target_function(
        test_integral + [0], len(test_integral), n_props_top_level, s_max_top_level
    )
    print(
        f"Test integral/operator: {test_integral} with operator {test_operator}, NN score: {nn_score:.6f}, Pretrain target: {pretrain_score:.6f}"
    )

    # Move model back to CPU so the rest of the script uses CPU tensors only
    network.to("cpu")


def save_optimization_result(
    network,
    seed_op_eq_list,
    vars_sorted_nonmasters,
    vars_sorted_masters,
    out_dir: str,
    optimize_seeds_only: bool = False,
    n_ibp_operators: int = None,
    n_eqs_used: int = None,
    freeze_variable_ordering: bool = False,
):
    """Save trained network parameters and resorted lists into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    network.eval()
    with torch.no_grad():
        model_path = os.path.join(out_dir, "integrals_sorting_network.pt")
        save_network_weights(network, model_path)
        resorted_seed_op_eq_list = reorder_seed_op_eq_list(
            seed_op_eq_list,
            network,
            optimize_seeds_only=optimize_seeds_only,
            n_ibp_operators=n_ibp_operators,
        )
        resorted_seed_op_eq_list = resorted_seed_op_eq_list[:n_eqs_used]
        seed_list_path = os.path.join(out_dir, "resorted_seed_op_list.txt")
        with open(seed_list_path, "w") as f:
            for integral, operator_number, _ in resorted_seed_op_eq_list:
                f.write(" ".join(map(str, integral)) + f" {operator_number}\n")
        vars_resorted_nonmasters_final = reorder_vars(
            vars_sorted_nonmasters,
            network,
            optimize_seeds_only=(optimize_seeds_only or freeze_variable_ordering),
            n_ibp_operators=n_ibp_operators,
        )
        vars_resorted = vars_resorted_nonmasters_final + vars_sorted_masters
        vars_path = os.path.join(out_dir, "vars_resorted.txt")
        with open(vars_path, "w") as f:
            for v in vars_resorted:
                f.write(" ".join(map(str, v)) + "\n")


def read_masters(path: str, trivial_sectors=None):
    """Read master integrals from a text file, one tuple per line.
    Accepts comma or whitespace separators; ignores empty lines and lines starting with '#'.
    """
    masters = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.replace(",", " ").split()
            try:
                masters.append(tuple(int(p) for p in parts))
            except ValueError as e:
                raise ValueError(f"Invalid masters line '{line.strip()}': {e}")
    return [m for m in masters if not pfg.is_trivial(m, trivial_sectors)]


def load_network_checkpoint(
    network: nn.Module, path: str, fallback: str = "integrals_sorting_network.pt"
):
    """Load network state_dict from 'path'; if that fails (e.g., it's an eqs cache),
    try 'fallback'. Uses weights_only=False for PyTorch 2.6 compatibility."""

    def try_load(p: str) -> bool:
        try:
            state = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(state, dict):
                network.load_state_dict(state)
                return True
        except Exception as e:
            return False
        return False

    if os.path.isfile(path) and try_load(path):
        print(f"Loaded network parameters from {path}")
        return
    if fallback and os.path.isfile(fallback) and try_load(fallback):
        print(f"Loaded network parameters from {fallback} (fallback)")
        return
    raise RuntimeError(
        f"Could not load network parameters from '{path}' or fallback '{fallback}'"
    )


def main():
    args = parse_args()

    def _resolve(path: str) -> str:
        """Resolve a path strictly under --folder."""
        if not path:
            return path
        return os.path.join(args.folder, path)

    trivial_sector_file = _resolve(args.trivial_sector_file)
    save_dir = _resolve(args.save_path)

    n_ibp_operators = int(args.n_ibp_operators)

    # Use CLI-provided variables mapping (required; has a sensible default)
    m_vals = _parse_variable_mapping(args.variables)
    if "d" not in m_vals:
        raise ValueError(
            "Missing required variable 'd' in --variables (e.g. include d->31293)."
        )

    # Build equation templates from the provided IBP file and variables
    ibp_path = _resolve(args.ibp_file)
    eq_templates = pfg.gen_eq_templates(ibp_path, m_vals)

    # Family-dependent options
    top_sector = _parse_index_tuple(args.top_sector)
    n_props_top_level = sum(1 for x in top_sector if x > 0)
    if not top_sector:
        raise ValueError("--top-sector cannot be empty")
    n_indices_cfg = len(top_sector)
    # Consistency check: top_sector length must match --n-indices
    if int(args.n_indices) != n_indices_cfg:
        raise ValueError(
            f"--n-indices ({args.n_indices}) does not match length of --top-sector ({n_indices_cfg})"
        )

    # s_max direct; r_max defaults to number of 1's in top_sector if not provided
    s_max = int(args.s_max)
    r_max = (
        int(args.r_max)
        if args.r_max is not None
        else sum(1 for x in top_sector if x == 1)
    )
    d_max = int(args.d_max)

    # target_integral defaults to top_sector with last entry set to -3
    if args.target_integral:
        target_integral = _parse_index_tuple(args.target_integral)
    else:
        if n_indices_cfg < 1:
            raise ValueError("Invalid --top-sector length for deriving target integral")
        target_integral = tuple(list(top_sector[:-1]) + [-3])

    cut = [int(x) for x in args.cut.split(",") if x.strip()]
    trivial_sectors = pfg.get_trivial_sectors(
        trivial_sector_file, cut=cut, n_indices=n_indices_cfg
    )
    masters_path = _resolve("masters")
    masters = read_masters(masters_path, trivial_sectors=trivial_sectors)
    print(f"{s_max=}, {r_max=}, {d_max=}")
    all_seeds = pfg.gen_all_seeds(top_sector, trivial_sectors, s_max, r_max, d_max)
    print(f"Generated {len(all_seeds)} seeds")

    # Optional rectangular post-filtering: restrict seeds to a1<=M, a2<=N.
    # Only valid for 2-index integral families (e.g., bubble integrals).
    if args.rectangular is not None:
        if n_indices_cfg != 2:
            raise ValueError(
                f"--rectangular is only valid for 2-index integral families, "
                f"but the current family has {n_indices_cfg} indices."
            )
        m_rect, n_rect = args.rectangular
        n_seeds_before = len(all_seeds)
        all_seeds = [
            s for s in all_seeds if s[0] <= m_rect and s[1] <= n_rect
        ]
        print(
            f"After rectangular filter (a1<={m_rect}, a2<={n_rect}): "
            f"{len(all_seeds)} seeds remain (removed {n_seeds_before - len(all_seeds)})"
        )
        if not all_seeds:
            raise ValueError(
                "No seeds remain after rectangular filtering. "
                "Relax --rectangular bounds or increase --s-max / --r-max."
            )

    # Equation generation (uses save_dir/saved_eps.pkl)
    seed_op_eq_list, all_variables, vars_sorted_nonmasters, vars_sorted_masters = (
        generate_equations(
            eq_templates, trivial_sectors, m_vals, all_seeds, save_dir, masters
        )
    )

    # Exit early if only generating equations
    if args.gen_eqs_only:
        print("gen-eqs-only is set; exiting after equation generation.")
        return

    # Initialize network and reorder
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n_indices_local = len(all_variables[0]) if all_variables else n_indices_cfg
    input_dim = (
        n_indices_local
        if args.optimize_seeds_only
        else (n_indices_local + (n_ibp_operators + 1))
    )
    network = IntegralScoringNetwork(input_dim=input_dim)
    initial_l2_threshold = float(np.linalg.norm(_flatten_params(network)))
    loaded_weights = bool(args.load_weights)
    if loaded_weights:
        load_path = _resolve(args.load_weights)
        load_network_checkpoint(
            network,
            load_path,
            fallback=os.path.join(save_dir, "integrals_sorting_network.pt"),
        )
        initial_l2_threshold = float(np.linalg.norm(_flatten_params(network)))

    def _save_interrupt_weights():
        save_network_weights(
            network, os.path.join(save_dir, "integrals_sorting_network.interrupted.pt")
        )
        save_network_weights(
            network, os.path.join(save_dir, "integrals_sorting_network.checkpoint.pt")
        )
        print(
            "Interrupted by Ctrl+C. Resume with "
            f"--load-weights {os.path.join(args.save_path, 'integrals_sorting_network.checkpoint.pt')}"
        )

    # --- Pretrain the scoring network to imitate improved seeding ---
    try:
        if loaded_weights and not args.pretrain_loaded_weights:
            print("Loaded weights provided; skipping pretraining before CMA-ES.")
        else:
            pretrain(
                network,
                pretrain_target_function,
                args.pretrain_epochs,
                args.pretrain_batch_size,
                all_variables,
                seed_op_eq_list,
                n_props_top_level=n_props_top_level,
                s_max_top_level=s_max,
                learning_rate=args.learning_rate,
                optimize_seeds_only=args.optimize_seeds_only,
                n_ibp_operators=n_ibp_operators,
            )

        init_cost = evaluate_cost(
            network,
            seed_op_eq_list,
            vars_sorted_nonmasters,
            vars_sorted_masters,
            target_integral,
            masters,
            optimize_seeds_only=args.optimize_seeds_only,
            freeze_variable_ordering=args.freeze_variable_ordering,
            n_ibp_operators=n_ibp_operators,
        )
    except KeyboardInterrupt:
        _save_interrupt_weights()
        return
    print(f"Initial IBP cost: {init_cost}")

    try:
        # CMA-ES optimization
        best_cost, n_eqs_used = cma_optimize_network(
            network,
            seed_op_eq_list,
            vars_sorted_nonmasters,
            vars_sorted_masters,
            target_integral,
            masters,
            max_evals=args.max_evals,
            sigma0=None,
            mutation_size=args.mutation_size,
            seed=args.seed,
            popsize=None,
            restarts=args.restarts,
            incpopsize=args.incpopsize,
            l2_threshold=initial_l2_threshold,
            optimize_seeds_only=args.optimize_seeds_only,
            freeze_variable_ordering=args.freeze_variable_ordering,
            n_ibp_operators=n_ibp_operators,
            cost_cutoff=args.cost_cutoff,
            checkpoint_dir=save_dir,
        )
    except KeyboardInterrupt:
        print(
            "Interrupted by Ctrl+C. Resume with "
            f"--load-weights {os.path.join(args.save_path, 'integrals_sorting_network.checkpoint.pt')}"
        )
        return
    print(f"After CMA-ES: best_cost={best_cost}, n_eqs_used={n_eqs_used}")

    # Save trained network parameters and current resorted lists into save_dir
    save_optimization_result(
        network,
        seed_op_eq_list,
        vars_sorted_nonmasters,
        vars_sorted_masters,
        out_dir=save_dir,
        optimize_seeds_only=args.optimize_seeds_only,
        n_ibp_operators=n_ibp_operators,
        n_eqs_used=n_eqs_used,
        freeze_variable_ordering=args.freeze_variable_ordering,
    )


if __name__ == "__main__":
    main()
