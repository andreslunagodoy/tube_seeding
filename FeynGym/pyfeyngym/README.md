# pyfeyngym

Python tools for integration-by-parts (IBP) reduction experiments with
FeynGym. The package exposes

- a `gymnasium` environment for reinforcement learning for the one-loop massive
  bubble family;
- helpers for generating concrete IBP equations from template files and seed
  integrals for arbitrary integral families;
- wrappers around the Julia finite-field linear solvers in `FeynGym.jl` and
  `SparseSolveExact.jl`;
- a command-line tool `optimize-ordering-NN` for neural-network/CMA-ES
  optimization of equation and variable orderings.

Importing `pyfeyngym` initializes Julia through `juliacall`, so the first import
may take a little while while Julia packages are resolved and precompiled.

## Installation

From the repository root:

```bash
cd pyfeyngym
pip install -e .
python install_julia_packages.py
```

The second command registers the local Julia packages `FeynGym.jl` and
`SparseSolveExact.jl` with `juliapkg` (part of `juliacall`).

The neural-network ordering optimizer additionally imports `torch` and `cma`;
install them in the same Python environment before using
`optimize-ordering-NN`.

## Massive Bubble Gymnasium Environment

This is a reinforcement learning environment with the standard gymnasium
interface.
The registered environment ID is `pyfeyngym-v0`. Each observation is an
image-like array with shape `(channels, height, width)`, where the last two axes
are a grid over the two bubble-integral indices `(a1, a2)`. For
`max_seed_propagator_power = M`, both grid axes cover the coordinate range
`-1, 0, ..., M + 1`. `info["action_mask"]` contains the valid-action mask.

Example usage without training:

```python
import gymnasium as gym
import numpy as np
import pyfeyngym

env = gym.make(
    "pyfeyngym-v0",
    target_integral=[6, 6],
    max_seed_propagator_power=6,
)

obs, info = env.reset()
done = False
total_reward = 0.0

while not done:
    valid_actions = np.flatnonzero(info["action_mask"])
    action = int(valid_actions[0])
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(total_reward)
```

Useful environment helpers include:

- `env.action_masks()` for the flattened valid-action mask;
- `env.action_log()` for the sequence of chosen raw IBP actions;
- `env.all_seed_candidates()` for the bubble-family seed candidates;
- `env.visualize()` for a Matplotlib visualization of the chosen seeds.

For non-RL experiments on the bubble family, the package also provides
`test_ibp_with_priority_function` and `test_ibp_with_priority_list`.

## Generating Equations

For general integral families, `pyfeyngym` expects a family-specific IBP
template file and a trivial-sector file. The template format is the Kira-style
triangular-equation format used by `examples/triangle/IBP`: each blank-line
separated group is an IBP operator, and each line has an integral shift vector
and a coefficient polynomial.

```python
import pyfeyngym as pfg

m_vals = {"d": 31293, "m2": 17917, "m3": 22913}
top_sector = (1, 1, 1, 1, 1, 1, 0)

eq_templates = pfg.gen_eq_templates("examples/triangle/IBP", m_vals)
trivial_sectors = pfg.get_trivial_sectors(
    "examples/triangle/trivialsector",
    cut=[1, 2, 3, 4, 5, 6],
    n_indices=7,
)

seeds = pfg.gen_all_seeds(
    top_sector,
    trivial_sectors,
    s_max=3,
    r_max=7,
    d_max=1,
)

seed_op_eq_list, variables = pfg.gen_eqs(
    eq_templates,
    trivial_sectors,
    m_vals,
    seeds,
)
```

`seed_op_eq_list` contains `(seed, operator_number, equation)` triples, where an
equation is a list of `[integral, coefficient]` terms. `variables` contains the
integrals appearing in the generated equations.

The most commonly used utility functions are:

- `gen_eq_templates(path, m_vals)` parses the IBP template file;
- `get_trivial_sectors(path, cut=..., n_indices=...)` loads and augments the
  trivial-sector list;
- `gen_all_seeds(top_sector, trivial_sectors, s_max, r_max, d_max=-1)` generates
  seed integrals in nontrivial sectors below a top sector;
- `sort_integrals_desc(integrals)` applies the built-in integral ordering;
- `r_level`, `s_level`, `d_level`, `t_level`, `to_sector`, and
  `to_sector_list` expose small sector/order helpers.

## Running a Reduction

Continuing from the equation-generation example above, call the Julia-backed
reduction routine with a target integral, a master-integral list, and a
finite-field modulus.

```python
modulus = 2**31 - 1
target_integral = (1, 1, 1, 1, 1, 1, -3)
masters = [
    (1, 0, 1, 1, 1, 1, 0),
    (1, 1, 0, 1, 1, 1, 0),
    (1, 0, 0, 1, 1, 1, 0),
    (1, -1, 1, 0, 1, 1, 0),
    (1, 0, 1, 0, 1, 1, 0),
    (1, 1, 0, 1, 0, 1, 0),
    (1, 0, 1, 0, 0, 1, 0),
    (1, 1, 1, 1, 1, 0, 0),
    (0, 1, 1, 1, 0, 0, 0),
    (1, 0, 1, 0, 1, 0, 0),
    (1, 1, 0, 1, 1, 0, 0),
    (0, 0, 1, 1, 1, 0, 0),
    (1, 0, 1, 1, 1, 0, 0),
    (1, -1, 1, 1, 1, 0, 0),
    (0, 1, 1, 1, 1, 0, 0),
    (-1, 1, 1, 1, 1, 0, 0),
]

equations = [eq for _, _, eq in seed_op_eq_list]
variables_sorted = pfg.sort_integrals_desc(variables)

reduction_complete, cost, n_eqs_used = pfg.run_ibp_no_reordering(
    equations,
    variables_sorted,
    target_integral,
    masters,
    modulus,
)

print(reduction_complete, cost, n_eqs_used)
```

For direct finite-field linear solves, use `solve_eqs_modulo(eqs, variables,
modulus, ...)`.

## Optimizing Equation Orderings

The `optimize-ordering-NN` command builds equations from seeds, scores
seed/operator pairs and variables with a small neural network, and optimizes the
network parameters with CMA-ES using the IBP reduction cost as the objective.

The input folder must contain:

- `IBP`: the IBP template file, unless overridden with `--ibp-file`;
- `trivialsector`: the trivial-sector file, unless overridden with
  `--trivial-sector-file`;
- `masters`: one master integral per line, with comma or whitespace separated
  indices.

Example:

```bash
optimize-ordering-NN \
  --folder examples/triangle \
  --ibp-file IBP \
  --trivial-sector-file trivialsector \
  --top-sector 1,1,1,1,1,1,0 \
  --target-integral 1,1,1,1,1,1,-3 \
  --n-indices 7 \
  --n-ibp-operators 9 \
  --variables 'd->31293,m2->17917,m3->22913' \
  --s-max 3 \
  --r-max 7 \
  --d-max 1 \
  --save-path save
```

TOML configuration files are also supported with `--config`; see
`examples/triangle_config.toml` for the available fields. CLI flags override
values loaded from the config. Use `--gen-eqs-only` to generate or load the
cached equation data under `save_path/saved_eps.pkl` and exit before training.

The optimizer writes:

- `saved_eps.pkl`: cached seeds, generated equations, and variables;
- `integrals_sorting_network.pt`: trained PyTorch parameters;
- `resorted_seed_op_list.txt`: selected seed/operator order;
- `vars_resorted.txt`: selected variable order.

## Examples

- `examples/IBPEnv_example.ipynb`: direct use of the bubble Gym environment;
- `../examples/pyfeyngym_test_ibp_cost.ipynb`: a small direct test of
  `run_ibp_no_reordering`, comparing several seed/equation orderings by the
  resulting IBP reduction cost. This is the Julia-backed backend function that
  supplies the objective value optimized by `optimize-ordering-NN`;
- `examples/solve_eqs_finite_field.ipynb`: finite-field linear solves.
