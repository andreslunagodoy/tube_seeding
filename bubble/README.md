# Bubble — IBP ordering optimization

This directory contains a workflow for optimizing Integration-By-Parts (IBP) reduction of
**massive bubble** integrals using a neural network trained with CMA-ES.
The topology is the one-loop massive bubble.

## Files

| File | Description |
|---|---|
| `IBP` | IBP equation templates (8 equations for the massive bubble) |
| `masters` | Master integrals: `(1,0)`, `(0,1)`, `(1,1)` |
| `trivialsector` | Trivial sector flag (`0` — no trivial sectors) |
| `run1.sh` | Run 1: full seed region (triangular, `a1 + a2 ≤ 12`) |
| `run2.sh` | Run 2: rectangular seed region (`a1 ≤ 6, a2 ≤ 6`) |
| `plot.sh` | Visualize the resulting seed orderings from both runs |
| `plot_resorted_seed_op_list.py` | Python script called by `plot.sh` |
| `run1/` | Output artifacts from run 1 |
| `run2/` | Output artifacts from run 2 |

## Workflow

### Step 1 — `run1.sh`

```bash
mkdir -p run1
optimize-ordering-NN \
    --ibp-file IBP \
    --trivial-sector-file trivialsector \
    --n-indices 2 \
    --n-ibp-operators 2 \
    --save-path run1 \
    --top-sector 1,1 \
    --s-max 0 \
    --r-max 12 \
    --d-max 12 \
    --target-integral 6,6 \
    --variables 'm0->37,d->13' \
    --max-evals 9000 \
    --mutation-size 0.025 \
    --learning-rate 0.02
```

This runs the full pipeline:

1. **Seed generation** — All seeds in the top sector `(1,1)` are generated with:
   - `s_max = 0` — no negative-index shifts
   - `r_max = 12` — sum of positive indices (`a1 + a2`) up to 12 → triangular region
   - `d_max = 12` — "dots" level up to 12
   - This produces seeds like `(0,0)`, `(0,1)`, `(1,0)`, …, `(12,0)`, `(0,12)`, etc.

2. **Equation generation** — For each seed and each IBP operator (2 operators), an IBP
   equation is instantiated from the templates in `IBP`. The results are cached in
   `run1/saved_eps.pkl` so subsequent runs with the same seeds skip this step.

3. **Neural network pretraining** — A small feedforward net (3→16→8→1 for the 2-index
   case with 2 operators) is pretrained to imitate a heuristic scoring function that
   ranks integrals by index magnitude and dot/s/t levels. Input is either:
   - `[a1, a2, onehot(operator)]` for (integral, operator) pairs
   - `[a1, a2, onehot(special_bit)]` for variables

4. **CMA-ES optimization** — The network weights are optimized with CMA-ES
   (with restarts) to minimize the total number of arithmetic operations in the IBP
   reduction to the target integral `(6,6)`. The network scores all (integral, operator)
   pairs and variables, and the IBP solver processes them in descending score order.

5. **Output** — Saved to `run1/`:
   - `resorted_seed_op_list.txt` — ordered list of seeds with operators
   - `vars_resorted.txt` — ordered list of variables (elimination ordering)
   - `integrals_sorting_network.pt` — trained network weights
   - `saved_eps.pkl` — cache of generated equations

### Step 2 — `run2.sh`

```bash
mkdir -p run2
optimize-ordering-NN --rectangular 6 6 ...  # same parameters otherwise
```

Identical to run 1 except for the `--rectangular 6 6` flag, which filters the seed
space to `a1 ≤ 6, a2 ≤ 6` instead of the triangular `a1 + a2 ≤ 12`. This changes the
set of available seeds and thus the IBP system that the NN must learn to optimize.

Output goes to `run2/`.

### Step 3 — `plot.sh`

```bash
python plot_resorted_seed_op_list.py --data-file run1/resorted_seed_op_list.txt --save bubble_ibp_eqs_scheme1.png
python plot_resorted_seed_op_list.py --data-file run2/resorted_seed_op_list.txt --save bubble_ibp_eqs_scheme2.png
```

Visualizes the NN-optimized seed orderings. Each seed `(a1, a2)` is plotted on a grid,
color-coded by operator (1 or 2), and animated in the order chosen by the network.
The target integral `(6,6)` is highlighted with a marker.

## Algorithm Details

The core idea is: **the order in which IBP equations are applied and variables are
eliminated dramatically affects the computational cost of IBP reduction.** The network
learns to score each (integral, operator) pair so that processing them in descending
score order minimizes arithmetic operations.

The pipeline (`optimize_ordering_NN.py`) uses:
- **PyTorch** for the neural network (small: 3-layer MLP with GELU activations)
- **pycma** for CMA-ES optimization with restarts and increasing population size
- **pyfeyngym** for the actual IBP reduction cost evaluation (`run_ibp_no_reordering`)
- An L2 penalty on network weights to prevent unbounded growth during CMA-ES

The cost function is the normalized number of arithmetic operations modulo
`2^31 - 1`, evaluated by running the actual IBP reduction with the current ordering.

## Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `n_indices` | 2 | Number of integral indices (`a1`, `a2`) |
| `n_ibp_operators` | 2 | Number of IBP operators |
| `top_sector` | `1,1` | Both propagators are present |
| `s_max` | 0 | No negative-index shifts |
| `r_max` | 12 | Sum of positive indices ≤ 12 |
| `d_max` | 12 | Dot level ≤ 12 |
| `target_integral` | `6,6` | Integral to reduce to |
| `variables` | `m0→37, d→13` | Mass parameter = 37, spacetime dim = 13 |
| `max_evals` | 9000 | CMA-ES objective function evaluations |
| `mutation_size` | 0.025 | CMA-ES initial standard deviation scale |
| `learning_rate` | 0.02 | Adam LR for pretraining |
