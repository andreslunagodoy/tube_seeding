# Massless nonplanar double pentagon integrals

## Directory layout

```
doublePentagon/
├── gen_eqs_and_solve.ipynb         minimal worked example notebook
├── evolutionary_strategies/        CMA-ES maximal-cut seeding (Sec. 4.1 + 4.3.1)
├── rank20/                         rank-20 reductions (Sec. 4.2.4 + 4.3.2)
├── rank10/                         all rank-10 experiments (Sec. 4.2–4.3.3)
├── benchmarks/                     timing/memory scans (Sec. 4.2.3–4.2.4)
├── optimize_seeding/               subsector-exception search (Sec. 4.2.5)
├── validation/                     finite-field artifact check
├── kira_equation_template_generation/  how IBP_LI was generated from Kira
├── IBP_LI                          topology data (IBP + LI equation templates)
└── trivialsector                   trivial sector list

```

---

## A minimal worked example

Open `gen_eqs_and_solve.ipynb` at the top level. It walks through the basic
`pyfeyngym` workflow on this topology — building IBP/LI equation templates from
`IBP_LI`, looking up trivial sectors, generating equations for a chosen seed
set, and solving them over a finite field — both without a cut and on a cut.
Run from this directory so the bare `"IBP_LI"` / `"trivialsector"` paths resolve.

---

## evolutionary_strategies/

### Maximal-cut CMA-ES (Sec. 4.1)

```bash
cd evolutionary_strategies
bash run_cma_es.sh
```

Runs `optimize-ordering-NN` (the `pyfeyngym` CLI entry point) with CMA-ES to
find an optimal seed ordering for reducing `(1,...,1,0,0,-12)` on the maximal
cut. Outputs are written to `maxhigh/resorted_seed_op_list.txt` and
`maxhigh/seeds.png`. The two post-processing helpers can also be run
independently:

```bash
python summarize_maxcut_seeds.py
python plot_seeds.py --seed-op-file maxhigh/resorted_seed_op_list.txt \
    --export-figure maxhigh/seeds.png \
    --title "Maximal-cut seeds for reducing (1,...,1,0,0,-12)"
```

### Multi-ISP seed selection with CMA-ES (Sec. 4.3.1)

```bash
cd evolutionary_strategies/cma_es_seed_selection
python cmaes_seed_select_066.py --workers 16 --popsize 16 --max-evals 2000
```

The five scripts cover the targets and cuts behind the multi-ISP seed-set
figures: maximal cut for `(0,-6,-6)` and `(-8,-6,-6)`, and hexa cut
`[1,3,4,6,7,8]` for `(0,-6,-6)` and `(0,-8,-8)`. Run any script with `--help`
for the full option list. Results (`result.json`, `best_x.npy`) are written to
`outputs/` next to the script.

---

## rank20/

### Rank-20 reductions on a quadruple cut (Sec. 4.2.4, 4.3.2)

Open `high_rank.ipynb`. Reduces `(1,...,1,0,0,-20)` (single-ISP) and
`(1,...,1,-8,-6,-6)` (three-ISP) on the quadruple cut `[1,3,6,8]`.
Run the notebook from inside `rank20/` so the bare `"IBP_LI"` path resolves.

### 3-ISP rank-20 reduction on all spanning cuts (Sec. 4.3.2)

```bash
cd rank20
bash run_all_cuts.sh
```

Drives `combined_solve.py` across all 11 spanning cuts. `combined_solve.py`
imports `high_rank_one_cut_firstpass.py` (primary zigzag tube seeds) and
`high_rank_one_cut_secondpass.py` (leftover seeds). Per-cut logs are written to
`log_combined_cut{cut_lines}.txt` (e.g. `log_combined_cut246.txt`). Each log
contains resource consumption information and final IBP reduction results at
numerical dimension and kinematic values over a finite field. This is the full
reduction reported in Sec. 4.3.2.

---

## rank10/

Shared module: `path_coverage_rank10.py` defines the rank rule, the 66
rank-10 targets, the zigzag tube paths, and exports `IBP_FILE`/`TRIV_FILE`
pointing to the `IBP_LI`/`trivialsector` copies in `rank10/`.

### Reducing all 66 rank-10 targets individually on a quadruple cut (App. B)

```bash
cd rank10/all_rank10_quad
python reduce_all_rank10_individual.py
```

### Path coverage on a quadruple cut (Sec. 4.3.3 cover table)

```bash
cd rank10/all_rank10_cover
bash run_per_path_rss.sh          # 66 workers, one per path
bash run_phase3_pinned10.sh       # 10 reps × 5 bottleneck paths, CPU-pinned
python analyze_path_coverage.py
python analyze_minmax_cover.py    # requires pulp
```

`per_path_rss_worker.py` records which targets each path reduces and its peak
memory. `run_per_path_rss.sh` runs it over all 66 paths. `run_phase3_pinned10.sh`
re-measures the five chosen cover paths with CPU pinning (ten repeats each) for
stable memory numbers. `analyze_path_coverage.py` and `analyze_minmax_cover.py`
then select the bottleneck-optimal five-path cover.

### Path coverage across all spanning cuts (Sec. 4.3.3)

```bash
cd rank10/all_rank10_spanning
bash run_rank10_allcuts.sh        # 5 paths × 11 cuts, CPU-pinned
bash run_braced_central.sh        # braced W=4 for prop-4 cut (2,4,6)
bash run_braced_remaining.sh      # braced W=4 for remaining prop-4 cuts
bash run_improved_triples.sh      # decreasing-rank baseline on triple cuts
```

`per_path_rss_worker_allcuts.py` extends `per_path_rss_worker.py` to all
spanning cuts, reading the cut from the `CUT` environment variable (set by the
launcher scripts). `all66_improved_decr_allcuts.py` measures the decreasing-rank
baseline across all triple cuts, driven by `run_improved_triples.sh`;
`run_improved_oom2.sh` reruns the two most memory-heavy cuts (`[1,5,6]` and
`[3,5,8]`) under a RAM monitor after expanding the memory budget. This is the
data of Sec. 4.3.3.

`improved_decreasing_worker.py` (single target) and
`all66_improved_decreasing_worker.py` (all 66 rank-10 targets), located in
`benchmarks/`, measure the decreasing-rank baseline on its own for direct
comparison with the zigzag tube strategy.

---

## benchmarks/

The heavy solves are memory-intensive; pin a worker to a single core
(`taskset -c <cpu>` or `numactl --physcpubind=<cpu>`) and watch available memory.

### Linear scaling on a quadruple cut (Sec. 4.2.3)

```bash
cd benchmarks
SCAN_N=5,10,15,20 python scan_n_feyngym_metrics.py
python aggregate_rerun.py
```

`scan_n_feyngym_metrics.py` reduces the single-ISP target `(1,...,1,0,0,-n)` on
the quadruple cut `[1,3,6,8]` over a range of ranks `n`, recording seed count,
number of equations, generation/solve time and peak memory (averaged over
repeated runs). Results are written to `scan_n_feyngym_results_*.json`.
`aggregate_rerun.py` fits these to straight lines. This is the linear-scaling
data of Sec. 4.2.3 (the quadruple-cut benchmark table and plot).

### Tube vs decreasing-rank seeding, single-ISP (Sec. 4.2.3)

```bash
cd benchmarks
python scan_n_paper_metrics.py
```

Runs the same target with both tube seeding and conventional decreasing-rank
seeding and reports them side by side. Reproduces the head-to-head comparison
of Sec. 4.2.3.

### Full reduction across spanning cuts, single ISP (Sec. 4.2.4)

```bash
cd benchmarks
bash run_phase2_pinned10.sh       # 10 cuts × 10 reps, each pinned
python aggregate_triple_rerun.py
```

`scan_triple_cuts.py` reduces `(1,...,1,0,0,-n)` on one spanning cut (chosen
by the `CUT_INDEX` environment variable) for `n=5..20`; `run_phase2_pinned10.sh`
launches all ten triple cuts in parallel, each pinned to its own core, and
`aggregate_triple_rerun.py` collects the per-cut timings and memory. This is the
data behind the spanning-cut summary table and the per-cut plots of Sec. 4.2.4.

`scan_triple_cuts.py` writes an inline worker to a subdirectory
`triple_{cut}_{n}/worker.py` and spawns it as a subprocess.

---

## optimize_seeding/

Subsector-exception search (Sec. 4.2.5). Starting from decreasing-rank seeding
with `s_max=2`, the scripts walk through subsectors in order of how many
propagators are uncut — one uncut propagator, then two, then three — and for
each subsector find the smallest seeding rank that still reduces the target,
re-solving the full system at every step so the running configuration always
closes. In the code a subsector is keyed by its set of uncut propagators and
referred to as a "corner" of that order; its seeding rank is the `length`. The
resulting per-subsector exceptions are what make the optimized `s_max=2`
strategies grow more slowly with rank than the plain `s_max=4` tube — the
seed-count formulas of Sec. 4.2.5.

```bash
cd optimize_seeding
python quad_optimize.py           # quadruple cut [1,3,6,8]
python three_uncut_optimize.py    # triple cuts
```

Starting point: `es_seeds_maxcut_rank12.txt` (maximal-cut seed lattice at rank 12).

---

## validation/

```bash
cd validation
python validate_tube_reduction.py
```

Checks that tube seeding gives the genuine reduction and not a finite-field
artifact: for `(1,...,1,0,0,-n)` on the quadruple cut it verifies, at several
primes and kinematic points, that the target closes to the master basis with no
extra survivors and that the reconstructed master coefficients agree term by term
with the much larger decreasing-rank seed set. This is the cross-check of
Sec. 4.2.2.

---

## kira_equation_template_generation/

The `IBP_LI` file was generated from [Kira](https://gitlab.com/kira-pyred/kira)
via `kira_equation_template_generation/`. To regenerate:

```bash
cd kira_equation_template_generation
kira initiate.yaml
bash generate_IBP_LI.sh
cp IBP_LI ../
```

`kira initiate.yaml` reads the topology definition in `config/`
(`integralfamilies.yaml` and `kinematics.yaml`). Because the job sets
`run_initiate: true`, Kira only generates the IBP and Lorentz-invariance (LI)
identities and the trivial sectors — it does **not** perform the reduction.
This writes `sectormappings/doublePentagon/IBP` and `.../LI`.

`generate_IBP_LI.sh` concatenates those two files into `IBP_LI`, renames the
Mandelstam invariants (`s23→m1`, `s34→m2`, `s45→m3`, `s51→m4`), and removes
Kira's temporary files (`kira.log`, `results/`, `tmp/`, `sectormappings/`).
Note that it deletes `sectormappings/` as part of the cleanup, so it must be
run *after* `kira initiate.yaml`.

The renaming was done for the historic reason that an earlier version of the
Python code for generating IBP equations expected all variables to begin with
`m`. This restriction has since been lifted, but the existing scripts and
notebooks have not been updated to use the more natural variable names.

