# SparseSolveExact.jl

SparseSolveExact.jl is a Julia package for solving sparse linear systems
over exact fields, such as finite fields, via Gaussian elimination.

In this repository, it serves the purpose of powering integration-by-parts
(IBP) reduction experiments in FeynGym for generic topologies other than
the one-loop bubble. The solver support tracking the costs of arithmetic
operations over the entire solve process or at individual steps, which
provides feedback for optimizing the IBP reduction.

The package provides:

- sparse row-vector and matrix types specialized for elimination;
- finite-field arithmetic through `FF{p, T}`;
- sparse pivoting heuristics that try to control fill-in;
- high-level equation-solving wrappers returning replacement rules;
- a Python-friendly `solve_eqs_modulo` interface used by `pyfeyngym`.

## Installation

From the package root (currently `FeynGym/SparseSolveExact.jl`), use the package as a local Julia project:

```bash
julia --project=.
```

Inside Julia:

```julia
using SparseSolveExact
```

When using FeynGym through Python, `pyfeyngym/install_julia_packages.py`
registers this local package with `juliapkg`, so users normally do not need to
install it separately.

## Data Model

An equation is represented as a sparse row vector:

```julia
using SparseSolveExact

const BigRat = Rational{BigInt}

eq = SparseVec([
    "a" => big(2//1),
    "b" => big(3//1),
])
```

This represents the homogeneous equation

```text
2a + 3b = 0
```

The key type can be any variable label type convenient for the problem, such as
`String`, `Symbol`, tuples of integral indices, or integers. The coefficient
type can be an exact field-like type such as `Rational{BigInt}` or `FF{p, T}`.

For lower-level elimination, sparse rows are collected into a `SparseMat`:

```julia
mat = SparseMat{Int, BigRat}([
    BigRat[3, 1, 0, -2],
    BigRat[2, 3, 1, 0],
])

pivots = rref!(mat)
```

Most users should prefer the higher-level `solve_eqs` API.

## Solving Equations

`solve_eqs(equations, variables)` solves a homogeneous linear system with
respect to an ordered variable list and returns a dictionary of replacement
rules.

```julia
using SparseSolveExact

const BigRat = Rational{BigInt}

eqs = [
    SparseVec(["a" => big(2//1), "b" => big(3//1)]),   # 2a + 3b = 0
    SparseVec(["a" => big(1//1), "c" => big(-2//1)]),  # a - 2c = 0
]
variables = ["a", "b", "c"]

solution = solve_eqs(eqs, variables)

solution["a"]  # SparseVec(["c" => 2])
solution["b"]  # SparseVec(["c" => -4//3])
```

By default the solver runs back-substitution, so right-hand sides are expressed
in terms of the remaining free variables. To keep selected variables on the
right-hand side, use `keep_on_rhs`:

```julia
solution = solve_eqs(eqs, variables, keep_on_rhs = ["b"])

solution["a"]  # a = -3//2 b
solution["c"]  # c = -3//4 b
```

For best performance, use immutable types such as integers or tuples, rather than strings, as variable names.

For already ordered systems where diagonal pivots are known to work, use
`solve_eqs_nopivoting`:

```julia
solution = solve_eqs_nopivoting(eqs, variables)
```

## Finite Fields

`FF{p, T}` represents a finite field modulo a prime `p`, with values stored in
integer type `T`.

```julia
const F = FF{2^31 - 1, Int}

a = convert(F, 4542//4135)
b = convert(F, 7193//314)

a + b
a * b
a // b
inv(a)
```

For the default machine-integer implementation, the modulus must satisfy

```text
(p - 1)^2 + (p - 1) <= typemax(Int)
```

so that multiplication-plus-addition operations stay in range.

## Pivoting And Cost Information

The high-level solver supports several pivoting options:

- default: sparsity-oriented partial pivoting;
- `complete_pivoting = true`: searches pivot columns more globally;
- `naive_pivoting = true`: uses a simple first-row/first-column strategy;
- `keep_on_rhs = [...]`: forbids selected variables from becoming pivots.

Pass `return_info = true` to get the arithmetic-operation cost and the row/column
orders chosen during elimination:

```julia
solution, cost, eqs_in_order, vars_in_order =
    solve_eqs(eqs, variables, return_info = true)
```

This is useful in FeynGym-style optimization, where different equation and
variable orderings can have very different solve costs.

## Python-Friendly Interface

`solve_eqs_modulo` accepts packed equations and returns plain Julia/Python
container data, making it convenient to call through `juliacall`.

```julia
eqs = [
    [("a", 2), ("b", 3)],
    [("a", 1), ("c", -2)],
]
variables = ["a", "b", "c"]
modulus = 2^31 - 1

solution = solve_eqs_modulo(eqs, variables, modulus)
```

Each returned entry has the form

```julia
(eliminated_variable, [(rhs_variable, coefficient), ...])
```

where coefficients are represented as integers modulo `modulus`. The keyword
arguments mirror the Julia solver where applicable:

```julia
solution, cost, eq_order, var_order = solve_eqs_modulo(
    eqs,
    variables,
    modulus;
    return_info = true,
    keep_on_rhs = ["b"],
)
```

Supported options include `run_back_subst`, `keep_on_rhs`,
`complete_pivoting`, `naive_pivoting`, `nopivoting`, and `return_info`.

## Low-Level Utilities

The package also exports lower-level routines for custom workflows:

- `SparseVec`, `to_sparse_vec`, `to_dense_vec`;
- `SparseMat`;
- `echelonize!`, `echelonize_nopivoting!`;
- `back_subst!`;
- `rref!`, `rref_nopivoting!`;
- pivot selectors such as `findpivot!`, `findpivot_partial!`,
  `findpivot_no_optimization!`, and `FindPivotWithPreference`;
- `reduce_with_ref_mat!` and `reduce_with_preordered_equations!`;
- `trace_needed_equations!` for tracing which original equations are needed to
  reduce selected rows.

## Tests

The available tests cover sparse vectors, sparse matrices, finite-field
arithmetic, generic rational arithmetic, and equation solving.

The standard Julia test command is:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
