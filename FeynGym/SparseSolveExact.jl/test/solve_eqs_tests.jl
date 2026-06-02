using Test
using SparseSolveExact

v1 = SparseVec(["a" => big(2//1), "b" => big(3//1)]) # linear equation 2a + 3b == 0
v2 = SparseVec(["a" => big(1//1), "c" => big(-2//1)]) # a - 2c == 0
equations = [v1, v2]
variables = ["a", "b", "c"]

v10 = deepcopy(v1)
v20 = deepcopy(v2)
solution1 = solve_eqs(equations, variables)
@test solution1["b"] == SparseVec(["c" => big(-4//3)]) &&
      solution1["a"] == SparseVec(["c" => big(2//1)]) # solution is a=2c, b=(-4/3)c
@test v1 == v10 && v2 == v20 # solve_eqs should not mutate arguments

solution1a = solve_eqs_nopivoting(equations, variables)
@test solution1a["b"] == SparseVec(["c" => big(-4//3)]) &&
      solution1a["a"] == SparseVec(["c" => big(2//1)]) # solution is a=2c, b=(-4/3)c

solution2 = solve_eqs(equations, variables, keep_on_rhs = ["b"])
@test solution2["c"] == SparseVec(["b" => big(-3//4)]) &&
      solution2["a"] == SparseVec(["b" => big(-3//2)]) # solution is c=(-3/4)b, a=(-3/2)b

solution3 = solve_eqs(equations, variables, run_back_subst = false)
@test solution3["b"] == SparseVec(["c" => big(-4//3)]) &&
      solution3["a"] == SparseVec(["b" => big(-3//2)]) # solution is a=(-3/2)b, b=(-4/3)c. The RHS is not in terms of a minimal list of "master" variables, since `run_back_subst=false` is used