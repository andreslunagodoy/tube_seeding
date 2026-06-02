using SafeTestsets

@safetestset "Sparse Vector Tests" include("sparsevec_tests.jl")

@safetestset "Sparse Matrix Tests" include("sparsemat_tests.jl")

@safetestset "Generic Rational Field Tests" include("abstractfield_tests.jl")

@safetestset "Finite Field Tests" include("finitefield_tests.jl")

@safetestset "Equation Solving Tests" include("solve_eqs_tests.jl")
