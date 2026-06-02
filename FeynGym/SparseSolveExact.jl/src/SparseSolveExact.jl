module SparseSolveExact

import Base: show, hash

# forward Dict methods to SparseVec
import Base: getindex, setindex!, delete!, haskey, get, iterate, values, keys, length, empty!, sizehint!

import Base: convert

import Base: zero, one, ==, +, -, *, //, ^, inv, isless

include("sparse_vec.jl")
export SparseVec, to_sparse_vec, to_dense_vec, map_sparsevec_indices, map_sparsevec_values

include("sparse_mat.jl")
export SparseMat
export echelonize!, echelonize_nopivoting!, back_subst!, rref!, rref_nopivoting!
export findpivot!, findpivot_with_forbidden!, findpivot_partial!, findpivot_partial_with_forbidden!, findpivot_no_optimization!, FindPivotWithPreference
export reduce_with_preordered_equations!, reduce_with_ref_mat!
export trace_needed_equations!

include("abstract_field.jl")

include("finite_field.jl")
export FF

include("solve_equations.jl")
export solve_eqs, solve_eqs_nopivoting, eq_to_rule

include("python_interface.jl")
export solve_eqs_modulo

end # module SparseSolveExact
