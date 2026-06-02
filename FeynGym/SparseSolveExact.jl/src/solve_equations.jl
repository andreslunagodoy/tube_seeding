"""
    eq_to_rule(eq::SparseVec{IndexType, T}, eliminated::IndexType)

Turn an equation, i.e. a sparse vector, `eq`, into a replacement rule which expresses the variable `eliminated` in terms of other variables.
"""
function eq_to_rule(eq::SparseVec{IndexType, T}, eliminated::IndexType)::Pair{IndexType, SparseVec{IndexType, T}} where {IndexType, T}
    coeff = get(eq.data, eliminated, zero(T))
    @assert coeff != zero(T)
    minus_coeff_inv = - inv(coeff)
    newdata = deepcopy(eq.data)
    delete!(newdata, eliminated)
    for key in keys(newdata)
        newdata[key] = minus_coeff_inv * newdata[key]
    end
    return eliminated => SparseVec{IndexType, T}(newdata)
end

"""
    solve_eqs(eqs::Vector{SparseVec{IndexType, T}}, variables::Vector{IndexType}, run_back_subst = true, keep_on_rhs = nothing)

A wrapper function for solving a system of linear equations, `eqs`, w.r.t. `variables`. The functions constructs a sparse matrix internally and performs echelonization, and returns the result as a dictionary of solution rules.
"""
function solve_eqs(
        eqs::Vector{SparseVec{IndexType, T}}, variables::Vector{IndexType};
        run_back_subst = true,
        keep_on_rhs::Union{Nothing, Vector{IndexType}} = nothing,
        complete_pivoting = false,
        naive_pivoting = false,
        return_info::Bool = false
    ) where {IndexType, T}
    variables_to_numbering = Dict{IndexType, Int}(variables[i] => i for i in eachindex(variables))
    sparsevecs = map(Base.Fix1(map_sparsevec_indices, variables_to_numbering), eqs)
    log = Val(return_info)
    if keep_on_rhs isa Nothing
        mat = SparseMat{Int, T}(sparsevecs, length(sparsevecs), length(variables))
        pivs = if naive_pivoting
            echelonize!(mat, pivotfunc = findpivot_no_optimization!, log = log)
        elseif !complete_pivoting
            echelonize!(mat, pivotfunc = findpivot_partial!, log = log)
        else
            echelonize!(mat, pivotfunc = findpivot!, log = log)
        end
    else
        @assert keep_on_rhs isa Vector{IndexType}
        forbidden_col_pivots = Set(variables_to_numbering[var] for var in keep_on_rhs)
        mat = SparseMat{Int, T}(sparsevecs, length(sparsevecs), length(variables), forbidden_col_pivots)
        if naive_pivoting
            @warn "naive_pivoting is ignored when keep_on_rhs is provided, since reorder must be enabled in the presence of forbidden pivots."
        end
        pivs = if !complete_pivoting
            echelonize!(mat, pivotfunc = findpivot_partial_with_forbidden!, log = log)
        else
            echelonize!(mat, pivotfunc = findpivot_with_forbidden!, log = log)
        end
    end
    if run_back_subst
        back_subst!(mat, pivs)
    end
    solution_dict = Dict(
        eq_to_rule(
                map_sparsevec_indices(variables, mat.row_vectors[i]),
                variables[pivs.pivotcols[i]]
            ) for
            i in eachindex(pivs.pivotcols)
    )
    if return_info
        eqs_in_order = pivs.pivotrows
        vars_in_order = vcat(pivs.pivotcols, setdiff(1:length(variables), pivs.pivotcols))
        return solution_dict, mat.forward_elimination_cost, eqs_in_order, vars_in_order
    else
        return solution_dict
    end
end

"""
    solve_eqs(eqs_packed::Vector{Vector{Pair{IndexType, T}}}, variables::Vector{IndexType}, run_back_subst = true, keep_on_rhs = nothing)

A wrapper function for solving a system of linear equations in a packed format, `eqs_packed`, w.r.t. `variables`. The functions constructs a sparse matrix internally and performs echelonization, and returns the result as a dictionary of solution rules.
"""
function solve_eqs(
        eqs_packed::Vector{Vector{Pair{IndexType, T}}}, variables::Vector{IndexType};
        run_back_subst = true,
        keep_on_rhs::Union{Nothing, Vector{IndexType}} = nothing,
        complete_pivoting = false,
        return_info::Bool = false
    ) where {IndexType, T}
    eqs = map(SparseVec, eqs_packed)
    return solve_eqs(
        eqs, variables; run_back_subst = run_back_subst, keep_on_rhs = keep_on_rhs,
        complete_pivoting = complete_pivoting, return_info = return_info
    )
end

"""
    solve_eqs_nopivoting(eqs::Vector{SparseVec{IndexType, T}}, variables::Vector{IndexType}, run_back_subst = true, keep_on_rhs = nothing)

Similar to `solve_eqs`, but requires `eqs` to be pre-ordered (in rows and columns) so that diagonal pivots can be used in every step.
"""
function solve_eqs_nopivoting(
        eqs::Vector{SparseVec{IndexType, T}}, variables::Vector{IndexType};
        run_back_subst = true,
        return_info::Bool = false
    ) where {T, IndexType}
    variables_to_numbering = Dict{IndexType, Int}(variables[i] => i for i in eachindex(variables))
    sparsevecs = map(Base.Fix1(map_sparsevec_indices, variables_to_numbering), eqs)
    mat = SparseMat{Int, T}(sparsevecs, length(sparsevecs), length(variables))
    pivs = echelonize_nopivoting!(mat)
    if run_back_subst
        back_subst!(mat, pivs)
    end
    solution_dict = Dict(
        eq_to_rule(
                map_sparsevec_indices(variables, mat.row_vectors[i]),
                variables[pivs.pivotcols[i]]
            ) for
            i in eachindex(pivs.pivotcols)
    )
    if return_info
        eqs_in_order = pivs.pivotrows
        vars_in_order = vcat(pivs.pivotcols, setdiff(1:length(variables), pivs.pivotcols))
        return solution_dict, mat.forward_elimination_cost, eqs_in_order, vars_in_order
    else
        return solution_dict
    end
end
