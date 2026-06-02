"""
    function solve_eqs_modulo(eqs_packed::Vector{<:AbstractVector}, variables::Vector{IndexType}, modulus::Integer; run_back_subst = true, keep_on_rhs::Union{Nothing, Vector{IndexType}}

Convert input linear system to one over a 32-bit finite field and solve the system
"""
function solve_eqs_modulo(
        eqs_packed::AbstractVector, variables::AbstractVector,
        modulus::Integer; run_back_subst = true,
        keep_on_rhs::Union{Nothing, AbstractVector} = nothing,
        complete_pivoting = false,
        return_info::Bool = false,
        naive_pivoting = false,
        nopivoting = false
    )
    @assert length(eqs_packed) > 0 && length(variables) > 0
    @assert modulus > 0 && (modulus - 1)^2 + (modulus - 1) <= typemax(Int) "Modulus out of range!"
    fftype = FF{modulus, Int}
    eqs = [
        SparseVec(Dict(var => convert(fftype, coeff) for (var, coeff) in eq))
            for eq in eqs_packed
    ]
    variables1 = variables isa Vector ? variables : identity.(variables)
    solution = if !nopivoting
        keep_on_rhs1 = keep_on_rhs isa Union{Nothing, Vector} ? keep_on_rhs : identity.(keep_on_rhs)
        solve_eqs(
            eqs, variables1, run_back_subst = run_back_subst,
            keep_on_rhs = keep_on_rhs1,
            complete_pivoting = complete_pivoting,
            naive_pivoting = naive_pivoting,
            return_info = return_info
        )
    else
        if keep_on_rhs !== nothing
            @warn "keep_on_rhs is ignored when nopivoting = true"
        end
        solve_eqs_nopivoting(eqs, variables1, run_back_subst = run_back_subst, return_info = return_info)
    end
    if return_info
        solution_dict, cost, eqs_in_order, vars_in_order = solution
        return [(k, [(var, coeff.value) for (var, coeff) in v]) for (k, v) in solution_dict], cost, eqs_in_order, vars_in_order
    else
        return [(k, [(var, coeff.value) for (var, coeff) in v]) for (k, v) in solution]
    end
end
