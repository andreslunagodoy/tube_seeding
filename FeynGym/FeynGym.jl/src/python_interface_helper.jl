# Variants of constants and functions, using plain arrays instead of static arrays, for easier Python interop.
# const integral_list1 = [integral_list[i][j] for i in eachindex(integral_list), j in 1:2]

# const all_actions1 = [all_actions[i][j] for i in eachindex(all_actions), j in 1:3]

"""
    all_actions_python_friendly(all_actions)

Convert a vector of length-3 static-vector actions into a dense matrix-like
array whose rows are plain integer triples for easier Python interop.
"""
all_actions_python_friendly(all_actions) = [all_actions[i][j] for i in eachindex(all_actions), j in 1:3]

# const all_possibly_valid_actions1 = [all_possibly_valid_actions[i][j] for i in eachindex(all_possibly_valid_actions), j in 1:3]

"""
    action_log1(e)

Return an environment action log as a plain two-dimensional integer array,
instead of a vector of `SVector`s.
"""
function action_log1(e)
    l = e.action_log
    [l[i][j] for i in eachindex(l), j in 1:3]
end

"""
    run_reduction(eqs, integrals, targets, masters, modulus, cost_cutoff=-1)

Python-facing helper for running an ordered IBP reduction over a supplied list
of equations. Inputs use Python-friendly vectors; the function converts them to
Julia `Integral`, `Eq`, and finite-field types, then returns completion status,
total cost, number of equations consumed, and reduced targets.
"""
function run_reduction(eqs::AbstractVector, integrals::AbstractVector, targets::AbstractVector,
    masters::AbstractVector, modulus::Integer, cost_cutoff::Integer = -1)
    N = length(integrals[1])
    CoeffType = FF{modulus, Int}
    eqs = [[collect(term[1])::Vector{Int} => term[2] for term in eq] for eq in eqs]
    eqs = convert(Vector{Eq{N, CoeffType}}, eqs)
    integrals = convert(Vector{Integral{N}}, integrals)
    # targets = convert(Vector{<:Union{SparseVector{CoeffType, Int}, Integral{N}}}, targets)
    targets = if length(targets) == 0
        Integral{N}[]
    else
        [t isa SparseVector ? t : convert(Integral{N}, t) for t in targets]
    end
    masters = convert(Vector{Integral{N}}, masters)
    s = ReductionStateWithTargets{N, CoeffType}(integrals, targets, masters)
    total_reduction_cost = 0
    reduction_complete = false
    n_eqs_used = 0
    for eq in eqs
        (; created_nonzero_equation, reduction_cost, reduction_incomplete) = insert_eq_elim_one_integral!(s, eq, :from_integral_list)
        n_eqs_used += 1
        total_reduction_cost += reduction_cost
        if !reduction_incomplete
            reduction_complete = true
            break
        end
        if total_reduction_cost > cost_cutoff > 0
            total_reduction_cost = cost_cutoff
            break
        end
    end
    return (; reduction_complete = reduction_complete, total_reduction_cost = total_reduction_cost, n_eqs_used = n_eqs_used, targets_reduced = s.targets)
end
