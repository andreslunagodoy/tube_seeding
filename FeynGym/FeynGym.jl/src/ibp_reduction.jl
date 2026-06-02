"""
    MAX_N_INTEGRALS_DEFAULT

Default sparse-vector length used when no tighter upper bound on the number of
integrals is supplied.
"""
const MAX_N_INTEGRALS_DEFAULT = 10^9

"""
    AbstractReductionState{N, CoeffType}

Common supertype for IBP reduction states over `N`-index integrals with
coefficients of type `CoeffType`.
"""
abstract type AbstractReductionState{N, CoeffType} end

"""
    ReductionState{N, CoeffType}

State for incrementally inserting IBP equations and choosing eliminated
integrals. Equations are stored as sparse vectors over the state's integral
numbering.
"""
struct ReductionState{N, CoeffType} <: AbstractReductionState{N, CoeffType}
    integral_list::Vector{Integral{N}}
    integral_numbering_dict::Dict{Integral{N}, Int}
    elim_order::Vector{Int}
    integral_number_to_elim_order::Dict{Int, Int}
    ibp_equations::Vector{SparseVector{CoeffType, Int}}
    ibp_equation_lengths::Vector{Int}
    integral_appearance_count::Vector{Int}
    fixed_integral_numbering::Bool
    max_n_integrals::Int
end

"""
    ReductionStateWithTargets{N, CoeffType}

Reduction state that also tracks target expressions and master integrals. Each
chosen reduction rule is immediately applied to the targets, allowing callers
to stop once all targets are reduced to only master integrals.
"""
struct ReductionStateWithTargets{N, CoeffType} <: AbstractReductionState{N, CoeffType}
    integral_list::Vector{Integral{N}}
    integral_numbering_dict::Dict{Integral{N}, Int}
    elim_order::Vector{Int}
    integral_number_to_elim_order::Dict{Int, Int}
    ibp_equations::Vector{SparseVector{CoeffType, Int}}
    ibp_equation_lengths::Vector{Int}
    integral_appearance_count::Vector{Int}
    fixed_integral_numbering::Bool
    max_n_integrals::Int
    targets::Vector{SparseVector{CoeffType, Int}}
    masters::Set{Integral{N}}
end

"""
    ReductionState{N, CoeffType}(integral_list, fixed_integral_numbering, max_n_integrals=MAX_N_INTEGRALS_DEFAULT)

Create an empty reduction state. When `fixed_integral_numbering` is true, every
integral appearing in later equations must already be present in `integral_list`.
"""
function ReductionState{N, CoeffType}(
        integral_list::Vector{Integral{N}}, fixed_integral_numbering::Bool,
        max_n_integrals = MAX_N_INTEGRALS_DEFAULT
    ) where {N, CoeffType}
    integral_numbering_dict = Dict(integral => i for (i, integral) in enumerate(integral_list))
    elim_order = Int[]
    integral_number_to_elim_order = Dict{Int, Int}()
    ibp_equations = SparseVector{CoeffType, Int}[]
    ibp_equation_lengths = Int[]
    integral_appearance_count = Int[]
    return ReductionState{N, CoeffType}(
        integral_list, integral_numbering_dict, elim_order, integral_number_to_elim_order,
        ibp_equations, ibp_equation_lengths, integral_appearance_count,
        fixed_integral_numbering, max_n_integrals
    )
end

"""
    ReductionState{N, CoeffType}(integral_list, max_n_integrals=MAX_N_INTEGRALS_DEFAULT)

Create a reduction state with a fixed integral numbering from `integral_list`.
"""
function ReductionState{N, CoeffType}(
        integral_list::Vector{Integral{N}},
        max_n_integrals = MAX_N_INTEGRALS_DEFAULT
    ) where {N, CoeffType}
    return ReductionState{N, CoeffType}(integral_list, true, max_n_integrals)
end

"""
    ReductionState{N, CoeffType}(max_n_integrals=MAX_N_INTEGRALS_DEFAULT)

Create a reduction state with dynamic integral registration.
"""
function ReductionState{N, CoeffType}(max_n_integrals = MAX_N_INTEGRALS_DEFAULT) where {N, CoeffType}
    return ReductionState{N, CoeffType}(Integral{N}[], false, max_n_integrals)
end

"""
    ReductionStateWithTargets{N, CoeffType}(integral_list, targets, masters,
                                            fixed_integral_numbering,
                                            max_n_integrals=MAX_N_INTEGRALS_DEFAULT)

Create a target-tracking reduction state. Targets can be supplied as sparse
vectors or as single integrals, and `masters` defines which integrals are
considered terminal.
"""
function ReductionStateWithTargets{N, CoeffType}(
        integral_list::Vector{Integral{N}},
        targets::Vector{<:Union{SparseVector{CoeffType, Int}, Integral{N}}},
        masters::Union{Set{Integral{N}}, Vector{Integral{N}}},
        fixed_integral_numbering::Bool,
        max_n_integrals = MAX_N_INTEGRALS_DEFAULT
    ) where {N, CoeffType}
    masters = if masters isa Vector
        Set{Integral{N}}(masters)
    else
        masters
    end
    ilist = copy(integral_list)
    integral_numbering_dict = Dict(integral => i for (i, integral) in enumerate(ilist))
    elim_order = Int[]
    integral_number_to_elim_order = Dict{Int, Int}()
    ibp_equations = SparseVector{CoeffType, Int}[]
    ibp_equation_lengths = Int[]
    integral_appearance_count = Int[]

    # Normalize targets to SparseVector form
    svec_targets = Vector{SparseVector{CoeffType, Int}}(undef, length(targets))
    for (k, t) in enumerate(targets)
        if t isa Integral{N}
            idx = get(integral_numbering_dict, t, 0)
            if idx == 0
                if fixed_integral_numbering
                    error("Target integral not present in the integral list")
                else
                    push!(ilist, t)
                    idx = length(ilist)
                    integral_numbering_dict[t] = idx
                end
            end
            svec_targets[k] = SparseVector(max_n_integrals, [idx], [one(CoeffType)])
        else
            tv = copy(t)::SparseVector{CoeffType, Int} # avoid mutating the input target vector
            svec_targets[k] = tv.n == max_n_integrals ? tv : SparseVector(max_n_integrals, tv.nzind, tv.nzval)
        end
    end

    return ReductionStateWithTargets{N, CoeffType}(
        ilist, integral_numbering_dict, elim_order, integral_number_to_elim_order,
        ibp_equations, ibp_equation_lengths, integral_appearance_count,
        fixed_integral_numbering, max_n_integrals, svec_targets, masters
    )
end

"""
    ReductionStateWithTargets{N, CoeffType}(integral_list, targets, masters,
                                            max_n_integrals=MAX_N_INTEGRALS_DEFAULT)

Create a target-tracking reduction state with a fixed integral numbering.
"""
function ReductionStateWithTargets{N, CoeffType}(
        integral_list::Vector{Integral{N}},
        targets::Vector{<:Union{SparseVector{CoeffType, Int}, Integral{N}}},
        masters::Union{Set{Integral{N}}, Vector{Integral{N}}},
        max_n_integrals = MAX_N_INTEGRALS_DEFAULT
    ) where {N, CoeffType}
    return ReductionStateWithTargets{N, CoeffType}(integral_list, targets, masters, true, max_n_integrals)
end

"""
    register_integral!(s, i)

Add integral `i` to the state's numbering if it is not already present.
"""
function register_integral!(s::AbstractReductionState{N, CoeffType}, i::Integral{N}) where {N, CoeffType}
    if !haskey(s.integral_numbering_dict, i)
        push!(s.integral_list, i)
        s.integral_numbering_dict[i] = length(s.integral_list)
    end
    return
end

"""
    register_all_integrals!(s, eq)

Register every integral appearing in the dense equation `eq`.
"""
function register_all_integrals!(s::AbstractReductionState{N, CoeffType}, eq::Eq{N, CoeffType}) where {N, CoeffType}
    for (i, _) in eq
        register_integral!(s, i)
    end
    return
end

# all integrals in `eq` must be registered
"""
    _convert_eq_to_sparse_vector(eq, s)

Convert a dense `Eq` into the sparse-vector representation used by reduction
states. All integrals in `eq` must already be registered in `s`.
"""
function _convert_eq_to_sparse_vector(eq::Eq{N, CoeffType}, s::AbstractReductionState{N, CoeffType}) where {N, CoeffType}
    numbering = Int[]
    coeff_list = CoeffType[]
    for (integral, coeff) in eq
        integral_number = s.integral_numbering_dict[integral]
        push!(numbering, integral_number)
        push!(coeff_list, coeff)
    end
    numbering_coeff_list = zip(numbering, coeff_list) |> collect |> sort
    numbering_sorted = map(Base.Fix2(Base.getindex, 1), numbering_coeff_list)
    coeff_list_sorted = map(Base.Fix2(Base.getindex, 2), numbering_coeff_list)
    return SparseVector(s.max_n_integrals, numbering_sorted, coeff_list_sorted)
end

"""
    convert_sparse_vector_to_eq(vec, s)

Convert a sparse equation vector back into a dense list of integral/coefficient
pairs using the state's integral numbering.
"""
function convert_sparse_vector_to_eq(
        vec::SparseVector{CoeffType, Int},
        s::AbstractReductionState{N, CoeffType}
    ) where {N, CoeffType}
    return [s.integral_list[ind] => val for (ind, val) in zip(vec.nzind, vec.nzval)]
end

"""
    _reduce_eq(eq_vec, s)

Reduce `eq_vec` by all elimination rules already stored in `s`. Returns the
reduced sparse equation and a simple arithmetic-operation cost estimate.
"""
function _reduce_eq(eq_vec::EqVector{CoeffType}, s::AbstractReductionState{N, CoeffType}) where {N, CoeffType}
    reduced_eq_vec = deepcopy(eq_vec)
    integral_number_to_reduce = 0
    candidate_eq_to_reduce_against = 0
    final_eq_to_reduce_against = 0
    integral_number_to_be_eliminated = 0
    reduction_cost = 0
    while true
        need_reduction = false
        for (integral_number, coeff) in zip(reduced_eq_vec.nzind, reduced_eq_vec.nzval)
            if haskey(s.integral_number_to_elim_order, integral_number)
                candidate_eq_to_reduce_against = s.integral_number_to_elim_order[integral_number]
                if !need_reduction || candidate_eq_to_reduce_against < final_eq_to_reduce_against
                    need_reduction = true
                    final_eq_to_reduce_against = candidate_eq_to_reduce_against
                    integral_number_to_be_eliminated = integral_number
                end
            end
        end
        if !need_reduction
            break
        end
        reduced_eq_vec = reduced_eq_vec -
            reduced_eq_vec[integral_number_to_be_eliminated] * s.ibp_equations[final_eq_to_reduce_against]
        reduction_cost += length(s.ibp_equations[final_eq_to_reduce_against].nzind)
    end
    return reduced_eq_vec, reduction_cost
end

"""
    insert_eq!(s, eq, verbose=false)

Register and reduce an IBP equation, then store it if it is not reduced to zero.
Returns a named tuple with `created_nonzero_equation` and `reduction_cost`.
"""
function insert_eq!(s::AbstractReductionState{N, CoeffType}, eq::Eq{N, CoeffType}, verbose = false) where {N, CoeffType}
    if !s.fixed_integral_numbering
        register_all_integrals!(s, eq)
    end
    eq_vec = _convert_eq_to_sparse_vector(eq, s)
    if verbose
        print("IBP equation: ")
        print_as_equation(s.integral_list, eq_vec)
    end
    reduced_eq_vec, reduction_cost = _reduce_eq(eq_vec, s)
    if verbose
        print("IBP equation reduced by previous rules: ")
        print_as_equation(s.integral_list, reduced_eq_vec)
    end
    if length(reduced_eq_vec.nzind) > 0
        push!(s.ibp_equations, reduced_eq_vec)
        push!(s.ibp_equation_lengths, length(reduced_eq_vec.nzind))
        return (; created_nonzero_equation = true, reduction_cost = reduction_cost)
    else
        return (; created_nonzero_equation = false, reduction_cost = reduction_cost)
    end
end

"""
    print_as_equation(integral_list, v)

Print a sparse vector as a human-readable homogeneous equation in `G[...]`
notation.
"""
function print_as_equation(integral_list, v::SparseVector)
    printed_one_term = false
    for (integral_number, coeff) in zip(v.nzind, v.nzval)
        coeff = denominator(coeff) == 1 ? numerator(coeff) : coeff
        integral = integral_list[integral_number]
        if printed_one_term
            print("+ ")
        end
        print("(" * replace(string(coeff), "//" => "/") * ")*", "G[$(integral[1]),$(integral[2])] ")
        printed_one_term = true
    end
    println("== 0")
    return
end

"""
    print_as_rule(integral_list, v, eliminated_integral)

Print a normalized sparse equation as a replacement rule for
`eliminated_integral`.
"""
function print_as_rule(integral_list, v::SparseVector, eliminated_integral)
    encountered_eliminated_integral = false
    @assert length(v.nzind) > 0
    print("G[$(eliminated_integral[1]),$(eliminated_integral[2])] -> ")
    if length(v.nzind) == 1
        println(0)
        return
    end
    printed_one_term = false
    for (integral_number, coeff) in zip(v.nzind, v.nzval)
        coeff = denominator(coeff) == 1 ? numerator(coeff) : coeff
        integral = integral_list[integral_number]
        if integral == eliminated_integral
            encountered_eliminated_integral = true
            @assert coeff == 1
            continue # do not print the term on RHS
        end
        if printed_one_term
            print("+ ")
        end
        print("(" * replace(string(-coeff), "//" => "/") * ")*", "G[$(integral[1]),$(integral[2])] ")
        printed_one_term = true
    end
    println()
    return
end

"""
    reduce_eq(eq, s)

Return the equation obtained by reducing `eq` against the existing elimination
rules in `s` without mutating `s`.
"""
function reduce_eq(eq::Eq{N, CoeffType}, s1::AbstractReductionState{N, CoeffType}) where {N, CoeffType}
    s = ReductionState{N, CoeffType}(
        deepcopy(s1.integral_list), deepcopy(s1.integral_numbering_dict),
        s1.elim_order, s1.integral_number_to_elim_order,
        s1.ibp_equations, s1.ibp_equation_lengths,
        s1.integral_appearance_count, s1.fixed_integral_numbering,
        s1.max_n_integrals
    )
    if !s.fixed_integral_numbering
        register_all_integrals!(s, eq)
    end
    eq_vec = _convert_eq_to_sparse_vector(eq, s)
    (reduced_eq_vec, _) = _reduce_eq(eq_vec, s)
    return convert_sparse_vector_to_eq(reduced_eq_vec, s)
end

"""
    _unsafe_choose_integral_number_to_elim!(s, to_elim)

Normalize the most recently inserted equation so that integral number
`to_elim` is eliminated by it. This helper assumes all validity checks have
already been performed.
"""
function _unsafe_choose_integral_number_to_elim!(s::AbstractReductionState, to_elim::Int)
    push!(s.elim_order, to_elim)
    s.integral_number_to_elim_order[to_elim] = length(s.elim_order)
    s.ibp_equations[end] = s.ibp_equations[end] // s.ibp_equations[end][to_elim]
    return
end

"""
    choose_integral_to_elim!(s, integral)

Turn the most recently inserted nonzero equation into a reduction rule that
eliminates `integral`. For target-tracking states, also applies the rule to the
targets and returns whether reduction is still incomplete.
"""
function choose_integral_to_elim!(s::AbstractReductionState{N, CoeffType}, integral::Integral{N}) where {N, CoeffType}
    if length(s.elim_order) != length(s.ibp_equations) - 1
        error("There must be exactly one stored IBP equation without a specified\
 integral that is eliminated by the equation")
    end
    integral_number = get(s.integral_numbering_dict, integral, -1)
    if integral_number == -1
        error("Supplied integral not present in the integral list")
    end
    if !(integral_number in last(s.ibp_equations).nzind)
        error("Integral not present in the last IBP equation")
    end

    _unsafe_choose_integral_number_to_elim!(s, integral_number)

    if s isa ReductionStateWithTargets{N, CoeffType}
        reduction_incomplete = false    
        last_rule = last(s.ibp_equations)
        for i in eachindex(s.targets)
            t_vec = s.targets[i]
            coeff = t_vec[integral_number]
            if coeff != zero(CoeffType)
                s.targets[i] = t_vec - coeff * last_rule
            end
            if !reduction_incomplete && has_non_master_integrals(s.targets[i], s)
                reduction_incomplete = true
            end
        end
        return reduction_incomplete
    else
        return true
    end
end

"""
    has_non_master_integrals(vec, s)

Return whether sparse expression `vec` contains any integral that is not listed
as a master in the target-tracking state `s`.
"""
function has_non_master_integrals(vec::SparseVector{CoeffType, Int}, s::ReductionStateWithTargets{N, CoeffType}) where {N, CoeffType}
    for ind in vec.nzind
        integral = s.integral_list[ind]
        if !(integral in s.masters)
            return true
        end
    end
    return false
end

"""
    auto_choose_integral_to_elim!(s, integral_greater_than)

Choose an eliminated integral from the most recent equation according to either
`:from_integral_list` or a comparison predicate, then call
`choose_integral_to_elim!`.
"""
function auto_choose_integral_to_elim!(s::AbstractReductionState{N, CoeffType},
    integral_greater_than) where {N, CoeffType}
    if length(s.elim_order) != length(s.ibp_equations) - 1
        error("There must be exactly one stored IBP equation without a specified integral\
 that is eliminated by the equation")
    end
    if last(s.ibp_equations).n <= 0
        error("The last equation must have length > 0")
    end
    most_complex_integral, most_complex_ind = nothing, 0
    for ind in last(s.ibp_equations).nzind
        integral = s.integral_list[ind]
        if most_complex_integral isa Nothing
            most_complex_integral, most_complex_ind = integral, ind
        else
            if integral_greater_than == :from_integral_list
                if ind < most_complex_ind # complex integrals have smaller indices in the integral list
                    most_complex_integral, most_complex_ind = integral, ind
                end
            else
                if integral_greater_than(integral, most_complex_integral)
                    most_complex_integral, most_complex_ind = integral, ind
                end
            end
        end
    end
    @assert !(most_complex_integral isa Nothing)
    # println("most_complex_integral = ", most_complex_integral)
    return choose_integral_to_elim!(s, most_complex_integral)
end

"""
    insert_eq_elim_one_integral!(s, eq, integral_greater_than)

Insert an equation and, when it remains nonzero after reduction, immediately
choose one integral from it to eliminate. Returns status, cost, and target
completion information.
"""
function insert_eq_elim_one_integral!(
        s::AbstractReductionState{N, CoeffType}, eq::Eq{N, CoeffType},
        integral_greater_than
    ) where {N, CoeffType}
    (; created_nonzero_equation, reduction_cost) = insert_eq!(s, eq)
    if created_nonzero_equation
        reduction_incomplete = auto_choose_integral_to_elim!(s, integral_greater_than)
        reduction_cost += length(last(s.ibp_equations).nzind)
        return (; created_nonzero_equation = true, reduction_cost = reduction_cost, reduction_incomplete = reduction_incomplete)
    else
        return (; created_nonzero_equation = false, reduction_cost = reduction_cost, reduction_incomplete = true)
    end
end
