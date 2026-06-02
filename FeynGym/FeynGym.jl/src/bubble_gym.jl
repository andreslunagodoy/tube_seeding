"""
    FLOAT_TYPE

Floating-point type used for observations and costs returned to learning code.
"""
const FLOAT_TYPE = Float32

"""
    n_ibp_operators

Number of non-elimination IBP seed operators in the bubble environment.
"""
const n_ibp_operators = 2

"""
    actual_data_length_per_integral

Number of observation channels stored for each integral.
"""
const actual_data_length_per_integral = n_ibp_operators+2

"""
    mask_data_length_per_integral

Number of action-mask channels stored for each integral.
"""
const mask_data_length_per_integral = n_ibp_operators+1

"""
    FiniteField

Finite field used for the default bubble-environment arithmetic.
"""
const FiniteField = FF{2^31-1, Int}

"""
    p2, m2, d

Default finite-field kinematic and dimension parameters used when generating
bubble IBP equations.
"""
const p2, m2, d = FiniteField(7381), FiniteField(1), FiniteField(1009)

"""
    masters

Master integrals treated as terminal targets in the default bubble environment.
"""
const masters = Integral{2}[[1,0], [0,1], [1,1]]

# const max_seed_propagator_power = 5 # maximum complexity of seed integral
# const integral_list_unsorted = [SVector(a,b) for a in -1:(max_seed_propagator_power+1) for b in -1:(max_seed_propagator_power+1) if a+b<=max_seed_propagator_power+1 && (a>0 || b>0) && !(a==1 && b==-1) && !(a==-1 && b==1)] # integrals (1,-1) and (-1,1) don't get generated in IBP equations
# const integral_list = sort(integral_list_unsorted, by=bubble_integral_complexity_info)
# const integral_list_length = length(integral_list)
# const integral_numbering_dict = Dict(integral => i for (i, integral) in enumerate(integral_list))

"""
    valid_seed(a, b, max_seed_propagator_power, board_shape)

Return whether `(a, b)` is a valid seed integral for the configured board.
Triangle boards impose `a + b <= max_seed_propagator_power`; square boards
bound each coordinate independently.
"""
function valid_seed(a, b, max_seed_propagator_power, board_shape)
    a>=0 && b>=0 && !(a==b==0) &&
        ((board_shape == "triangle" && a+b <= max_seed_propagator_power) ||
         (board_shape == "square" && a <= max_seed_propagator_power && b <= max_seed_propagator_power)
         )
end
# const seed_candidate_mask::Array{Bool} = [valid_seed(a, b) for (a,b) in integral_list, j in 1:n_ibp_operators]
# const seed_candidate_mask_2d::Array{Bool} = [valid_seed(a, b) for j in 1:n_ibp_operators, a in -1:(max_seed_propagator_power+1), b in -1:(max_seed_propagator_power+1)]

"""
    bubble_gym_episode_counter

Mutable counter incremented when a `BubbleEnv` is constructed or reset.
"""
const bubble_gym_episode_counter = Ref(0)

"""
    bubble_gym_step_counter

Mutable counter incremented on each environment action.
"""
const bubble_gym_step_counter = Ref(0)



# Const all_actions = [SVector(integral[1], integral[2], n) for integral in integral_list, n in 0:n_ibp_operators] |> vec
# const all_actions_2d = [[i, j, n] for n in 0:n_ibp_operators, i in -1:(max_seed_propagator_power+1), j in -1:(max_seed_propagator_power+1)] # using plain vector rather than SVector for easier Python interop
# const all_possibly_valid_actions = [SVector(integral[1], integral[2], n) for integral in integral_list, n in 0:n_ibp_operators if (n==0 && !(integral in masters)) || (valid_seed(integral[1], integral[2]) && n>0)] |> vec
# const possibly_valid_action_to_numbering = Dict(a => i for (i, a) in enumerate(all_possibly_valid_actions))

"""
    BubbleEnvProperties(max_seed_propagator_power, board_shape)

Immutable collection of geometry-dependent lookup tables for a bubble
environment. It stores the integral grid, integral numbering, seed masks, and
flat/2D action lists used by Julia and Python callers.
"""
struct BubbleEnvProperties
    max_seed_propagator_power::Int
    board_shape::String # either "triangle" or "square"
    integral_list::Vector{Integral{2}}
    integral_numbering_dict::Dict{Integral{2}, Int}
    seed_candidate_mask::Matrix{Bool}
    seed_candidate_mask_2d::Array{Bool, 3}
    all_actions::Vector{SVector{3, Int}}
    all_actions_2d::Array{Vector{Int}, 3}
    all_possibly_valid_actions::Vector{SVector{3, Int}}
    possibly_valid_action_to_numbering::Dict{SVector{3, Int}, Int}
    function BubbleEnvProperties(max_seed_propagator_power, board_shape)
        if board_shape != "triangle" && board_shape != "square"
            error("board_shape must be either \"triangle\" or \"square\"")
        end
        integral_list_unsorted = if board_shape == "triangle"
            [SVector(a,b) for a in -1:(max_seed_propagator_power+1) for b in -1:(max_seed_propagator_power+1) if a+b<=max_seed_propagator_power+1 && (a>0 || b>0)]
        else # square
            [SVector(a,b) for a in -1:(max_seed_propagator_power+1) for b in -1:(max_seed_propagator_power+1) if (a>0 || b>0)]
        end
        integral_list = sort(integral_list_unsorted, by=bubble_integral_complexity_info)      
        integral_numbering_dict = Dict(integral => i for (i, integral) in enumerate(integral_list))
        seed_candidate_mask::Array{Bool} = [valid_seed(a, b, max_seed_propagator_power, board_shape) for (a,b) in integral_list, j in 1:n_ibp_operators]
        seed_candidate_mask_2d::Array{Bool} = [valid_seed(a, b, max_seed_propagator_power, board_shape) for j in 1:n_ibp_operators, a in -1:(max_seed_propagator_power+1), b in -1:(max_seed_propagator_power+1)]
        all_actions = [SVector(integral[1], integral[2], n) for integral in integral_list, n in 0:n_ibp_operators] |> vec
        all_actions_2d = [[i, j, n] for n in 0:n_ibp_operators, i in -1:(max_seed_propagator_power+1), j in -1:(max_seed_propagator_power+1)] # using plain vector rather than SVector for easier Python interop
        all_possibly_valid_actions = [SVector(integral[1], integral[2], n) for integral in integral_list, n in 0:n_ibp_operators if (n==0 && !(integral in masters)) || (valid_seed(integral[1], integral[2], max_seed_propagator_power, board_shape) && n>0)] |> vec
        possibly_valid_action_to_numbering = Dict(a => i for (i, a) in enumerate(all_possibly_valid_actions))
        new(max_seed_propagator_power, board_shape, integral_list, integral_numbering_dict, seed_candidate_mask, seed_candidate_mask_2d, all_actions, all_actions_2d, all_possibly_valid_actions, possibly_valid_action_to_numbering)
    end
end

"""
    BubbleEnv(target_integral0 = SVector(6,6), skip_redundant_equation_cost = false,
              max_seed_propagator_power = 5, board_shape = "triangle")

Mutable IBP-reduction environment for the two-propagator massive bubble family.
The state alternates between adding a seed/operator IBP equation and choosing a
non-master integral from that equation to eliminate.
"""
mutable struct BubbleEnv
    target_integral0::SVector{2, Int}
    s::ReductionState
    to_reduce::Vector{SparseVector{FiniteField, Int}}
    seeded_step::Dict{SVector{3, Int}, Int} # length-3 vector contains the length-2 seed integral and the numbeing of the IBP opertor
    eliminated_step::Dict{Integral{2}, Int}
    to_eliminate_in_last_eq::Set{Integral{2}}
    terminated::Bool
    seed_count::Int
    elim_count::Int
    last_action::SVector{3, Int}
    skip_redundant_equation_cost::Bool
    action_log::Vector{SVector{3, Int}}
    properties::BubbleEnvProperties
    function BubbleEnv(target_integral0 = SVector(6,6), skip_redundant_equation_cost = false, max_seed_propagator_power = 5, board_shape = "triangle")
        target_integral = convert(SVector{2, Int}, target_integral0)
        bubble_gym_episode_counter[] += 1
        properties = BubbleEnvProperties(max_seed_propagator_power, board_shape)
        s = ReductionState{2, FiniteField}(properties.integral_list)
        to_reduce = [_convert_eq_to_sparse_vector([target_integral => FiniteField(1)], s)]
        new(target_integral0, s, to_reduce, Dict{Integral{2}, Int}(), Dict{Integral{2}, Int}(), Set{Integral{2}}(), false, 0, 0, SVector(0, 0, 0), skip_redundant_equation_cost, SVector{3,Int}[], properties)
    end
end

"""
    reset!(e::BubbleEnv)

Reset an existing environment to its initial state while preserving its target
integral and geometry properties.
"""
function reset!(e::BubbleEnv)
    bubble_gym_episode_counter[] += 1
    e.s = ReductionState{2, FiniteField}(e.properties.integral_list)
    e.to_reduce = [_convert_eq_to_sparse_vector([e.target_integral0 => FiniteField(1)], e.s)]
    empty!(e.seeded_step)
    empty!(e.eliminated_step)
    empty!(e.to_eliminate_in_last_eq)
    e.terminated = false
    e.seed_count = 0
    e.elim_count = 0
    empty!(e.action_log)
    e
end

"""
    act_automatic_elim!(e, a; comparison=bubble_integral_greater_than)

Choose a seed/operator pair, generate and reduce its IBP equation, then
automatically eliminate the most complex eligible integral from the new
equation. Returns the total cost of the seed and elimination actions.
"""
function act_automatic_elim!(e::BubbleEnv, a::SVector{3, Int}; comparison = bubble_integral_greater_than)
    @assert a[3] > 0 # a[3]==0 corresponds to choosing integral to eliminate when forming a new reduction rule,
    # which is automated in this function
    @assert length(e.s.elim_order) == length(e.s.ibp_equations)
    cost = 0f0
    cost += act!(e, a)
    if length(e.s.elim_order) < length(e.s.ibp_equations) # Created new nonzero equation, need to choose an integral in it to reduce
        max_integral = nothing
        for index in last(e.s.ibp_equations).nzind
            integral = e.properties.integral_list[index]
            if max_integral isa Nothing || comparison(integral, max_integral)
                max_integral = integral
            end
        end
        @assert !(max_integral isa Nothing)
        cost += act!(e, SVector(max_integral[1], max_integral[2], 0)) # choose this integral to eliminate
    end
    cost
end

"""
    act_automatic_with_seeds!(e, seeds; sorted=false, rev=false, rev_ops=false, comparison=bubble_integral_greater_than)

Apply all IBP operators to a list of seed integrals, automatically selecting an
integral to eliminate after each nonzero generated equation. Returns the total
reduction cost accumulated before termination or seed exhaustion.
"""
function act_automatic_with_seeds!(e::BubbleEnv, seeds; sorted =  false, rev=false, rev_ops=false, comparison = bubble_integral_greater_than)
    cost = 0f0
    seeds_in_order = sorted ? sort(seeds, by = bubble_integral_complexity_info) : seeds
    if rev
        reverse!(seeds_in_order)
    end
    for seed in seeds_in_order
        if e.terminated
            break
        end
        for op in (rev_ops ? reverse(1:n_ibp_operators) : (1:n_ibp_operators))
            if e.terminated
                break
            end
            cost += act_automatic_elim!(e, SVector(seed[1], seed[2], op), comparison=comparison)
        end
    end
    cost
end

"""
    _play_using_preference_scores(e, map_to_elim_action_index, scores; verbose=false)

Choose and apply the valid action with the highest supplied preference score.
This is the internal single-step implementation behind
`play_using_preference_scores`.
"""
function _play_using_preference_scores(e::BubbleEnv, map_to_elim_action_index, scores; verbose = false)
    chosen_action, max_score = nothing, nothing
    need_seeding = length(e.s.elim_order) == length(e.s.ibp_equations)
    if need_seeding
        for (i, a) in enumerate(e.properties.all_possibly_valid_actions)
            if a[3] != 0 && !haskey(e.seeded_step, a) && (max_score isa Nothing || scores[i] > max_score)
                chosen_action, max_score = a, scores[i]
            end
        end
        @assert !(chosen_action isa Nothing)
    else
        # choose integral to eliminate
        for ind in e.s.ibp_equations[end].nzind
            integral_at_ind = e.s.integral_list[ind]
            if integral_at_ind in masters
                continue
            end
            i = map_to_elim_action_index[integral_at_ind]
            if max_score isa Nothing || scores[i] > max_score
                chosen_action, max_score = SVector(integral_at_ind[1], integral_at_ind[2], 0), scores[i]
            end
        end
    end
    if verbose
        @show chosen_action
    end
    cost = act!(e, chosen_action, verbose)
    verbose ? (@show cost) : cost
end
        
"""
    play_using_preference_scores(scores; verbose=false, steps_only=false, integral=[6,6])

Run a fresh bubble environment using a fixed score for each possibly valid
action. At each step the highest-scoring currently valid action is chosen.
Returns either total cost or the number of steps.
"""
function play_using_preference_scores(scores; verbose = false, steps_only = false, integral=[6,6], show_seeds_only=false)      
    e = BubbleEnv(convert(SVector{2, Int}, integral), false)
    @assert length(scores) == length(e.properties.all_possibly_valid_actions)
    steps = 0
    map_to_elim_action_index = Dict{SVector{2, Int}, Int}()
    for (i, a) in enumerate(e.properties.all_possibly_valid_actions)
        if a[3] == 0
            map_to_elim_action_index[SVector(a[1], a[2])] = i
        end
    end
    cost = 0f0
    while !e.terminated
        cost += _play_using_preference_scores(e, map_to_elim_action_index, scores, verbose=verbose)
        steps += 1
    end
    if verbose
        @show steps
    end
    steps_only ? FLOAT_TYPE(steps) : cost
end

"""
    act!(e::BubbleEnv, a::AbstractArray, verbose=false)

Array-friendly wrapper around `act!` that converts `a` to `SVector{3, Int}`.
"""
function act!(e::BubbleEnv, a::AbstractArray, verbose=false)
    act!(e, convert(SVector{3, Int}, a), verbose)
end

"""
    act!(e::BubbleEnv, a::SVector{3, Int}, verbose=false)

Apply one environment action. Actions with third component `1` or `2` seed an
IBP equation with that operator; actions with third component `0` choose an
integral from the most recent equation to eliminate. The return value is the
reduction cost charged for the action.
"""
function act!(e::BubbleEnv, a::SVector{3, Int}, verbose=false)
    if e.terminated
        error("The environment has already terminated. Please reset first.")
    end
    bubble_gym_step_counter[] += 1
    integral::SVector{2, Int} = a[1:2]
    integral_number = e.properties.integral_numbering_dict[integral]
    ibp_operator = a[3]
    local created_nonzero_equation, reduction_cost
    if length(e.s.elim_order) == length(e.s.ibp_equations)
        @assert ibp_operator in (1, 2)
        if haskey(e.seeded_step, integral)
            error("integral already seeded!")
        end
        let (a,b) = integral
            if !valid_seed(a, b, e.properties.max_seed_propagator_power, e.properties.board_shape)
                error("Seed integral outside range")
            end
        end
        e.seed_count += 1
        ibp_eq = ibp_operator == 1 ? bubble_ibp_equation_1(integral, p2, m2, d) : bubble_ibp_equation_2(integral, p2, m2, d)
        (;created_nonzero_equation, reduction_cost) = insert_eq!(e.s, ibp_eq, verbose)
        if e.skip_redundant_equation_cost && !created_nonzero_equation
            reduction_cost = zero(reduction_cost)
        end
        e.seeded_step[a] = e.seed_count
        if created_nonzero_equation
            for ind in e.s.ibp_equations[end].nzind
                integral_at_ind = e.s.integral_list[ind]
                if !(integral_at_ind in masters)
                    push!(e.to_eliminate_in_last_eq, integral_at_ind)
                end
            end
            @assert length(e.to_eliminate_in_last_eq) > 0
        end
    else
        @assert ibp_operator == 0
        if !(integral in e.to_eliminate_in_last_eq)
            error("Can only eliminate integrals appearing in last equation")
        end
        if integral in masters
            error("Cannot eliminate master integrals")
        end
        e.elim_count += 1
        choose_integral_to_elim!(e.s, integral)
        for i in eachindex(e.to_reduce)
            if e.to_reduce[i][integral_number] != FiniteField(0)
                e.to_reduce[i] = e.to_reduce[i] - e.to_reduce[i][integral_number] * last(e.s.ibp_equations)
            end
        end
        reduction_cost = length(last(e.s.ibp_equations).nzind)
        e.eliminated_step[integral] = e.elim_count
        reduction_complete = true
        for a in e.to_reduce
            for ind in a.nzind
                integral = e.s.integral_list[ind]
                if !(integral in masters)
                    reduction_complete = false
                end
            end
        end
        if reduction_complete
            e.terminated = true
        end
        empty!(e.to_eliminate_in_last_eq)
    end
    e.last_action = a
    push!(e.action_log, a)
    return reduction_cost
end

"""
    terminated(e::BubbleEnv)

Return whether all tracked target expressions have reduced to master integrals.
"""
function terminated(e::BubbleEnv)
    result = e.terminated
    result
end

"""
    observation_and_mask(e::BubbleEnv)

Return flattened observation and action-mask arrays. The observation records
eliminated integrals, candidate eliminations from the last equation, seeded
operator choices, and the target marker; the mask marks currently valid actions.
"""
function observation_and_mask(e::BubbleEnv)::Tuple{Vector{FLOAT_TYPE}, Vector{Bool}}
    obs = zeros(FLOAT_TYPE, length(e.properties.integral_list), actual_data_length_per_integral)
    need_to_choose_integral_to_elim = false
    if length(e.s.ibp_equations) != length(e.s.elim_order)
        @assert length(e.s.ibp_equations) == length(e.s.elim_order) + 1
        need_to_choose_integral_to_elim = true
    end
    for integral in e.to_eliminate_in_last_eq
        integral_number = e.properties.integral_numbering_dict[integral]
        obs[integral_number, 1] = -one(FLOAT_TYPE)
    end    
    for (integral, step) in e.eliminated_step
        integral_number = e.properties.integral_numbering_dict[integral]
        obs[integral_number, 1] = one(FLOAT_TYPE)
    end
    for (integral_operator, step) in e.seeded_step
        integral = SVector(integral_operator[1], integral_operator[2])
        operator_number = integral_operator[3]
        integral_number = e.properties.integral_numbering_dict[integral]
        obs[integral_number, operator_number+1] = one(FLOAT_TYPE)
    end
    
    # Now produce the action mask (for choosing integrals to eliminate and choosing seed integrals) and append to the end of the observation data
    mask = zeros(Bool, length(e.properties.integral_list), mask_data_length_per_integral) # Bool array with `false` initialization
    if need_to_choose_integral_to_elim
        # mask[:, 1] .= false
        for integral in e.to_eliminate_in_last_eq
            integral_number = e.properties.integral_numbering_dict[integral]
            mask[integral_number, 1] = true
        end
        # mask[:, 2:(n_ibp_operators+1)] .= false
    else
        # mask[:, 1] .= false
        mask[:, 2:(n_ibp_operators+1)] .= e.properties.seed_candidate_mask
        for (integral_operator, step) in e.seeded_step
            integral = SVector(integral_operator[1], integral_operator[2])
            operator_number = integral_operator[3]
            integral_number = e.properties.integral_numbering_dict[integral]
            mask[integral_number, 1+operator_number] = false
        end
    end
    vec(obs), vec(mask)
end

"""
    observation_and_mask_2d(e::BubbleEnv)

Return image-like observation and action-mask tensors with channel-first layout.
The spatial axes cover integral coordinates from `-1` through
`max_seed_propagator_power + 1`.
"""
function observation_and_mask_2d(e::BubbleEnv)::Tuple{<:Array{FLOAT_TYPE}, <:Array{Bool}}
    # the coordinate range is from -1 to (max_seed_propagator_power+1), so the number of cells per axis is (max_seed_propagator_power+3), with an offset of 2 so that the first index is -1+2=1
    offset = 2
    obs = zeros(FLOAT_TYPE, actual_data_length_per_integral, e.properties.max_seed_propagator_power+3, e.properties.max_seed_propagator_power+3)
    need_to_choose_integral_to_elim = false
    if length(e.s.ibp_equations) != length(e.s.elim_order)
        @assert length(e.s.ibp_equations) == length(e.s.elim_order) + 1
        need_to_choose_integral_to_elim = true
    end
    for integral in e.to_eliminate_in_last_eq
        obs[1, integral[1] + offset, integral[2] + offset] = -one(FLOAT_TYPE)
    end    
    for (integral, step) in e.eliminated_step
        obs[1, integral[1] + offset, integral[2] + offset] = one(FLOAT_TYPE)
    end
    for (integral_operator, step) in e.seeded_step
        integral = SVector(integral_operator[1], integral_operator[2])
        operator_number = integral_operator[3]
        obs[operator_number+1, integral[1] + offset, integral[2] + offset] = one(FLOAT_TYPE)
    end
    obs[n_ibp_operators+2, e.target_integral0[1] + offset, e.target_integral0[2] + offset] = 1f0
    # for i in -1:(e.properties.max_seed_propagator_power+1), j in -1:(e.properties.max_seed_propagator_power+1)
    #     obs[n_ibp_operators+3, i+offset, j+offset] = convert(FLOAT_TYPE, i) / e.properties.max_seed_propagator_power
    #     obs[n_ibp_operators+4, i+offset, j+offset] = convert(FLOAT_TYPE, j) / e.properties.max_seed_propagator_power
    # end
    
    # Now produce the action mask (for choosing integrals to eliminate and choosing seed integrals) and append to the end of the observation data
    mask = zeros(Bool, mask_data_length_per_integral, e.properties.max_seed_propagator_power+3, e.properties.max_seed_propagator_power+3) # Bool array with `false` initialization
    if need_to_choose_integral_to_elim
        for integral in e.to_eliminate_in_last_eq
            mask[1, integral[1] + offset, integral[2] + offset] = true
        end
    else
        mask[2:(n_ibp_operators+1), :, :] .= e.properties.seed_candidate_mask_2d
        for (integral_operator, step) in e.seeded_step
            integral = SVector(integral_operator[1], integral_operator[2])
            operator_number = integral_operator[3]
            mask[1+operator_number, integral[1] + offset, integral[2] + offset] = false
        end
    end
    obs, mask
end

"""
    remove_mask_from_obs(data2d)

Drop mask channels from a combined observation/mask tensor and flatten the
remaining observation data.
"""
function remove_mask_from_obs(data2d)
    non_mask_part = selectdim(data2d, 2, 1:actual_data_length_per_integral)
    ndims(data2d) > 2 ? Flux.flatten(non_mask_part) : reshape(non_mask_part, :)
end
