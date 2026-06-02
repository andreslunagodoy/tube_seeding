"""
    FeynGym

Julia core for a reinforcement-learning environment for integration-by-parts
(IBP) reduction of Feynman integrals.

The package represents integrals by integer index vectors, IBP equations by
sparse linear relations, and exposes a one-loop massive bubble RL environment
that can be wrapped from Python through `pyfeyngym`.

For generic integral families other than the one-loop bubble, the generic
linear solver in the package enables IBP reduction and black-box optimization
of the total cost, but not an RL environment with step-by-step feedback. You'll
need to generate the IBP equations for your integral family of interest
yourself, and then call the linear solver on the equations.
"""
module FeynGym

import StaticArrays: SVector
import SparseArrays: SparseVector

"""
    Integral{N}

Static integer vector identifying an `N`-index Feynman integral.
"""
const Integral{N} = SVector{N, Int}

"""
    Eq{N, CoeffType}

Dense list representation of an IBP equation. Each term maps an `Integral{N}`
to its coefficient in the equation.
"""
const Eq{N, CoeffType} = Vector{Pair{Integral{N}, CoeffType}}

"""
    EqVector{CoeffType}

Sparse vector representation of an IBP equation after integral indices have
been mapped to integer column numbers.
"""
const EqVector{CoeffType} = SparseVector{CoeffType, Int}

include("ibp_equation_generator.jl")

include("finite_field.jl")

include("ibp_reduction.jl")

include("bubble_gym.jl")

include("python_interface_helper.jl")

"""
    __init__()

Reset mutable module-level defaults each time the package is loaded.
"""
function __init__()
    bubble_gym_episode_counter[] = 0
    bubble_gym_step_counter[] = 0
    asymmetric_ibp[] = true
end

end # module FeynGym
