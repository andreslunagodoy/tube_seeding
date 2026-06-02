# abstract type AbstractField end
# We need to use built-in types such as Rational{Int}, so we won't formally declare an abstract type.
# Nevertheless, we expect the interface to have functions `zero`, `one`, `==`, `-`, `*`, `minus_mult`,
# and `is_zero`

"""
Generic `minus_mult(a, b, c)` which returns `a - b * c` by, but specialized fused implementations can be defined.
"""
function minus_mult(a, b, c) 
    return a - b * c
end

"""
Generic `is_zero(a)` which returns `true` if `a==zero(T)`.
"""
function is_zero(a::T) where {T}
    return a == zero(T)
end

"""
Generic `is_not_zero(a)` which returns `true` if `a!=zero(T)`.
"""
function is_not_zero(a::T) where {T}
    return a != zero(T)
end

"""
Generic `is_one(a)` which returns `true` if `a==one(T)`.
"""
function is_one(a::T) where {T}
    return a == one(T)
end

"""
Generic `is_not_one(a)` which returns `true` if `a!=one(T)`.
"""
function is_not_one(a::T) where {T}
    return a != one(T)
end
