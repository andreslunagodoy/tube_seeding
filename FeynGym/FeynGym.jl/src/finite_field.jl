"""
    FF{p, T}

Finite field with prime modulus `p`, stored using integer type `T`. Arithmetic
assumes `p^2 + p < typemax(T)` so fused operations such as `a*b + c` fit in the
underlying integer type before reduction.
"""
struct FF{p, T} <: Number
    value::T
end

"""
    show(io::IO, a::FF)

Custom printing method for finite-field numbers.
"""
show(io::IO, a::FF) = print(io, a.value)

"""
    zero(::Type{FF{p,T}})

Zero element of the finite field.
"""
function Base.zero(::Type{FF{p,T}}) where {p,T}
    FF{p,T}(0)
end

"""
    zero(::FF{p,T})

Zero element of the finite field.
"""
function Base.zero(::FF{p,T}) where {p,T}
    FF{p,T}(0)
end

"""
    one(::Type{FF{p,T}})

Unit element of the finite field.
"""
function Base.one(::Type{FF{p,T}}) where {p,T}
    FF{p,T}(1)
end

"""
    one(::FF{p,T})

Unit element of the finite field.
"""
function Base.one(::FF{p,T}) where {p,T}
    FF{p,T}(1)
end

"""
    convert(::Type{FF{p,T}}, a::Integer)

Convert an integer to the finite-field type.
"""
function Base.convert(::Type{FF{p,T}}, a::Integer) where {p,T}
    FF{p,T}(a % p)
end

"""
    convert(::Type{FF{p,T}},  a::Rational{<:Integer})

Convert a rational number to the finite-field type using modular division.
"""
function Base.convert(::Type{FF{p,T}}, a::Rational{<:Integer}) where {p,T}
    FF{p,T}(numerator(a) % p) * inv(FF{p,T}(denominator(a) % p))
end

# Base.isless(a::FF, b::FF) = isless(a.value, b.value)

"""
    _field_inverse(a, prime)

Compute the inverse of `a` modulo `prime`.
"""
function _field_inverse(a, prime)
    if a == zero(a)
        error("Cannot compute inverse of zero")
    end
    if (a<0)
        a += prime
    end
    b0 = prime
    x0 = 0
    x1 = 1
    # local t, q
    while a>1
        q = a ÷ prime
        t = prime
        prime = a % prime
        a = t
        t = x0
        x0 = x1 - q * x0
        x1 = t
    end
    if x1<0
        x1 += b0
    end
    return x1
end

"""
    ==(a::FF{p, T}, b::FF{p, T})

Equality test for two elements of the same finite field.
"""
Base.:(==)(a::FF{p, T}, b::FF{p, T}) where {p, T} = (a.value - b.value) % p == 0

"""
    ==(a::FF{p, T}, b::Integer)

Equality test against an integer representative.
"""
Base.:(==)(a::FF{p, T}, b::Integer) where {p, T} = (a.value % p - b) % p ==0

"""
    ==(b::Integer, a::FF{p, T})

Equality test against an integer representative.
"""
Base.:(==)(b::Integer, a::FF{p, T}) where {p, T} = (a==b)

"""
    -(a::FF{p, T})

Compute the additive inverse in a finite field.
"""
Base.:-(a::FF{p, T}) where {p, T} = FF{p, T}(-a.value)

"""
    -(a::FF{p, T}, b::FF{p, T})

Subtract two finite-field elements.
"""
Base.:-(a::FF{p, T}, b::FF{p, T}) where {p, T} = FF{p, T}((a.value - b.value) % p)

"""
    +(a::FF{p, T}, b::FF{p, T})

Add two finite-field elements.
"""
Base.:+(a::FF{p, T}, b::FF{p, T}) where {p, T} = FF{p, T}((a.value + b.value) % p)

"""
    *(a::FF{p, T}, b::FF{p, T})

Multiply two finite-field elements.
"""
Base.:*(a::FF{p, T}, b::FF{p, T}) where {p, T} = FF{p, T}(_mult(a.value, b.value, p))

"""
    *(a::FF{p, T}, b::Union{Integer, Rational{<:Integer}})

Multiply a finite-field element by an integer or rational scalar.
"""
Base.:*(a::FF{p, T}, b::Union{Integer, Rational{<:Integer}}) where {p, T} = convert(FF{p, T}, a.value * b)

"""
    *(a::Union{Integer, Rational{<:Integer}}, b::FF{p, T})

Multiply an integer or rational scalar by a finite-field element.
"""
Base.:*(a::Union{Integer, Rational{<:Integer}}, b::FF{p, T}) where {p, T} = convert(FF{p, T}, b.value * a)

"""
    //(a::FF{p, T}, b::FF{p, T})

Exact division in the finite field.
"""
Base.://(a::FF{p, T}, b::FF{p, T}) where {p, T} = a * inv(b)

"""
    ^(a::FF{p, T}, b::Integer)

Raise a finite-field element to an integer power.
"""
Base.:^(a::FF{p, T}, b::Integer) where {p, T} = convert(FF{p, T}, big(a.value)^b)

"""
    _mult(a, b, p)

Compute `a*b` modulo `p`.
"""
_mult(a, b, p) = (a * b) % p

"""
    _minus_mult(a, b, c, p)

Compute `a - b*c` modulo `p`.
"""
_minus_mult(a, b, c, p) = (a - b*c) % p

"""
    inv(a::FF{p, T})

Compute the multiplicative inverse in a finite field.
"""
Base.:inv(a::FF{p, T}) where {p, T} = FF{p, T}(_field_inverse(a.value, p))

"""
    minus_mult(a::FF{p, T}, b::FF{p, T}, c::FF{p, T})

Compute `a - b*c` in a finite field.
"""
minus_mult(a::FF{p, T}, b::FF{p, T}, c::FF{p, T}) where {p, T} = FF{p, T}(_minus_mult(a.value, b.value, c.value, p))
