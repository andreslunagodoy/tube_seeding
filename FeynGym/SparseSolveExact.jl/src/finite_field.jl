"""
    FF{p, T}

Finite field with prime modulus p, stored with Integer type T. We impose `p^2 + p < typemax(T)`, i.e. `2^63-1` for `Int64`, `2^127-1` for `Int128`, to ensure expressions like `a*b+c` never exceeds the limit of an unsigned integer.
"""
struct FF{p, T}
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
function zero(::Type{FF{p,T}}) where {p,T}
    FF{p,T}(0)
end

"""
    zero(::FF{p,T})

Zero element of the finite field.
"""
function zero(::FF{p,T}) where {p,T}
    FF{p,T}(0)
end

"""
    one(::Type{FF{p,T}})

Unit element of the finite field.
"""
function one(::Type{FF{p,T}}) where {p,T}
    FF{p,T}(1)
end

"""
    one(::FF{p,T})

Unit element of the finite field.
"""
function one(::FF{p,T}) where {p,T}
    FF{p,T}(1)
end

"""
    is_zero(a::FF{p,T})

Fast zero test for finite-field elements.

This intentionally checks the wrapped integer directly instead of reducing modulo `p`.
Because the representation may be non-canonical (e.g. negative representatives are
allowed by [`-(a::FF{p,T})`](src/finite_field.jl:128)), direct equality is only
safe for testing zero.
"""
is_zero(a::FF{p,T}) where {p,T} = (a.value == zero(T))

"""
    is_not_zero(a::FF{p,T})

Fast nonzero test for finite-field elements.
"""
is_not_zero(a::FF{p,T}) where {p,T} = (a.value != zero(T))

"""
    is_one(a::FF{p,T})

Fast one test for finite-field elements.

Since the representation is not guaranteed to be canonical, this checks the
wrapped integer against two common representatives of 1:

- `1`
- `-p + 1` (i.e. `1 - p`)
"""
@inline function is_one(a::FF{p,T}) where {p,T}
    v = a.value
    v == one(T) || v == (one(T) - T(p))
end

"""
    is_not_one(a::FF{p,T})

Fast non-one test for finite-field elements.
"""
@inline function is_not_one(a::FF{p,T}) where {p,T}
    v = a.value
    v != one(T) && v != (one(T) - T(p))
end

"""
    convert(::Type{FF{p,T}}, a::Integer)

Convert an integer to the finite field type
"""
function convert(::Type{FF{p,T}}, a::Integer) where {p,T}
    FF{p,T}(a % p)
end

"""
    convert(::Type{FF{p,T}},  a::Rational{<:Integer})

Convert a rational number to the finite field type, using modular division
"""
function convert(::Type{FF{p,T}}, a::Rational{<:Integer}) where {p,T}
    FF{p,T}(numerator(a) % p) * inv(FF{p,T}(denominator(a) % p))
end

# Base.isless(a::FF, b::FF) = isless(a.value, b.value)

"""
    _field_inverse(a, prime)

Compute the inverse of `a` modulo `prime`
"""
function _field_inverse(a, prime)
    @assert a!=0
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

Equality test
"""
==(a::FF{p, T}, b::FF{p, T}) where {p, T} = (a.value - b.value) % p == 0

"""
    ==(a::FF{p, T}, b::Integer)

Equality test
"""
==(a::FF{p, T}, b::Integer) where {p, T} = (a.value % p - b) % p ==0

"""
    ==(b::Integer, a::FF{p, T})

Equality test
"""
==(b::Integer, a::FF{p, T}) where {p, T} = (a==b)

"""
    -(a::FF{p, T})

Compute the negative in a finite field
"""
-(a::FF{p, T}) where {p, T} = FF{p, T}(-a.value)

"""
    -(a::FF{p, T}, b::FF{p, T})

Binary minus operator
"""
-(a::FF{p, T}, b::FF{p, T}) where {p, T} = FF{p, T}((a.value - b.value) % p)

"""
    +(a::FF{p, T}, b::FF{p, T})

Binary plus operator
"""
+(a::FF{p, T}, b::FF{p, T}) where {p, T} = FF{p, T}((a.value + b.value) % p)

"""
    *(a::FF{p, T}, b::FF{p, T})

Binary times operator
"""
*(a::FF{p, T}, b::FF{p, T}) where {p, T} = FF{p, T}(_mult(a.value, b.value, p))

"""
    //(a::FF{p, T}, b::FF{p, T})

Binary exact division operator
"""
//(a::FF{p, T}, b::FF{p, T}) where {p, T} = a * inv(b)

"""
    ^(a::FF{p, T}, b::Integer)

Exponential operator
"""
^(a::FF{p, T}, b::Integer) where {p, T} = convert(FF{p, T}, big(a.value)^b)

"""
    _mult(a, b, p)

Compute `a*b` modulo `p`
"""
_mult(a, b, p) = (a * b) % p

"""
_minus_mult(a, b, c, p)

Compute `a - b*c` modulo `p`
"""
_minus_mult(a, b, c, p) = (a - b*c) % p

"""
    inv(a::FF{p, T})

Compute the inverse in a finite field
"""
inv(a::FF{p, T}) where {p, T} = FF{p, T}(_field_inverse(a.value, p))

"""
    minus_mult(a::FF{p, T}, b::FF{p, T}, c::FF{p, T})

Compute `a - b*c`` in a finite field
"""
minus_mult(a::FF{p, T}, b::FF{p, T}, c::FF{p, T}) where {p, T} = FF{p, T}(_minus_mult(a.value, b.value, c.value, p))
