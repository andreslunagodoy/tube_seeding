"""
    asymmetric_ibp

Mutable switch selecting the asymmetric form of the second bubble IBP operator.
The asymmetric form is the usual form found in the literature, unless special
care is taken to symmetrize the two IBP operators.

`__init__` resets it to `true` when the module is loaded.
"""
const asymmetric_ibp = Ref(true)

"""
    bubble_ibp_equation_1(seed, p2, m2, d)

Generate the first IBP equation for the two-propagator massive bubble family at
the supplied seed integral. Terms with zero coefficient or no positive
propagator power are removed.
"""
function bubble_ibp_equation_1(seed::SVector{2, Int}, p2::CoeffType, m2::CoeffType, d::CoeffType)::Eq{2, CoeffType} where CoeffType
    nu1, nu2 = convert(CoeffType, seed[1]), convert(CoeffType, seed[2])
    eq =[
        seed => d - 2*nu1 - nu2,
        seed + SVector(1, 0) => 2*nu1*m2,
        seed + SVector(-1, 1) => -nu2,
        seed + SVector(0, 1) => nu2*(2*m2 - p2)
    ]
    filter(a -> a[2] != 0 && (a[1][1] > 0 || a[1][2] > 0), eq)
end

"""
    bubble_ibp_equation_2(seed, p2, m2, d)

Generate the second IBP equation for the two-propagator massive bubble family.
The equation uses the asymmetric operator when `asymmetric_ibp[]` is true and
the mirror-symmetric operator otherwise.
"""
function bubble_ibp_equation_2(seed::SVector{2, Int}, p2::CoeffType, m2::CoeffType, d::CoeffType)::Eq{2, CoeffType} where CoeffType
    nu1, nu2 = convert(CoeffType, seed[1]), convert(CoeffType, seed[2])
    eq = if asymmetric_ibp[]
        [
            seed => nu2-nu1,
            seed + SVector(1, -1) => nu1,
            seed + SVector(-1, 1) => -nu2,
            seed + SVector(1, 0) => nu1*p2,
            seed + SVector(0, 1) => -nu2*p2
        ]
    else
    # symmetric version; mirror image of `bubble_ibp_equation_1`
        [
            seed => d - 2*nu2 - nu1,
            seed + SVector(0, 1) => 2*nu2*m2,
            seed + SVector(1, -1) => -nu1,
            seed + SVector(1, 0) => nu1*(2*m2 - p2)
        ]
    end
    filter(a -> a[2] != 0 && (a[1][1] > 0 || a[1][2] > 0), eq)
end

"""
    bubble_masters

Canonical master integrals for the massive bubble family.
"""
const bubble_masters = (SVector(1,0), SVector(0, 1), SVector(1, 1))

"""
    bubble_integral_greater_than(i1, i2)

Return whether `i1` is more complex than `i2` under
`bubble_integral_complexity_info`.
"""
function bubble_integral_greater_than(i1::Integral{2}, i2::Integral{2})::Bool
    bubble_integral_complexity_info(i1) > bubble_integral_complexity_info(i2) # lexicographic ordering of the vector returned by `bubble_integral_complexity_info`
end

"""
    bubble_integral_complexity_info(integral)

Return a suitably defined complexity, encoded in a vector to be compared lexicographically, for ordering bubble integrals.
"""
function bubble_integral_complexity_info(integral::Integral{2})
    [count(>(0), integral), sum(abs.(integral)), integral...]
end

"""
    bubble_integral_complexity_info_alt(integral)

Alternative bubble complexity key that treats positive powers relative to one
and non-positive powers by absolute value.
"""
function bubble_integral_complexity_info_alt(integral::Integral{2})
    [count(>(0), integral), sum(map(a -> a<=0 ? -a : a-1, integral)), integral...]
end

"""
    bubble_integral_greater_than_alt(i1, i2)

Alternative ordering predicate that keeps master integrals below non-masters and
then compares the alternative complexity key.
"""
function bubble_integral_greater_than_alt(i1::Integral{2}, i2::Integral{2})::Bool
    if i1 in bubble_masters && !(i2 in bubble_masters)
        false
    elseif !(i1 in bubble_masters) && i2 in bubble_masters
        true
    else
        bubble_integral_complexity_info(i1) < bubble_integral_complexity_info(i2) # lexicographic ordering of the vector returned by `bubble_integral_complexity_info`
    end
end
