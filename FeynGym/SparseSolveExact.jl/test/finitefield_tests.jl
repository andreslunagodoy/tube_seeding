using Test
using SparseSolveExact
import SparseSolveExact: minus_mult

# test printing
@test string(FF{13, Int128}(2)) == "2"

# zero and one
@test zero(FF{13, Int64}) == FF{13, Int64}(0)
@test zero(FF{13, Int64}(3)) == FF{13, Int64}(0)
@test one(FF{13, Int64}) == FF{13, Int64}(1)
@test one(FF{13, Int64}(3)) == FF{13, Int64}(1)

p = 2^31 - 1
a0 = 4542//4135
b0 = 7193//314
c0 = -128//97432
ff = FF{p, Int128}
a, b, c = convert(ff, a0), convert(ff, b0), convert(ff, c0)
@test a + b == convert(ff, a0 + b0)
@test a - b == convert(ff, a0 - b0)
@test a * b == convert(ff, a0 * b0)
@test a // b == convert(ff, a0 // b0)
@test a^3891 == convert(ff, big(a0)^3891)
@test minus_mult(a, b, c) == convert(ff, a0 - b0 * c0)
@test inv(a) == convert(ff, inv(a0))
@test -a == convert(ff, -a0)
@test -b == convert(ff, -b0)
@test -c == convert(ff, -c0)
