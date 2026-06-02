using Test
using SparseSolveExact

@test SparseSolveExact.minus_mult(3//11, 4//9, -9//7) == 3//11 - (4//9) * (-9//7)