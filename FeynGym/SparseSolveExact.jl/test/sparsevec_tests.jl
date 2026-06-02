using Test
using SparseSolveExact

# test SparseVec constructors
v1 = SparseVec(Dict(["a" => 2//1, "b" => 3//1]))
v1a = SparseVec(["a" => 2//1, "b" => 3//1])
@test v1 == v1a

# test printing, accept either ordering of elements
@test string(v1) == """SparseVec(["b" => 3//1, "a" => 2//1])""" || string(v1) == """SparseVec(["a" => 2//1, "b" => 3//1])"""

# test forwarded Dict methods. `collect` tests e.g. `iterate`
@test collect(v1) == ["b" => 3//1, "a" => 2//1] || collect(v1) == ["a" => 2//1, "b" => 3//1]

v2 = SparseVec(Dict(["a" => 1//1, "c" => 3//1]))
@test SparseSolveExact.minus_mult!(v1, v2, 2) == SparseVec(["b" => 3//1, "c" => -6//1])
@test v1.data == Dict(["b" => 3//1, "c" => -6//1]) # test that the above returned result is also saved into v1

v1 = SparseVec(["a" => 2//1, "b" => 3//1])
v2 = SparseVec(["a" => 1//1, "c" => 3//1])
SparseSolveExact.minus_mult!(v1, v2, 2)
@test v1 == SparseVec(["b" => 3//1, "c" => -6//1])

@test to_sparse_vec([3, 0, 4, 5, 0]) == SparseVec([1 => 3, 3 => 4, 4 => 5])
@test to_dense_vec(SparseVec([1 => 3, 3 => 4, 4 => 5])) == [3, 0, 4, 5]
@test to_dense_vec(SparseVec([1 => 3, 3 => 4, 4 => 5]), padded_length = 7) == [3, 0, 4, 5, 0, 0, 0]

v1 = SparseVec(["a" => 2//1, "b" => 3//1])
SparseSolveExact.normalize!(v1, "b")
@test v1 == SparseVec(["a" => 2//3, "b" => 1//1])

v1 == SparseVec(["a" => 2//3, "b" => 1//1])
@test map_sparsevec_indices(Dict("a" => 1, "b" => 2), v1) == SparseVec([1 => 2//3, 2 => 1//1])
@test map_sparsevec_indices(s -> s * s, String, v1) == SparseVec(["aa" => 2//3, "bb" => 1//1])
@test map_sparsevec_indices(s -> s[1], Char, v1) == SparseVec(['a' => 2//3, 'b' => 1//1])
@test map_sparsevec_indices(s -> s[1], v1) == SparseVec(['a' => 2//3, 'b' => 1//1])

@test map_sparsevec_values(Dict(2//3 => -2//3, 1//1 => -1//1), v1) == SparseVec(["a" => -2//3, "b" => -1//1])
@test map_sparsevec_values(v -> v^2, Rational{Int}, v1) == SparseVec(["a" => 4//9, "b" => 1//1])
@test map_sparsevec_values(v -> v^2/1.0, v1) == SparseVec(["a" => 4.0/9, "b" => 1.0])