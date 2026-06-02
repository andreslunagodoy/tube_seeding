using Test
using SparseSolveExact
import SparseSolveExact: minus_mult!

import Base: ==

function ==(a::SparseMat{SizeType, T}, b::SparseMat{SizeType, T}) where {SizeType, T}
    for field in fieldnames(SparseMat)
        if field != :pivot_row_packed && getproperty(a, field) != getproperty(b, field)
            return false
        end
    end
    return true
end

const BigRat = Rational{BigInt}

v1dense = BigRat[1, 0, 0, -2, -3, 0]
v1 = to_sparse_vec(v1dense)
v2dense = BigRat[2, 1, 0, 0, -3, 9]
v2 = to_sparse_vec(v2dense)

mat1 = SparseMat{Int, BigRat}([v1,v2], 2, 6) # 2 and 6 give the sparse matrix size
mat2 = SparseMat{Int, BigRat}([v1dense, v2dense])

@test mat1 == mat2

@test mat1[1,4] == -2
@test mat1[2,2] == 1
@test mat1[2,4] == mat1[3,1] == mat1[2,100] == 0

minus_mult!(mat1, 1, 2, mat1[1, 1] // mat1[2, 1])
@test mat1[1, 1] == 0 // 1

v1dense = BigRat[1, 0, 0, -2, -3, 0]
v2dense = BigRat[2, 1, 0, 0, -3, 9]
mat1 = SparseMat{Int, BigRat}([v1dense, v2dense])
@test mat1.rowdensity == [3, 4]
@test (mat1.pivot_row, mat1.pivot_col) == (0, 0)
@test true === SparseSolveExact.next_diag_pivot!(mat1)
@test (mat1.pivot_row, mat1.pivot_col) == (1, 1)
@test true === SparseSolveExact.next_diag_pivot!(mat1)
@test (mat1.pivot_row, mat1.pivot_col) == (2, 2)
@test false === SparseSolveExact.next_diag_pivot!(mat1) # Have run out of pivots

v1dense = BigRat[1, 0, 0, -2, -3, 0]
v2dense = BigRat[2, 0, 1, 0, 0, 0]
mat1 = SparseMat{Int, BigRat}([v1dense, v2dense])
findpivot!(mat1) # choose sparsest row 2, then within the nonzero elements of this row, choose the sparsest column 3
@test mat1.pivot_row == 2 && mat1.pivot_col == 3

v1dense = BigRat[0, 1, 0, -2, -3, 0]
v2dense = BigRat[2, 0, 1, 0, 0, 0]
mat1 = SparseMat{Int, BigRat}([v1dense, v2dense])
findpivot_no_optimization!(mat1) # choose the first nonzero row 1, then the first nonzero column 2
@test mat1.pivot_row == 1 && mat1.pivot_col == 2

v1dense = BigRat[1, 6, 0, -2, -3, 0]
v1 = to_sparse_vec(v1dense)
v2dense = BigRat[2, 1, 0, 0, -3, 0]
v2 = to_sparse_vec(v2dense)
mat1 = SparseMat([v1, v2], 2, 6, Set([2])) # forbid column 2 as a pivot
@test (mat1.pivot_row, mat1.pivot_col) == (0, 0)
findpivot_with_forbidden!(mat1)
@test mat1.pivot_row == 2 && (mat1.pivot_col == 1 || mat1.pivot_col == 5)

v1dense = BigRat[0, 1, 0, -2, -3, 0]
v2dense = BigRat[2, 0, 1, 0, 0, 0]
mat1 = SparseMat{Int, BigRat}([v1dense, v2dense])
findpivot_partial!(mat1) # choose the sparsest row 2, then the first nonzero column 1
@test mat1.pivot_row == 2 && mat1.pivot_col == 1

v1dense = BigRat[1, 6, 0, -2, -3, 0]
v1 = to_sparse_vec(v1dense)
v2dense = BigRat[2, 1, 0, 0, -3, 0]
v2 = to_sparse_vec(v2dense)
mat1 = SparseMat([v1, v2], 2, 6, Set([1])) # forbid column 1 as a pivot
@test (mat1.pivot_row, mat1.pivot_col) == (0, 0)
findpivot_partial_with_forbidden!(mat1) # choose sparsest row 2, then the firtst non-forbidden column 2
@test mat1.pivot_row == 2 && (mat1.pivot_col == 2 || mat1.pivot_col == 2)

v1dense = BigRat[1, 0, 0, -2, -3, 5]
v2dense = BigRat[2, 3, 1, 0, 0, 0]
mat1 = SparseMat{Int, BigRat}([v1dense, v2dense])
FindPivotWithPreference([1, 10, 5, 1, 1, 1])(mat1) # choose sparsest row 2, then within the nonzero elements of this row, choose the column with the largest preference value, i.e. column 2
@test mat1.pivot_row == 2 && mat1.pivot_col == 2

v1dense = BigRat[1, 1, 0, -2, -3, 0]
v2dense = BigRat[2, 3, 1, 0, 0, 0]
mat1 = SparseMat{Int, BigRat}([v1dense, v2dense])
echelonize_nopivoting!(mat1)
@test to_dense_vec(mat1.row_vectors[1], padded_length = 6) == v1dense
@test to_dense_vec(mat1.row_vectors[2], padded_length = 6) == BigRat[0, 1, 1, 4, 6, 0]

v1dense = BigRat[1, 1, 0, -2, -3, 0]
v1 = to_sparse_vec(v1dense)
v2dense = BigRat[2, 3, 1, 0, 0, 0]
v2 = to_sparse_vec(v2dense)
to_reduce1_dense = BigRat[1, 1, 0, -2, -3, 1]
to_reduce1 = to_sparse_vec(to_reduce1_dense)
to_reduce2_dense = BigRat[0, 1, 1, 4, 7, 0]
to_reduce2 = to_sparse_vec(to_reduce2_dense)
reduce_with_preordered_equations!([to_reduce1, to_reduce2], [v1, v2], 6) # to_reduce1 and to_reduce2 are mutated
@test to_dense_vec(to_reduce1, padded_length = 6) == BigRat[0, 0, 0, 0, 0, 1]
@test to_dense_vec(to_reduce2, padded_length = 6) == BigRat[0, 0, 0, 0, 1, 0]

v1dense = BigRat[3, 1, 0, -2, -3, 0]
v2dense = BigRat[2, 3, 1, 0, 0, 0]
mat1 = SparseMat{Int, BigRat}([v1dense, v2dense])
pivs = echelonize!(mat1)
@test pivs.pivotrows == [2, 1] # Refers to original row numbers, not reordered row numbers
@test pivs.pivotcols == [3, 1]
@test to_dense_vec(mat1.row_vectors[1], padded_length = 6) == BigRat[2, 3, 1, 0, 0, 0]
@test to_dense_vec(mat1.row_vectors[2], padded_length = 6) == v1dense .// 3 == BigRat[1, 1//3, 0, -2//3, -1, 0]
SparseSolveExact.back_subst!(mat1, pivs)
@test to_dense_vec(mat1.row_vectors[1], padded_length = 6) == BigRat[0, 7//3, 1, 4//3, 2, 0]
@test to_dense_vec(mat1.row_vectors[2], padded_length = 6) == v1dense .// 3 == BigRat[1, 1//3, 0, -2//3, -1, 0]

v1dense = BigRat[3, 0, 0, -2, -3, 9]
v2dense = BigRat[2, 3, 1, 0, 0, 0]
mat1 = SparseMat{Int, BigRat}([v1dense, v2dense])
pivs = echelonize!(mat1, pivotfunc = FindPivotWithPreference([10, 10, 5, 1, 1, 1]))
@test pivs.pivotrows == [2, 1] # Refers to original row numbers, not reordered row numbers
@test pivs.pivotcols[1] == 2 # For the first step, row 2 is the pivot row. Both columns 1 and 2 have the largest preference values, but column 2 is the sparsest and gets chosen

v1dense = BigRat[3, 0, 0, -2, -3, 9]
v2dense = BigRat[2, 3, 1, 0, 0, 0]
mat1 = SparseMat{Int, BigRat}([v1dense, v2dense])
pivs = echelonize!(mat1, pivotfunc = FindPivotWithPreference([20, 10, 5, 1, 1, 1]))
@test pivs.pivotrows == [2, 1] # Refers to original row numbers, not reordered row numbers
@test pivs.pivotcols[1] == 1 # For the first step, row 2 is the pivot row. Column 1 has the largest preference value and gets chosen, even though columns 2 and 3 are the sparsest

v1dense = BigRat[3, 0, 0, -2, -3, 9]
v2dense = BigRat[2, 3, 1, 0, 0, 0]
mat1 = SparseMat{Int, BigRat}([v1dense, v2dense])
pivs = echelonize!(mat1, pivotfunc = FindPivotWithPreference([0, 5, 5, 1, 1, 1]))
@test pivs.pivotrows == [2, 1] # Refers to original row numbers, not reordered row numbers
@test pivs.pivotcols[1] == 2 # For the first step, row 2 is the pivot row. Columns 2 and 3 have identical sparsity and preference values, so the smaller column number, 2, is chosen

# Same as above, but calls `rref!` which combines `echelonize!` and `back_subst!`
v1dense = BigRat[3, 1, 0, -2, -3, 0]
v2dense = BigRat[2, 3, 1, 0, 0, 0]
mat1 = SparseMat{Int, BigRat}([v1dense, v2dense])
pivs = rref!(mat1)
@test pivs.pivotrows == [2, 1] # Refers to original row numbers, not reordered row numbers
@test pivs.pivotcols == [3, 1]
@test to_dense_vec(mat1.row_vectors[1], padded_length = 6) == BigRat[0, 7//3, 1, 4//3, 2, 0]
@test to_dense_vec(mat1.row_vectors[2], padded_length = 6) == v1dense .// 3 == BigRat[1, 1//3, 0, -2//3, -1, 0]
remainder = BigRat[0, 2, 0, -3//4, 0, 0] # remainder has zero entries at the pivot columns 1 and 3

# same test matrix as above. After echelonizing the matrix, we reduce a vector against the matrix and records which rows (as numbered in the pre-echelonized matrix) are needed.
v1dense = BigRat[3, 1, 0, -2, -3, 0]
v1 = to_sparse_vec(v1dense)
v2dense = BigRat[2, 3, 1, 0, 0, 0]
mat1 = SparseMat{Int, BigRat}([v1dense, v2dense])
@test trace_needed_equations!(mat1, [v1]).eqs == Int[1]

# same test as above, but reducing [v2] instead of [v1]
v1dense = BigRat[3, 1, 0, -2, -3, 0]
v1 = to_sparse_vec(v1dense)
v2dense = BigRat[2, 3, 1, 0, 0, 0]
v2 = to_sparse_vec(v2dense)
mat1 = SparseMat{Int, BigRat}([v1dense, v2dense])
@test trace_needed_equations!(mat1, [v2]).eqs == Int[2]

# same test as above, but reducing [v1 - v2]
v1dense = BigRat[3, 1, 0, -2, -3, 0]
v2dense = BigRat[2, 3, 1, 0, 0, 0]
mat1 = SparseMat{Int, BigRat}([v1dense, v2dense])
@test sort(trace_needed_equations!(mat1, [to_sparse_vec(v1dense - v2dense)]).eqs) == Int[1, 2]

v1dense = BigRat[3, 1, 0, -2, -3, 0]
v2dense = BigRat[2, 3, 1, 0, 0, 0]
mat1 = SparseMat{Int, BigRat}([v1dense, v2dense])
# pivs = rref!(mat1) # a little overkill, use the line below
pivs = echelonize!(mat1)
remainder = BigRat[0, 2, 0, -3//4, 0, 0] # remainder has zero entries at the pivot columns 1 and 3
to_reduce = to_sparse_vec(3//4 * to_dense_vec(mat1.row_vectors[1], padded_length = 6)  - 7//8 * to_dense_vec(mat1.row_vectors[2], padded_length = 6) + remainder)
to_reduce_copy = deepcopy(to_reduce)
needed_rows = reduce_with_ref_mat!(to_reduce, mat1, pivs)
@test to_dense_vec(to_reduce, padded_length = 6) == remainder
@test needed_rows == [1, 2]
@test SparseSolveExact.get_needed_rows_fast([to_reduce_copy], mat1, pivs.pivotcols) == needed_rows # the fast method should give the same result in this case

to_reduce = to_sparse_vec(- 7//8 * to_dense_vec(mat1.row_vectors[2], padded_length = 6) + remainder)
to_reduce_copy = deepcopy(to_reduce)
needed_rows = reduce_with_ref_mat!(to_reduce, mat1, pivs)
@test to_dense_vec(to_reduce, padded_length = 6) == remainder
@test needed_rows == [2]
@test SparseSolveExact.get_needed_rows_fast([to_reduce_copy], mat1, pivs.pivotcols) == needed_rows # the fast method should give the same result in this case

v1dense = BigRat[2, 1, 0]
v2dense = BigRat[4, 3, 1]
mat1 = SparseMat{Int, BigRat}([v1dense, v2dense])
pivs = rref_nopivoting!(mat1)
@test pivs.pivotrows == [1, 2]
@test pivs.pivotcols == [1, 2]
@test to_dense_vec(mat1.row_vectors[1], padded_length = 3) == BigRat[1, 0, -1//2]
@test to_dense_vec(mat1.row_vectors[2], padded_length = 3) == BigRat[0, 1, 1]

mock_instructions = [(2,0), (1,2), (1,3), (2,0), (2,3), (3,0)] # Suppose the original rows are [rowA, rowB, rowC]. In step k=1 of the forward elimination, the first entry of `mock_instructions`, (2,0) means that we choose the 2nd row, i.e. rowB as the pivot row. Therefore we swap it with the kth row, and the rows are now ordered as [rowB, rowA, rowC]. The second entry of `mock_instructions`, (1,2) means that we use the 1st row, which is now rowB, to reduce the 2nd row, which is now rowA. The 3rd entry (1,3) means that we use the 1st row to reduce the 3rd row. The 4th entry (2,0) ends with 0, so we know that we're at the next step k=2 of the forward elimination. (2,0) means that we choose the 2nd row, now rowA, as the pivot row. (2,3) means with use the 2nd row, i.e. rowA, to reduce the 3rd row, rowC. (3,0) means we choose the 3rd row, rowC, as the pivot row in step k=3. There are no more entries in `mock_instructions`, meaning that there are no more nonzero rows left, so we didnt' use the 3rd row to reduce anything
mock_pivotrows, mock_pivotcols = [2, 1, 3], [1, 2, 3] # `pivotrows` refers to rows in the original ordering, so [2, 1] corresponds to [rowB, rowA]
@test SparseSolveExact._extract_reduction_steps(mock_instructions, 3, (pivotrows = mock_pivotrows , pivotcols = mock_pivotcols)) == [(1,2), (1,3), (2,3)]
# To use the result above, first reorder the rows according to pivotrows==(2,1,3), as (rowB, rowA, rowC). Then apply the result of _extract_reduction_steps(), which is [(1, 2), (1, 3), (2, 3)]. Each entry in the list is in the form (pivotRow, reducedRow)

@test SparseSolveExact.dependent_rows([(1,2), (1,3), (2,3)], [3]) == [1, 2, 3] # Suppose the reduction instruction, as computed by the previous test, is [(1,2), (1,3), (2,3)], and the actual row used to reduce some expressions is just [3], we still need all rows, i.e. [1, 2, 3], as the reduced form of row 3 depends on rows 1 and 2 which were used in the reduction

@test SparseSolveExact.dependent_rows([(1,2), (1,4), (2,3)], [2, 4]) == [1, 2, 4]

# test an edge case, where the second argument is an empty list []
@test SparseSolveExact.dependent_rows([(1,2), (1,4), (2,3)], Int[]) == Int[]

# test another edge case which can show up in practise: if the input matrix is already echelonized, the reduction intruction will be an emtpy list, and the needed rows are exactly the ones used the reduction
@test SparseSolveExact.dependent_rows(Tuple{Int, Int}[], Int[3]) == Int[3]
