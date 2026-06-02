# const SEARCH_SINGLET_COLUMNS = false

"Abstract Sparse matrix type"
abstract type AbstractSparseMatBare{SizeType, T} end

"Abstract Sparse matrix type containing both data and metadata for Gaussian elimination"
abstract type AbstractSparseMat{SizeType, T} <: AbstractSparseMatBare{SizeType, T} end

################################
"""
`SparseMat{SizeType, T}` is the data structure for sparse matrix over rational numbers or finite fields, to be used in Gaussian elimination. The type parameter `SizeType` is the type of integer indexes for row / column numbers. `T` is the type of matrix entries. The data fields will be documented using `matrix` as the name of the `SparseMat` instance.

The constructor is

    SparseMat{SizeType, T}(row_vectors::Vector{SparseVec{SizeType, T}}, nrows::SizeType, ncols::SizeType, forbidden_col_pivots = Set{SizeType}())

The outer constructor allows you to skip to type annotation

    SparseMat(row_vectors::Vector{SparseVec{SizeType, T}}, nrows::SizeType, ncols::SizeType, forbidden_col_pivots = Set{SizeType}())

Another outer constructor

    SparseMat{SizeType, T}(row_vectors::Vector{Vector{T}})
    
constructs a sparse matrix from a dense matrix given as array of arrays.
"""
mutable struct SparseMat{SizeType, T} <: AbstractSparseMat{SizeType, T}
    "`row_vectors[i]` is the i-th row of the sparse matrix, given as a [`SparseVec`](@ref) object which is a wrapper type for a dictionary which maps column numbers to nonzero matrix entries."
    row_vectors::Vector{SparseVec{SizeType, T}}
    "number of rows."
    nrows::SizeType
    "number of columns."
    ncols::SizeType
    "`nonzero_rows_in_col[j]` is the set of row indices `i` such that `matrix[i, j]` is nonzero."
    nonzero_rows_in_col::Vector{Set{SizeType}}
    "`rowdensity[i]` is the number of nonzero elements in row i."
    rowdensity::Vector{SizeType}
    "`coldensity[j]` is the number of nonzero elements in column j."
    coldensity::Vector{SizeType}
    "temporary buffer used in `elim_one_step` to store the rows that need to be reduced against the pivot rows"
    rows_to_reduce::Vector{SizeType}
    "`nrows_pivoted`: number of rows that has been used as a pivot row to reduce other rows, in echelonization"
    nrows_pivoted::SizeType
    "`pivot_row`: `mat.row_vectors[pivot_row]` is the row selected as the pivot row in Gaussian elimination, which will be subsequently swapped to the position `mat.row_vectors[nrows_pivoted]` before other rows are reduced against this row."
    pivot_row::SizeType
    "`pivot_col`: the column number used as the pivot column in Gaussian elimination. We will not perform column swaps, but will keep track of the needed column permutations."
    pivot_col::SizeType
    "`forbidden_col_pivots`: the column numbers that should not be chosen as pivots. Only used when using kwarg `pivotfunc = findpivot_with_forbidden!` in `echelonize!()`."
    forbidden_col_pivots::Set{SizeType}
    "a vector of tuples (pivotrow, reducedrow), which records the steps taken during echelonization. At each step, we reduce the reducedrow against pivotrow. Here both reducedrow and pivotrow are integer labels. If pivotrow is not equal to the step number i, then we first swap the row i and the row pivotrow. In subsequent steps, the row labels refer to the reordered rows, not the original rows."
    reduction_log::Vector{Tuple{SizeType, SizeType}}
    "temporary buffer for storing a packed copy of the pivot row, for fast access"
    pivot_row_packed::Vector{Pair{SizeType, T}}
    "Number of arithmetic operations in forward elimination. Only recorded when `log = Val(true)` is passed to `echelonize!()`."
    forward_elimination_cost::Int64
    function SparseMat{SizeType, T}(row_vectors::Vector{SparseVec{SizeType, T}}, nrows::SizeType, ncols::SizeType, forbidden_col_pivots = Set{SizeType}()) where {SizeType, T}
        @assert 0 < nrows < typemax(SizeType)
        @assert 0 < ncols < typemax(SizeType)
        rowdensity = map(length, row_vectors)
        nonzero_rows_in_col = [Set{SizeType}() for _ in 1:ncols]
        for i in 1:nrows
            for column_number in keys(row_vectors[i])
                push!(nonzero_rows_in_col[column_number], i)
            end
        end
        coldensity = map(length, nonzero_rows_in_col)
        new(row_vectors, nrows, ncols, nonzero_rows_in_col, rowdensity, coldensity, fill(zero(SizeType), nrows), zero(SizeType), zero(SizeType), zero(SizeType), forbidden_col_pivots, Tuple{SizeType, SizeType}[], Vector{Pair{SizeType, T}}(undef, nrows), 0)
    end
end

function SparseMat(row_vectors::Vector{SparseVec{SizeType, T}}, nrows::SizeType, ncols::SizeType, forbidden_col_pivots = Set{SizeType}()) where {SizeType, T}
    SparseMat{SizeType, T}(row_vectors, nrows, ncols, forbidden_col_pivots)
end

SparseMat{SizeType, T}(row_vectors::Vector{Vector{T}}) where {SizeType, T} = SparseMat{SizeType, T}(map(to_sparse_vec, row_vectors), length(row_vectors), length(row_vectors[1]))

"""
    show(io::IO, ::MIME"text/plain", s::AbstractSparseMat)

Custom display method for REPL and IJulia
"""
function show(io::IO, ::MIME"text/plain", s::AbstractSparseMat)
    print(io, s.nrows, " times ", s.ncols, " sparse matrix with current pivot ", (s.pivot_row, s.pivot_col))
end

"""
    function getindex(s::AbstractSparseMat{SizeType, T}, i::Integer, j::Integer)

Return the (i, j) entry of the sparse matrix. If the entry is zero or if the indices are out of range, return 0.
"""
function getindex(s::AbstractSparseMat{SizeType, T}, i::Integer, j::Integer) where {SizeType, T}
    if 1 <= i <= s.nrows
        get(s.row_vectors[i], j, zero(T))
    else
        zero(T)
    end
end

"""
    function minus_mult!(mat::AbstractSparseMat{SizeType, T}, row1::Integer, row2::Integer) 

turns row1 into row1-row2*a. The return value is the mutated `mat`.
"""
function minus_mult!(mat::AbstractSparseMat{SizeType, T}, row1::Integer, row2::Integer, a) where {SizeType, T}
    v1 = mat.row_vectors[row1]
    v2 = mat.row_vectors[row2]
    for (column_number, value) in v2
        oldvalue = get(v1, column_number, zero(T))
        newvalue = minus_mult(oldvalue, value, a)
        if is_not_zero(newvalue)
            v1[column_number] = newvalue
            if is_zero(oldvalue)
                push!(mat.nonzero_rows_in_col[column_number], row1)
                mat.coldensity[column_number] += 1
            end
        elseif is_not_zero(oldvalue) # for newvalue==0 case, only do something if oldvalue was not already zero
            delete!(v1, column_number)
            pop!(mat.nonzero_rows_in_col[column_number], row1)
            mat.coldensity[column_number] -= 1
        end
    end
    mat.rowdensity[row1] = length(v1)
    mat
end

"""
    function minus_mult!(mat::AbstractSparseMat{SizeType, T}, row1::Integer, v2::Vector{Pair{SizeType, T}}, a) 

turns row1 into row1-v2*a. The return value is the mutated `mat`.
"""
function minus_mult!(mat::AbstractSparseMat{SizeType, T}, row1::Integer, v2::Vector{Pair{SizeType, T}}, a) where {SizeType, T}
    v1 = mat.row_vectors[row1]
    len = length(v2)
    for i0 in 1:len
        i = ((i0 + row1) % len) + 1 # randomize iteration order to avoid threads competing for resources
        (column_number, value) = v2[i]
        oldvalue = get(v1, column_number, zero(T))
        newvalue = minus_mult(oldvalue, value, a)
        if is_not_zero(newvalue)
            v1[column_number] = newvalue
            if is_zero(oldvalue)
                push!(mat.nonzero_rows_in_col[column_number], row1)
                mat.coldensity[column_number] += 1
            end
        elseif is_not_zero(oldvalue) # for newvalue==0 case, only do something if oldvalue was not already zero
            delete!(v1, column_number)
            pop!(mat.nonzero_rows_in_col[column_number], row1)
            mat.coldensity[column_number] -= 1
        end
    end
    mat.rowdensity[row1] = length(v1)
    mat
end

"""
    function next_diag_pivot!(mat::AbstractSparseMat{SizeType, T})::Bool where {SizeType, T}

Move pivot from (k,k) to (k+1,k+1); only works for pre-ordered linear system (e.g. obtained by solving at some random parameter value) without any need for nontrivial pivoting.
"""
function next_diag_pivot!(mat::AbstractSparseMat{SizeType, T})::Bool where {SizeType, T}
    if mat.nrows_pivoted == mat.nrows
        return false
    end
    mat.pivot_row += 1
    mat.pivot_col += 1
    mat.nrows_pivoted += 1
    return true
end

const MIN_CHUNK_PER_THREAD = 2048

function _find_min_nonzero_elem_and_position(v::Vector{SizeType}, lower::SizeType, upper::SizeType)::Tuple{SizeType, SizeType} where SizeType
    min_elem = typemax(SizeType)
    min_position = zero(SizeType)
    for i in lower:upper
        if zero(SizeType) < v[i] < min_elem
            min_elem = v[i]
            min_position = i
        end
    end
    (min_elem, min_position)
end

function _find_min_nonzero_elem_and_position(v::Vector{SizeType}, lower::SizeType, upper::SizeType, filter_function)::Tuple{SizeType, SizeType} where SizeType
    min_elem = typemax(SizeType)
    min_position = zero(SizeType)
    for i in lower:upper
        if zero(SizeType) < v[i] < min_elem && filter_function(i)
            min_elem = v[i]
            min_position = i
        end
    end
    (min_elem, min_position)
end

function find_min_nonzero_elem_and_position(v::Vector{SizeType}, lower::SizeType, upper::SizeType)::Tuple{SizeType, SizeType} where SizeType
    if upper - lower <= MIN_CHUNK_PER_THREAD
        _find_min_nonzero_elem_and_position(v, lower, upper)
    else
        middle = (lower + upper) ÷ SizeType(2)
        result1 = Threads.@spawn find_min_nonzero_elem_and_position(v, lower, middle)
        result2 = Threads.@spawn find_min_nonzero_elem_and_position(v, middle + one(SizeType), upper)
        (elem1, pos1) = fetch(result1)
        (elem2, pos2) = fetch(result2)
        elem1 < elem2 ? (elem1, pos1) : (elem2, pos2)
    end
end

function find_min_nonzero_elem_and_position(v::Vector{SizeType}, lower::SizeType, upper::SizeType, filter_function)::Tuple{SizeType, SizeType} where SizeType
    if upper - lower <= MIN_CHUNK_PER_THREAD
        _find_min_nonzero_elem_and_position(v, lower, upper, filter_function)
    else
        middle = (lower + upper) ÷ SizeType(2)
        result1 = Threads.@spawn find_min_nonzero_elem_and_position(v, lower, middle, filter_function)
        result2 = Threads.@spawn find_min_nonzero_elem_and_position(v, middle + one(SizeType), upper, filter_function)
        (elem1, pos1) = fetch(result1)
        (elem2, pos2) = fetch(result2)
        elem1 < elem2 ? (elem1, pos1) : (elem2, pos2)
    end
end

"""
    function findpivotrow!(mat::AbstractSparseMat{SizeType, T})::Bool where {SizeType, T}

Find a sparsity-preserving pivot row
"""
@inline function findpivotrow!(mat::AbstractSparseMat{SizeType, T})::Bool where {SizeType, T}
    if mat.nrows_pivoted == mat.nrows # reduction has finished
        return false
    end

    foundpivot = false

    # if SEARCH_SINGLET_COLUMNS
    #     for i in eachindex(mat.coldensity)
    #         if mat.coldensity[i] == one(SizeType)
    #             for row in mat.nonzero_rows_in_col[i]
    #                 if row > mat.nrows_pivoted
    #                     mat.pivot_row = row
    #                     break
    #                 end
    #             end
    #             mat.pivot_col = i
    #             mat.nrows_pivoted += one(SizeType)
    #             foundpivot = true
    #             break
    #         end
    #     end
    #     if foundpivot
    #         return true
    #     end
    # end

    # select sparsest row as pivot row
    min_nonzero_elem, min_nonzero_position = find_min_nonzero_elem_and_position(mat.rowdensity, mat.nrows_pivoted + one(SizeType), mat.nrows)
    if min_nonzero_elem < typemax(SizeType)
        mat.pivot_row = min_nonzero_position # the 2nd element in the tuple is the position
        foundpivot = true
    end
    foundpivot
end

"""
    function findpivot!(mat::AbstractSparseMat{SizeType, T})::Bool where {SizeType, T}

Find a sparsity-preserving pivot (i,j) for the next step of Gaussian elimination
"""
function findpivot!(mat::AbstractSparseMat{SizeType, T})::Bool where {SizeType, T}
    foundpivot = findpivotrow!(mat)
    
    # within the selected pivot row, select the sparsest column as pivot column; for columns with the same sparsity, prefer the columns with smaller numbers
    if foundpivot
        mindensity::SizeType = mat.nrows + one(SizeType)
        for column_number in keys(mat.row_vectors[mat.pivot_row])
            current_density  = mat.coldensity[column_number]
            if current_density < mindensity
                mindensity = current_density
                mat.pivot_col = column_number
            elseif current_density == mindensity
                if column_number < mat.pivot_col
                    mat.pivot_col = column_number
                end
            end
            # The code below was intended to improve performance, but didn't seem to do so in some tested examples
            # if mindensity == 1
            #     break
            # end
        end
        mat.nrows_pivoted += one(SizeType)
    end
    return foundpivot
end

"""
    function findpivot_no_optimization!(mat::AbstractSparseMat{SizeType, T})::Bool where {SizeType, T}

Naive pivot-finding function identifying the next non-zero row and its first nonzero column. This generally destroys sparsity.
"""
function findpivot_no_optimization!(mat::AbstractSparseMat{SizeType, T})::Bool where {SizeType, T}
    if mat.nrows_pivoted == mat.nrows
        return false
    end

    foundpivot = false
    # select next nonzero row as pivot row
    for i in (mat.nrows_pivoted + one(SizeType)):mat.nrows
        if zero(SizeType) < mat.rowdensity[i]
            mat.pivot_row = i
            foundpivot = true
            break
        end
    end

    # within the selected pivot row, select the column with the lowest index as the pivot column
    if foundpivot
        mat.pivot_col = minimum(keys(mat.row_vectors[mat.pivot_row]))
        mat.nrows_pivoted += one(SizeType)
    end

    return foundpivot
end


"""
Helper function.
"""
function _has_non_forbidden_column(row_vector::SparseVec, forbidden_cols)::Bool
    result = false
    for col in keys(row_vector)
        if !(col in forbidden_cols)
            result = true
            break
        end
    end
    result
end

struct HasNonForbidden{SizeType, T}
    mat::SparseMat{SizeType, T}
end

function (h::HasNonForbidden{SizeType, T})(i::SizeType) where {SizeType, T}
    _has_non_forbidden_column(h.mat.row_vectors[i], h.mat.forbidden_col_pivots)
end

"""
Similar to `findpivotrow!`, but avoid columns in `mat.forbidden_col_pivots`.
"""
function findpivotrow_with_forbidden!(mat::AbstractSparseMat{SizeType, T})::Bool where {SizeType, T}
    if mat.nrows_pivoted == mat.nrows
        return false
    end
    foundpivot = false
    
    # select sparsest row, with at least one nonzero entry in a non-forbidden column position, as pivot row
    min_nonzero_elem, min_nonzero_position = find_min_nonzero_elem_and_position(mat.rowdensity, mat.nrows_pivoted + one(SizeType), mat.nrows, HasNonForbidden(mat))
    if min_nonzero_elem < typemax(SizeType)
        mat.pivot_row = min_nonzero_position # the 2nd element in the tuple is the position
        foundpivot = true
    end
    foundpivot
end

"""
Similar to `findpivot!`, but avoid columns in `mat.forbidden_col_pivots`.
"""
function findpivot_with_forbidden!(mat::AbstractSparseMat{SizeType, T})::Bool where {SizeType, T}
    foundpivot = findpivotrow_with_forbidden!(mat)

    # within the selected pivot row, select the sparsest column as pivot column
    if foundpivot
        local mindensity::SizeType = mat.nrows + one(SizeType)
        for column_number in keys(mat.row_vectors[mat.pivot_row])
            if !in(column_number, mat.forbidden_col_pivots)
                current_density  = mat.coldensity[column_number]
                if current_density < mindensity
                    mindensity = current_density
                    mat.pivot_col = column_number
                elseif current_density == mindensity
                    if column_number < mat.pivot_col
                        mat.pivot_col = column_number
                    end
                end
            end
            # if mindensity == one(SizeType)
            #     break
            # end
        end
        # @assert mindensity<mat.nrows+1 "Error in finding column pivot. Inconsistent choice of forbidden pivots?"
        mat.nrows_pivoted += one(SizeType)
    end
    return foundpivot
end

"""
    function findpivot_partial!(mat::AbstractSparseMat{SizeType, T})::Bool where {SizeType, T}

Select the sparsest row but picks the first nonzero column, i.e. without optimizing column choice in pivot.
"""
function findpivot_partial!(mat::AbstractSparseMat{SizeType, T})::Bool where {SizeType, T}
    foundpivot = findpivotrow!(mat)

    # within the selected pivot row, select the leftmost nonzero column (i.e. with smallest column number) as pivot column
    if foundpivot
        mincol = mat.ncols+one(SizeType)
        for column_number in keys(mat.row_vectors[mat.pivot_row])
            if column_number < mincol
                mincol = column_number
            end
        end
        mat.pivot_col = mincol
        mat.nrows_pivoted += one(SizeType)
    end
    return foundpivot
end

"""
    Wrapper type holding a list of preferred columns in an indexable member variable `column_preference_list`.
"""
struct FindPivotWithPreference{T}
    column_preference_list::T
end

"""
    (p::FindPivotWithPreference)(mat::AbstractSparseMat)
    
Use a callable object of type `FindPivotWithPreference` to find the pivots for the next step of Gaussian elimination.
"""
function (p::FindPivotWithPreference)(mat::AbstractSparseMat{SizeType, T})::Bool where {SizeType, T}
    foundpivot = findpivotrow!(mat)

    # within the selected pivot row, select the column with the largest preference value
    if foundpivot
        mindensity::SizeType = mat.nrows + one(SizeType)
        max_preference = nothing
        for column_number in keys(mat.row_vectors[mat.pivot_row])
            current_preference = p.column_preference_list[column_number]
            if isa(max_preference, Nothing) || current_preference > max_preference
                mindensity = mat.coldensity[column_number]
                max_preference = current_preference
                mat.pivot_col = column_number
            elseif current_preference == max_preference # this branch can only be entered when analyzing the 2nd column candidate onwards, not the first one
                current_density = mat.coldensity[column_number]
                if current_density < mindensity
                    mindensity = current_density
                    # max_preference = current_preference # no need to have this line, as they're already equal
                    mat.pivot_col = column_number
                elseif current_density == mindensity
                    if column_number < mat.pivot_col
                        mat.pivot_col = column_number
                    end
                end
            end
        end
        mat.nrows_pivoted += one(SizeType)
    end
    return foundpivot
end

"""
Similar to `findpivot_partial!`, but avoid columns in `mat.forbidden_col_pivots`.
"""
function findpivot_partial_with_forbidden!(mat::AbstractSparseMat{SizeType, T})::Bool where {SizeType, T}
    foundpivot = findpivotrow_with_forbidden!(mat)

    # within the selected pivot row, select the leftmost nonzero column (i.e. with smallest column number) as pivot column
    if foundpivot
        mincol = mat.ncols+one(SizeType)
        for column_number in keys(mat.row_vectors[mat.pivot_row])
            if !in(column_number, mat.forbidden_col_pivots) && column_number < mincol
                mincol = column_number
            end
        end
        mat.pivot_col = mincol
        mat.nrows_pivoted += one(SizeType)
    end
    return foundpivot
end

"""
Custom macro for swapping.
`@swap x y` is equivalent to `x,y = y,x`
"""
macro swap(x,y)
    quote
        $(esc(x)), $(esc(y)) = $(esc(y)), $(esc(x))
    end
end

"""
    function elim_one_step!(mat::AbstractSparseMat{SizeType, T}; log::Union{Val{true}, Val{false}} = Val(false)) where {SizeType, T}

Perform one step of Gaussian elimination. The pivot row is moved to the top, below previously used pivot rows but above all other rows. Subtract a multiple of the pivot rows from other rows (which are now situated below it) to set the j-th column to zero in these rows. The return value is the mutated `mat`.
"""
function elim_one_step!(mat::AbstractSparseMat{SizeType, T}; log::Union{Val{true}, Val{false}} = Val(false)) where {SizeType, T}
    if mat.pivot_row != mat.nrows_pivoted
        # swap rows to make the pivot row the n-th row, in 3 steps below.

        pivotrow, otherrow = mat.pivot_row, mat.nrows_pivoted

        # 1st step: remove entries from nonzero_rows_in_col
        for column_number in keys(mat.row_vectors[otherrow])
            pop!(mat.nonzero_rows_in_col[column_number], otherrow)
        end
        for column_number in keys(mat.row_vectors[pivotrow])
            pop!(mat.nonzero_rows_in_col[column_number], pivotrow)
            mat.coldensity[column_number] -= one(SizeType) # The row to be used as pivot is removed from coldensity counting, since the previously used pivot rows are not touched any more in computing the (unreduced) row echelon form. However, we do not remove the pivot row from mat.nonzero_rows_in_col, to keep the information available for back substitution.
        end

        # 2nd step: add entries into nonzero_rows_in_col, with pivotrow and otherrow swapped
        for column_number in keys(mat.row_vectors[otherrow])
            push!(mat.nonzero_rows_in_col[column_number], pivotrow)
        end
        for column_number in keys(mat.row_vectors[pivotrow])
            push!(mat.nonzero_rows_in_col[column_number], otherrow)
        end

        # 3rd step: swap row vectors and rowdensity
        @swap mat.rowdensity[otherrow] mat.rowdensity[pivotrow]
        @swap mat.row_vectors[otherrow] mat.row_vectors[pivotrow]

    else # the case of mat.pivot_row == mat.nrows_pivoted, only need to adjust coldensity
        for column_number in keys(mat.row_vectors[mat.nrows_pivoted])
            mat.coldensity[column_number] -= one(SizeType)
        end
    end

    normalize!(mat.row_vectors[mat.nrows_pivoted], mat.pivot_col)

    number_of_rows_to_reduce = 0
    for i in mat.nonzero_rows_in_col[mat.pivot_col]
        if i > mat.nrows_pivoted
            number_of_rows_to_reduce += 1
            mat.rows_to_reduce[number_of_rows_to_reduce] = i
        end
    end

    if log isa Val{true}
        for j in 1:number_of_rows_to_reduce
            i = mat.rows_to_reduce[j]
            push!(mat.reduction_log, (mat.nrows_pivoted, i))
        end
        mat.forward_elimination_cost += (1 + number_of_rows_to_reduce) * length(mat.row_vectors[mat.nrows_pivoted])
        # This is a very rough estimate of the cost of forward elimination
    end

    resize!(mat.pivot_row_packed, length(mat.row_vectors[mat.nrows_pivoted].data))
    counter = 0
    for index_and_value in mat.row_vectors[mat.nrows_pivoted].data
        counter += 1
        mat.pivot_row_packed[counter] = index_and_value
    end
    # pivot_row_packed = collect(mat.row_vectors[mat.nrows_pivoted].data)::Vector{Pair{SizeType, T}} # Turn the sparse vector into a packed (index, value) pair array
    for j in 1:number_of_rows_to_reduce
        i = mat.rows_to_reduce[j]
        minus_mult!(mat, i, mat.pivot_row_packed, mat.row_vectors[i][mat.pivot_col])
    end

    mat
end

"""
    PivotInfoType{SizeType}

Type alias for a named tuple with fields `pivotrows` and `pivotcols`, each of which is a vector of indices of `SizeType`
"""
const PivotInfoType{SizeType} = NamedTuple{(:pivotrows, :pivotcols), Tuple{Vector{SizeType}, Vector{SizeType}}}

"""
    function echelonize_nopivoting!(mat::AbstractSparseMat{SizeType, T}; printfreq = 100) where {SizeType, T}

Echelonize a pre-ordered matrix where diagonal pivots are guaranteed to be valid.
"""
function echelonize_nopivoting!(mat::AbstractSparseMat{SizeType, T}; printfreq::Int = 100)::PivotInfoType{SizeType} where {SizeType, T}
    while next_diag_pivot!(mat)
        if mat.nrows_pivoted % printfreq == 1
            print(stderr, "\rcurrent row = ", mat.nrows_pivoted)
        end        
        elim_one_step!(mat)
    end
    println(stderr)
    return (pivotrows = collect(one(SizeType):mat.nrows), pivotcols = collect(one(SizeType):mat.nrows)) # nrows is used instead of ncols here, as the pivots are diagonal
end

"""
    function reduce_nopivoting!(mat::AbstractSparseMat{SizeType, T}, n_steps::Integer, n_columns_skip::Integer=zero(SizeType); printfreq=100) where {SizeType, T}

Similar to `echelonize_nopivoting!`, but the starting pivot position is `(1, n_columns_skip+1)`, and only exactly `n_steps` of Gaussian elimination is performed.
"""
function reduce_nopivoting!(mat::AbstractSparseMat{SizeType, T}, n_steps::Integer, n_columns_skip::Integer = zero(SizeType); printfreq = 100) where {SizeType, T}
    mat.pivot_col = n_columns_skip
    for i in 1:n_steps
        next_diag_pivot!(mat)
        if mat.nrows_pivoted % printfreq == 1
            print(stderr, "\rcurrent row = ", mat.nrows_pivoted)
        end
        elim_one_step!(mat)
        mat.row_vectors[i] = SparseVec{SizeType, T}() # After using the row to eliminate other rows, we can discard the original row as we're only interested in reducing the last rows
    end
    println(stderr)
    return (pivotrows = collect(one(SizeType):mat.nrows), pivotcols = collect(one(SizeType):mat.nrows))
end

"""
    function reduce_with_preordered_equations!(to_reduce::Vector{SparseVec{SizeType, T}}, relations::Vector{SparseVec{SizeType, T}}, n_variables::Integer, n_columns_skip::Integer = zero(SizeType); printfreq = 100) where {SizeType, T}

Reduce row vectors `to_reduce` against the preordered `relations`, using diagonal pivots.
"""
function reduce_with_preordered_equations!(to_reduce::Vector{SparseVec{SizeType, T}}, equations::Vector{SparseVec{SizeType, T}}, n_variables::Integer, n_columns_skip::Integer = zero(SizeType); printfreq = 100) where {SizeType, T}
    n_eqs = length(equations)
    sparse_mat = SparseMat{SizeType, T}(vcat(equations, to_reduce), SizeType(n_eqs + length(to_reduce)), SizeType(n_variables))
    reduce_nopivoting!(sparse_mat, n_eqs, n_columns_skip, printfreq = printfreq)
    sparse_mat.row_vectors[(n_eqs + 1):end]
end

"""
    function echelonize!(mat::AbstractSparseMat{SizeType, T}; pivotfunc = findpivot!, printfreq = 100, log::Union{Val{true}, Val{false}}=Val(false)) where {SizeType, T}

Turn matrix into the row echelon form (REF), and return the pivots. If `log = Val(true)`, detailed reduction operations are recorded.
"""
function echelonize!(mat::AbstractSparseMat{SizeType, T}; pivotfunc = findpivot!, printfreq = 100, log::Union{Val{true}, Val{false}} = Val(false))::PivotInfoType where {SizeType, T}
    row_labels = collect(one(SizeType):mat.nrows)
    pivotcols = Vector{SizeType}(undef, mat.nrows)
    while pivotfunc(mat)
        if mat.nrows_pivoted % printfreq == 1
        # if true
            print(stderr, "\rcurrent row = ", mat.nrows_pivoted)
        end
        if log isa Val{true}
            push!(mat.reduction_log, (mat.pivot_row, zero(SizeType)))
        end
        elim_one_step!(mat, log=log)
        if mat.pivot_row != mat.nrows_pivoted
            row_labels[mat.nrows_pivoted], row_labels[mat.pivot_row] = row_labels[mat.pivot_row], row_labels[mat.nrows_pivoted]
        end
        pivotcols[mat.nrows_pivoted] = mat.pivot_col
    end
    resize!(mat.row_vectors, mat.nrows_pivoted)
    sizehint!(mat.row_vectors, mat.nrows_pivoted)
    resize!(row_labels, mat.nrows_pivoted)
    resize!(pivotcols, mat.nrows_pivoted)
    println(stderr)
    return (pivotrows = row_labels, pivotcols = pivotcols)
end

"""
    function back_subst!(mat::AbstractSparseMat{SizeType, T}, pivots::PivotInfoType) where {SizeType, T}
    
Turn `mat` from REF form to RREF form, using column permutations stored in `pivots` The return value is the mutated `mat`.
"""
function back_subst!(mat::AbstractSparseMat{SizeType, T}, pivots::PivotInfoType) where {SizeType, T}
    rank = length(pivots.pivotcols)
    colpivots = pivots.pivotcols
    for i in rank:-1:2
        pivotcol = colpivots[i]
        @assert is_one(mat.row_vectors[i][pivotcol])
        affected_rows = mat.nonzero_rows_in_col[pivotcol]
        for j in collect(affected_rows)
            if j<i
                if haskey(mat.row_vectors[j], pivotcol)
                    colvalue = mat.row_vectors[j][pivotcol]
                    # @assert !haskey(mat.row_vectors[j], pivotcol)
                    minus_mult!(mat, j, i, colvalue)
                end
            end
        end
    end
    mat
end

"""
    rref!(mat::AbstractSparseMat{SizeType, T}; pivotfunc = findpivot!, printfreq=100) where {SizeType, T}

Echelonize the matrix, perform back substitution, and return the pivots.
"""
function rref!(mat::AbstractSparseMat{SizeType, T}; pivotfunc = findpivot!, printfreq = 100)::PivotInfoType where {SizeType, T}
    pivots = echelonize!(mat, pivotfunc = pivotfunc, printfreq = printfreq)
    back_subst!(mat, pivots)
    return pivots
end

"""
    function rref_nopivoting!(mat::AbstractSparseMat{SizeType, T}; printfreq=100) where {SizeType, T}

Similar to `rref!` but uses trivial diagonal pivots.
"""
function rref_nopivoting!(mat::AbstractSparseMat{SizeType, T}; printfreq = 100)::PivotInfoType where {SizeType, T}
    pivots = echelonize_nopivoting!(mat, printfreq=printfreq)
    back_subst!(mat, pivots)
    return pivots
end


####################################
#Code related to dependency tracking

"""
    function reduce_with_ref_mat!(vecs::Vector{SparseVec{SizeType, T}}, mat::AbstractSparseMat{SizeType, T}, pivots::PivotInfoType) where {SizeType, T}

Reduce `vecs` against `mat` which has already been turned into row-echelon form (with column permutations). The recorded `pivots`, e.g. from `echelonize!()` must be supplied. The return value is a vector of needed rows from `mat`, for the purpose of dependence tracking.
"""
function reduce_with_ref_mat!(vecs::Vector{SparseVec{SizeType, T}}, mat::AbstractSparseMat{SizeType, T}, pivots::PivotInfoType)::Vector{SizeType} where {SizeType, T}
    needed_rows = SizeType[]
    rank = length(pivots.pivotcols)
    colpivots = pivots.pivotcols
    for i in 1:rank
        pivotcol = colpivots[i]
        @assert is_one(mat.row_vectors[i][pivotcol])
        row_i_is_need = false
        for j in 1:length(vecs)   
            if haskey(vecs[j], pivotcol)
                row_i_is_need = true
                colvalue = vecs[j][pivotcol]
                minus_mult!(vecs[j], mat.row_vectors[i], colvalue)
            end
        end
        if row_i_is_need
            push!(needed_rows, i)
        end
    end
    needed_rows
end

"""
    function reduce_with_ref_mat!(vec::SparseVec{SizeType, T}, mat::AbstractSparseMat{SizeType, T}, pivots::PivotInfoType) where {SizeType, T}

Overloaded version of function, which turns vec into [vec] in order to call the original version. The return value is the mutated `vec`.
"""
function reduce_with_ref_mat!(vec::SparseVec{SizeType, T}, mat::AbstractSparseMat{SizeType, T}, pivots::PivotInfoType) where {SizeType, T}
    reduce_with_ref_mat!([vec], mat, pivots)
    # need_rows is returned
end

"""
    get_needed_rows_fast(to_reduce_vecs::Vector{<:SparseVec}, eqs::AbstractSparseMat, pivotcols::Vector{<:Integer})

Given a pre-echelonized sparse matrix `eqs` and the ordering of pivot columns `pivotcols`, determine the rows of the matrix that are needed to reduce a sparse vector `to_reduce_vecs` by looking at the nonzero patterns without running an actual reduction, i.e. skipping detection of possible cancellation during reduction.
"""
function get_needed_rows_fast(to_reduce_vecs::Vector{<:SparseVec}, eqs::AbstractSparseMat, pivotcols::Vector{<:Integer})
    pivotcol_numbering = Dict(pivotcols[i] => i for i in eachindex(pivotcols))
    needed_rows_set = Set{Int}()
    needed_rows_vector = Int[]
    for v in to_reduce_vecs
        for column in keys(v)
            row_to_reduce_against = get(pivotcol_numbering, column, 0)
            if row_to_reduce_against != 0 && !in(row_to_reduce_against, needed_rows_set)
                push!(needed_rows_set, row_to_reduce_against)
                push!(needed_rows_vector, row_to_reduce_against)
            end
        end
    end
    first_unsearched_row = 1
    while first_unsearched_row <= length(needed_rows_vector)
        for column in keys(eqs.row_vectors[needed_rows_vector[first_unsearched_row]])
            row_to_reduce_against = get(pivotcol_numbering, column, 0)
            if row_to_reduce_against != 0 && !in(row_to_reduce_against, needed_rows_set)
                push!(needed_rows_set, row_to_reduce_against)
                push!(needed_rows_vector, row_to_reduce_against)
            end
        end
        first_unsearched_row += 1
    end
    needed_rows_vector
end

"""
    function _extract_reduction_steps(reduction_log1::Vector{Tuple{SizeType, SizeType}}, nrows, pivots::PivotInfoType) where{SizeType}

Process the reduction log to turns it into a set of instructions for reducing a rearranged matrix again at different parameter values, assuming that the rearranged matrix does not contain redundant rows (which became zero in the reduction process) and that the rearranged matrix adopts a new ordering of rows to make sure that the kth row is always the pivot row during the kth elimination step. This function is used internally inside another function `trace_needed_equations!`.
"""
function _extract_reduction_steps(reduction_log1::Vector{Tuple{SizeType, SizeType}}, nrows, pivots::PivotInfoType) where{SizeType}

    instructions = Tuple{SizeType, SizeType}[]
    pivotrows = pivots.pivotrows
    pivotrowsSet = Set(pivotrows)
    pivotrows_ordering = Dict{SizeType, SizeType}()
    for i in 1:length(pivotrows)
        pivotrows_ordering[pivotrows[i]] = i
    end
    # if the pivot rows, numbered according to the original matrix, are [a1, a2, a3 ...], then e.g. `pivotrows_ordering[a2]` is equal to 2.
    pivotrows_echelonize_ordering = map(x->x[1], filter(t->t[2] == zero(SizeType), reduction_log1)) # the pivot rows numbered not according to the ordering of the input matrix, but the ordering during `echelonize!()`.
    reduction_log = filter(t -> t[2] != zero(SizeType), reduction_log1)

    if length(reduction_log) == 0
        return instructions # empty result returned
    end

    row_labels = collect(1:nrows)
    current_position = 1 # current position in processing the vector instructions
    for i in 1:length(pivotrows)
        if pivotrows_echelonize_ordering[i] != i
            row_labels[pivotrows_echelonize_ordering[i]], row_labels[i] = row_labels[i], row_labels[pivotrows_echelonize_ordering[i]]
        end
        # now row_labels[i] == j means the ith row during echelonization comes from the jth row in the original matrix
        
        if reduction_log[current_position][1] == i
            while current_position<=length(reduction_log) && reduction_log[current_position][1] == i
                reduced_row = row_labels[reduction_log[current_position][2]] # reduced_row is according to the numbering of the original matrix
                if reduced_row in pivotrowsSet
                    push!(instructions, (i, pivotrows_ordering[reduced_row])) # pivotrows_ordering[reduced_row] is the step number at which the row is used at a pivot, since eventually `instructions` uses a numbering scheme which assumes that the rows have been ordered so that we can always use the kth row as the pivot row in the kth elimination step
                end
                current_position += 1
            end
        end
        if current_position > length(reduction_log)
            break
        end
    end
    instructions
end

"""
    dependent_rows(reduction_steps::Vector{Tuple{SizeType, SizeType}}, needed_rows::Vector{SizeType})

Given a record of reduction steps (used to perform echelonization), `reduction_steps`, and a vector of needed rows to reduce some sparse vector, `needed_rows`, e.g. returned by `get_needed_rows_fast` or `reduce_with_ref_mat!`, determine the rows in the original matrix that are actually needed (to be echelonized to produce an echelonized matrix for use in reducing the sparse vector).
"""
function dependent_rows(reduction_steps::Vector{Tuple{SizeType, SizeType}}, needed_rows::Vector{SizeType}) where {SizeType}
    if length(needed_rows) == 0
        return Int[]
    end
    
    @assert all(x->x[1]<x[2], reduction_steps)

    # all_rows = Set(vcat(map(x->x[1], reduction_steps), map(x->x[2], reduction_steps)))
    all_rows = Set{SizeType}()
    for s in reduction_steps
        push!(all_rows, s[1])
        push!(all_rows, s[2])
    end

    dependency_table = Dict{SizeType, Set{SizeType}}()
    for row in all_rows
        dependency_table[row] = Set{SizeType}()
    end
    for step in reduction_steps
        push!(dependency_table[step[2]], step[1])
    end
    last_row = maximum(needed_rows)
    dependent_flags = fill(false, last_row)
    for row in needed_rows
        dependent_flags[row] = true
    end
    for i in last_row:-1:1
        if dependent_flags[i]
            if haskey(dependency_table, i)
                for dependency in dependency_table[i]
                    dependent_flags[dependency] = true
                end
            end
        end
    end
    findall(dependent_flags) # return a vector of indices with boolean value "true"
end

"""
    function trace_needed_equations!(eqs::AbstractSparseMat{SizeType, T}, to_reduce_vecs::Union{Vector{SparseVec{SizeType, T}}, Nothing} = nothing; pivotfunc = findpivot!, printfreq=100, refine_pivots=false, fastmode::Bool = false) where {SizeType,T}

Reduce `to_reduce_vecs` against `eqs` to find the minimal set of equations needed. Return `(eqs=needed_eqs_numbers, vars=all_columns_ordered)`. If `fastmode = true` is used, nonzero patterns of `to_reduce_vecs` and the post-echelonized `eqs` will be used to determine the needed equations (rows), without running the actual reduction.
"""
function trace_needed_equations!(eqs::AbstractSparseMat{SizeType, T}, to_reduce_vecs::Union{Vector{SparseVec{SizeType, T}}, Nothing} = nothing; pivotfunc = findpivot!, printfreq = 100, refine_pivots = false, fastmode::Bool = false) where {SizeType,T}
    if !refine_pivots
        pivots = echelonize!(eqs, pivotfunc = pivotfunc, log = Val(true), printfreq=printfreq)
    else
        eqs_copy = deepcopy(eqs)
        pivots_crude = echelonize!(eqs_copy, pivotfunc = pivotfunc, log=Val(false), printfreq=printfreq)
        eqs.forbidden_col_pivots = Set{SizeType}(setdiff(1:eqs_copy.ncols, pivots_crude.pivotcols)) 
        pivots = echelonize!(eqs, pivotfunc = findpivot_with_forbidden!, log=Val(true), printfreq=printfreq)
    end
    reduction_steps = _extract_reduction_steps(eqs.reduction_log, eqs.nrows, pivots)
    # reduction_steps tell you, for example, to use Eq. 1 to reduce Eq. 100, where all the numberings are w.r.t. the subset of rows as in pivots.pivotrows
    dependencies = if typeof(to_reduce_vecs) == Nothing
        collect(1:length(pivots.pivotrows))
    else
        needed_rows_in_reduction = fastmode ? get_needed_rows_fast(to_reduce_vecs, eqs, pivots.pivotcols) : reduce_with_ref_mat!(to_reduce_vecs, eqs, pivots)
        # note that `to_reduce_vecs`, even though mutated in the 2nd branch of the line above, will not be used subsequently, and it only matter that the correct `needed_rows_in_reduction` is returned.
        
        dependent_rows(reduction_steps, needed_rows_in_reduction) # the output of dependent_rows is always sorted from small to large, and is numbered from 1,2,3... for the subset of rows in pivots.pivotrows rather than the original rows in eqs before echelonize!()
    end

    needed_eqs_numbers = pivots.pivotrows[dependencies] # a list of needed equations as numbered in the order of the original matrix
    
    column_ordering = pivots.pivotcols[dependencies] # a list of needed variables as numbered in the order of the original matrix
    all_columns_ordered = vcat(column_ordering, setdiff(collect(1:eqs.ncols), column_ordering))

    (eqs=needed_eqs_numbers, vars=all_columns_ordered)
end
