################################
# code related to sparse vector

"""
`SparseVec{IndexType, T}` is a sparse row vector, internally represented as a dictinary mapping column indices (keys) of type `IndexType` to values of type `T`. The constructor accepts a dictinary for the mapping, or alternatively a vector of pairs for the mapping.
"""
struct SparseVec{IndexType, T}
    data::Dict{IndexType, T}
    SparseVec{IndexType, T}() where {IndexType, T} = new(Dict{IndexType, T}())
    SparseVec{IndexType, T}(data::Dict{IndexType, T}) where {IndexType, T} = new(data)
end

SparseVec(data::Dict{IndexType, T}) where {IndexType, T} = SparseVec{IndexType, T}(data)

SparseVec(data::Vector{Pair{IndexType, T}}) where {IndexType, T} = SparseVec{IndexType, T}(Dict(data))

SparseVec(data::AbstractVector) = SparseVec(Dict(data))

"""
    ==(a::SparseVec, b::SparseVec)

tests equality between two sparse vectors.
"""
function ==(a::SparseVec, b::SparseVec)
    ==(a.data, b.data)
end

"""
    hash(a::SparseVec, b)

Custom hash function for `SparseVec` objects; uses only the information from the `data` member variable.
"""
function hash(a::SparseVec, b)
    hash(a.data, b)
end

"""
    show(io::IO, v::SparseVec)

Custom REPL display for `SparseVec` objects
"""
function show(io::IO, v::SparseVec)
    print(io, "SparseVec([")
    printed_one_term = false
    for entry in v.data
        if printed_one_term
            print(io, ", ")
        end
        print(io, entry)
        printed_one_term = true
    end
    print(io, "])")
end

"""
    show(io::IO, mime::MIME"text/plain", v::SparseVec)

Custom printing for `SparseVec` objects
"""
function show(io::IO, mime::MIME"text/plain", v::SparseVec)
    print(io, "Sparse vector with data\n")
    show(io, mime, collect(v.data))
end

"""
    forwarded method that operates on the `data` member variable, which is a dictionary, of `SparseVec` objects
"""
function setindex!(v::SparseVec, i, x)
    setindex!(v.data, i, x)
    v
end

"""
    forwarded method that operates on the `data` member variable, which is a dictionary, of `SparseVec` objects
"""
function getindex(v::SparseVec, i)
    getindex(v.data, i)
end

"""
    forwarded method that operates on the `data` member variable, which is a dictionary, of `SparseVec` objects
"""
function delete!(v::SparseVec, i)
    delete!(v.data, i)
    v
end

"""
    forwarded method that operates on the `data` member variable, which is a dictionary, of `SparseVec` objects
"""
function haskey(v::SparseVec, i)
    haskey(v.data, i)
end

"""
    forwarded method that operates on the `data` member variable, which is a dictionary, of `SparseVec` objects
"""
function get(v::SparseVec, i, x)
    get(v.data, i, x)
end

"""
    forwarded method that operates on the `data` member variable, which is a dictionary, of `SparseVec` objects
"""
function iterate(v::SparseVec)
    iterate(v.data)
end

"""
    forwarded method that operates on the `data` member variable, which is a dictionary, of `SparseVec` objects
"""
function iterate(v::SparseVec, state)
    iterate(v.data, state)
end

"""
    forwarded method that operates on the `data` member variable, which is a dictionary, of `SparseVec` objects
"""
function values(v::SparseVec)
    values(v.data)
end

"""
    forwarded method that operates on the `data` member variable, which is a dictionary, of `SparseVec` objects
"""
function keys(v::SparseVec)
    keys(v.data)
end

"""
    forwarded method that operates on the `data` member variable, which is a dictionary, of `SparseVec` objects
"""
function length(v::SparseVec)
    length(v.data)
end

"""
    forwarded method that operates on the `data` member variable, which is a dictionary, of `SparseVec` objects
"""
function empty!(v::SparseVec)
    empty!(v.data)
    v
end

"""
    forwarded method that operates on the `data` member variable, which is a dictionary, of `SparseVec` objects
"""
function sizehint!(v::SparseVec, size::Integer)
    sizehint!(v.data, size)
    v
end

"""
    function to_sparse_vec(dense_array::Vector{T})

converts a dense vector of type `Vector{T}` to a sparse vector of type `SparseVec{Int, T}`.
"""
function to_sparse_vec(dense_array::Vector{T}) where {T}
    SparseVec(
        map(
            pair -> pair[1] => pair[2],
            filter(pair -> is_not_zero(pair[2]), [enumerate(dense_array)...])
        )
    )
end

"""
    function minus_mult!(v1::SparseVec{IndexType, T}, v2::SparseVec{IndexType, T}, a)

calculates `v1 - v2 * a` and overwrite the result into `v1`.
"""
function minus_mult!(v1::SparseVec{IndexType, T}, v2::SparseVec{IndexType, T}, a) where {IndexType, T}
    for (column_number, value) in v2
        newvalue = minus_mult(get(v1, column_number, zero(T)), value, a)
        if is_not_zero(newvalue)
            v1[column_number] = newvalue
        else
            delete!(v1, column_number)
        end
    end
    v1
end

"""
    function normalize!(v::SparseVec{IndexType, T}, n::IndexType) where {IndexType, T}

normalize a sparse row vector v such that the n-th column is 1
"""
function normalize!(v::SparseVec{IndexType, T}, n::IndexType) where {IndexType, T}
    inverse = inv(v[n])
    map!(x -> x * inverse, values(v))
    v
end

"""
    function to_dense_vec(v::SparseVec{IndexType, T})::Vector{T}

converts a sparse vector `v` into a dense vector. We require that `IndexType` is an integer type.
"""
function to_dense_vec(v::SparseVec{IndexType, T}; padded_length = nothing)::Vector{T} where {IndexType<:Integer, T}
    current_length = maximum(keys(v))
    if padded_length isa Integer
        if padded_length > current_length
            current_length = padded_length
        end
    end
    result = zeros(T, current_length)
    for (key, value) in v
        result[key] = value
    end
    result
end

"""
    map_sparsevec_indices(mapping_dict, v::SparseVec)

Return a new `SparseVec` object where the values are unchanged from the ones in `v` but the keys are transformed according to `mapping_dict`, which is either a dictionary or a vector (in the latter case, the keys of `v` must be positive integers).
"""
function map_sparsevec_indices(mapping_dict::Union{Dict{IndexType, S}, Vector{S}}, v::SparseVec{IndexType, T})::SparseVec{S, T} where {IndexType, S, T}
    new_vec_data = Dict{S, T}()
    for (index, value) in v.data
        try
            new_vec_data[mapping_dict[index]] = value
        catch
            error("mapping_dict does not contain a necessary key?")
        end
    end
    SparseVec{S, T}(new_vec_data)
end

"""
    map_sparsevec_indices(mapping_func::Function, newvaluetype, v::SparseVec)

Return a new `SparseVec` object where values are unchanged from the ones in `v` but they keys have been transformed by a function `mapping_func`. The type of keys after the transformation is specified by `newindextype`.
"""
function map_sparsevec_indices(mapping_func::Function, newindextype::Type{S}, v::SparseVec{IndexType, T})::SparseVec{S, T} where {IndexType, S, T}
    new_vec_data = Dict{S, T}()
    for (index, value) in v.data
        try
            new_vec_data[mapping_func(index)] = value
        catch
            error("mapping_func cannot map the necessary key?")
        end
    end
    SparseVec{S, T}(new_vec_data)
end

"""
    map_sparsevec_indices(mapping_func::Function, v::SparseVec)

Return a new `SparseVec` object where values are unchanged from the ones in `v` but they keys have been transformed by a function `mapping_func`. The type of keys does not need to be specified in this overloaded method, but is determined dynamically, which may incur a runtime cost.
"""
function map_sparsevec_indices(mapping_func::Function, v::SparseVec{IndexType, T})::SparseVec{<:Any, T} where {IndexType, T}
    new_vec_data = Dict{Any, T}()
    for (index, value) in v.data
        try
            new_vec_data[mapping_func(index)] = value
        catch
            error("mapping_func cannot map the necessary key?")
        end
    end
    new_vec_data_narrowed = Dict(k => v for (k, v) in new_vec_data) # turn the key type from Any to the actual type that occurs
    SparseVec(new_vec_data_narrowed)
end

"""
    map_sparsevec_values(mapping_dict, v::SparseVec)

Transform the values of `v` by the dictionary `mapping_dict`, and return the transformed `SparseVec` object.
"""
function map_sparsevec_values(mapping_dict::Dict{T, S}, v::SparseVec{IndexType, T})::SparseVec{IndexType, S} where {IndexType, S, T}
    new_vec_data = Dict{IndexType, S}()
    for (index, value) in v.data
        try
            new_vec_data[index] = mapping_dict[value]
        catch
            error("mapping_dict does not contain a necessary key?")
        end
    end
    SparseVec{IndexType, S}(new_vec_data)
end

"""
    map_sparsevec_values(mapping_func::Function, newvaluetype, v::SparseVec)

Transform the value of of `v` using the funtion `mapping_func`. The new type of values is supplied as `newvaluetype`.
"""
function map_sparsevec_values(mapping_func::Function, newvaluetype::Type{S}, v::SparseVec{IndexType, T})::SparseVec{IndexType, S} where {IndexType, S, T}
    new_vec_data = Dict{IndexType, S}()
    for (index, value) in v.data
        try
            new_vec_data[index] = mapping_func(value)
        catch
            error("mapping_func cannot map the necessary key?")
        end
    end
    SparseVec{IndexType, S}(new_vec_data)
end

"""
    map_sparsevec_values(mapping_func::Function, v::SparseVec)

Transform the value of of `v` using the funtion `mapping_func`. In this overloaded version, the new value type is not supplied by the user but dynamically determined.
"""
function map_sparsevec_values(mapping_func::Function, v::SparseVec{IndexType, T})::SparseVec{IndexType} where {IndexType, T}
    new_vec_data = Dict{IndexType, Any}()
    for (index, value) in v.data
        try
            new_vec_data[index] = mapping_func(value)
        catch
            error("mapping_func cannot map the necessary key?")
        end
    end
    new_vec_data_narrowed = Dict(k => v for (k, v) in new_vec_data)
    SparseVec(new_vec_data_narrowed)
end
