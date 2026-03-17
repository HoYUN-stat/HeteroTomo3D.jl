"""
    EvaluationGrid{I<:Integer}

3D array of size `(2, s, r, n)` storing the discrete 2D detector voxel coordinates for X-ray evaluation, i.e.,
``X[i, j, k] \\in \\mathbb{Z}^{2}``.

# Fields
- `blocks::Array{NTuple{2,I}, 3}`: 3D array of size `(s, r, n)` containing the coordinates.
- `s::Int64`: Number of evaluation points per view.
- `r::Int64`: Number of projection angles.
- `n::Int64`: Number of 3D functions.
- `m::Int64`: Voxel grid resolution/dimension parameter.

```jldoctest
julia> using HeteroTomo3D;

julia> s, r, n, m = 2, 3, 2, 64;

julia> blocks = fill((1, 1), s, r, n);

julia> eval_grid = EvaluationGrid(blocks, s, r, n, m);

julia> size(eval_grid) == (s, r, n)
true
```

See also [`QuaternionGrid`](@ref).
"""
struct EvaluationGrid{I<:Integer}
    blocks::Array{NTuple{2,I},3}
    s::Int64
    r::Int64
    n::Int64
    m::Int64
end
Base.eltype(X::EvaluationGrid) = eltype(X.blocks)
Base.size(X::EvaluationGrid) = (X.s, X.r, X.n)
Base.size(X::EvaluationGrid, i::Int) = size(X)[i]
Base.getindex(X::EvaluationGrid, i::Int) = X.blocks[i]
Base.getindex(X::EvaluationGrid, i::Int, j::Int, k::Int) = X.blocks[i, j, k]
Base.setindex!(X::EvaluationGrid, v, i::Int) = (X.blocks[i] = v)
Base.setindex!(X::EvaluationGrid, v, i::Int, j::Int, k::Int) = (X.blocks[i, j, k] = v)

function Base.show(io::IO, mime::MIME"text/plain", X::EvaluationGrid)
    println(io, "EvaluationGrid{$(eltype(X))} with $(X.s) points, $(X.r) views, $(X.n) functions (Resolution=$(X.m)):")
    show(io, mime, X.blocks)
end

"""
    rand_evaluation_grid(s, r, n, m; seed=nothing)

Create a random `EvaluationGrid` with random integer coordinates.

The block array is filled with random integers in the range [1, m].

# Arguments
- `s::Int64`: Number of evaluation points per view.
- `r::Int64`: Number of projection angles.
- `n::Int64`: Number of 3D functions.
- `m::Int64`: Voxel grid resolution/dimension parameter.

# Keyword Argument
- `seed::Union{Nothing, Int}`: Random seed for reproducibility. Defaults to `nothing`.

# Examples
```jldoctest
julia> using HeteroTomo3D, Random;

julia> eval_grid = rand_evaluation_grid(2, 3, 1, 64; seed=123);

julia> size(eval_grid)
(2, 3, 1)

julia> eltype(eval_grid)
Tuple{Int64, Int64}
```

See also [`EvaluationGrid`](@ref), [`rand_quaternion_grid`](@ref).
"""
function rand_evaluation_grid(s::Int, r::Int, n::Int, m::Int; seed::Union{Nothing,Int}=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end

    blocks = Array{NTuple{2,Int}}(undef, s, r, n)

    # Compute center and squared radius
    c = (m + 1) / 2.0
    R2 = ((m - 1) / 2.0)^2

    @inbounds for i in eachindex(blocks)
        # Edge case handler for m=1 to avoid infinite loop
        if m == 1
            blocks[i] = (1, 1)
            continue
        end

        # Rejection sampling
        while true
            x = rand(1:m)
            y = rand(1:m)
            if (x - c)^2 + (y - c)^2 <= R2
                blocks[i] = (x, y)
                break
            end
        end
    end

    return EvaluationGrid(blocks, s, r, n, m)
end

"""
    QuaternionGrid{T<:Real}

Matrix of `UnitQuaternion{T}` of size `(r, n)` for singularity-free 3D rotation angles.
```math
\\mathbf{q}_{jk} = Q[k, j] \\in \\mathbb{S}^{3}
```

# Fields
- `block::Matrix{UnitQuaternion{T}}`: Matrix of size `(r, n)` containing the rotation geometry.
- `r::Int64`: Number of projection angles.
- `n::Int64`: Number of 3D functions.

See also [`EvaluationGrid`](@ref).
"""
struct QuaternionGrid{T<:Real}
    blocks::Matrix{UnitQuaternion{T}}
    r::Int64
    n::Int64
end

Base.eltype(Q::QuaternionGrid) = eltype(Q.blocks)
Base.size(Q::QuaternionGrid) = (Q.r, Q.n)
Base.getindex(Q::QuaternionGrid, i::Int) = Q.blocks[i]
Base.getindex(Q::QuaternionGrid, i::Int, j::Int) = Q.blocks[i, j]
Base.setindex!(Q::QuaternionGrid, v, i::Int) = (Q.blocks[i] = v)
Base.setindex!(Q::QuaternionGrid, v, i::Int, j::Int) = (Q.blocks[i, j] = v)

function Base.show(io::IO, mime::MIME"text/plain", Q::QuaternionGrid)
    println(io, "QuaternionGrid{$(eltype(Q))} with $(Q.r) views, $(Q.n) functions:")
    show(io, mime, Q.blocks)
end



"""
    rand_quaternion_grid(r, n; seed=nothing)

Create a random `QuaternionGrid` with random unit quaternions.

Each quaternion is randomly sampled from the uniform distribution on the 3-sphere.

# Arguments
- `r::Int64`: Number of projection angles.
- `n::Int64`: Number of 3D functions.

# Keyword Argument
- `seed::Union{Nothing, Int}`: Random seed for reproducibility. Defaults to `nothing`.

# Examples
```jldoctest
julia> using HeteroTomo3D, Random;

julia> quat_grid = rand_quaternion_grid(3, 2; seed=123);

julia> size(quat_grid)
(3, 2)

julia> eltype(quat_grid)
UnitQuaternion{Float64}
```

See also [`QuaternionGrid`](@ref), [`rand_evaluation_grid`](@ref).
"""
function rand_quaternion_grid(r::Int, n::Int; seed::Union{Nothing,Int}=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    # Generate random unit quaternions by normalizing random samples
    quats = Matrix{UnitQuaternion{Float64}}(undef, r, n)
    @inbounds for i in eachindex(quats)
        quats[i] = rand(UnitQuaternion)
    end
    return QuaternionGrid(quats, r, n)
end

"""
    BlockDiagonal{T, R<:(AbstractArray{<:AbstractArray{T, 2}, 1})}

Blocked array of square matrices of type `T` arranged in a block diagonal structure.

# Fields
- `blocks::R`: stores the vector of matrices of type `T` being wrapped.

# Supertype Hierarchy
- `BlockDiagonal{T, R<:(AbstractArray{<:AbstractArray{T, 2}, 1})} <: AbstractArray{T, 1} <: Any`

# Examples
```jldoctest
julia> using HeteroTomo3D;

julia> A = [1 2; 3 4];

julia> B = [5 6; 8 9];

julia> bd = BlockDiagonal([A, B]);

julia> bd.blocks[1]
2×2 Matrix{Int64}:
 1  2
 3  4
```

See also [`blocksizes`](@ref), [`zero_block_diag`](@ref), [`undef_block_diag`](@ref), [`rand_block_diag`](@ref).
"""
struct BlockDiagonal{T,R<:AbstractVector{<:AbstractMatrix{T}}} <: AbstractVector{T}
    blocks::R
    # Inner constructor
    function BlockDiagonal(blocks_input::R) where {T,R<:AbstractVector{<:AbstractMatrix{T}}}
        return new{T,R}(blocks_input)
    end
end
Base.eltype(::Type{<:BlockDiagonal{T}}) where {T} = T

function Base.similar(D::BlockDiagonal)
    blocks = [similar(block) for block in D.blocks]
    return BlockDiagonal(blocks)
end

Base.length(D::BlockDiagonal) = sum(length(block) for block in D.blocks)

Base.size(D::BlockDiagonal) = (length(D),)

function Base.getindex(D::BlockDiagonal{T}, i::Int) where {T}
    if !(1 <= i <= length(D))
        throw(BoundsError(D, i))
    end

    currentIndex = 0
    for block in D.blocks
        block_len = length(block)
        if i <= currentIndex + block_len
            idx_in_block = i - currentIndex
            return @inbounds block[idx_in_block]
        end
        currentIndex += block_len
    end
    error("Internal error: Index $i out of bounds or logic error in BlockDiagonal.getindex")
end

function Base.setindex!(D::BlockDiagonal{T}, v, i::Int) where {T}
    if !(1 <= i <= length(D))
        throw(BoundsError(D, i)) # Use `i` directly for linear indexing BoundsError
    end

    currentIndex = 0
    for block in D.blocks
        block_len = length(block)
        if i <= currentIndex + block_len
            idx_in_block = i - currentIndex
            @inbounds block[idx_in_block] = v
            return D # Conventional return for setindex! on arrays
        end
        currentIndex += block_len
    end
    # This should not be reached if i is within bounds and length(D) is correct.
    error("Internal error: Index $i out of bounds or logic error in BlockDiagonal.setindex!")
end

function Base.show(io::IO, mime::MIME"text/plain", D::BlockDiagonal)
    # Use the show method defined by BlockArrays.jl. Does not interact with the BlockArrays package directly.
    matrix_representation = BlockArrays.mortar(Diagonal(D.blocks))
    show(io, mime, matrix_representation)
end

# Initializing the BlockDiagonal type
"""
    zero_block_diag([T=Float64], block_sizes)

Create a `BlockDiagonal` matrix with all elements set to zero.

The blocks are created as square matrices according to the specified sizes.

# Arguments
- `T::Type`: The element type of the blocks. Defaults to `Float64`.
- `block_sizes::Vector{Int}`: A vector of integers specifying the size of each square block.


# Examples
```jldoctest
julia> using HeteroTomo3D;

julia> block_sizes = [2, 1];

julia> bd = zero_block_diag(block_sizes);

julia> eltype(bd)
Float64

julia> bd.blocks[1]
2×2 Matrix{Float64}:
 0.0  0.0
 0.0  0.0
```

See also [`undef_block_diag`](@ref), [`rand_block_diag`](@ref).
"""
function zero_block_diag(T::Type, BS::Vector{Int})
    # Create a zero block diagonal matrix with specified block sizes
    myblocks = [zeros(T, r, r) for r in BS]
    return BlockDiagonal(myblocks)
end

function zero_block_diag(BS::Vector{Int})
    # Create a zero block diagonal matrix with specified block sizes
    myblocks = [zeros(r, r) for r in BS]
    return BlockDiagonal(myblocks)
end

"""
    undef_block_diag([T=Float64], block_sizes)

Create a `BlockDiagonal` matrix with uninitialized elements.

The blocks are created as square matrices according to the specified sizes.

# Arguments
- `T::Type`: The element type of the blocks. Defaults to `Float64`.
- `block_sizes::Vector{Int}`: A vector of integers specifying the size of each square block.

# Examples
```jldoctest
julia> using HeteroTomo3D;

julia> block_sizes = [2, 3];

julia> bd = undef_block_diag(UInt8, block_sizes);

julia> eltype(bd)
UInt8

julia> size(bd.blocks[1])
(2, 2)

julia> size(bd.blocks[2])
(3, 3)
```

See also [`zero_block_diag`](@ref), [`rand_block_diag`](@ref).
"""
function undef_block_diag(T::Type, BS::Vector{Int})
    # Create a zero block diagonal matrix with specified block sizes
    myblocks = [Matrix{T}(undef, r, r) for r in BS]
    return BlockDiagonal(myblocks)
end

function undef_block_diag(BS::Vector{Int})
    # Create a zero block diagonal matrix with specified block sizes
    myblocks = [Matrix{Float64}(undef, r, r) for r in BS]
    return BlockDiagonal(myblocks)
end

"""
    rand_block_diag([T=Float64], block_sizes; seed=nothing)

Create a `BlockDiagonal` matrix with random elements.

The blocks are created as square matrices according to the specified sizes.

# Arguments
- `T::Type`: The element type of the blocks. Defaults to `Float64`.
- `block_sizes::Vector{Int}`: A vector of integers specifying the size of each square block.

# Keyword Arguments
- `seed::Union{Nothing, Int}`: An integer seed for the random number generator to ensure
  reproducibility. Defaults to `nothing`.

# Examples
```jldoctest
julia> using Random;

julia> using HeteroTomo3D;

julia> block_sizes = [2, 1];

julia> bd = rand_block_diag(block_sizes; seed=123);

julia> size(bd.blocks[1])
(2, 2)
```

See also [`zero_block_diag`](@ref), [`undef_block_diag`](@ref).
"""
function rand_block_diag(T::Type, BS::Vector{Int}; seed::Union{Nothing,Int}=nothing)
    # Create a random block diagonal matrix with specified block sizes
    if seed !== nothing
        Random.seed!(seed)
    end
    myblocks = [rand(T, r, r) for r in BS]
    return BlockDiagonal(myblocks)
end

function rand_block_diag(BS::Vector{Int}; seed::Union{Nothing,Int}=nothing)
    # Create a random block diagonal matrix with specified block sizes
    if seed !== nothing
        Random.seed!(seed)
    end
    myblocks = [rand(r, r) for r in BS]
    return BlockDiagonal(myblocks)
end

"""
    block_outer(y)
    y ⊙ y

Computes the block-wise outer product of a `BlockVector`, returning a `BlockDiagonal` matrix.

For a `BlockVector` `y` with blocks `y₁, y₂, ...`, this function computes a `BlockDiagonal`
matrix where the i-th block is the outer product `yᵢ * yᵢ'`.

The infix operator `⊙` is an alias for this function. It must be used on the same object (e.g., `y ⊙ y`).

# Arguments
- `y::AbstractBlockVector`: The input block vector.

# Examples
```jldoctest
julia> using HeteroTomo3D, BlockArrays;

julia> v = [1.0, 2.0, 3.0];

julia> y = BlockVector(v, [2, 1]);

julia> (y ⊙ y).blocks[1]
2×2 Matrix{Float64}:
 1.0  2.0
 2.0  4.0
```
"""
function block_outer(y::AbstractBlockVector{T})::BlockDiagonal{T} where {T}
    blocks = [y_block * y_block' for y_block in y.blocks]
    return BlockDiagonal(blocks)
end


"""
    y ⊙ y

Computes the block-wise outer product of a `BlockVector`, returning a `BlockDiagonal` matrix.

The infix operator `⊙` is an alias for `block_outer(y)`, and must be used on the same object (e.g., `y ⊙ y`).

See also [`block_outer`](@ref).
```
"""
function ⊙(y1::AbstractBlockVector{T}, y2::AbstractBlockVector{T}) where {T}
    @assert y1 === y2 "The ⊙ operator is only defined for the same block vector, e.g., y ⊙ y."
    return block_outer(y1)
end

"""
    blocksizes(D::BlockDiagonal, d::Int)

Returns the sizes of the blocks in a `BlockDiagonal` matrix along the specified dimension `d`.

# Arguments
- `D::BlockDiagonal`: The block diagonal matrix.
- `d::Int`: The dimension along which to get the block sizes (1 for rows, 2 for columns).

# Examples
```jldoctest
julia> using HeteroTomo3D, Random;

julia> block_sizes = [2, 1];

julia> bd = rand_block_diag(block_sizes; seed=123);

julia> blocksizes(bd, 1)
2-element Vector{Int64}:
 2
 1
```
"""
BlockArrays.blocksizes(D::BlockDiagonal, d::Int) = [size(block, d) for block in D.blocks]

#########################################################
# Compatibility with Krylov.jl
# Check https://jso.dev/Krylov.jl/v0.10/custom_workspaces/
function Krylov.kdot(n::Integer, D1::BlockDiagonal{T}, D2::BlockDiagonal{T}) where T<:FloatOrComplex
    dot_product = zero(T)
    @inbounds for i in eachindex(D1.blocks)
        dot_product += dot(D1.blocks[i], D2.blocks[i])
    end
    return dot_product
end

function Krylov.knorm(n::Integer, D::BlockDiagonal{T}) where T<:FloatOrComplex
    norm_value = zero(real(T))
    @inbounds for i in eachindex(D.blocks)
        norm_value += dot(D.blocks[i], D.blocks[i])
    end
    return sqrt(norm_value)
end

function Krylov.kscal!(n::Integer, s::T, D::BlockDiagonal{T}) where T<:FloatOrComplex
    @inbounds @simd for i in eachindex(D.blocks)
        D.blocks[i] .*= s
    end
    return D
end

function Krylov.kdiv!(n::Integer, D::BlockDiagonal{T}, s::T) where T<:FloatOrComplex
    @inbounds @simd for i in eachindex(D.blocks)
        D.blocks[i] ./= s
    end
    return D
end

function Krylov.kaxpy!(n::Integer, s::T, D1::BlockDiagonal{T}, D2::BlockDiagonal{T}) where T<:FloatOrComplex
    @inbounds @simd for i in eachindex(D1.blocks)
        axpy!(s, D1.blocks[i], D2.blocks[i])
    end
    return D2
end

function Krylov.kaxpby!(n::Integer, s::T, D1::BlockDiagonal{T}, t::T, D2::BlockDiagonal{T}) where T<:FloatOrComplex
    @inbounds @simd for i in eachindex(D1.blocks)
        axpby!(s, D1.blocks[i], t, D2.blocks[i])
    end
    return D2
end

function Krylov.kcopy!(n::Integer, D2::BlockDiagonal{T}, D1::BlockDiagonal{T}) where T<:FloatOrComplex
    @inbounds @simd for i in eachindex(D1.blocks)
        D2.blocks[i] .= D1.blocks[i]
    end
    return D2
end

function Krylov.kscalcopy!(n::Integer, D2::BlockDiagonal{T}, s::T, D1::BlockDiagonal{T}) where T<:FloatOrComplex
    @inbounds @simd for i in eachindex(D1.blocks)
        D2.blocks[i] .= s .* D1.blocks[i]
    end
    return D2
end

function Krylov.kdivcopy!(n::Integer, D2::BlockDiagonal{T}, s::T, D1::BlockDiagonal{T}) where T<:FloatOrComplex
    @inbounds @simd for i in eachindex(D1.blocks)
        D2.blocks[i] .= D1.blocks[i] ./ s
    end
    return D2
end

function Krylov.kfill!(D::BlockDiagonal{T}, val::T) where T<:FloatOrComplex
    @inbounds @simd for i in eachindex(D.blocks)
        fill!(D.blocks[i], val)
    end
    return D
end

# Only required for the function minres_qlp
function Krylov.kref!(n::Integer, D1::BlockDiagonal{T}, D2::BlockDiagonal{T}, c::T, s::T) where T<:FloatOrComplex
    s_conj = conj(s)
    @inbounds @simd for i in eachindex(D1.blocks)
        B1 = D1.blocks[i]
        B2 = D2.blocks[i]

        @inbounds @simd for k in eachindex(B1)
            b1k_original = B1[k]
            b2k_original = B2[k]
            B1[k] = c * b1k_original + s * b2k_original
            B2[k] = s_conj * b1k_original - c * b2k_original
        end
    end
    return D1, D2
end
