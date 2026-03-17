"""
    AbstractBlockTensor{T}

Element-free tensor type for block-wise operations.

See also [`BlockOuter`](@ref), [`CovFwdTensor`](@ref), [`AdjointBlockOuter`](@ref), and [`AdjointCovFwdTensor`](@ref).
"""
abstract type AbstractBlockTensor{T} end


# --------------------------------------------------------------------
#                      CONCRETE BLOCK TENSORS
# --------------------------------------------------------------------
"""
    BlockOuter(blocks, [workspace])
    K ⊙ K

Represents a linear operator that performs a block-wise outer product. Specifically, for a `BlockDiagonal` matrix `A`, the output `B = (K ⊙ K)(A)` is computed as
`B.blocks[j] = ∑ᵢ K.blocks[j, i] * A.blocks[i] * K.blocks[i, j]`.

The infix operator `⊙` is an alias for the constructor `BlockOuter(K)`. It requires
that both operands are the same object (e.g., `K ⊙ K`).

# Fields
- `K::AbstractBlockMatrix`: The block matrix that defines the operator.
- `workspace::Matrix`: Pre-allocated matrix buffer for intermediate iterations.

# Examples
```jldoctest
julia> using HeteroTomo3D, BlockArrays;

julia> K = BlockMatrix(rand(3, 3), [2, 1], [2, 1]);

julia> L = K ⊙ K
Block Outer Product Tensor: L = K ⊙ K
 Action: B = L(A), where B[j] = ∑_i K[j, i] A[i] K[i, j]'
K: 2×2-blocked 3×3 BlockMatrix{Float64}:
 0.390663  0.830517  │  0.264633
 0.802763  0.666519  │  0.670233
 ────────────────────┼──────────
 0.928832  0.133315  │  0.713226

julia> L isa BlockOuter
true

julia> size(L)
(5, 5)
```

See also [`AdjointBlockOuter`](@ref).
"""
struct BlockOuter{T,R<:AbstractBlockMatrix{T}} <: AbstractBlockTensor{T}
    K::R
    workspace::Matrix{T}

    # Inner Constructor
    function BlockOuter(K_input::R) where {T,R<:AbstractBlockMatrix{T}}
        @assert eltype(K_input) == T "Element type mismatch: K has eltype $(eltype(K_input)), but expected $T."
        workspace = Matrix{T}(undef, maximum(blocksizes(K_input, 1)), maximum(blocksizes(K_input, 2)))
        return new{T,R}(K_input, workspace)
    end
    function BlockOuter(K_input::R, workspace::Matrix{T}) where {T,R<:AbstractBlockMatrix{T}}
        @assert eltype(K_input) == T "Element type mismatch: K has eltype $(eltype(K_input)), but expected $T."
        @assert size(workspace, 1) >= maximum(blocksizes(K_input, 1)) "workspace has insufficient rows for K."
        @assert size(workspace, 2) >= maximum(blocksizes(K_input, 2)) "workspace has insufficient columns for K."

        return new{T,R}(K_input, workspace)
    end
end
Base.eltype(::Type{<:BlockOuter{T}}) where {T} = T
Base.size(L::BlockOuter) = (sum(blocksizes(L.K, 1) .^ 2), sum(blocksizes(L.K, 2) .^ 2))
Base.size(L::BlockOuter, d::Int) = size(L)[d]

"""
    K ⊙ K

The infix operator `⊙` is an alias for the constructor `BlockOuter(K)`. It requires
that both operands are the same object (e.g., `K ⊙ K`).

See also [`BlockOuter`](@ref).
"""
function ⊙(K1::AbstractBlockMatrix, K2::AbstractBlockMatrix)
    @assert K1 === K2 "The ⊙ operator is only defined for the same matrix, e.g., K ⊙ K."
    return BlockOuter(K1)
end


"""
    CovFwdTensor(K, [workspace])

Represents a covariance-based forward operator `L = O * (K ⊙ K) * O'`, where `O` is the eliminiation operator.

# Fields
- `K::AbstractBlockMatrix`: The block matrix that defines the core of the operator.
- `workspace::Matrix`: Pre-allocated matrix buffer for intermediate iterations.

# Examples
```jldoctest
julia> using HeteroTomo3D, BlockArrays;

julia> K = BlockMatrix(rand(3, 3), [2, 1], [2, 1]);

julia> L = CovFwdTensor(K);

julia> L_adj = adjoint(L); # Or L'

julia> L_adj isa AdjointCovFwdTensor
true
```

See also [`AdjointCovFwdTensor`](@ref).
"""
struct CovFwdTensor{T,R<:AbstractBlockMatrix{T}} <: AbstractBlockTensor{T}
    K::R
    workspace::Matrix{T}

    # Inner Constructor
    function CovFwdTensor(K_input::R) where {T,R<:AbstractBlockMatrix{T}}
        @assert eltype(K_input) == T "Element type mismatch: K has eltype $(eltype(K_input)), but expected $T."
        workspace = Matrix{T}(undef, maximum(blocksizes(K_input, 1)), maximum(blocksizes(K_input, 2)))
        return new{T,R}(K_input, workspace)
    end
    function CovFwdTensor(K_input::R, workspace::Matrix{T}) where {T,R<:AbstractBlockMatrix{T}}
        @assert eltype(K_input) == T "Element type mismatch: K has eltype $(eltype(K_input)), but expected $T."
        @assert size(workspace, 1) >= maximum(blocksizes(K_input, 1)) "workspace has insufficient rows for K."
        @assert size(workspace, 2) >= maximum(blocksizes(K_input, 2)) "workspace has insufficient columns for K."

        return new{T,R}(K_input, workspace)
    end
end
Base.eltype(::Type{<:CovFwdTensor{T}}) where {T} = T
Base.size(L::CovFwdTensor) = (sum(blocksizes(L.K, 1) .^ 2), sum(blocksizes(L.K, 2) .^ 2))
Base.size(L::CovFwdTensor, d::Int) = size(L)[d]


"""
    AdjointBlockOuter(E, [workspace])

Represents the adjoint of the `BlockOuter` operator.

This type is typically not constructed directly, but rather by taking the adjoint of a
`BlockOuter` object (e.g., `L'`).

The operator `L'` is defined by its action on a `BlockDiagonal` matrix `A`:
`A = L'(B)`, where the i-th block of `A` is computed as
`A_i = ∑ⱼ Eⱼᵢ' Bⱼ Eᵢⱼ`.

# Fields
- `F::AbstractBlockMatrix`: The block matrix that defines the original operator.
- `workspace::Matrix`: Pre-allocated matrix to be used for intermediate calculations.

# Examples
```jldoctest
julia> using HeteroTomo3D, BlockArrays;

julia> E = BlockMatrix(rand(3, 2), [2, 1], [1, 1]);

julia> L_adj = (E ⊙ E)';

julia> L_adj isa AdjointBlockOuter
true
```

See also [`BlockOuter`](@ref).
"""
struct AdjointBlockOuter{T,R<:AbstractBlockMatrix{T}} <: AbstractBlockTensor{T}
    K::R
    workspace::Matrix{T}

    # Inner Constructor
    function AdjointBlockOuter(K_input::R) where {T,R<:AbstractBlockMatrix{T}}
        @assert eltype(K_input) == T "Element type mismatch: K has eltype $(eltype(K_input)), but expected $T."
        workspace = Matrix{T}(undef, maximum(blocksizes(K_input, 1)), maximum(blocksizes(K_input, 2)))
        return new{T,R}(K_input, workspace)
    end
    function AdjointBlockOuter(K_input::R, workspace::Matrix{T}) where {T,R<:AbstractBlockMatrix{T}}
        @assert eltype(K_input) == T "Element type mismatch: K has eltype $(eltype(K_input)), but expected $T."
        @assert size(workspace, 1) >= maximum(blocksizes(K_input, 1)) "workspace has insufficient rows for K."
        @assert size(workspace, 2) >= maximum(blocksizes(K_input, 2)) "workspace has insufficient columns for K."
        return new{T,R}(K_input, workspace)
    end
end
Base.eltype(::Type{<:AdjointBlockOuter{T}}) where {T} = T
Base.size(L::AdjointBlockOuter) = (sum(blocksizes(L.K, 2) .^ 2), sum(blocksizes(L.K, 1) .^ 2))
Base.size(L::AdjointBlockOuter, d::Int) = size(L)[d]

Base.adjoint(L::BlockOuter) = AdjointBlockOuter(L.K, L.workspace)
Base.adjoint(L::AdjointBlockOuter) = BlockOuter(L.K, L.workspace)


"""
    AdjointCovFwdTensor(F, [workspace])

Represents the adjoint of the `CovFwdTensor` operator.

# Fields
- `F::AbstractBlockMatrix`: The block matrix that defines the core of the operator.
- `workspace::Matrix`: Pre-allocated matrix to be used for intermediate calculations.

# Examples
```jldoctest
julia> using HeteroTomo3D, BlockArrays;

julia> F = BlockMatrix(rand(3, 2), [2, 1], [1, 1]);

julia> L = CovFwdTensor(F);

julia> L_adj = L';

julia> L_adj isa AdjointCovFwdTensor
true

julia> L_adj' === L
true
```

See also [`CovFwdTensor`](@ref).
"""
struct AdjointCovFwdTensor{T,R<:AbstractBlockMatrix{T}} <: AbstractBlockTensor{T}
    K::R
    workspace::Matrix{T}

    # Inner Constructor
    function AdjointCovFwdTensor(K_input::R) where {T,R<:AbstractBlockMatrix{T}}
        @assert eltype(K_input) == T "Element type mismatch: K has eltype $(eltype(K_input)), but expected $T."
        workspace = Matrix{T}(undef, maximum(blocksizes(K_input, 1)), maximum(blocksizes(K_input, 2)))
        return new{T,R}(K_input, workspace)
    end
    function AdjointCovFwdTensor(K_input::R, workspace::Matrix{T}) where {T,R<:AbstractBlockMatrix{T}}
        @assert eltype(K_input) == T "Element type mismatch: K has eltype $(eltype(K_input)), but expected $T."
        @assert size(workspace, 1) >= maximum(blocksizes(K_input, 1)) "workspace has insufficient rows for K."
        @assert size(workspace, 2) >= maximum(blocksizes(K_input, 2)) "workspace has insufficient columns for K."

        return new{T,R}(K_input, workspace)
    end
end
Base.eltype(::Type{<:AdjointCovFwdTensor{T}}) where {T} = T
Base.size(L::AdjointCovFwdTensor) = (sum(blocksizes(L.K, 2) .^ 2), sum(blocksizes(L.K, 1) .^ 2))
Base.size(L::AdjointCovFwdTensor, d::Int) = size(L)[d]
Base.adjoint(L::CovFwdTensor) = AdjointCovFwdTensor(L.K, L.workspace)
Base.adjoint(L::AdjointCovFwdTensor) = CovFwdTensor(L.K, L.workspace)


# --------------------------------------------------------------------
#                      SHOW METHODS
# --------------------------------------------------------------------
function Base.show(io::IO, mime::MIME"text/plain", L::BlockOuter)
    println(io, "Block Outer Product Tensor: L = K ⊙ K")
    println(io, " Action: B = L(A), where B[j] = ∑_i K[j, i] A[i] K[i, j]'")
    print(io, "K: ")
    show(io, mime, L.K)
end

function Base.show(io::IO, mime::MIME"text/plain", L::AdjointBlockOuter)
    println(io, "Adjoint Block Outer Product Tensor: L = (K ⊙ K)'")
    println(io, " Action: A = L(B), where A = ∑_j K[j]' B[j] K[j]")
    print(io, "K: ")
    show(io, mime, L.K)
end


function Base.show(io::IO, mime::MIME"text/latex", L::CovFwdTensor)
    print(io, "Covariance Forward Tensor: L = O (K ⊙ K) O'")
    print(io, " Action: B = L(A), where B[j] = O[j] (∑_i K[j, i] (O[i]A[i]) K[i, j])'")
    print(io, "K: ")
    show(io, mime, L.K)
end

function Base.show(io::IO, mime::MIME"text/latex", L::AdjointCovFwdTensor)
    print(io, "Adjoint Covariance Forward Tensor: L = O (K ⊙ K) O' (Redundant due to symmetry)")
    print(io, "K: ")
    show(io, mime, L.K)
end

# --------------------------------------------------------------------
#                      IN-PLACE SANDWICH PRODUCT
# --------------------------------------------------------------------
function sandwich!(B::AbstractMatrix{T}, F::AbstractMatrix{T}, A::AbstractMatrix{T},
    workspace::AbstractMatrix{T}) where {T}
    n, m = size(F)

    @assert size(A) == (m, m) "Dimension mismatch for A"
    @assert size(B) == (n, n) "Dimension mismatch for B"
    temp_view = view(workspace, 1:n, 1:m)
    mul!(temp_view, F, A) # Compute F * A and store in temp_view
    mul!(B, temp_view, F', one(T), one(T)) # Compute B = F * A * F' + B * β
    return B
end


function adjoint_sandwich!(A::AbstractMatrix{T}, F::AbstractMatrix{T}, B::AbstractMatrix{T},
    workspace::AbstractMatrix{T}) where {T}
    n, m = size(F)

    @assert size(A) == (m, m) "Dimension mismatch for A"
    @assert size(B) == (n, n) "Dimension mismatch for B"
    temp_view = view(workspace, 1:n, 1:m)
    mul!(temp_view, B, F) # Compute B * F and store in temp_view
    mul!(A, F', temp_view, one(T), one(T)) # Compute A = F' * B * F + A * β
    return A
end


function blockdiagelim!(B_block::AbstractMatrix{T}) where {T}
    # Remove diagonal entries of B_block
    for i in 1:size(B_block, 1)
        B_block[i, i] = zero(T)
    end
    return B_block
end

# --------------------------------------------------------------------
#                      IN-PLACE MULTIPLICATION
# --------------------------------------------------------------------
#1. Block Outer Product
function LinearAlgebra.mul!(D::BlockDiagonal{T}, L::BlockOuter{T}, C::BlockDiagonal{T}) where {T}
    A = C.blocks
    B = D.blocks
    K = L.K
    workspace = L.workspace

    n1 = length(A)
    n2 = length(B)

    for j in 1:n2
        @inbounds B_block = B[j]
        B_block .= zero(T)
        for i in 1:n1
            @inbounds A_block = A[i]
            @inbounds K_block = view(K, Block(j, i))
            # Compute the sandwich product B_block += K_block * A_block * K_block'
            sandwich!(B_block, K_block, A_block, workspace)
        end
    end
    return D
end

#2. Covariance Forward Tensor
function LinearAlgebra.mul!(D::BlockDiagonal{T}, L::CovFwdTensor{T}, C::BlockDiagonal{T}) where {T}
    A = C.blocks
    B = D.blocks
    K = L.K
    workspace = L.workspace

    n1 = length(A)
    n2 = length(B)

    @assert n1 == n2 "Dimension mismatch: n1 should be equal to n2 (RKHS), got n1=$n1 and n2=$n2"

    for i in 1:n1
        @inbounds A_block = A[i]
        blockdiagelim!(A_block)  # Remove diagonal entries of A_block
    end
    for j in 1:n2
        @inbounds B_block = B[j]
        # Make this block zero matrix
        B_block .= zero(T)
        for i in 1:n1
            @inbounds A_block = A[i]
            @inbounds K_block = view(K, Block(j, i))
            # Compute the sandwich product B_block += K_block * A_block * K_block'
            sandwich!(B_block, K_block, A_block, workspace)
        end
        blockdiagelim!(B_block)  # Remove diagonal entries of B_block
    end
    return D
end


#3. Adjoint of Block Outer Product
function LinearAlgebra.mul!(C::BlockDiagonal{T}, adjL::AdjointBlockOuter{T}, D::BlockDiagonal{T}) where {T}
    A = C.blocks
    B = D.blocks
    K = adjL.K
    workspace = adjL.workspace

    n1 = length(A)
    n2 = length(B)

    for i in 1:n1
        @inbounds A_block = A[i]
        # Make this block zero matrix
        A_block .= zero(T)
        for j in 1:n2
            @inbounds B_block = B[j]
            @inbounds K_block = view(K, Block(j, i))
            # Compute the sandwich product B_block += K_block * A_block * K_block'
            adjoint_sandwich!(A_block, K_block, B_block, workspace)
        end
    end
    return C
end

#4. Adjoint of Covariance Forward Tensor
function LinearAlgebra.mul!(C::BlockDiagonal{T}, adjL::AdjointCovFwdTensor{T}, D::BlockDiagonal{T}) where {T}
    A = C.blocks
    B = D.blocks
    K = adjL.K
    workspace = adjL.workspace

    n1 = length(A)
    n2 = length(B)

    @assert n1 == n2 "Dimension mismatch: n1 should be equal to n2, got n1=$n1 and n2=$n2"

    for j in 1:n2
        @inbounds B_block = B[j]
        blockdiagelim!(B_block)  # Remove diagonal entries of B_block
    end
    for i in 1:n1
        @inbounds A_block = A[i]
        # Make this block zero matrix
        A_block .= zero(T)
        for j in 1:n2
            @inbounds B_block = B[j]
            @inbounds K_block = view(K, Block(j, i))
            # Compute the sandwich product B_block += K_block * A_block * K_block'
            adjoint_sandwich!(A_block, K_block, B_block, workspace)
        end
        blockdiagelim!(A_block)  # Remove diagonal entries of A_block
    end
    return C
end
