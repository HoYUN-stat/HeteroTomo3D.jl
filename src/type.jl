"""
        EvaluationGrid{T<:Real}

    4D array of size `(2, s, r, n)` storing the detector coordinates for X-ray evaluation, i.e.,
    ```math
    \\mathbf{x}_{ijk} = X[:, k, j, i] \\in \\mathbb{B}^{2}
    ```
    # Fields
    - `block::Array{T, 4}`: 4D array of size `(2, s, r, n)` for 2D detector plane coordinates.
    - `s::Int64`: Number of evaluation points per view.
    - `r::Int64`: Number of projection angles.
    - `n::Int64`: Number of 3D functions.

    # Examples
    ```julia-repl
    julia> s, r, n = 20, 5, 100;
        block = rand(2, s, r, n);
        eval_grid = EvaluationGrid(block, s, r, n);
        size(eval_grid)
    (2, 20, 5, 100)
    ```
"""
struct EvaluationGrid{T<:Real}
    block::Array{T,4}
    s::Int64
    r::Int64
    n::Int64
end
Base.eltype(X::EvaluationGrid) = eltype(X.block)
Base.size(X::EvaluationGrid) = (2, X.s, X.r, X.n)
Base.size(X::EvaluationGrid, i::Int) = size(X)[i]

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
"""
struct QuaternionGrid{T<:Real}
    block::Matrix{UnitQuaternion{T}}
    r::Int64
    n::Int64
end

Base.eltype(Q::QuaternionGrid) = eltype(Q.block)
Base.size(Q::QuaternionGrid) = (Q.r, Q.n)


"""
        BlockDiag{T<:Real}

    Block diagonal matrix representing the coefficients with respect to the tensorized basis functions for covariance estimation, i.e.,
    ```math
    \\hat{\\mathbf{B}} \\in \\bigoplus_{i=1}^{n} \\mathbb{R}^{(s \\cdot r) \\times (s \\cdot r)} \\implies 
    \\hat{\\mathbf{\\Sigma}} = \\sum_{i=1}^{n} \\sum_{(j_{1}, k_{1})} \\sum_{(j_{2}, k_{2})} \\hat{B}_{i, (j_{1}, k_{1}), (j_{2}, k_{2})} \\varphi_{i j_{1} k_{1}} \\otimes \\varphi_{i j_{2} k_{2}} \\in \\mathbb{H} \\otimes \\mathbb{H}.
    ```

    Thread-safe pre-allocated buffers are included for zero-allocation `mul!` operations in Krylov solvers, to guarantee no heap allocations during multithreaded linear algebra operations.

    # Fields
    - `block::Array{T, 3}`: Saving the diagonal blocks `[:, :, i]`, with each of size `(s * r, s * r)`.
    - `s::Int64`: Number of evaluation points.
    - `r::Int64`: Number of projection angles.
    - `n::Int64`: Number of 3D functions.
    - `temp::Vector{Matrix{T}}`: Thread-local temp buffers for `mul!` of length `Threads.nthreads()`, isolating memory per CPU core.
    - `iter::Int64`: Iteration count for Krylov solvers, initialized to 0.
"""
mutable struct BlockDiag{T<:Real}
    block::Array{T,3}
    s::Int64
    r::Int64
    n::Int64
    temp::Vector{Matrix{T}}
    iter::Int64
end

# Custom constructor to dynamically allocate isolated temp buffers for each CPU thread
function BlockDiag(block::Array{T,3}, s::Int64, r::Int64, n::Int64) where {T<:Real}
    n_threads = Threads.nthreads()
    temp_buffers = [Matrix{T}(undef, s * r, s * r) for _ in 1:n_threads]
    return BlockDiag(block, s, r, n, temp_buffers, 0)
end

Base.eltype(B::BlockDiag) = eltype(B.block)
Base.size(B::BlockDiag) = (B.s * B.r * B.n, B.s * B.r * B.n)
Base.size(B::BlockDiag, i::Int) = size(B)[i]


"""
        LazyKhatri{T<:Real}

    Khatri-Rao product of Gram matrix `K`, i.e., `K ⊙ K`.
    Bypasses generating massive dense matrices by evaluating on-the-fly in the matrix-free manner.
    In other words, given a block diagonal matrix `B = \\operatorname{Diag}[B_{i}]_{i=1}^{n}`, `mul!(C, khatK, B)` updates a block diagonal matrix `C = \\operatorname{Diag}[C_{i}]_{i=1}^{n}` in-place according to the formula:
    ```math
    \\mathbf{C}_{i} = \\sum_{i'=1}^{n} \\mathbf{K}_{i, i'} \\mathbf{B}_{i'} \\mathbf{K}_{i', i}
    ```

    # Fields
    - `K::Matrix{T}`: Base Gram matrix of size `(s * r * n, s * r * n)`.
    - `s::Int64`: Number of evaluation points.
    - `r::Int64`: Number of projection angles.
    - `n::Int64`: Number of 3D functions.

    # Examples
    ```julia-repl
    julia> s, r, n = 2, 4, 5;
        K = rand(s * r * n, s * r * n);
        khatK = LazyKhatri(K, s, r, n);
        size(khatK)
    (320, 320)
    ```
"""
struct LazyKhatri{T<:Real}
    K::Matrix{T}
    s::Int64
    r::Int64
    n::Int64
end

Base.eltype(khatK::LazyKhatri) = eltype(khatK.K)
Base.size(khatK::LazyKhatri) = ((khatK.s * khatK.r)^2 * khatK.n, (khatK.s * khatK.r)^2 * khatK.n)
Base.size(khatK::LazyKhatri, i::Int) = size(khatK)[i]
