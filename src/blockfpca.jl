
function LinearAlgebra.mul!(b::AbstractBlockVector{T}, D::BlockDiagonal{T}, a::AbstractBlockVector{T}) where {T}
    b_blocks = b.blocks
    D_blocks = D.blocks
    a_blocks = a.blocks
    n = length(D_blocks)
    for i in 1:n
        LinearAlgebra.mul!(b_blocks[i], D_blocks[i], a_blocks[i])
    end
    return b
end

function LinearAlgebra.mul!(b::AbstractBlockVector{T}, C::AbstractBlockMatrix{T}, a::AbstractBlockVector{T}) where {T}
    b_blocks = b.blocks
    C_blocks = C.blocks
    a_blocks = a.blocks

    for i in 1:blocksize(C, 1)
        b_block = b_blocks[i]
        b_block .= zero(eltype(b_block))  # Clear the block before accumulation
        for j in 1:blocksize(C, 2)
            C_block = C_blocks[i, j]
            a_block = a_blocks[j]
            mul!(b_block, C_block, a_block, one(T), one(T))  # Multiply the block
        end
    end
    return b
end

function conj_dot(b::AbstractVector{T}, C::AbstractMatrix{T}, a::AbstractVector{T}) where {T}
    dot(b, C, a)
end

function conj_dot(b::AbstractBlockVector{T}, C::AbstractBlockMatrix{T}, a::AbstractBlockVector{T}) where {T}
    b_blocks = b.blocks
    C_blocks = C.blocks
    a_blocks = a.blocks
    q = zero(T)
    for i in eachindex(b_blocks)
        b_block = b_blocks[i]
        for j in eachindex(a_blocks)
            C_block = C_blocks[i, j]
            a_block = a_blocks[j]
            q += conj_dot(b_block, C_block, a_block)
        end
    end
    return q
end

"""
    conj_lanczos(b0, D, C; ...)

Performs the C-Lanczos algorithm for the generalized eigenvalue problem `D*C*q = λ*q`.

This iterative method generates a C-orthonormal basis `Q` for the Krylov subspace
and a symmetric tridiagonal matrix `T` that represents the projection of the
operator `D*C` onto that subspace.

# Arguments
- `b0::AbstractBlockVector{T}`: The starting vector for the iteration.
- `D::BlockDiagonal{T}`: A block-diagonal matrix to factorize.
- `C::AbstractBlockMatrix{T}`: A symmetric, positive-definite matrix defining the
  inner product (`C`-inner product).

# Keyword Arguments
- `itmax::Int=100`: Maximum number of iterations.
- `tol::Float64=1e-8`: Tolerance for the C-norm of the residual to determine convergence.
- `history::Bool=false`: If `true`, the history of residual C-norms is stored and returned.
- `reortho_level::Symbol=:full`: Level of reorthogonalization. Use `:full` for numerical
  stability, or `:none` to observe loss of orthogonality.

# Returns
- `NamedTuple`: A named tuple `(T, Q, history)` containing:
    - `T`: A `SymTridiagonal` matrix.
    - `Q`: A matrix whose columns are the C-orthonormal Lanczos basis vectors.
    - `history`: A vector of residual C-norms, or `nothing` if `history=false`.
"""
function conj_lanczos(
    b0::AbstractBlockVector{T},
    D::BlockDiagonal{T},
    C::AbstractBlockMatrix{T};
    itmax::Int=50,
    tol::Float64=1e-8,
    history::Bool=false,
    reortho_level::Symbol=:full
) where {T}

    alphas = Vector{T}(undef, itmax)
    betas = Vector{T}(undef, itmax)
    beta_history = history ? Vector{T}() : nothing
    Q_vectors = Vector{typeof(b0)}(undef, itmax)

    t, w, r = similar(b0), similar(b0), similar(b0)

    q_prev = similar(b0)
    fill!(q_prev, 0)

    beta_prev = sqrt(conj_dot(b0, C, b0))
    if beta_prev < tol
        @warn "Initial vector has C-norm < tol."
        # Return empty/minimal results
        T_k = SymTridiagonal(T[], T[])
        Q_k = hcat(Vector{typeof(b0)}[])
        return (T=T_k, Q=Q_k, history=beta_history)
    end

    q_curr = b0 / beta_prev

    k = 0 # Actual number of iterations performed

    for i in 1:itmax
        k = i
        Q_vectors[i] = copy(q_curr) # Store a copy

        # Apply the C-symmetric operator M = D*C
        mul!(t, C, q_curr)
        mul!(w, D, t)

        # Calculate alpha
        alpha_i = conj_dot(q_curr, C, w)
        alphas[i] = alpha_i

        # Calculate residual vector r = w - α*q_curr - β*q_prev
        copyto!(r, w)
        r .-= alpha_i .* q_curr
        if i > 1
            r .-= beta_prev .* q_prev
        end

        # Full reorthogonalization to maintain C-orthogonality
        if reortho_level == :full
            for j in 1:i
                projection = conj_dot(Q_vectors[j], C, r)
                r .-= projection .* Q_vectors[j]
            end
        end

        # Calculate beta from the C-norm of the (now cleaner) residual
        beta_i = sqrt(conj_dot(r, C, r))

        if history
            push!(beta_history, beta_i)
        end

        if beta_i < tol
            break
        end

        # Only store beta if there is a next iteration
        if i < itmax
            betas[i] = beta_i
        end

        # Prepare for the next iteration
        beta_prev = beta_i
        copyto!(q_prev, q_curr)
        q_curr = r / beta_i
    end

    # Trim arrays to the actual number of iterations
    resize!.((alphas, Q_vectors), k)
    resize!(betas, k - 1)

    T_k = SymTridiagonal(alphas, betas)
    Q_k = hcat(Q_vectors...)

    return (T=T_k, Q=Q_k, history=beta_history)
end