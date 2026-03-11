"""
    grid_to_real(x::NTuple{2, I}, m::Int, ::Type{T}) where {I<:Integer, T<:Real} -> NTuple{2, T}

Maps discrete voxel integer coordinates natively back to the continuous unit ball `[-1, 1]`.
"""
function grid_to_real(x::NTuple{2,I}, m::Int, ::Type{T}=Float64) where {I<:Integer,T<:AbstractFloat}
    c = T(m + 1) / 2
    R = T(m - 1) / 2
    return ((T(x[1]) - c) / R, (T(x[2]) - c) / R)
end

"""
    build_mean_gram!(K::Matrix{Float64}, X::EvaluationGrid{Float64}, Q::QuaternionGrid{Float64}, γ::Float64)

Constructs the `(s * r * n) x (s * r * n)` Gram matrix `K` for the mean representer theorem.
Iterates over functions (i), quaternions (j), and evaluation points (k), adhering to the column-major indexing `k + (j - 1) s + (i - 1) s r`.

See also [`collinear_inner_product`](@ref), [`noncollinear_inner_product`](@ref).
"""
function build_gram_matrix!(
    K::BlockMatrix{T},
    X::EvaluationGrid{I},
    Q::QuaternionGrid{T},
    γ::T
) where {T<:Real,I<:Integer}
    @assert X.r == Q.r "Inconsistent dimensions"
    @assert X.n == Q.n "Inconsistent dimensions"

    N = size(K, 1)
    @assert N == X.s * X.r * X.n "Inconsistent dimensions"
    s = X.s
    r = X.r
    n = X.n
    m = X.m

    # Multithread over the independent column blocks i2
    Threads.@threads for i2 in 1:n
        # Only compute the upper block-triangle
        for i1 in 1:i2
            @inbounds K_block = view(K, Block(i1, i2))

            for j2 in 1:r
                q2 = Q[j2, i2]

                # If diagonal block, j1 goes up to j2. Otherwise, j1 goes up to r.
                j1_end = (i1 == i2) ? j2 : r

                for j1 in 1:j1_end
                    q1 = Q[j1, i1]
                    q_rel = q2 * inv(q1)

                    is_same_view = (i1 == i2) && (j1 == j2)

                    if is_same_view
                        # Diagonal sub-block: Compute only the upper triangle of k1, k2
                        for k2 in 1:s
                            x2_int = X[k2, j2, i2]
                            x2_real = grid_to_real(x2_int, m, T)
                            J_local = (j2 - 1) * s + k2

                            @simd for k1 in 1:k2
                                x1_int = X[k1, j1, i1]
                                x1_real = grid_to_real(x1_int, m, T)
                                I_local = (j1 - 1) * s + k1

                                val = collinear_inner_product(q_rel, x2_real, x1_real, γ)
                                K_block[I_local, J_local] = val
                                if k1 != k2
                                    K_block[J_local, I_local] = val # Mirror across diagonal
                                end
                            end
                        end
                    else
                        # Off-diagonal sub-block: Compute the full s x s grid
                        for k2 in 1:s
                            x2_int = X[k2, j2, i2]
                            x2_real = grid_to_real(x2_int, m, T)
                            J_local = (j2 - 1) * s + k2

                            @simd for k1 in 1:s
                                x1_int = X[k1, j1, i1]
                                x1_real = grid_to_real(x1_int, m, T)
                                I_local = (j1 - 1) * s + k1

                                val = inner_product(q_rel, x2_real, x1_real, γ)
                                K_block[I_local, J_local] = val

                                # Mirror off-diagonal sub-blocks inside the diagonal block
                                if i1 == i2
                                    K_block[J_local, I_local] = val
                                end
                            end
                        end
                    end
                end
            end

            # If off-diagonal block, mirror the fully populated block to the lower triangle
            if i1 != i2
                @inbounds K_sym_block = view(K, Block(i2, i1))
                rs = r * s
                for c in 1:rs
                    @simd for row in 1:rs
                        K_sym_block[c, row] = K_block[row, c]
                    end
                end
            end

        end
    end

    return K
end

n = 2
r = 3
s = 2
m = 50
γ = 5.0
X = rand_evaluation_grid(s, r, n, m)
Q = rand_quaternion_grid(r, n)

block_sizes = repeat([s * r], n)
K = BlockMatrix{Float64}(undef, block_sizes, block_sizes)
build_gram_matrix!(K, X, Q, γ)

issymmetric(K)
isposdef(K)

#Sanity check
i1, i2 = rand(1:n, 2)
j1, j2 = rand(1:r, 2)
k1, k2 = rand(1:s, 2)

x1 = grid_to_real(X[k1, j1, i1], m)
x2 = grid_to_real(X[k2, j2, i2], m)
q1 = Q[j1, i1]
q2 = Q[j2, i2]
q_rel = q2 * inv(q1)

val = (i1 == i2 && j1 == j2) ? collinear_inner_product(q_rel, x2, x1, γ) : noncollinear_inner_product(q_rel, x2, x1, γ)

isapprox(val, K.blocks[i1, i2][(j1-1)*s+k1, (j2-1)*s+k2], atol=1e-5)


t = randn(1000)
erf.(t) == -erf.(-t)
antid_erf.(t) == antid_erf.(-t)

"""
    solve_mean!(a::Vector{Float64}, K::Matrix{Float64}, y::Vector{Float64}, λ::Float64)

Solves the regularized system `(K + λI) a = y` for the mean representer theorem.
"""
function solve_mean!(a::Vector{Float64}, K::Matrix{Float64}, y::Vector{Float64}, λ::Float64)
    N = size(K, 1)

    # Allocate one copy so K is not destroyed by the factorization
    K_reg = copy(K)

    # Tikhonov regularization
    @inbounds @simd for i in 1:N
        K_reg[i, i] += λ
    end

    # Overwrite the copied matrix with its Cholesky factors
    # F = cholesky!(Symmetric(K_reg))

    # In-place Linear Solve: Overwrites data vector `y` with coefficient vector `c`
    # ldiv!(a, F, y)
    ldiv!(a, K_reg, y)

    return a
end