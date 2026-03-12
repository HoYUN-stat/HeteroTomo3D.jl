



"""
    solve_mean!(a::Vector{Float64}, K::Matrix{Float64}, y::Vector{Float64}, λ::Float64)

Solves the regularized system `(K + λI) a = y` for the mean representer theorem.
"""
function solve_mean!(a::AbstractVector{T}, K::AbstractMatrix{T}, y::AbstractVector{T}, λ::T) where {T<:Real}
    N = size(K, 1)

    # Allocate one copy so K is not destroyed by the factorization
    K_reg = copy(K)

    # Tikhonov regularization
    @inbounds @simd for i in 1:N
        K_reg[i, i] += λ
    end

    # Use Bunch-Kaufman factorization which is stable for symmetric indefinite matrices
    F = bunchkaufman!(Symmetric(K_reg))
    ldiv!(a, F, y)

    return a
end

n = 50
r = 5
s = 5
m = 50
γ = 5.0
λ = 1e-2
@time X = rand_evaluation_grid(s, r, n, m);
@time Q = rand_quaternion_grid(r, n);

X_real = grid_to_real.(X.blocks, m)

@time block_sizes = repeat([s * r], n);
@time K = BlockMatrix{Float64}(undef, block_sizes, block_sizes);
@time build_gram_matrix!(K, X, Q, γ);
@time issymmetric(K)
eigvals(K)[1]

a = BlockVector{Float64}(undef, block_sizes)
y = BlockVector{Float64}(undef, block_sizes)
@time solve_mean!(a, K, y, λ)