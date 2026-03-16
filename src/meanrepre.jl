



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
