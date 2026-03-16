"""
    xray_recons(z::NTuple{3,T}, coefficients::Vector{T}, X::EvaluationGrid{I}, Q::QuaternionGrid{T}, γ::T) where {T<:Real, I<:Integer}

Evaluates the reconstruction at a single point `z` in the unit ball using the mean representer theorem.

# Arguments
- `coefficients::Vector{T}`: Coefficients of length `s * r * n` for the representer theorem expansion.
- `X::EvaluationGrid{I}`: The evaluation grid of size `(s, r, n)`.
- `Q::QuaternionGrid{T}`: The rotation grid of size `(r, n)`.
- `γ::T`: Bandwidth parameter for Gaussian kernel.
"""
function xray_recons(z::NTuple{3,T}, coefficients::Vector{T}, X::EvaluationGrid{I}, Q::QuaternionGrid{T}, γ::T) where {T<:Real,I<:Integer}
    s, r, n = X.s, X.r, X.n
    z1, z2, z3 = z

    # Unit ball cutoff
    if z1 * z1 + z2 * z2 + z3 * z3 > 1.0
        return zero(T)
    end

    val = zero(T)

    # Accumulate the sum across all N = s*r*n observations
    for i in 1:n
        for j in 1:r
            q = Q[j, i]
            for k in 1:s
                x_int = X[k, j, i]
                x_real = grid_to_real(x_int, X.m, T)
                a = coefficients[k+(j-1)*s+(i-1)*s*r]
                val += a * backproject(q, x_real, z, γ)
            end
        end
    end

    return val
end

"""
    xray_recons!(F::AbstractArray{T}, coefficients::Vector{T}, X::EvaluationGrid{I}, Q::QuaternionGrid{T}, γ::T) where {T<:Real, I<:Integer}

Evaluates the reconstruction across 3D grid points using the mean representer theorem. 

# Arguments
- `F::AbstractArray{T}`: The output array for the reconstruction of size `(m, m, m)`.
- `coefficients::Vector{T}`: Coefficients of length `s * r * n` for the representer theorem expansion.
- `X::EvaluationGrid{I}`: The evaluation grid of size `(s, r, n)`.
- `Q::QuaternionGrid{T}`: The rotation grid of size `(r, n)`.
- `γ::T`: Bandwidth parameter for Gaussian kernel.
"""
function xray_recons!(F::AbstractArray{T}, coefficients::Vector{T}, X::EvaluationGrid{I}, Q::QuaternionGrid{T}, γ::T) where {T<:Real,I<:Integer}
    m = size(F, 1)
    for iz in 1:m
        z3 = 2.0 * (iz - 1) / (m - 1) - 1.0
        for iy in 1:m
            z2 = 2.0 * (iy - 1) / (m - 1) - 1.0
            for ix in 1:m
                z1 = 2.0 * (ix - 1) / (m - 1) - 1.0
                F[ix, iy, iz] = xray_recons((z1, z2, z3), coefficients, X, Q, γ)
            end
        end
    end
    return F
end