
"""
    backproject(q::UnitQuaternion{T}, x::NTuple{2, T}, z::NTuple{3, T}, γ::T) where {T<:Real} -> T

Evaluates the point of the tomographic feature map analytically, i.e., computes
```math
\\varphi_{\\gamma}(\\mathbf{R}_{\\mathbf{q}}, \\mathbf{x})(\\mathbf{z}) = \\int_{-W(\\mathbf{x})}^{W(\\mathbf{x})} \\exp(-\\gamma \\| \\mathbf{R}_{\\mathbf{q}} \\mathbf{z} - [\\mathbf{x} : z] \\|^2) \\, dz
```

# Arguments
- `q::UnitQuaternion{T}`: Rotation of the kernel
- `x::NTuple{2, T}`: Coordinates of the kernel center in the plane
- `z::NTuple{3, T}`: Coordinates of the evaluation point in 3D space
- `γ::T`: Bandwidth parameter for Gaussian kernel

# Examples
```julia-repl
julia> q = shortest_arc(0.0, 1.0, 0.0)
UnitQuaternion{Float64}(0.7071067811865476, 0.7071067811865475, -0.0, 0.0)

julia> backproject(q, (0.1, -0.2), (0.3, -0.4, 0.5), 10.0)
0.15197718857623901
```
"""
function backproject(q::UnitQuaternion{T}, x::NTuple{2,T}, z::NTuple{3,T}, γ::T) where {T<:Real}
    dist2_x = x[1]^2 + x[2]^2

    if dist2_x >= one(T)
        return zero(T)
    end

    W_x = sqrt(one(T) - dist2_x)
    p1, p2, p3 = rotate(q, z)

    dist2 = (x[1] - p1)^2 + (x[2] - p2)^2
    term = affine_erf(W_x, p3, γ)

    return exp(-γ * dist2) * term
end


"""
    collinear_inner_product(q::UnitQuaternion{T}, x1::NTuple{2, T}, x2::NTuple{2, T}, γ::T) where {T<:Real} -> T

Evaluates the inner product between two tomographic feature maps when the viewing directions are strictly parallel or anti-parallel.

# Arguments
- `q::UnitQuaternion{T}`: Relative rotation between the two kernels
- `x1::NTuple{2, T}`: Coordinates of the first kernel center in the plane
- `x2::NTuple{2, T}`: Coordinates of the second kernel center in the plane
- `γ::T`: Bandwidth parameter for Gaussian kernel

# Examples
```julia-repl
julia> q_parallel = UnitQuaternion(sqrt(2) / 2, 0.0, 0.0, sqrt(2) / 2);

julia> q_antiparallel = UnitQuaternion(0.0, sqrt(2) / 2, sqrt(2) / 2, 0.0);

julia> x1 = (0.1, -0.2);

julia> x2 = (0.3, -0.4);

julia> γ = 5.0;

julia> collinear_inner_product(q_parallel, x1, x2, γ)
0.339948592300871

julia> collinear_inner_product(q_antiparallel, x1, x2, γ)
0.10239054834872334

julia> q = shortest_arc(0.0, 1.0, 0.0);

julia> collinear_inner_product(q, x1, x2, γ)
ERROR: AssertionError: Relative rotation must be parallel or anti-parallel.
```

See also [`noncollinear_inner_product`](@ref).
"""
function collinear_inner_product(q::UnitQuaternion{T}, x1::NTuple{2,T}, x2::NTuple{2,T}, γ::T) where {T<:Real}
    w1_sq = one(T) - (x1[1]^2 + x1[2]^2)
    w2_sq = one(T) - (x2[1]^2 + x2[2]^2)

    # Grid-safe boundary handling
    if w1_sq <= zero(T) || w2_sq <= zero(T)
        return zero(T)
    end

    w1 = sqrt(w1_sq)
    w2 = sqrt(w2_sq)

    ω, x, y, z = q.ω, q.x, q.y, q.z
    ρ = one(T) - 2 * (x^2 + y^2)

    # if ρ > 0 # Parallel case
    #     c = ω^2 - z^2
    #     s = 2 * ω * z
    #     # Inverse of a 2D rotation matrix is its transpose
    #     x2_rot1 = c * x2[1] + s * x2[2]
    #     x2_rot2 = -s * x2[1] + c * x2[2]
    # else # Anti-parallel case
    #     c = x^2 - y^2
    #     s = 2 * x * y
    #     # The symmetric 2D reflection matrix is its own inverse
    #     x2_rot1 = c * x2[1] + s * x2[2]
    #     x2_rot2 = s * x2[1] - c * x2[2]
    # end

    # Ternary Operation
    c = ρ > 0 ? ω^2 - z^2 : x^2 - y^2
    s = ρ > 0 ? 2 * ω * z : 2 * x * y
    x2_rot1 = c * x2[1] + s * x2[2]
    x2_rot2 = ρ > 0 ? -s * x2[1] + c * x2[2] : s * x2[1] - c * x2[2]

    # Squared distance of the orthogonal components
    dist2 = (x1[1] - x2_rot1)^2 + (x1[2] - x2_rot2)^2
    sqrt_γ = sqrt(γ)

    term1 = antid_erf(sqrt_γ * (w1 + w2))
    term2 = antid_erf(sqrt_γ * (w1 - w2))

    return (one(T) / γ) * exp(-γ * dist2) * (term1 - term2)
end


"""
    noncollinear_inner_product(q::UnitQuaternion{T}, x1::NTuple{2, T}, x2::NTuple{2, T}, γ::T) where {T<:Real} -> T

Evaluates the inner product between two tomographic feature maps when the viewing directions are non-collinear.

# Arguments
- `q::UnitQuaternion{T}`: Relative rotation between the two kernels
- `x1::NTuple{2, T}`: Coordinates of the first kernel center in the plane
- `x2::NTuple{2, T}`: Coordinates of the second kernel center in the plane
- `γ::T`: Bandwidth parameter for Gaussian kernel

# Examples
```julia-repl
julia> q_parallel = UnitQuaternion(sqrt(2) / 2, 0.0, 0.0, sqrt(2) / 2);

julia> x1 = (0.1, -0.2);

julia> x2 = (0.3, -0.4);

julia> γ = 5.0;

julia> noncollinear_inner_product(q_parallel, x1, x2, γ)
ERROR: AssertionError: Correlation coefficient must satisfy |ρ| < 1 for non-collinear case.

julia> q = shortest_arc(0.0, 1.0, 0.0);

julia> noncollinear_inner_product(q, x1, x2, γ)
0.48743950572926514
```

See also [`collinear_inner_product`](@ref).
"""
function noncollinear_inner_product(q::UnitQuaternion{T}, x1::NTuple{2,T}, x2::NTuple{2,T}, γ::T) where {T<:Real}
    w1_sq = one(T) - (x1[1]^2 + x1[2]^2)
    w2_sq = one(T) - (x2[1]^2 + x2[2]^2)

    if w1_sq <= zero(T) || w2_sq <= zero(T)
        return zero(T)
    end

    w1 = sqrt(w1_sq)
    w2 = sqrt(w2_sq)

    # 1. Evaluate the displacement d = [x1; 0] - R_q^{-1} [x2; 0]
    p2 = rotate(inv(q), (x2[1], x2[2], zero(T)))
    d1 = x1[1] - p2[1]
    d2 = x1[2] - p2[2]
    d3 = -p2[3]

    # 2. Extract projection axis and correlation
    r_q = projection_axis(q)
    ρ = r_q[3] # Mathematically equal to e_3^T r_q

    # 3. Compute dot products
    b1 = d3
    b2 = r_q[1] * x1[1] + r_q[2] * x1[2]

    # 4. Compute BVN parameters
    onemρ2 = one(T) - ρ * ρ
    inv_onemρ2 = one(T) / onemρ2

    μ1 = (ρ * b2 - b1) * inv_onemρ2
    μ2 = (b2 - ρ * b1) * inv_onemρ2

    scale = sqrt(2 * γ * onemρ2)
    u1_plus = scale * (w1 - μ1)
    u1_minus = scale * (-w1 - μ1)
    u2_plus = scale * (w2 - μ2)
    u2_minus = scale * (-w2 - μ2)

    # 5. Evaluate custom BVN CDF block
    # Note: Explicit Float64 conversion ensures type stability with your bvncdf implementation
    bvn_pp = bvncdf(Float64(u1_plus), Float64(u2_plus), Float64(ρ))
    bvn_mp = bvncdf(Float64(u1_minus), Float64(u2_plus), Float64(ρ))
    bvn_pm = bvncdf(Float64(u1_plus), Float64(u2_minus), Float64(ρ))
    bvn_mm = bvncdf(Float64(u1_minus), Float64(u2_minus), Float64(ρ))

    prob_mass = bvn_pp - bvn_mp - bvn_pm + bvn_mm

    # 6. Apply constant multipliers
    d_norm2 = d1 * d1 + d2 * d2 + d3 * d3
    shift_penalty = (b1 * b1 - 2 * ρ * b1 * b2 + b2 * b2) * inv_onemρ2
    multiplier = (T(π) / γ) * sqrt(inv_onemρ2) * exp(-γ * (d_norm2 - shift_penalty))

    return multiplier * prob_mass
end

"""
    inner_product(q::UnitQuaternion{T}, x1::NTuple{2, T}, x2::NTuple{2, T}, γ::T) where {T<:Real} -> T

Evaluates the inner product between two tomographic feature maps exactly.

# Arguments
- `q::UnitQuaternion{T}`: Relative rotation between the two kernels
- `x1::NTuple{2, T}`: Coordinates of the first kernel center in the plane
- `x2::NTuple{2, T}`: Coordinates of the second kernel center in the plane
- `γ::T`: Bandwidth parameter for Gaussian kernel

See also [`collinear_inner_product`](@ref), [`noncollinear_inner_product`](@ref).
"""
function inner_product(q::UnitQuaternion{T}, x1::NTuple{2,T}, x2::NTuple{2,T}, γ::T) where {T<:Real}
    # Extract ρ = 1 - 2*(x^2 + y^2) to safely branch
    ρ = one(T) - 2 * (q.x^2 + q.y^2)

    if abs(ρ) > one(T) - 1e-12
        return collinear_inner_product(q, x1, x2, γ)
    else
        return noncollinear_inner_product(q, x1, x2, γ)
    end
end

