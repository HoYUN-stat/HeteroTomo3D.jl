
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
- `q::UnitQuaternion{T}`: Relative rotation between the two kernels, i.e., `q2 * inv(q1)`
- `x1::NTuple{2, T}`: Coordinates of the first kernel center in the plane
- `x2::NTuple{2, T}`: Coordinates of the second kernel center in the plane
- `γ::T`: Bandwidth parameter for Gaussian kernel

# Examples
```julia-repl
julia> q_parallel = UnitQuaternion(sqrt(2) / 2, 0.0, 0.0, sqrt(2) / 2);

julia> x1 = (0.1, -0.2); x2 = (0.3, -0.4);

julia> γ = 5.0;

julia> val1 = collinear_inner_product(q_parallel, x1, x2, γ)
0.33994859230087116

julia> val2 = collinear_inner_product(inv(q_parallel), x2, x1, γ)
0.33994859230087116

julia> isapprox(val1, val2, atol=1e-10) # Symmetry check
true
```

See also [`noncollinear_inner_product`](@ref).
"""
function collinear_inner_product(q::UnitQuaternion{T}, x1::NTuple{2,T}, x2::NTuple{2,T}, γ::T) where {T<:Real}
    sq1 = x1[1]^2 + x1[2]^2
    sq2 = x2[1]^2 + x2[2]^2
    w1_sq = one(T) - sq1
    w2_sq = one(T) - sq2

    # Grid-safe boundary handling
    if w1_sq <= zero(T) || w2_sq <= zero(T)
        return zero(T)
    end

    w1 = sqrt(w1_sq)
    w2 = sqrt(w2_sq)

    ω, x, y, z = q.ω, q.x, q.y, q.z
    ρ = one(T) - 2 * (x^2 + y^2)

    # Ternary Operation
    c = ρ > 0 ? ω^2 - z^2 : x^2 - y^2
    s = ρ > 0 ? 2 * ω * z : 2 * x * y

    # Squared distance of the orthogonal components
    if ρ > 0 # Parallel
        # dot = x1' * R(q^{-1}) * x2, where R is rotation
        dot_prod = c * (x1[1] * x2[1] + x1[2] * x2[2]) + s * (x1[1] * x2[2] - x1[2] * x2[1])
    else # Anti-parallel
        # dot = x1' * S_refl * x2, where S is reflection
        dot_prod = c * (x1[1] * x2[1] - x1[2] * x2[2]) + s * (x1[1] * x2[2] + x1[2] * x2[1])
    end
    dist2 = sq1 + sq2 - 2 * dot_prod
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
julia> q = rand(UnitQuaternion);

julia> x1 = (0.1, -0.2); x2 = (0.3, -0.4); γ = 5.0;

julia> val1 = noncollinear_inner_product(q, x1, x2, γ)
0.1864041860183954

julia> val2 = noncollinear_inner_product(inv(q), x2, x1, γ)
0.1864041860183954

julia> isapprox(val1, val2, atol=1e-10) # Symmetry check
true
```

See also [`collinear_inner_product`](@ref).
"""
function noncollinear_inner_product(q::UnitQuaternion{T}, x1::NTuple{2,T}, x2::NTuple{2,T}, γ::T) where {T<:Real}
    sq1 = x1[1]^2 + x1[2]^2
    sq2 = x2[1]^2 + x2[2]^2
    w1_sq = one(T) - sq1
    w2_sq = one(T) - sq2

    if w1_sq <= zero(T) || w2_sq <= zero(T)
        return zero(T)
    end

    w1 = sqrt(w1_sq)
    w2 = sqrt(w2_sq)

    ω, x, y, z = q.ω, q.x, q.y, q.z
    ρ = one(T) - 2 * (x^2 + y^2)
    b1 = 2 * (x * z + ω * y) * x2[1] + 2 * (y * z - ω * x) * x2[2]
    b2 = 2 * (x * z - ω * y) * x1[1] + 2 * (y * z + ω * x) * x1[2]

    # 4. Compute BVN parameters
    onemρ2 = one(T) - ρ * ρ
    inv_onemρ2 = one(T) / onemρ2

    μ1 = (b1 + ρ * b2) * inv_onemρ2
    μ2 = (b2 + ρ * b1) * inv_onemρ2

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
    M11 = one(T) - 2 * (y^2 + z^2)
    M12 = 2 * (x * y + ω * z)
    M21 = 2 * (x * y - ω * z)
    M22 = one(T) - 2 * (x^2 + z^2)

    d_norm2 = sq1 + sq2 - 2 * (x1[1] * (M11 * x2[1] + M12 * x2[2]) + x1[2] * (M21 * x2[1] + M22 * x2[2]))
    shift_penalty = b1 * μ1 + b2 * μ2
    multiplier = (T(π) / γ) * sqrt(inv_onemρ2) * exp(-γ * (d_norm2 - shift_penalty))

    return multiplier * prob_mass
end

using Random;
Random.seed!(42);
q = rand(UnitQuaternion);
x1 = (0.1, -0.2);
x2 = (0.3, -0.4);
γ = 5.0;
val1 = noncollinear_inner_product(q, x1, x2, γ)
val2 = noncollinear_inner_product(inv(q), x2, x1, γ)
isapprox(val1, val2, atol=1e-10) # Symmetry check


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
