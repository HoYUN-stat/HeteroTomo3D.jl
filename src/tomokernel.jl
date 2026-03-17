
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
```jldoctest
julia> using HeteroTomo3D;

julia> q = shortest_arc(0.0, 1.0, 0.0)
UnitQuaternion{Float64}(0.7071067811865476, 0.7071067811865475, -0.0, 0.0)

julia> backproject(q, (0.1, -0.2), (0.3, -0.4, 0.5), 10.0)
0.15197718857623901
```
"""
function backproject(q::UnitQuaternion{T}, x::NTuple{2,T}, z::NTuple{3,T}, γ::T) where {T<:Real}
    dist2_x = x[1]^2 + x[2]^2
    dist2_z = z[1]^2 + z[2]^2 + z[3]^2
    if dist2_x >= one(T) || dist2_z >= one(T)
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
```jldoctest
julia> using HeteroTomo3D;

julia> q_parallel = UnitQuaternion(sqrt(2) / 2, 0.0, 0.0, sqrt(2) / 2);

julia> x1 = (0.1, -0.2); x2 = (0.3, -0.4); γ = 5.0;

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
- `q::UnitQuaternion{T}`: Relative rotation between the two kernels, i.e., `q2 * inv(q1)`
- `x1::NTuple{2, T}`: Coordinates of the first kernel center in the plane
- `x2::NTuple{2, T}`: Coordinates of the second kernel center in the plane
- `γ::T`: Bandwidth parameter for Gaussian kernel

# Examples
```jldoctest
julia> using HeteroTomo3D;

julia> using Random; Random.seed!(123);

julia> q = rand(UnitQuaternion)
UnitQuaternion{Float64}(-0.28204731317456205, -0.6391304265753365, -0.7091703923721498, -0.09507347442567807)

julia> x1 = (0.1, -0.2); x2 = (0.3, -0.4); γ = 5.0;

julia> val1 = noncollinear_inner_product(q, x1, x2, γ)
0.2589394161448219

julia> val2 = noncollinear_inner_product(inv(q), x2, x1, γ)
0.2589394161448219

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



"""
    inner_product(q::UnitQuaternion{T}, x1::NTuple{2, T}, x2::NTuple{2, T}, γ::T) where {T<:Real} -> T

Evaluates the inner product between two tomographic feature maps exactly.

# Arguments
- `q::UnitQuaternion{T}`: Relative rotation between the two kernels, i.e., `q2 * inv(q1)`.
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
    build_gram_matrix!(K::BlockMatrix{T}, X::EvaluationGrid, Q::QuaternionGrid{T}, γ::T) where {T<:Real}

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

                                val = collinear_inner_product(q_rel, x1_real, x2_real, γ)
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

                                val = inner_product(q_rel, x1_real, x2_real, γ)
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
