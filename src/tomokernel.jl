const SQRT2 = sqrt(2.0)
const SQRT_PI = sqrt(π)

"""
    unicdf(x::Float64)

CDF of the standard normal distribution, i.e., 
```math
\\Phi(x) = \\mathbb{P}(Z \\le x) = \\frac{1}{2} \\left( 1 + \\operatorname{erf}\\left( \\frac{x}{\\sqrt{2}} \\right) \\right).
```
"""
unicdf(x::Float64) = 0.5 * (1 + erf(x / SQRT2)) #CDF of N(0, 1)



"""
    antid_erf(z::Float64)::Float64

Compute the antiderivative ``\\Phi(z)`` of ``f(z) = \\sqrt{\\pi} \\operatorname{erf}(z)``, i.e.,
```math
\\Phi(z) =  \\int_{0}^{z} f(z) \\, dz + e^{-1} = \\sqrt{\\pi} z \\operatorname{erf}(z) + \\exp(- z^{2}).
```

    # Examples
    ```
    julia> antid_erf(2.0)
    3.5466412019384204
    ```
"""
function antid_erf(z::Float64)::Float64
    return SQRT_PI * z * erf(z) + exp(-z^2)
end

# Precomputed constants for the bivariate normal CDF approximation
const c1 = -1.0950081470333
const c2 = -0.75651138383854

"""
    bvncdf(p::Float64, q::Float64, ρ::Float64)::Float64

Output the CDF of the bivariate standard normal distribution with correlation coefficient ρ, i.e.,
```math
\\Phi_{2}(p, q; \\rho) = \\mathbb{P}(Z_1 \\le p, Z_2 \\le q) \\quad \\text{where} \\quad \\begin{bmatrix} Z_1 \\\\ Z_2 \\end{bmatrix} \\sim \\mathcal{N}\\left(\\mathbf{0}, \\begin{bmatrix} 1 & \\rho \\\\ \\rho & 1 \\end{bmatrix}\\right)
```

    # Arguments
    - `p::Float64`: First input
    - `q::Float64`: Second input
    - `ρ::Float64`: Correlation coefficient in [-1, 1]

    # Examples
    ```
    julia> @btime 10^5 * bvncdf(-2.0, -2.0 , 0.0)
    34.323 ns (0 allocations: 0 bytes)
    51.75685036595643
    ```

    # Reference
    Tsay, Wen-Jen, and Peng-Hsuan Ke, A simple approximation for the bivariate normal integral (2021)
"""
function bvncdf(p::Float64, q::Float64, ρ::Float64)::Float64
    @assert -1 ≤ ρ ≤ 1

    sqrt1mρ2 = sqrt(1 - ρ^2)  # Precompute sqrt(1 - ρ^2)
    a = -ρ / sqrt1mρ2
    b = p / sqrt1mρ2

    if a > 0
        if a * q + b ≥ 0
            inv_sqrt1ma2c2 = 1 / sqrt(1 - a^2 * c2)  # Reused often
            exp_term1 = exp((a^2 * c1^2 - 2 * SQRT2 * b * c1 + 2 * b^2 * c2) / (4 * (1 - a^2 * c2)))
            exp_term2 = exp((a^2 * c1^2 + 2 * SQRT2 * b * c1 + 2 * b^2 * c2) / (4 * (1 - a^2 * c2)))

            erf1 = erf(q / SQRT2)
            erf2 = erf(b / (SQRT2 * a))
            erf3 = erf((SQRT2 * b - a^2 * c1) / (2 * a * sqrt(1 - a^2 * c2)))
            erf4 = erf((SQRT2 * q - SQRT2 * a^2 * c2 * q - SQRT2 * a * b * c2 - a * c1) / (2 * sqrt(1 - a^2 * c2)))
            erf5 = erf((a^2 * c1 + SQRT2 * b) / (2 * a * sqrt(1 - a^2 * c2)))

            cdf = 0.5 * (erf1 + erf2) +
                  0.25 * inv_sqrt1ma2c2 * exp_term1 * (1 - erf3) -
                  0.25 * inv_sqrt1ma2c2 * exp_term2 * (erf4 + erf5)
        else
            inv_sqrt1ma2c2 = 1 / sqrt(1 - a^2 * c2)
            exp_term = exp((a^2 * c1^2 - 2 * SQRT2 * b * c1 + 2 * b^2 * c2) / (4 * (1 - a^2 * c2)))
            erf_term = erf((SQRT2 * q - SQRT2 * a^2 * c2 * q - SQRT2 * a * b * c2 + a * c1) / (2 * sqrt(1 - a^2 * c2)))

            cdf = 0.25 * inv_sqrt1ma2c2 * exp_term * (1 + erf_term)
        end
    elseif a == 0
        cdf = unicdf(p) * unicdf(q)
    else
        if a * q + b ≥ 0
            inv_sqrt1ma2c2 = 1 / sqrt(1 - a^2 * c2)
            exp_term = exp((a^2 * c1^2 + 2 * SQRT2 * b * c1 + 2 * b^2 * c2) / (4 * (1 - a^2 * c2)))
            erf_term = erf((SQRT2 * q - SQRT2 * a^2 * c2 * q - SQRT2 * a * b * c2 - a * c1) / (2 * sqrt(1 - a^2 * c2)))

            cdf = 0.5 + 0.5 * erf(q / SQRT2) -
                  0.25 * inv_sqrt1ma2c2 * exp_term * (1 + erf_term)
        else
            inv_sqrt1ma2c2 = 1 / sqrt(1 - a^2 * c2)
            exp_term1 = exp((a^2 * c1^2 + 2 * SQRT2 * b * c1 + 2 * b^2 * c2) / (4 * (1 - a^2 * c2)))
            exp_term2 = exp((a^2 * c1^2 - 2 * SQRT2 * b * c1 + 2 * b^2 * c2) / (4 * (1 - a^2 * c2)))

            erf1 = erf((SQRT2 * b + a^2 * c1) / (2 * a * sqrt(1 - a^2 * c2)))
            erf2 = erf((-a^2 * c1 + SQRT2 * b) / (2 * a * sqrt(1 - a^2 * c2)))
            erf3 = erf((SQRT2 * q - SQRT2 * a^2 * c2 * q - SQRT2 * a * b * c2 + a * c1) / (2 * sqrt(1 - a^2 * c2)))

            cdf = 0.5 - 0.5 * erf(b / (SQRT2 * a)) -
                  0.25 * inv_sqrt1ma2c2 * exp_term1 * (1 - erf1) +
                  0.25 * inv_sqrt1ma2c2 * exp_term2 * (erf3 + erf2)
        end
    end

    return cdf
end


"""
    backproject(q::UnitQuaternion, x1::Float64, x2::Float64, z1::Float64, z2::Float64, z3::Float64, γ::Float64)

Evaluates the point of the tomographic feature map analytically, i.e., computes
```math
\\varphi_{\\gamma}(\\mathbf{R}_{\\mathbf{q}}, \\mathbf{x})(\\mathbf{z}) = \\int_{-W(\\mathbf{x})}^{W(\\mathbf{x})} \\exp(-\\gamma \\| \\mathbf{R}_{\\mathbf{q}} \\mathbf{z} - [\\mathbf{x} : z] \\|^2) \\, dz
```

    # Arguments
    - `q::UnitQuaternion`: Rotation of the kernel
    - `x1::Float64`: First coordinate of the kernel center in the plane
    - `x2::Float64`: Second coordinate of the kernel center in the plane
    - `z1::Float64`: First coordinate of the evaluation point in 3D space
    - `z2::Float64`: Second coordinate of the evaluation point in 3D space
    - `z3::Float64`: Third coordinate of the evaluation point in 3D space
    - `γ::Float64`: Bandwidth parameter for Gaussian kernel

    # Examples
    ```
    julia> q = shortest_arc(0.0, 1.0, 0.0)
    UnitQuaternion{Float64}(0.7071067811865476, 0.7071067811865475, -0.0, 0.0)

    julia> backproject(q, 0.1, -0.2, 0.3, -0.4, 0.5, 10.0)
    0.15197718857623893
    ```
"""
@inline function backproject(q::UnitQuaternion, x1::Float64, x2::Float64, z1::Float64, z2::Float64, z3::Float64, γ::Float64)
    ω, x, y, z_q = q.ω, q.x, q.y, q.z

    # 1. Compute projection axis r_q
    rq_x = 2.0 * (x * z_q - ω * y)
    rq_y = 2.0 * (y * z_q + ω * x)
    rq_z = 1.0 - 2.0 * (x * x + y * y)

    z_dot_rq = z1 * rq_x + z2 * rq_y + z3 * rq_z

    # 2. Extract first two rows of R_q applied to z
    R11 = 1.0 - 2.0 * (y * y + z_q * z_q)
    R12 = 2.0 * (x * y - ω * z_q)
    R13 = 2.0 * (x * z_q + ω * y)

    R21 = 2.0 * (x * y + ω * z_q)
    R22 = 1.0 - 2.0 * (x * x + z_q * z_q)
    R23 = 2.0 * (y * z_q - ω * x)

    # Projected coordinates in the plane of x1, x2
    pi1 = R11 * z1 + R12 * z2 + R13 * z3
    pi2 = R21 * z1 + R22 * z2 + R23 * z3

    # 3. Squared distance
    dist2 = (x1 - pi1)^2 + (x2 - pi2)^2

    # 4. Integration limits (W(x) function)
    W_x = sqrt(max(0.0, 1.0 - x1 * x1 - x2 * x2))

    sqrt_γ = sqrt(γ)
    term0 = sqrt_γ * (W_x - z_dot_rq)
    term1 = sqrt_γ * (-W_x - z_dot_rq)

    return (sqrt(π) * exp(-γ * dist2) / (2.0 * sqrt_γ)) * (erf(term0) - erf(term1))
end

"""
    inner_product(q::UnitQuaternion, x1_1::Float64, x1_2::Float64, q2::UnitQuaternion, x2_1::Float64, x2_2::Float64, γ::Float64)

Analytically computes the inner product between two tomographic feature maps.
```math
\\langle \\varphi_{\\gamma} (\\mathbf{R}_{\\mathbf{q}_{1}}, \\mathbf{x}_{1}), \\varphi_{\\gamma} (\\mathbf{R}_{\\mathbf{q}_{2}}, \\mathbf{x}_{2}) \\rangle_{\\mathcal{H}} =\\int_{-W(\\mathbf{x}_{1})}^{W(\\mathbf{x}_{1})} \\int_{-W(\\mathbf{x}_{2})}^{W(\\mathbf{x}_{2})} \\exp \\left( - \\gamma \\|[\\mathbf{x}_{1}:z_{1}] - \\mathbf{R}_{\\mathbf{q}}^{-1} [\\mathbf{x}_{2}:z_{2}]\\|^{2} \\right) \\, dz_{2} \\, d z_{1},
```
where ``\\mathbf{q} = \\mathbf{q}_{1} \\mathbf{q}_{2}^{-1}`` is the relative rotation between the two kernels.
Automatically branches between the collinear and non-collinear integrations.

    # Arguments
    - `q1::UnitQuaternion`: Rotation of the first kernel
    - `x1_1::Float64`: First coordinate of the first kernel center in the plane
    - `x1_2::Float64`: Second coordinate of the first kernel center in the plane
    - `q2::UnitQuaternion`: Rotation of the second kernel
    - `x2_1::Float64`: First coordinate of the second kernel center in the plane
    - `x2_2::Float64`: Second coordinate of the second kernel center in the plane
    - `γ::Float64`: Bandwidth parameter for Gaussian kernel

    # Examples
    ```
    julia> q_id = UnitQuaternion(1.0, 0.0, 0.0, 0.0)
    UnitQuaternion{Float64}(1.0, 0.0, 0.0, 0.0)

    julia> inner_product(q_id, 0.1, 0.2, q_id, 0.1, 0.2, 10.0)
    0.9926139338138249
    ```
"""
@inline function inner_product(q1::UnitQuaternion, x1_1::Float64, x1_2::Float64, q2::UnitQuaternion, x2_1::Float64, x2_2::Float64, γ::Float64)
    # 1. Relative rotation q = q1 * q2^{-1}
    q = q1 * inv(q2)
    ω, x, y, z = q.ω, q.x, q.y, q.z

    x2py2 = x * x + y * y
    ω2pz2 = ω * ω + z * z
    rq = 1.0 - 2.0 * x2py2

    # Limit variables
    W1 = sqrt(max(0.0, 1.0 - x1_1 * x1_1 - x1_2 * x1_2))
    W2 = sqrt(max(0.0, 1.0 - x2_1 * x2_1 - x2_2 * x2_2))

    if rq ≈ 1.0
        # --- COLLINEAR AXES ---
        inv_n2 = 1.0 / ω2pz2
        R11 = (ω * ω - z * z) * inv_n2
        R12 = 2.0 * ω * z * inv_n2

        # Apply inverse planar rotation to x2
        rot_x2_1 = R11 * x2_1 + R12 * x2_2
        rot_x2_2 = -R12 * x2_1 + R11 * x2_2

        dist2 = (x1_1 - rot_x2_1)^2 + (x1_2 - rot_x2_2)^2

        sqrt_γ = sqrt(γ)

        # antid_erf is identical to Phi(z) up to the -1 constant, which cancels perfectly in this alternating sum.
        sum_phi = antid_erf(sqrt_γ * (W1 + W2)) -
                  antid_erf(sqrt_γ * (W1 - W2)) -
                  antid_erf(sqrt_γ * (-W1 + W2)) +
                  antid_erf(sqrt_γ * (-W1 - W2))

        return (1.0 / (2.0 * γ)) * exp(-γ * dist2) * sum_phi

    else
        # --- NON-COLLINEAR AXES ---
        inv_ω2pz2 = 1.0 / ω2pz2
        inv_w = 0.5 / sqrt(x2py2 * ω2pz2) # Precompute for scaling
        w_rq = 2.0 * sqrt(x2py2 * ω2pz2)

        # Extract 1D planar components for x2
        x2_1_mapped = ((ω * ω - z * z) * x2_1 + 2.0 * ω * z * x2_2) * inv_ω2pz2
        x2_2_mapped = (-2.0 * ω * z * x2_1 + (ω * ω - z * z) * x2_2) * inv_ω2pz2

        # Extract 1D planar components for x1
        x1_1_mapped = ((y * z + ω * x) * x1_1 + (ω * y - x * z) * x1_2) * inv_w
        x1_2_mapped = ((x * z - ω * y) * x1_1 + (y * z + ω * x) * x1_2) * inv_w

        μ1 = rq * x1_2_mapped - x2_2_mapped
        μ2 = x1_2_mapped - rq * x2_2_mapped

        sqrt_2γ = sqrt(2.0 * γ)

        cdf00 = bvncdf(sqrt_2γ * (w_rq * W1 - μ1), sqrt_2γ * (w_rq * W2 - μ2), rq)
        cdf01 = bvncdf(sqrt_2γ * (w_rq * W1 - μ1), sqrt_2γ * (-w_rq * W2 - μ2), rq)
        cdf10 = bvncdf(sqrt_2γ * (-w_rq * W1 - μ1), sqrt_2γ * (w_rq * W2 - μ2), rq)
        cdf11 = bvncdf(sqrt_2γ * (-w_rq * W1 - μ1), sqrt_2γ * (-w_rq * W2 - μ2), rq)

        sum_cdf = cdf00 - cdf01 - cdf10 + cdf11

        return (π * exp(-γ * (x1_1_mapped - x2_1_mapped)^2)) / (γ * w_rq) * sum_cdf
    end
end