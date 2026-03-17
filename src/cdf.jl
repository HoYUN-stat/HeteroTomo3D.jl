const SQRT2 = sqrt(2.0)
const SQRT_PI = sqrt(π)

"""
    unicdf(x::Float64) -> Float64

CDF of the standard normal distribution, i.e., 
```math
\\Phi(x) = \\mathbb{P}(Z \\le x) = \\frac{1}{2} \\left( 1 + \\operatorname{erf}\\left( \\frac{x}{\\sqrt{2}} \\right) \\right).
```

```jldoctest
julia> using HeteroTomo3D;

julia> unicdf(0.0)
0.5
julia> unicdf(1.0)
0.8413447460685429
```
"""
unicdf(x::Float64) = 0.5 * (1 + erf(x / SQRT2)) #CDF of N(0, 1)


"""
    antid_erf(z::Float64)::Float64

Compute the antiderivative of the scaled error function, i.e.,
```math
\\Phi(z) =  \\sqrt{\\pi} \\int_{0}^{z} \\operatorname{erf}(t) \\, dt + 1 = \\sqrt{\\pi} z \\operatorname{erf}(z) + \\exp(- z^{2}).
```

# Examples
```jldoctest
julia> using HeteroTomo3D;

julia> antid_erf(2.0)
3.5466412019384204
```
"""
function antid_erf(z::Float64)::Float64
    return SQRT_PI * z * erf(z) + exp(-z^2)
end


"""
    affine_erf(w::Float64, c::Float64, γ::Float64) -> Float64

Compute the antiderivative of the Gaussian function with an affine argument, i.e.,
```math
\\int_{-w}^{w} \\exp(- \\gamma (z - c)^2) \\, dz.
```

# Arguments
- `w::Float64`: Integration limit
- `c::Float64`: Shift parameter in the affine argument
- `γ::Float64`: Bandwidth parameter for the Gaussian


# Examples
```jldoctest
julia> using HeteroTomo3D;

julia> w = 0.5; c = 0.1; γ = 10.0;

julia> sign(affine_erf(w, c, γ)) == sign(w)
true
```
"""
function affine_erf(w::Float64, c::Float64, γ::Float64)::Float64
    sqrt_γ = sqrt(γ)
    y = sqrt_γ * (w - c)
    x = sqrt_γ * (-w - c)
    return (SQRT_PI / (2 * sqrt_γ)) * (erf(x, y))
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
```jldoctest
julia> using HeteroTomo3D;

julia> 10^5 * bvncdf(-2.0, -2.0 , 0.0)
51.75685036595643

julia> 10^5 * bvncdf(-2.0, -2.0 , 0.9)
1336.5628669461514
```

# Reference
Tsay, Wen-Jen, and Peng-Hsuan Ke, A simple approximation for the bivariate normal integral (2021)
"""
function bvncdf(p::Float64, q::Float64, ρ::Float64)::Float64
    @assert -1 < ρ < 1

    # Enforce symmetry: Φ2(p, q, ρ) == Φ2(q, p, ρ)
    if p > q
        p, q = q, p
    end

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
