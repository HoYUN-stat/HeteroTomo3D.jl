unicdf(x::Float64) = 0.5 * (1 + erf(x / SQRT2)) #CDF of N(0, 1)

"""
        bvncdf(p::Float64, q::Float64, ρ::Float64)::Float64

    Output the CDF of the bivariate standard normal distribution.

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