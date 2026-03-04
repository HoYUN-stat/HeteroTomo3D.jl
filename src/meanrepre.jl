"""
    build_mean_gram!(K::Matrix{Float64}, X::EvaluationGrid{Float64}, Q::QuaternionGrid{Float64}, γ::Float64)

Constructs the `(s * r * n) x (s * r * n)` Gram matrix `K` for the mean representer theorem.
Iterates over functions (i), quaternions (j), and evaluation points (k), i.e.,
```math
K[k1 + (j1 - 1) * s + (i1 - 1) * s * r, k2 + (j2 - 1) * s + (i2 - 1) * s * r] = \\langle \\varphi(\\mathbf{R}_{\\mathbf{q}_{i1, j1}} \\mathbf{x}_{i1, j1, k1}), \\varphi(\\mathbf{R}_{\\mathbf{q}_{i2, j2}} \\mathbf{x}_{i2, j2, k2}) \\rangle_{\\mathbb{H}}
```
"""
function build_mean_gram!(K::Matrix{Float64}, X::EvaluationGrid{Float64}, Q::QuaternionGrid{Float64}, γ::Float64)
    s, r, n = X.s, X.r, X.n
    N = s * r * n

    # Multithread over the outermost function index
    Threads.@threads for i2 in 1:n
        base_J_i2 = (i2 - 1) * s * r
        @inbounds for j2 in 1:r
            base_J = (j2 - 1) * s + base_J_i2
            q2 = Q.block[j2, i2]
            for k2 in 1:s
                J = k2 + base_J
                x2_1 = X.block[1, k2, j2, i2]
                x2_2 = X.block[2, k2, j2, i2]

                for i1 in 1:n
                    base_I_i1 = (i1 - 1) * s * r
                    for j1 in 1:r
                        base_I = (j1 - 1) * s + base_I_i1
                        q1 = Q.block[j1, i1]
                        @simd for k1 in 1:s
                            I = k1 + base_I

                            # Compute only the upper triangle
                            if I <= J
                                x1_1 = X.block[1, k1, j1, i1]
                                x1_2 = X.block[2, k1, j1, i1]

                                val = inner_product(q1, x1_1, x1_2, q2, x2_1, x2_2, γ)

                                K[I, J] = val
                            end
                        end
                    end
                end
            end
        end
    end

    @inbounds for J in 1:N
        @simd for I in 1:(J-1)
            K[J, I] = K[I, J] # Fill the lower triangle by symmetry
        end
    end

    return K
end

"""
    solve_mean!(a::Vector{Float64}, K::Matrix{Float64}, y::Vector{Float64}, λ::Float64)

Solves the regularized system `(K + λI) a = y` for the mean representer theorem.
"""
function solve_mean!(a::Vector{Float64}, K::Matrix{Float64}, y::Vector{Float64}, λ::Float64)
    N = size(K, 1)

    # Allocate exactly one copy so K is not destroyed by the factorization
    K_reg = copy(K)

    # Add Tikhonov regularization strictly to the diagonal elements
    @inbounds @simd for i in 1:N
        K_reg[i, i] += λ
    end

    # Overwrite the copied matrix with its Cholesky factors
    F = cholesky!(Symmetric(K_reg))

    # In-place Linear Solve: Overwrites data vector `y` with coefficient vector `c`
    ldiv!(a, F, y)

    return a
end