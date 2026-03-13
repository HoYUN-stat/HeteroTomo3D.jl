"""
    reconstruct_mean(a::Vector{Float64}, X::EvaluationGrid{Float64}, Q::QuaternionGrid{Float64}, m::Int, γ::Float64)

Reconstructs the `m x m x m` 3D mean volume from the representer theorem coefficients `a`, i.e., computes
```math
\\hat{f}(\\mathbf{z}) = \\sum_{i=1}^{n} \\sum_{j=1}^{r} \\sum_{k=1}^{s} a_{i j k} \\varphi_{\\gamma}(\\mathbf{R}_{\\mathbf{q}_{i j}}, \\mathbf{x}_{i j k})(\\mathbf{z})
```
Computes the action of the evaluation tensor on `a` entirely on-the-fly to avoid massive memory allocations, strictly bound to O(m^3) space complexity.
"""
function reconstruct_mean(a::Vector{Float64}, X::EvaluationGrid{Float64}, Q::QuaternionGrid{Float64}, m::Int, γ::Float64)
    s, r, n = X.s, X.r, X.n

    # Pre-allocate the output 3D volume
    V = zeros(Float64, m, m, m)

    # Precompute global constants
    sqrt_γ = sqrt(γ)
    C = sqrt(π) / (2.0 * sqrt_γ)

    # Multithread over the Z-axis of the volume
    Threads.@threads for iz in 1:m
        z3 = 2.0 * (iz - 1) / (m - 1) - 1.0

        for iy in 1:m
            z2 = 2.0 * (iy - 1) / (m - 1) - 1.0

            for ix in 1:m
                z1 = 2.0 * (ix - 1) / (m - 1) - 1.0

                # 1. Unit ball cutoff: Skips ~47.6% of the voxels
                if z1 * z1 + z2 * z2 + z3 * z3 > 1.0
                    continue
                end

                acc = 0.0

                # Accumulate the sum across all N = s*r*n observations
                for i in 1:n
                    for j in 1:r
                        q = Q.block[j, i]
                        ω, x, y, z_q = q.ω, q.x, q.y, q.z

                        # 2. Hoist the rotation R_q^T * z outside the k-loop
                        R11 = 1.0 - 2.0 * (y * y + z_q * z_q)
                        R21 = 2.0 * (x * y + ω * z_q)
                        R31 = 2.0 * (x * z_q - ω * y)
                        z_tilde1 = R11 * z1 + R21 * z2 + R31 * z3

                        R12 = 2.0 * (x * y - ω * z_q)
                        R22 = 1.0 - 2.0 * (x * x + z_q * z_q)
                        R32 = 2.0 * (y * z_q + ω * x)
                        z_tilde2 = R12 * z1 + R22 * z2 + R32 * z3

                        R13 = 2.0 * (x * z_q + ω * y)
                        R23 = 2.0 * (y * z_q - ω * x)
                        R33 = 1.0 - 2.0 * (x * x + y * y)
                        z_tilde3 = R13 * z1 + R23 * z2 + R33 * z3

                        @simd for k in 1:s
                            I = k + (j - 1) * s + (i - 1) * s * r

                            x1 = X.block[1, k, j, i]
                            x2 = X.block[2, k, j, i]

                            # 3. Simplified 2D Euclidean distance
                            dist2 = (x1 - z_tilde1)^2 + (x2 - z_tilde2)^2
                            W_x = sqrt(max(0.0, 1.0 - x1 * x1 - x2 * x2))

                            term0 = sqrt_γ * (W_x - z_tilde3)
                            term1 = sqrt_γ * (-W_x - z_tilde3)

                            # 4. Evaluate map (deferred global constant)
                            acc += a[I] * exp(-γ * dist2) * (erf(term0) - erf(term1))
                        end
                    end
                end

                # Apply global constant C exactly once per voxel
                V[ix, iy, iz] = acc * C
            end
        end
    end

    return V
end





n = 10
r = 5
s = 5
m = 50
γ = 5.0
λ = 1e-2
@time X = rand_evaluation_grid(s, r, n, m)
@time Q = rand_quaternion_grid(r, n)

X_real = grid_to_real.(X.blocks, m)

@time block_sizes = repeat([s * r], n);
@time K = BlockMatrix{Float64}(undef, block_sizes, block_sizes)
@time build_gram_matrix!(K, X, Q, γ);
@time issymmetric(K)
eigvals(K)[1]

A = rand_block_diag(block_sizes)

k = 5
mypca = fpca(k, A, K; itmax=50)
Λ = mypca[1]
V = mypca[2]

# Similar to Idenity
V' * K * V ≈ I(k)
B = K * V 
A * B
≈ V * Diagonal(Λ)

# a = BlockVector{Float64}(undef, block_sizes)
# y = BlockVector{Float64}(undef, block_sizes)
# @time solve_mean!(a, K, y, λ)