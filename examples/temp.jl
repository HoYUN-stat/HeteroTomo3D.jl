# cd("examples")
# ] activate .

using HeteroTomo3D
using BlockArrays
using LinearAlgebra
using Krylov
using GLMakie


# --------------------------------------------------------------------
#          Global Parameters (Deterministic Noiseless Setting)
# --------------------------------------------------------------------
n = 1       # Single Deterministic Function
r = 50      # Number of quaternions
s = 100      # Number of evaluation points per viewing angles
m = 50      # Resolution for reconstruction
L = 4       # Number of Gaussian components in the phantom
γ = 10.0    # Kernel bandwidth for RKHS framework


# --------------------------------------------------------------------
#                      Data Generation
# --------------------------------------------------------------------
# Generate Wrappers for 3D Phantom
centers = [
    (0.3, 0.3, 0.3),
    (-0.3, -0.3, 0.3),
    (-0.4, 0.4, -0.4),
    (0.3, -0.3, -0.3)
]
weights = reshape([1.0, 0.8, 0.6, 0.4], L, n)
gammas = [5.0, 4.0, 6.0, 4.0]

phantom = KernelPhantom3D(weights, centers, gammas)

# Generate the forward setup
X = rand_evaluation_grid(s, r, n, m)    # Evaluation grid for the forward operator
Q = rand_quaternion_grid(r, n)          # Random quaternion grid for the forward operator
projections = xray_transform(phantom, X, Q) # Size: (s, r, n)
y = vec(projections) # Flatten the projections to a vector for the linear system


# --------------------------------------------------------------------
#                      Representer Solution
# --------------------------------------------------------------------
block_sizes = repeat([s * r], n);
@time K = BlockMatrix{Float64}(undef, block_sizes, block_sizes);
@time build_gram_matrix!(K, X, Q, γ);
@time issymmetric(K)
eigvals(K)[1]

a_zero = zeros(size(K, 1)) # Create a zero initial guess for MINRES
kc_mean = KrylovConstructor(a_zero)
workspace_mean = MinresWorkspace(kc_mean)
@time minres!(workspace_mean, K, y; history=true, itmax=20)

a_sol = Krylov.solution(workspace_mean)
stats_mean = Krylov.statistics(workspace_mean)

# --------------------------------------------------------------------
#                      Reconstruction from Representer Solution
# --------------------------------------------------------------------
@time F = Array{Float64}(undef, m, m, m);
@time xray_recons!(F, a_sol, X, Q, γ);

# --------------------------------------------------------------------
#                      Visualization (GLMakie)
# --------------------------------------------------------------------
# 1. Evaluate the Ground Truth Phantom on the 3D grid
@time F_true = zeros(Float64, m, m, m);
for iz in 1:m
    z3 = 2.0 * (iz - 1) / (m - 1) - 1.0
    for iy in 1:m
        z2 = 2.0 * (iy - 1) / (m - 1) - 1.0
        for ix in 1:m
            z1 = 2.0 * (ix - 1) / (m - 1) - 1.0

            # Unit ball cutoff matching the reconstruction
            if z1^2 + z2^2 + z3^2 > 1.0
                continue
            end

            val = 0.0
            for l in 1:L
                c = phantom.centers[l]
                dist2 = (z1 - c[1])^2 + (z2 - c[2])^2 + (z3 - c[3])^2
                val += phantom.weights[l, 1] * exp(-phantom.gammas[l] * dist2)
            end
            F_true[ix, iy, iz] = val
        end
    end
end

# Compute the L2 relative error strictly in-place
squared_diff = sum(abs2(F[i] - F_true[i]) for i in eachindex(F))
squared_norm = sum(abs2, F_true)

rel_error = sqrt(squared_diff / squared_norm)

# 2. Plot Side-by-Side Volumes using Isosurfaces
fig = Figure(size=(1000, 500))
bounds = (-1.0, 1.0)

ax1 = Axis3(fig[1, 1], title="True 3D Phantom", aspect=:data)
# levels=6 draws 6 distinct density shells. alpha=0.4 makes them glassy so you can see inside.
vol1 = contour!(ax1, bounds, bounds, bounds, F_true,
    levels=6,
    colormap=:viridis,
    alpha=0.4)

ax2 = Axis3(fig[1, 2], title="Reconstructed Phantom (Rel. Error (L2): $(round(rel_error * 100, digits=2))%)", aspect=:data)
vol2 = contour!(ax2, bounds, bounds, bounds, F,
    levels=6,
    colormap=:viridis,
    alpha=0.4)

Colorbar(fig[1, 3], vol2, label="Density")

display(fig)

# # 2. Plot Side-by-Side Volumes
# fig = Figure(size=(1000, 500))
# bounds = (-1.0, 1.0)

# # Extract the base colors from viridis
# base_cmap = to_colormap(:viridis)

# # Create a new colormap where the alpha scales with the value
# # cmap_alpha = [RGBAf(c.r, c.g, c.b, (i / length(base_cmap))^2) for (i, c) in enumerate(base_cmap)]
# cmap_alpha = [RGBAf(c.r, c.g, c.b, min(1.0f0, 1.5f0 * (i / length(base_cmap)))) for (i, c) in enumerate(base_cmap)]

# # Sync the maximum density across both plots so their colorbars match perfectly
# max_density = max(maximum(F_true), maximum(F))

# ax1 = Axis3(fig[1, 1], title="True 3D Phantom", aspect=:data)
# vol1 = volume!(ax1, bounds, bounds, bounds, F_true,
#     algorithm=:absorption,
#     colormap=cmap_alpha,
#     colorrange=(0.0, max_density), # Lock the bottom of the colormap to exactly 0.0
#     lowclip=:transparent)          # Force all negative values to be completely invisible

# ax2 = Axis3(fig[1, 2], title="Reconstructed Phantom", aspect=:data)
# vol2 = volume!(ax2, bounds, bounds, bounds, F,
#     algorithm=:absorption,
#     colormap=cmap_alpha,
#     colorrange=(0.0, max_density),
#     lowclip=:transparent)

# # Colorbar(fig[1, 2], vol1, label="Density")
# Colorbar(fig[1, 3], vol2, label="Density")

# display(fig)