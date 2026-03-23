# cd("examples")
# ] activate .

using HeteroTomo3D
using LinearAlgebra
using GLMakie
using Distributions
using Random
using BlockArrays
using Krylov

# --------------------------------------------------------------------
#          Global Parameters (Noisy Setting)
# --------------------------------------------------------------------
n = 30       # Single Deterministic Function
r = 5      # Number of quaternions
s = 20      # Number of evaluation points per viewing angles
m = 50      # Resolution for reconstruction
L = 4       # Number of Gaussian components in the phantom
λ = 0.2    # Covariance level for weights
σ = 0.01   # Noise level
gamma_val = 6.0    # Kernel bandwidth for RKHS framework

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
gammas = [5.0, 4.0, 6.0, 4.0] .* 2

# Draw random weights
mean_vec = [1.0, 0.8, 0.6, 0.4] .* 2
cov_matrix = λ * I(L) # Isotropic covariance for simplicity
Random.seed!(42)
weights = rand(MvNormal(mean_vec, cov_matrix), n) # Size: (L, n)

phantom = KernelPhantom3D(weights, centers, gammas)

# Generate the forward setup
X = rand_evaluation_grid(s, r, n, m; seed=123)    # Evaluation grid for the forward operator
Q = rand_quaternion_grid(r, n; seed=123)          # Random quaternion grid for the forward operator
projections = xray_transform(phantom, X, Q) # Size: (s, r, n)

# Add noise to the projections
Random.seed!(456)
noise = σ * randn(size(projections))
noisy_projections = projections + noise
y = vec(noisy_projections); # Flatten the projections to a vector for the linear system


# --------------------------------------------------------------------
#                      Evaluate True 3D Phantom
# --------------------------------------------------------------------

# Evaluate dense 3D phantom
function evaluate_phantom(w_vec, centers, gammas, m)
    F = zeros(Float64, m, m, m)
    for iz in 1:m, iy in 1:m, ix in 1:m
        z1 = 2.0 * (ix - 1) / (m - 1) - 1.0
        z2 = 2.0 * (iy - 1) / (m - 1) - 1.0
        z3 = 2.0 * (iz - 1) / (m - 1) - 1.0
        if z1^2 + z2^2 + z3^2 <= 1.0
            val = 0.0
            for l in eachindex(centers)
                c = centers[l]
                dist2 = (z1 - c[1])^2 + (z2 - c[2])^2 + (z3 - c[3])^2
                val += w_vec[l] * exp(-gammas[l] * dist2)
            end
            F[ix, iy, iz] = val
        end
    end
    return F
end

# Evaluate all 3D volumes
F_true = evaluate_phantom(mean_vec, centers, gammas, m);
squared_norm = sum(abs2, F_true)


# --------------------------------------------------------------------
#           Representer Solution & Reconstruction loop
# --------------------------------------------------------------------
block_sizes = repeat([s * r], n);
K = BlockMatrix{Float64}(undef, block_sizes, block_sizes);
println("Building Gram Matrix for γ = $gamma_val")
@time build_gram_matrix!(K, X, Q, gamma_val)

a_zero = zeros(size(K, 1)) # Create a zero initial guess for MINRES
kc_mean = KrylovConstructor(a_zero)
workspace_mean = MinresWorkspace(kc_mean)

println("Processing Mean Estimation for γ = $gamma_val")
fill!(workspace_mean.x, 0.0) # Ensure zero start across iterations
@time minres!(workspace_mean, K, y; history=true, itmax=20)
a_sol = Krylov.solution(workspace_mean)

F_recon = Array{Float64}(undef, m, m, m)
@time xray_recons!(F_recon, a_sol, X, Q, gamma_val)  # Computationally most expensive part

# Compute the L2 relative error strictly in-place
squared_diff = sum(abs2(F_recon[i] - F_true[i]) for i in eachindex(F_recon))
rel_error = sqrt(squared_diff / squared_norm)

# --------------------------------------------------------------------
#           Covariance Representer Solution
# --------------------------------------------------------------------
y_centered = y
y_block = BlockVector(y_centered, block_sizes)
Y = y_block ⊙ y_block

K_tens = CovFwdTensor(K)
Y_zero = zero_block_diag(block_sizes)
kc_cov = KrylovConstructor(Y_zero)
workspace_cov = MinresWorkspace(kc_cov)

println("Processing Covariance Estimation for γ = $gamma_val")
# MINRES solver
@time minres!(workspace_cov, K_tens, Y; history=true, itmax=20)
A_sol = Krylov.solution(workspace_cov)

# Functional PCA
println("Performing Functional PCA")
Λ, V = fpca(10, A_sol, K; itmax=100)

F_eigs = [Array{Float64}(undef, m, m, m) for _ in 1:3]
for k in 1:3
    println("Reconstructing Eigenfunction $k")
    @time xray_recons!(F_eigs[k], V[:, k], X, Q, gamma_val)
end

# --------------------------------------------------------------------
#                      Visualization
# --------------------------------------------------------------------
fig = Figure(size=(2400, 1000))
bounds = (-1.0, 1.0)

gl_top = fig[1:2, 1] = GridLayout()
gl_scree = fig[1, 2:3] = GridLayout()
gl_bot = fig[2, 2:3] = GridLayout()

# Top row
ax1 = Axis3(gl_top[1, 1], title="True", aspect=:data)
# alpha=0.4 to see inside.
vol1 = contour!(ax1, bounds, bounds, bounds, F_true,
    levels=6,
    colormap=:viridis,
    alpha=0.4)

ax_mean = Axis3(gl_top[2, 1], title="Reconstructed Mean\nRel. Error (L2): $(round(rel_error * 100, digits=2))%", aspect=:data)
vol_mean = contour!(ax_mean, bounds, bounds, bounds, F_recon, levels=6, colormap=:viridis, alpha=0.4)
Colorbar(gl_top[1:2, 2], vol_mean, label="Density")

ax_scree = Axis(gl_scree[1, 1], title="Scree Plot", xlabel="Principal Component", ylabel="Eigenvalue")
scatterlines!(ax_scree, 1:10, Λ, color=:blue, markersize=10)

# Bottom row (Eigenfunctions)
global_max_val = maximum(maximum(abs, F) for F in F_eigs)
global_max_val = global_max_val == 0 ? 1.0 : global_max_val

# Avoid zero level to prevent a massive block covering the whole volume
eig_levels = [-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8] .* global_max_val

vol_eig = nothing
for k in 1:3
    ax_eig = Axis3(gl_bot[1, k], title="Eigenfunction $k", aspect=:data)
    F_eig = F_eigs[k]
    vol_eig = contour!(ax_eig, bounds, bounds, bounds, F_eig, levels=eig_levels, colormap=:balance, colorrange=(-global_max_val, global_max_val), alpha=0.3)
end
Colorbar(gl_bot[1, 4], vol_eig, label="Value")

display(fig)

save_path = joinpath("..", "docs", "src", "assets", "cov_fpca_results.png")
save(save_path, fig)
