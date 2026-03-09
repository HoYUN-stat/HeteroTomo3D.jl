using HeteroTomo3D
using LinearAlgebra
using Random
using GLMakie

# Set random seed for reproducibility
Random.seed!(42)

# ==========================================
# 1. Parameter Setup
# ==========================================
s = 10        # Evaluation points per view
r = 500        # Number of projection angles
n = 1        # Number of 3D samples
N = s * r * n # Total observations (4,000)

m_fwd = 50    # Resolution of the true 3D volume for forward pass
m_rec = 50    # Resolution of the reconstructed 3D volume
γ = 10.0       # RKHS Bandwidth
λ = 1e-3      # Tikhonov Regularization

# ==========================================
# 2. Simulate Geometry and Data
# ==========================================
println("Setting up geometries for N = $N rays...")
# Generate random detector coordinates u, v in [-1, 1]
X_block = zeros(Float64, 2, s, r, n)
for i in 1:n
    for j in 1:r
        for k in 1:s
            rad = sqrt(rand())        # sqrt ensures uniform area density
            theta = rand() * 2π
            X_block[1, k, j, i] = rad * cos(theta)
            X_block[2, k, j, i] = rad * sin(theta)
        end
    end
end
X = EvaluationGrid(X_block, s, r, n)

# Generate random rotations (mostly around Z-axis for a typical scan)
Q_block = Matrix{UnitQuaternion{Float64}}(undef, r, n)
for i in 1:n
    for j in 1:r
        # Draw from 4D standard normal and normalize to S^3
        v1, v2, v3, v4 = randn(), randn(), randn(), randn()
        norm_v = sqrt(v1^2 + v2^2 + v3^2 + v4^2)

        Q_block[j, i] = UnitQuaternion(v1 / norm_v, v2 / norm_v, v3 / norm_v, v4 / norm_v)
    end
end
Q = QuaternionGrid(Q_block, r, n)

# ==========================================
# 3. Simulate Data via xray_transform
# ==========================================
println("Simulating forward projections...")

# Create a true 3D volume (e.g., a ball and a smaller offset sphere)
true_vol = zeros(Float64, m_fwd, m_fwd, m_fwd)
for z in 1:m_fwd, y in 1:m_fwd, x in 1:m_fwd
    cx = 2.0 * (x - 1) / (m_fwd - 1) - 1.0
    cy = 2.0 * (y - 1) / (m_fwd - 1) - 1.0
    cz = 2.0 * (z - 1) / (m_fwd - 1) - 1.0
    
    if cx^2 + cy^2 + cz^2 <= 0.4^2
        true_vol[x, y, z] = 1.0
    elseif (cx - 0.4)^2 + (cy - 0.4)^2 + cz^2 <= 0.2^2
        true_vol[x, y, z] = 0.5
    end
end

y = zeros(Float64, N)

# Project each sample and extract continuous evaluation points
# for i in 1:n
    # Get the r quaternions for this specific sample
    qs = Q.block[:, i]
    
    # Run the explicit ray-caster (returns m_fwd x m_fwd x r)
    projs = xray_transform(true_vol, qs; n_steps=m_fwd)
    
    # Extract the exact values at the continuous random (u,v) points
    for j in 1:r
        for k in 1:s
            I = k + (j - 1) * s + (i - 1) * s * r
            u = X.block[1, k, j, i]
            v = X.block[2, k, j, i]
            
            # Map (u, v) in [-1, 1] to the discrete index space [1, m_fwd]
            row_exact = (u + 1.0) * (m_fwd - 1) / 2.0 + 1.0
            col_exact = (v + 1.0) * (m_fwd - 1) / 2.0 + 1.0
            
            # Bilinear Interpolation
            r1 = clamp(floor(Int, row_exact), 1, m_fwd)
            c1 = clamp(floor(Int, col_exact), 1, m_fwd)
            r2 = clamp(r1 + 1, 1, m_fwd)
            c2 = clamp(c1 + 1, 1, m_fwd)
            
            dr = row_exact - r1
            dc = col_exact - c1
            
            val = (1 - dr) * (1 - dc) * projs[r1, c1, j] +
                       dr  * (1 - dc) * projs[r2, c1, j] +
                  (1 - dr) * dc  * projs[r1, c2, j] +
                       dr  * dc  * projs[r2, c2, j]
            
            # Assign to data vector with slight measurement noise
            y[I] = val + 0.02 * randn()
        end
    end
end

# ==========================================
# 4. Solve the Mean Representer Theorem
# ==========================================
println("Building $(N)x$(N) Gram matrix (Multithreaded)...")
K = zeros(Float64, N, N)
@time build_mean_gram!(K, X, Q, γ)


@inbounds @simd for i in 1:N
    K[i, i] += λ
end

println("Solving the regularized linear system...")
a = zeros(Float64, N)
a = K \ y

@time solve_mean!(a, K, y, λ)

# ==========================================
# 5. Reconstruct the 3D Mean Volume
# ==========================================
println("Reconstructing $(m_rec) x $(m_rec) x $(m_rec) volume (On-the-fly evaluation)...")
@time V = reconstruct_mean(a, X, Q, m_rec, γ)

# ==========================================
# 6. 3D Visualization with GLMakie
# ==========================================
println("Rendering true and reconstructed 3D volumes side-by-side...")

# Make the figure wider to accommodate two subplots
fig = Figure(size = (1200, 600))

# Left Plot: Ground Truth
ax1 = Axis3(fig[1, 1], 
    title = "Ground Truth Volume",
    xlabel = "X", ylabel = "Y", zlabel = "Z",
    elevation = π/6, azimuth = π/4,
    aspect = (1, 1, 1)
)

# Right Plot: RKHS Reconstruction
ax2 = Axis3(fig[1, 3], 
    title = "RKHS Estimated Mean Volume",
    xlabel = "X", ylabel = "Y", zlabel = "Z",
    elevation = π/6, azimuth = π/4,
    aspect = (1, 1, 1)
)

# GLMakie now strictly requires just the (start, stop) tuples for bounding boxes
bounds = (-1.0, 1.0)

# Render True Volume
true_int = contour!(ax1, bounds, bounds, bounds, true_vol, 
    levels = 8, 
    colormap = :viridis, 
    alpha = 0.3,
    transparency = true
)

Colorbar(fig[1, 2], true_int, label="Intensity", height=Relative(0.8))


# Render Reconstructed Volume
est_int = contour!(ax2, bounds, bounds, bounds, V, 
    levels = 8, 
    colormap = :viridis, 
    alpha = 0.3,
    transparency = true
)

Colorbar(fig[1, 4], est_int, label="Intensity", height=Relative(0.8))


save_path = joinpath("..", "docs", "src", "assets", "mean_reconstruction.png")
save(save_path, fig)
println("Saved reconstruction plot to: $save_path")

display(fig)