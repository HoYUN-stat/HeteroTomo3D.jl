n = 1
r = 10
s = 50
m = 50
L = 5

X = rand_evaluation_grid(s, r, n, m)
Q = rand_quaternion_grid(r, n)

weights = randn(L, n)
centers = rand_center_grid(L; seed=123)
gammas = repeat([5.0], L)
phantom = KernelPhantom3D(weights, centers, gammas)

projections = xray_transform(phantom, X, Q);

#Turn vec(projections) into a BlockVector
y = BlockVector{Float64}(undef, repeat([s * r], n))

n = 50
r = 5
s = 5
m = 50
γ = 5.0
λ = 1e-2
@time X = rand_evaluation_grid(s, r, n, m);
@time Q = rand_quaternion_grid(r, n);

X_real = grid_to_real.(X.blocks, m)

block_sizes = repeat([s * r], n);
@time K = BlockMatrix{Float64}(undef, block_sizes, block_sizes);
@time build_gram_matrix!(K, X, Q, γ);
@time issymmetric(K)
eigvals(K)[1]

a = BlockVector{Float64}(undef, block_sizes)
y = BlockVector{Float64}(undef, block_sizes)
@time solve_mean!(a, K, y, λ)



# Set the global parameters
σ::Float64 = 1e-1           #Standard deviation σ
jitter::Float64 = σ^2       #Perturbation level σ^2
g::Int64 = 500              #Resolution to plot the covariance
n::Int64 = 100               #Number of random functions
seed!(1234)                 #Seed for reproducibility
BS = rand(10:12, n)        #Number of locations for each function
r = median(BS)

# --------------------------------------------------------------------
#                      Data Generation
# --------------------------------------------------------------------
# Generate random locations
loc = loc_grid(BS, seed=345);
# Generate Gaussian process sample paths
process = BrownianBridge()
# process = IntegratedBM()
# process = BrownianMotion()
@time y = sample_gp(process, loc; jitter=jitter, seed=42);
@time Y = y ⊙ y;

myeval = range(start=0.0, step=1.0 / g, length=g)
Σ_true = covariancekernel(process, myeval)

# --------------------------------------------------------------------
#                Cubic B-Spline Covariance Smoothing
# --------------------------------------------------------------------
order = 4;  # Cubic B-splines
m1 = 8;  # Number of (internal) knots
m2 = 18;

p1 = m1 + order - 2
p2 = m2 + order - 2

knots1 = range(0, 1; length=m1);
knots2 = range(0, 1; length=m2);

myspline1 = BSplineMethod(order, knots1)
myspline2 = BSplineMethod(order, knots2)

Φ1 = mean_fwd(loc, myspline1);
Φ2 = mean_fwd(loc, myspline2);
Φ1_tens = CovFwdTensor(Φ1);
Φ2_tens = CovFwdTensor(Φ2);

# --------------------------------------------------------------------
#                Find L-curve Corner
# --------------------------------------------------------------------
A_spline1 = zero_block_diag([p1]);
A_spline2 = zero_block_diag([p2]);

kc_spline1 = KrylovConstructor(Y, A_spline1);
workspace_spline1 = LsqrWorkspace(kc_spline1);
kc_spline2 = KrylovConstructor(Y, A_spline2);
workspace_spline2 = LsqrWorkspace(kc_spline2);

G1 = galerkin_matrix(BSplineBasis(BSplineOrder(order), knots1));
G2 = galerkin_matrix(BSplineBasis(BSplineOrder(order), knots2));

function spline_sol_norm_callback(G_matrix::AbstractMatrix, results_vector::Vector{Float64})
    function callback(workspace)
        A_block = workspace.x.blocks[1]
        sol_norm = norm(G_matrix * A_block)
        push!(results_vector, sol_norm)
        return false
    end
    return callback
end

sol_norms1 = Vector{Float64}()
callback1 = spline_sol_norm_callback(G1, sol_norms1) # Create the first callback
lsqr!(workspace_spline1, Φ1_tens, Y; history=true, callback=callback1, itmax=50);
A_spline1 = solution(workspace_spline1)
stats_spline1 = statistics(workspace_spline1)
sol_norms1



sol_norms2 = Vector{Float64}()
callback2 = spline_sol_norm_callback(G2, sol_norms2) # Create the second callback
lsqr!(workspace_spline2, Φ2_tens, Y; history=true, callback=callback2, itmax=50);
A_spline2 = solution(workspace_spline2)
stats_spline2 = statistics(workspace_spline2)
sol_norms2

function find_l_curve_corner(x_coords, y_coords; min_index=1)
    # --- Start of the function is the same ---

    # Add a check for valid min_index to prevent errors
    if min_index >= length(x_coords)
        throw(ArgumentError("min_index ($min_index) must be less than the number of coordinates ($(length(x_coords)))."))
    end

    log_x = log10.(x_coords)
    log_y = log10.(y_coords)

    # Line endpoints
    p1 = (log_x[1], log_y[1])
    p_end = (log_x[end], log_y[end])

    # Vector for the line segment connecting start and end
    line_vec = (p_end[1] - p1[1], p_end[2] - p1[2])
    line_vec_norm_sq = line_vec[1]^2 + line_vec[2]^2

    # Find the perpendicular distance for each point to the line
    distances = zeros(length(log_x))
    for i in 2:(length(log_x)-1)
        point_vec = (log_x[i] - p1[1], log_y[i] - p1[2])
        cross_product = abs(point_vec[1] * line_vec[2] - point_vec[2] * line_vec[1])
        distances[i] = cross_product / sqrt(line_vec_norm_sq)
    end

    # --- This is the modified section ---

    # 1. Create a view of the array to search, starting from min_index.
    #    Using a @view is memory-efficient as it doesn't create a new array.
    search_region = @view distances[min_index:end]

    # 2. Find the maximum value and its index *within the view*.
    _, corner_idx_local = findmax(search_region)

    # 3. Convert the local index back to the index of the original `distances` array.
    corner_idx = corner_idx_local + min_index - 1

    return corner_idx
end

aresiduals1 = stats_spline1.Aresiduals[2:end]
aresiduals2 = stats_spline2.Aresiduals[2:end]

corner_idx1 = find_l_curve_corner(aresiduals1, sol_norms1)
corner_idx2 = find_l_curve_corner(aresiduals2, sol_norms2)

# --------------------------------------------------------------------
#                Rerun LSQR with the corner index
# --------------------------------------------------------------------
A_spline_early1 = zero_block_diag([p1]);
kc_spline_early1 = KrylovConstructor(Y, A_spline_early1);
workspace_spline_early1 = LsqrWorkspace(kc_spline_early1);
lsqr!(workspace_spline_early1, Φ1_tens, Y; history=true, itmax=corner_idx1);
A_spline_early1 = solution(workspace_spline_early1)
stats_spline_early1 = statistics(workspace_spline_early1)

A_spline_early2 = zero_block_diag([p2]);
kc_spline_early2 = KrylovConstructor(Y, A_spline_early2);
workspace_spline_early2 = LsqrWorkspace(kc_spline_early2);
lsqr!(workspace_spline_early2, Φ2_tens, Y; history=true, itmax=corner_idx2);
A_spline_early2 = solution(workspace_spline_early2)
stats_spline_early2 = statistics(workspace_spline_early2)

# Plot the smoothed covariance
E_spline1 = eval_fwd(myeval, myspline1)
Σ_spline_early1 = eval_covariance(E_spline1, A_spline_early1)

E_spline2 = eval_fwd(myeval, myspline2)
Σ_spline_early2 = eval_covariance(E_spline2, A_spline_early2)

Σ_spline_early1_error = Σ_spline_early1 - Σ_true
rel_MISE_spline_early1 = dot(Σ_spline_early1_error, Σ_spline_early1_error) / dot(Σ_true, Σ_true)

Σ_spline_early2_error = Σ_spline_early2 - Σ_true
rel_MISE_spline_early2 = dot(Σ_spline_early2_error, Σ_spline_early2_error) / dot(Σ_true, Σ_true)



# --------------------------------------------------------------------
#                RKHS Covariance Smoothing
# --------------------------------------------------------------------
γ_Gauss1 = 2.0
γ_Gauss2 = 10.0
γ_Lap1 = 1.0
γ_Lap2 = 3.0

kernel_Gauss1 = GaussianKernel(γ_Gauss1);
kernel_Gauss2 = GaussianKernel(γ_Gauss2);
kernel_Lap1 = LaplacianKernel(γ_Lap1);
kernel_Lap2 = LaplacianKernel(γ_Lap2);

K_Gauss1 = mean_fwd(loc, kernel_Gauss1);
K_Gauss2 = mean_fwd(loc, kernel_Gauss2);
K_Lap1 = mean_fwd(loc, kernel_Lap1);
K_Lap2 = mean_fwd(loc, kernel_Lap2);

K_Gauss1_tens = CovFwdTensor(K_Gauss1);
K_Gauss2_tens = CovFwdTensor(K_Gauss2);
K_Lap1_tens = CovFwdTensor(K_Lap1);
K_Lap2_tens = CovFwdTensor(K_Lap2);

Y_zero = zero_block_diag(BS); # Create a zero matrix for the initial guess

kc_Gauss1 = KrylovConstructor(Y_zero);
kc_Gauss2 = KrylovConstructor(Y_zero);
kc_Lap1 = KrylovConstructor(Y_zero);
kc_Lap2 = KrylovConstructor(Y_zero);

workspace_Gauss1 = MinresWorkspace(kc_Gauss1);
workspace_Gauss2 = MinresWorkspace(kc_Gauss2);
workspace_Lap1 = MinresWorkspace(kc_Lap1);
workspace_Lap2 = MinresWorkspace(kc_Lap2);

# --------------------------------------------------------------------
#                Find L-curve Corner
# --------------------------------------------------------------------
function kernel_sol_norm_callback(K_matrix, results_vector::Vector{Float64})
    function callback(workspace)
        A_blocks = workspace.x.blocks # Diagonal blocks of the solution
        K_blocks = K_matrix.blocks # blocks of the kernel matrix
        n = length(A_blocks)
        sol_norm = 0.0
        for i in 1:n
            A_block1 = A_blocks[i]
            for j in 1:n
                A_block2 = A_blocks[j]
                K_block = K_blocks[i, j]
                sol_norm += tr(K_block * A_block2 * K_block' * A_block1)
            end
        end
        sol_norm = sqrt(sol_norm)  # Take the square root for the norm
        # Store the solution norm in the results vector
        push!(results_vector, sol_norm)
        return false
    end
    return callback
end


sol_norms_Gauss1 = Vector{Float64}()
callback_Gauss1 = kernel_sol_norm_callback(K_Gauss1, sol_norms_Gauss1) # Create the first callback
minres!(workspace_Gauss1, K_Gauss1_tens, Y; history=true, callback=callback_Gauss1, itmax=50);
A_Gauss1 = solution(workspace_Gauss1)
stats_Gauss1 = statistics(workspace_Gauss1)
sol_norms_Gauss1
residuals_Gauss1 = stats_Gauss1.residuals[2:end]
corner_idx_Gauss1 = find_l_curve_corner(residuals_Gauss1, sol_norms_Gauss1; min_index=12)

function plot_Lcurve_annotated_Gauss1()
    corner_x = residuals_Gauss1[corner_idx_Gauss1]
    corner_y = sol_norms_Gauss1[corner_idx_Gauss1]

    fig = Figure(size=(1200, 800), fontsize=30)
    ax = Axis(fig[1, 1],
        xlabel="Residuals",
        ylabel="Solution norms",
        xscale=log10, # Use log10 for more intuitive tick labels
        yscale=log10
    )

    scatterlines!(ax, residuals_Gauss1, sol_norms_Gauss1, color=:blue, label="L-curve", markersize=12)
    scatter!(ax, corner_x, corner_y,
        color=:red,
        markersize=25,
        marker=:cross, # Use a cross or other symbol
        label="Corner (k = $corner_idx_Gauss1)"
    )

    axislegend(ax, position=:lb) # :ct is center top
    fig
end

plot_Lcurve_annotated_Gauss1()


sol_norms_Gauss2 = Vector{Float64}()
callback_Gauss2 = kernel_sol_norm_callback(K_Gauss2, sol_norms_Gauss2) # Create the first callback
minres!(workspace_Gauss2, K_Gauss2_tens, Y; history=true, callback=callback_Gauss2, itmax=50);
A_Gauss2 = solution(workspace_Gauss2)
stats_Gauss2 = statistics(workspace_Gauss2)
sol_norms_Gauss2
residuals_Gauss2 = stats_Gauss2.residuals[2:end]
corner_idx_Gauss2 = find_l_curve_corner(residuals_Gauss2, sol_norms_Gauss2; min_index=15)

function plot_Lcurve_annotated_Gauss2()
    corner_x = residuals_Gauss2[corner_idx_Gauss2]
    corner_y = sol_norms_Gauss2[corner_idx_Gauss2]

    fig = Figure(size=(1200, 800), fontsize=30)
    ax = Axis(fig[1, 1],
        xlabel="Residuals",
        ylabel="Solution norms",
        xscale=log10, # Use log10 for more intuitive tick labels
        yscale=log10
    )

    scatterlines!(ax, residuals_Gauss2, sol_norms_Gauss2, color=:blue, label="L-curve", markersize=12)
    scatter!(ax, corner_x, corner_y,
        color=:red,
        markersize=25,
        marker=:cross, # Use a cross or other symbol
        label="Corner (k = $corner_idx_Gauss2)"
    )

    axislegend(ax, position=:lb) # :ct is center top
    fig
end

plot_Lcurve_annotated_Gauss2()


sol_norms_Lap1 = Vector{Float64}()
callback_Lap1 = kernel_sol_norm_callback(K_Lap1, sol_norms_Lap1) # Create the first callback
minres!(workspace_Lap1, K_Lap1_tens, Y; history=true, callback=callback_Lap1, itmax=50);
A_Lap1 = solution(workspace_Lap1)
stats_Lap1 = statistics(workspace_Lap1)
sol_norms_Lap1
residuals_Lap1 = stats_Lap1.residuals[2:end]
corner_idx_Lap1 = find_l_curve_corner(residuals_Lap1, sol_norms_Lap1; min_index=10)

function plot_Lcurve_annotated_Lap1()
    corner_x = residuals_Lap1[corner_idx_Lap1]
    corner_y = sol_norms_Lap1[corner_idx_Lap1]

    fig = Figure(size=(1200, 800), fontsize=30)
    ax = Axis(fig[1, 1],
        xlabel="Residuals",
        ylabel="Solution norms",
        xscale=log10, # Use log10 for more intuitive tick labels
        yscale=log10
    )

    scatterlines!(ax, residuals_Lap1, sol_norms_Lap1, color=:blue, label="L-curve", markersize=12)
    scatter!(ax, corner_x, corner_y,
        color=:red,
        markersize=25,
        marker=:cross, # Use a cross or other symbol
        label="Corner (k = $corner_idx_Lap1)"
    )

    axislegend(ax, position=:lb) # :ct is center top
    fig
end

plot_Lcurve_annotated_Lap1()



sol_norms_Lap2 = Vector{Float64}()
callback_Lap2 = kernel_sol_norm_callback(K_Lap2, sol_norms_Lap2) # Create the first callback
minres!(workspace_Lap2, K_Lap2_tens, Y; history=true, callback=callback_Lap2, itmax=50);
A_Lap2 = solution(workspace_Lap2)
stats_Lap2 = statistics(workspace_Lap2)
sol_norms_Lap2
residuals_Lap2 = stats_Lap2.residuals[2:end]
corner_idx_Lap2 = find_l_curve_corner(residuals_Lap2, sol_norms_Lap2; min_index=10)

function plot_Lcurve_annotated_Lap2()
    corner_x = residuals_Lap2[corner_idx_Lap2]
    corner_y = sol_norms_Lap2[corner_idx_Lap2]

    fig = Figure(size=(1200, 800), fontsize=30)
    ax = Axis(fig[1, 1],
        xlabel="Residuals",
        ylabel="Solution norms",
        xscale=log10, # Use log10 for more intuitive tick labels
        yscale=log10
    )

    scatterlines!(ax, residuals_Lap2, sol_norms_Lap2, color=:blue, label="L-curve", markersize=12)
    scatter!(ax, corner_x, corner_y,
        color=:red,
        markersize=25,
        marker=:cross, # Use a cross or other symbol
        label="Corner (k = $corner_idx_Lap2)"
    )

    axislegend(ax, position=:lb) # :ct is center top
    fig
end

plot_Lcurve_annotated_Lap2()


# --------------------------------------------------------------------
#                Rerun MINRES with the corner index
# --------------------------------------------------------------------
kc_Gauss1 = KrylovConstructor(Y_zero);
workspace_Gauss1 = MinresWorkspace(kc_Gauss1);
minres!(workspace_Gauss1, K_Gauss1_tens, Y; history=true, itmax=corner_idx_Gauss1);
A_early_Gauss1 = solution(workspace_Gauss1)

kc_Gauss2 = KrylovConstructor(Y_zero);
workspace_Gauss2 = MinresWorkspace(kc_Gauss2);
minres!(workspace_Gauss2, K_Gauss2_tens, Y; history=true, itmax=corner_idx_Gauss2);
A_early_Gauss2 = solution(workspace_Gauss2)

kc_Lap1 = KrylovConstructor(Y_zero);
workspace_Lap1 = MinresWorkspace(kc_Lap1);
minres!(workspace_Lap1, K_Lap1_tens, Y; history=true, itmax=corner_idx_Lap1);
A_early_Lap1 = solution(workspace_Lap1)

kc_Lap2 = KrylovConstructor(Y_zero);
workspace_Lap2 = MinresWorkspace(kc_Lap2);
minres!(workspace_Lap2, K_Lap2_tens, Y; history=true, itmax=corner_idx_Lap2);
A_early_Lap2 = solution(workspace_Lap2)


E_Gauss1 = eval_fwd(myeval, loc, kernel_Gauss1)
Σ_Gauss1 = eval_covariance(E_Gauss1, A_Gauss1)
Σ_early_Gauss1 = eval_covariance(E_Gauss1, A_early_Gauss1)

Σ_Gauss1_error = Σ_Gauss1 - Σ_true
rel_MISE_Gauss1 = dot(Σ_Gauss1_error, Σ_Gauss1_error) / dot(Σ_true, Σ_true)
Σ_early_Gauss1_error = Σ_early_Gauss1 - Σ_true
rel_MISE_early_Gauss1 = dot(Σ_early_Gauss1_error, Σ_early_Gauss1_error) / dot(Σ_true, Σ_true)

E_Gauss2 = eval_fwd(myeval, loc, kernel_Gauss2)
Σ_Gauss2 = eval_covariance(E_Gauss2, A_Gauss2)
Σ_early_Gauss2 = eval_covariance(E_Gauss2, A_early_Gauss2)

Σ_Gauss2_error = Σ_Gauss2 - Σ_true
rel_MISE_Gauss2 = dot(Σ_Gauss2_error, Σ_Gauss2_error) / dot(Σ_true, Σ_true)
Σ_early_Gauss2_error = Σ_early_Gauss2 - Σ_true
rel_MISE_early_Gauss2 = dot(Σ_early_Gauss2_error, Σ_early_Gauss2_error) / dot(Σ_true, Σ_true)

E_Lap1 = eval_fwd(myeval, loc, kernel_Lap1)
Σ_Lap1 = eval_covariance(E_Lap1, A_Lap1)
Σ_early_Lap1 = eval_covariance(E_Lap1, A_early_Lap1)

Σ_Lap1_error = Σ_Lap1 - Σ_true
rel_MISE_Lap1 = dot(Σ_Lap1_error, Σ_Lap1_error) / dot(Σ_true, Σ_true)
Σ_early_Lap1_error = Σ_early_Lap1 - Σ_true
rel_MISE_early_Lap1 = dot(Σ_early_Lap1_error, Σ_early_Lap1_error) / dot(Σ_true, Σ_true)

E_Lap2 = eval_fwd(myeval, loc, kernel_Lap2)
Σ_Lap2 = eval_covariance(E_Lap2, A_Lap2)
Σ_early_Lap2 = eval_covariance(E_Lap2, A_early_Lap2)

Σ_Lap2_error = Σ_Lap2 - Σ_true
rel_MISE_Lap2 = dot(Σ_Lap2_error, Σ_Lap2_error) / dot(Σ_true, Σ_true)
Σ_early_Lap2_error = Σ_early_Lap2 - Σ_true
rel_MISE_early_Lap2 = dot(Σ_early_Lap2_error, Σ_early_Lap2_error) / dot(Σ_true, Σ_true)

function plot_spline()
    fig = Figure(size=(3000, 1500), fontsize=30)

    proc_string = string(nameof(typeof(process)))

    ax1 = Axis(fig[1, 1:2], xlabel="Time", ylabel="Value", title=proc_string)

    for i in 1:n
        # Use the colors array to select a color for each curve
        lines!(ax1, loc.blocks[i], y.blocks[i], color=:black, linewidth=1)
    end

    lines!(ax1, [NaN], [NaN], label="n = $n", color=:transparent)
    lines!(ax1, [NaN], [NaN], label="med(r) = $r", color=:transparent)
    lines!(ax1, [NaN], [NaN], label="σ = $σ", color=:transparent)
    axislegend(ax1, position=:lb)


    # Match the color limits of the heatmaps
    vmin = min(minimum(Σ_true), minimum(Σ_spline_early1), minimum(Σ_spline_early2), minimum(Σ_early_Gauss1))
    vmax = max(maximum(Σ_true), maximum(Σ_spline_early1), maximum(Σ_spline_early2), maximum(Σ_early_Gauss1))

    # Create a tuple for the color limits
    clims = (vmin, vmax)

    ax2 = Axis(fig[2, 1], xlabel="Time", ylabel="Time", title="True Covariance")
    h2 = heatmap!(ax2, myeval, myeval, Σ_true, colormap=:grays, colorrange=clims)
    Colorbar(fig[2, 2], h2)

    # 2. Plot the smoothed covariance for cubic B-splines
    ax3 = Axis(fig[1, 3], xlabel="Time", ylabel="Time", title="Cubic B-spline, p=$p1")
    h3 = heatmap!(ax3, myeval, myeval, Σ_spline_early1, colormap=:grays, colorrange=clims)
    Colorbar(fig[1, 4], h3)

    num_iter_early1 = "# of Iter: " * string(stats_spline_early1.niter)
    rel_MISE_string_early1 = "Rel.MISE: " * string(round(rel_MISE_spline_early1 * 100; digits=2)) * "%"

    lines!(ax3, [NaN], [NaN], label=num_iter_early1, color=:transparent)
    lines!(ax3, [NaN], [NaN], label=rel_MISE_string_early1, color=:transparent)
    axislegend(ax3, position=:lb)

    ax4 = Axis(fig[2, 3], xlabel="Time", ylabel="Time", title="Cubic B-spline, p=$p2")
    h4 = heatmap!(ax4, myeval, myeval, Σ_spline_early2, colormap=:grays, colorrange=clims)
    Colorbar(fig[2, 4], h4)

    num_iter_early2 = "# of Iter: " * string(stats_spline_early2.niter)
    rel_MISE_string_early2 = "Rel.MISE: " * string(round(rel_MISE_spline_early2 * 100; digits=2)) * "%"

    lines!(ax4, [NaN], [NaN], label=num_iter_early2, color=:transparent)
    lines!(ax4, [NaN], [NaN], label=rel_MISE_string_early2, color=:transparent)
    axislegend(ax4, position=:lb)

    # 3. Plot the smoothed covariance for Gaussian kernels
    γ_Gauss1_int = Int(γ_Gauss1)
    ax5 = Axis(fig[1, 5], xlabel="Time", ylabel="Time", title="Gaussian Kernel, γ=$γ_Gauss1_int")
    h5 = heatmap!(ax5, myeval, myeval, Σ_early_Gauss1, colormap=:grays, colorrange=clims)
    Colorbar(fig[1, 6], h5)

    num_iter_Gauss1 = "# of Iter: " * string(corner_idx_Gauss1)
    rel_MISE_string_Gauss1 = "Rel.MISE: " * string(round(rel_MISE_early_Gauss1 * 100; digits=2)) * "%"

    lines!(ax5, [NaN], [NaN], label=num_iter_Gauss1, color=:transparent)
    lines!(ax5, [NaN], [NaN], label=rel_MISE_string_Gauss1, color=:transparent)
    axislegend(ax5, position=:lb)

    γ_Gauss2_int = Int(γ_Gauss2)
    ax6 = Axis(fig[2, 5], xlabel="Time", ylabel="Time", title="Gaussian Kernel, γ=$γ_Gauss2_int")
    h6 = heatmap!(ax6, myeval, myeval, Σ_early_Gauss2, colormap=:grays, colorrange=clims)
    Colorbar(fig[2, 6], h6)

    num_iter_Gauss2 = "# of Iter: " * string(corner_idx_Gauss2)
    rel_MISE_string_Gauss2 = "Rel.MISE: " * string(round(rel_MISE_early_Gauss2 * 100; digits=2)) * "%"

    lines!(ax6, [NaN], [NaN], label=num_iter_Gauss2, color=:transparent)
    lines!(ax6, [NaN], [NaN], label=rel_MISE_string_Gauss2, color=:transparent)
    axislegend(ax6, position=:lb)

    #4. Plot the smoothed covariance for Laplacian kernels
    γ_Lap1_int = Int(γ_Lap1)
    ax7 = Axis(fig[1, 7], xlabel="Time", ylabel="Time", title="Laplacian Kernel, γ=$γ_Lap1_int")
    h7 = heatmap!(ax7, myeval, myeval, Σ_early_Lap1, colormap=:grays, colorrange=clims)
    Colorbar(fig[1, 8], h7)

    num_iter_Lap1 = "# of Iter: " * string(corner_idx_Lap1)
    rel_MISE_string_Lap1 = "Rel.MISE: " * string(round(rel_MISE_early_Lap1 * 100; digits=2)) * "%"

    lines!(ax7, [NaN], [NaN], label=num_iter_Lap1, color=:transparent)
    lines!(ax7, [NaN], [NaN], label=rel_MISE_string_Lap1, color=:transparent)
    axislegend(ax7, position=:lb)

    γ_Lap2_int = Int(γ_Lap2)
    ax8 = Axis(fig[2, 7], xlabel="Time", ylabel="Time", title="Laplacian Kernel, γ=$γ_Lap2_int")
    h8 = heatmap!(ax8, myeval, myeval, Σ_early_Lap2, colormap=:grays, colorrange=clims)
    Colorbar(fig[2, 8], h8)

    num_iter_Lap2 = "# of Iter: " * string(corner_idx_Lap2)
    rel_MISE_string_Lap2 = "Rel.MISE: " * string(round(rel_MISE_early_Lap2 * 100; digits=2)) * "%"

    lines!(ax8, [NaN], [NaN], label=num_iter_Lap2, color=:transparent)
    lines!(ax8, [NaN], [NaN], label=rel_MISE_string_Lap2, color=:transparent)
    axislegend(ax8, position=:lb)

    fig

    # file_name = "basis_choice"
    # file_date = Dates.format(now(), "yymmddHH")
    # CairoMakie.save("simul/" * proc_string * file_name * "_" * file_date * ".png", fig)
end

plot_spline()






