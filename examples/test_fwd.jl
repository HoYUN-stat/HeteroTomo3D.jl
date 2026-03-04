using HeteroTomo3D
using GLMakie # Switched to GLMakie for native 3D rendering!

# 1. Generate the Phantom and Projections
m = 100
n_projections = 60
angles_rad = range(0, π, length=n_projections)
angles_deg = rad2deg.(angles_rad) # Convert purely to Degrees!
quats = [shortest_arc(cos(θ), sin(θ), 0.0) for θ in angles_rad]

println("Generating deterministic phantom and ray-casting...")
phantom = rand_shepp_logan_3d(m; λ=0.0, trunc=HyperbolicTangent(10.0))
projections = xray_transform(phantom, quats; n_steps=m)

# 2. Extract Sinogram & Global Color Limits
center_slice = div(m, 2)
sinogram = projections[:, center_slice, :]

# Lock limits for the uniform projection colorbar
proj_clims = (minimum(projections), maximum(projections))

# 3. Setup the 2x2 Layout Figure
fig = Figure(size=(1600, 1200), fontsize=20)

# --- TOP LEFT: 3D Phantom & Isolated Colorbar ---
ax1 = Axis3(fig[1, 1],
    title="Original 3D Phantom",
    elevation=pi / 8, azimuth=pi / 4,
    aspect=(1, 1, 1)
)
# GLMakie will flawlessly render these semi-transparent density shells
hm_phantom = contour!(ax1, phantom, levels=[0.5, 2.0, 5.0, 10.0], colormap=:viridis, alpha=0.15)
Colorbar(fig[1, 2], hm_phantom, label="Phantom Density", height=Relative(0.8))

# --- TOP RIGHT: Sinogram ---
ax2 = Axis(fig[1, 3],
    title="Center Sinogram (Z = $center_slice)",
    xlabel="Projection Angle (Degrees)",
    ylabel="Detector X"
)
hm_proj = heatmap!(ax2, angles_deg, 1:m, sinogram', colorrange=proj_clims, colormap=:greys)

# --- BOTTOM LEFT: 0° Shadow ---
q1 = quats[1]
q1_str = "[$(round(q1.ω, digits=2)), $(round(q1.x, digits=2)), $(round(q1.y, digits=2)), $(round(q1.z, digits=2))]"
ax3 = Axis(fig[2, 1], title="Shadow at 0°\nq = $q1_str", aspect=1)
heatmap!(ax3, projections[:, :, 1], colorrange=proj_clims, colormap=:greys)

# --- BOTTOM RIGHT: 90° Shadow ---
q30 = quats[30]
q30_str = "[$(round(q30.ω, digits=2)), $(round(q30.x, digits=2)), $(round(q30.y, digits=2)), $(round(q30.z, digits=2))]"
ax4 = Axis(fig[2, 3], title="Shadow at 90°\nq = $q30_str", aspect=1)
heatmap!(ax4, projections[:, :, 30], colorrange=proj_clims, colormap=:greys)

# --- UNIFORM PROJECTION COLORBAR ---
# This spans the entire right side of the figure, tying the sinogram and shadows together
Colorbar(fig[1:2, 4], hm_proj, label="Integrated Intensity", height=Relative(0.8))

fig

save("docs/src/assets/forward_simulation.png", fig)
