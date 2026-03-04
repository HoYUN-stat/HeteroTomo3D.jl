"""
    trilinear_interp(vol::Array{Float64, 3}, x::Float64, y::Float64, z::Float64)::Float64

Performs fast trilinear interpolation for a 3D volume.
Coordinates `x, y, z` are expected in the range `[-1.0, 1.0]`. 
Returns `0.0` for out-of-bound requests.
"""
@inline function trilinear_interp(vol::Array{Float64,3}, x::Float64, y::Float64, z::Float64)::Float64
    if x^2 + y^2 + z^2 > 1.0
        return 0.0
    end

    m = size(vol, 1)
    scale = 0.5 * (m - 1.0)

    # Map from [-1.0, 1.0] to [1.0, m]
    xi = (x + 1.0) * scale + 1.0
    yi = (y + 1.0) * scale + 1.0
    zi = (z + 1.0) * scale + 1.0

    if xi < 1.0 || xi > m || yi < 1.0 || yi > m || zi < 1.0 || zi > m
        return 0.0
    end

    # Fast integer truncation
    x0 = trunc(Int, xi)
    y0 = trunc(Int, yi)
    z0 = trunc(Int, zi)

    # Upper bounds (avoid exceeding m)
    x1 = x0 + (x0 < m)
    y1 = y0 + (y0 < m)
    z1 = z0 + (z0 < m)

    xd = xi - x0
    yd = yi - y0
    zd = zi - z0

    @inbounds begin
        # Interpolate along X
        c00 = vol[x0, y0, z0] * (1.0 - xd) + vol[x1, y0, z0] * xd
        c10 = vol[x0, y1, z0] * (1.0 - xd) + vol[x1, y1, z0] * xd
        c01 = vol[x0, y0, z1] * (1.0 - xd) + vol[x1, y0, z1] * xd
        c11 = vol[x0, y1, z1] * (1.0 - xd) + vol[x1, y1, z1] * xd
    end

    # Interpolate along Y
    c0 = c00 * (1.0 - yd) + c10 * yd
    c1 = c01 * (1.0 - yd) + c11 * yd

    # Interpolate along Z
    return c0 * (1.0 - zd) + c1 * zd
end

"""
    xray_transform(volume::Array{Float64, 3}, quats::AbstractVector{UnitQuaternion{Float64}}; n_steps::Int=size(volume, 1))

Computes the 3D X-ray transform (projection) for an array of `UnitQuaternion` rotations:
```math
\\mathcal{P}(f)(\\mathbf{R}_{\\mathbf{q}})(\\mathbf{u}) = \\int_{-W(\\mathbf{u})}^{W(\\mathbf{u})} f(\\mathbf{R}_{\\mathbf{q}}^{-1} [\\mathbf{u} : z]) \\, dz
```
Returns a 3D array of size ``m \\times m \\times r`` where `r` is the number of projections (`UnitQuaternion`).

    # Arguments
    - `volume::Array{Float64, 3}`: The input 3D volume to be projected. Must be a cube of size `m x m x m`.
    - `quats::AbstractVector{UnitQuaternion{Float64}}`: An array of `UnitQuaternion` representing the projection angles. Each quaternion defines a unique projection direction.
    - `n_steps::Int`: The number of steps for ray marching along the projection direction. Default is `size(volume, 1)` for a balance between accuracy and performance.
"""
function xray_transform(volume::Array{Float64,3}, quats::AbstractVector{UnitQuaternion{Float64}}; n_steps::Int=size(volume, 1))
    m = size(volume, 1)
    @assert size(volume, 2) == m && size(volume, 3) == m "Volume must be a perfect cube."

    r = length(quats)
    projections = zeros(Float64, m, m, r)

    dw = 2.0 / (n_steps - 1)

    # Multithread over the projection angles (outermost parallelization)
    Base.Threads.@threads for proj_idx in 1:r
        # The detector views the volume from the rotated frame.
        # So we use the inverse quaternion to map detector coordinates back to the static volume.
        q_inv = inv(quats[proj_idx])
        ω, x, y, z = q_inv.ω, q_inv.x, q_inv.y, q_inv.z

        # Unroll the 3x3 rotation matrix
        R11 = 1.0 - 2.0 * (y^2 + z^2)
        R12 = 2.0 * (x * y - ω * z)
        R13 = 2.0 * (x * z + ω * y)

        R21 = 2.0 * (x * y + ω * z)
        R22 = 1.0 - 2.0 * (x^2 + z^2)
        R23 = 2.0 * (y * z - ω * x)

        R31 = 2.0 * (x * z - ω * y)
        R32 = 2.0 * (y * z + ω * x)
        R33 = 1.0 - 2.0 * (x^2 + y^2)

        for j in 1:m
            @inbounds v = 2.0 * (j - 1) / (m - 1) - 1.0
            for i in 1:m
                @inbounds u = 2.0 * (i - 1) / (m - 1) - 1.0

                # Base ray origin on detector at w = -1.0
                x0 = R11 * u + R12 * v - R13
                y0 = R21 * u + R22 * v - R23
                z0 = R31 * u + R32 * v - R33

                # Step vector for w
                dx = R13 * dw
                dy = R23 * dw
                dz = R33 * dw

                pixel_val = 0.0

                # Ray marching (inner SIMD loop)
                @simd for k in 1:n_steps
                    cx = x0 + (k - 1) * dx
                    cy = y0 + (k - 1) * dy
                    cz = z0 + (k - 1) * dz

                    pixel_val += trilinear_interp(volume, cx, cy, cz)
                end

                @inbounds projections[i, j, proj_idx] = pixel_val * dw
            end
        end
    end

    return projections
end