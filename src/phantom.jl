"""
    TruncationType

Abstract type for the truncation function type ``h`` with ``h(0; \\gamma)=0``.
"""
abstract type TruncationType end

struct HyperbolicTangent <: TruncationType
    γ::Float64
end

struct ArcTangent <: TruncationType
    γ::Float64
end

struct Gudermannian <: TruncationType
    γ::Float64
end

"""
    truncation(r2::Float64, trunc::TruncationType)

Computes the smooth boundary truncation value using multiple dispatch for zero-allocation performance.
The input `r2` is the squared radius ``r^2 = (x/a)^2 + (y/b)^2 + (z/c)^2``.
"""
@inline function truncation(r2::Float64, trunc::HyperbolicTangent)
    t1 = tanh(trunc.γ * (1.0 - r2))^2
    t2 = tanh(trunc.γ * r2)^2
    return t1 / (t1 + t2)
end

@inline function truncation(r2::Float64, trunc::ArcTangent)
    t1 = atan(trunc.γ * (1.0 - r2))^2
    t2 = atan(trunc.γ * r2)^2
    return t1 / (t1 + t2)
end

@inline function truncation(r2::Float64, trunc::Gudermannian)
    t1 = 2.0 * atan(trunc.γ * sinh((1.0 - r2) / 2.0))^2
    t2 = 2.0 * atan(trunc.γ * sinh(r2 / 2.0))^2
    return t1 / (t1 + t2)
end

"""
    rand_shepp_logan_3d(m::Int; λ=2.0, seed=nothing, trunc=nothing)

Generates an ``m \\times m \\times m`` random 3D Shepp-Logan phantom inscribed in a unit sphere.
Executes with zero inner-loop heap allocations via Tuple unrolling and multithreading.
"""
function rand_shepp_logan_3d(m::Int;
    λ::Float64=2.0,
    seed::Union{Int,Nothing}=nothing,
    trunc::T=nothing) where {T<:Union{Nothing,TruncationType}}

    # 1. Allocate the single 3D volume
    phantom = zeros(Float64, m, m, m)
    @assert m > 1 "Grid size m must be greater than 1"

    # 2. Hardcoded 3D Geometry as Tuples (Forces compiler unrolling = 0 allocations)
    # Extruding the 2D logic to 3D with standard Z-axis parameters
    cx = (0.0, 0.0, 0.22, -0.22, 0.0, 0.0)
    cy = (0.0, -0.0184, 0.0, 0.0, 0.35, -0.455)
    cz = (0.0, 0.0, 0.0, 0.0, 0.25, -0.25)

    a = (0.69, 0.55, 0.15, 0.20, 0.21, 0.20)
    b = (0.92, 0.75, 0.31, 0.41, 0.25, 0.20)
    c = (0.81, 0.70, 0.22, 0.25, 0.25, 0.20)

    # Precompute trig constants for the Z-axis rotations
    theta_z = (0.0, 0.0, -18.0, 18.0, 0.0, 0.0)
    cos_t = Tuple(cosd(th) for th in theta_z)
    sin_t = Tuple(sind(th) for th in theta_z)

    grayLevel = (10.0, -7.0, -2.0, -2.0, 4.0, 3.0)

    # 3. Generate Random Noise
    if seed !== nothing
        Random.seed!(seed)
    end

    noise = randn(6) .* λ
    noise[1] = 0.0 # Standard background remains static
    noise[2] = 0.0

    # Convert final intensities back to a Tuple for loop speed
    xi = Tuple(grayLevel[i] + noise[i] for i in 1:6)

    # 4. Multithreaded Voxel Generation
    Base.Threads.@threads for k in 1:m
        @inbounds z = 2.0 * (k - 1) / (m - 1) - 1.0

        for j in 1:m
            @inbounds y = 2.0 * (j - 1) / (m - 1) - 1.0

            @simd for i in 1:m
                @inbounds x = 2.0 * (i - 1) / (m - 1) - 1.0

                val = 0.0

                # Because we used Tuples, LLVM unrolls this 1..6 loop completely
                for l in 1:6
                    dx = x - cx[l]
                    dy = y - cy[l]
                    dz = z - cz[l]

                    rot_x = cos_t[l] * dx + sin_t[l] * dy
                    rot_y = -sin_t[l] * dx + cos_t[l] * dy
                    rot_z = dz

                    r2 = (rot_x / a[l])^2 + (rot_y / b[l])^2 + (rot_z / c[l])^2

                    if r2 < 1.0
                        if T === Nothing
                            val += xi[l]
                        else
                            val += xi[l] * truncation(r2, trunc)
                        end
                    end
                end

                @inbounds phantom[i, j, k] = val
            end
        end
    end

    return phantom
end