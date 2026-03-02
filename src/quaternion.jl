import Base: inv, conj, *

"""
    UnitQuaternion{T<:Real}

Represents a unit quaternion q = (ω, x, y, z) ∈ S^3 for singularity-free 3D rotations.
"""
struct UnitQuaternion{T<:Real}
    ω::T
    x::T
    y::T
    z::T
end

"""
    conj(q::UnitQuaternion)
    inv(q::UnitQuaternion)

For a unit quaternion, the inverse is equal to its conjugate (ω, -x, -y, -z).
"""
@inline Base.conj(q::UnitQuaternion) = UnitQuaternion(q.ω, -q.x, -q.y, -q.z)
@inline Base.inv(q::UnitQuaternion) = conj(q)

"""
    *(q1::UnitQuaternion, q2::UnitQuaternion)

Overloads the multiplication operator to compose two rotations: R_{q1} R_{q2} = R_{q1 q2}.
"""
@inline function Base.:*(q1::UnitQuaternion, q2::UnitQuaternion)
    ω1, x1, y1, z1 = q1.ω, q1.x, q1.y, q1.z
    ω2, x2, y2, z2 = q2.ω, q2.x, q2.y, q2.z

    @inbounds return UnitQuaternion(
        ω1 * ω2 - x1 * x2 - y1 * y2 - z1 * z2,
        ω1 * x2 + x1 * ω2 + y1 * z2 - z1 * y2,
        ω1 * y2 - x1 * z2 + y1 * ω2 + z1 * x2,
        ω1 * z2 + x1 * y2 - y1 * x2 + z1 * ω2
    )
end

"""
    projection_axis(q::UnitQuaternion)

Extracts the projection axis r_q = R_{q}^{-1} e_3 ∈ S^2.
Returns a Tuple (x, y, z) to guarantee 0 heap allocations.
"""
@inline function projection_axis(q::UnitQuaternion)
    ω, x, y, z = q.ω, q.x, q.y, q.z
    return (
        2 * (x * z - ω * y),
        2 * (y * z + ω * x),
        1 - 2 * (x * x + y * y)
    )
end

"""
    planar_rotation(q::UnitQuaternion)

Extracts the first column of the 2D in-plane rotation matrix (SO(2) component), 
i.e., it returns a Tuple (c, s) representing [c -s; s c] ∈ SO(2).
"""
@inline function planar_rotation(q::UnitQuaternion)
    ω, z = q.ω, q.z
    n2 = ω * ω + z * z
    @assert n2 > 0
    # Avoid division by zero by assuming non-parallel views (n2 > 0)
    inv_n2 = 1.0 / n2

    # Precompute elements
    c = (ω * ω - z * z) * inv_n2
    s = (2 * ω * z) * inv_n2

    # Return only the first column for zero-allocation
    return (c, s)
end

"""
    shortest_arc(ux::Real, uy::Real, uz::Real)

Computes the shortest-arc rotation E(u) ∈ SO(3) mapping e_3 to u ∈ S^2.
Requires u ∈ S^2 - {-e_3}.
"""
@inline function shortest_arc(ux::Real, uy::Real, uz::Real)
    # Used one(uz) and typeof(uz)(2) for flawless type stability.
    @assert uz != -1
    T = typeof(uz)
    two = T(2)

    denom = sqrt(two * (one(T) + uz))
    return UnitQuaternion(
        sqrt((one(T) + uz) / two),
        uy / denom,
        -ux / denom,
        zero(T)
    )
end