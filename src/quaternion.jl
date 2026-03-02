import Base: inv, conj, *

"""
    UnitQuaternion{T<:Real}

Represents a unit quaternion ``\\mathbf{q} = (\\omega, x, y, z) \\in \\mathbb{S}^{3}`` for singularity-free 3D rotations.
"""
struct UnitQuaternion{T<:Real}
    ω::T
    x::T
    y::T
    z::T
end

"""
    conj(q::UnitQuaternion)

For a unit quaternion ``\\mathbf{q} = (\\omega, x, y, z)``, the inverse is equal to its conjugate ``\\mathbf{q}^{-1} = (\\omega, -x, -y, -z)``.
"""
@inline Base.conj(q::UnitQuaternion) = UnitQuaternion(q.ω, -q.x, -q.y, -q.z)

"""
    inv(q::UnitQuaternion)

For a unit quaternion ``\\mathbf{q} = (\\omega, x, y, z)``, the inverse is equal to its conjugate ``\\mathbf{q}^{-1} = (\\omega, -x, -y, -z)``.
"""
@inline Base.inv(q::UnitQuaternion) = conj(q)

"""
    *(q1::UnitQuaternion, q2::UnitQuaternion)

Overloads the multiplication operator to compose two rotations: ``\\mathbf{R}_{\\mathbf{q}_1} \\mathbf{R}_{\\mathbf{q}_2} = \\mathbf{R}_{\\mathbf{q}_1 \\mathbf{q}_2} \\in SO(3)``.
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

Extracts the projection axis ``\\mathbf{r}_{\\mathbf{q}} = \\mathbf{R}_{\\mathbf{q}}^{-1} \\mathbf{e}_3 \\in \\mathbb{S}^2``.
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

Extracts the first column of the 2D in-plane rotation matrix, 
i.e., it returns a Tuple (c, s) representing ``\\begin{bmatrix} c & -s \\ s & c \\end{bmatrix} \\in SO(2)`` that rotates the detector plane around the projection axis.
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

Computes the shortest-arc rotation ``\\mathbf{E}(\\mathbf{u}) \\in SO(3)`` mapping ``\\mathbf{e}_3`` to ``\\mathbf{u} \\in \\mathbb{S}^2``. Requires ``\\mathbf{u} \\in \\mathbb{S}^2 \\backslash \\{-\\mathbf{e}_3\\}`` for uniqueness.
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