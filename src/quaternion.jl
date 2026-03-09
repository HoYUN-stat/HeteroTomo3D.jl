import Base: inv, conj, *

"""
    UnitQuaternion{T<:Real}

Represents a unit quaternion ``\\mathbf{q} = (\\omega, x, y, z) \\in \\mathbb{S}^{3}`` for singularity-free 3D rotations.
"""
struct UnitQuaternion{T<:Real} <: Number
    ω::T
    x::T
    y::T
    z::T
end


"""
    conj(q::UnitQuaternion{T}) -> UnitQuaternion{T}

For a unit quaternion ``\\mathbf{q} = (\\omega, x, y, z)``, the inverse is equal to its conjugate ``\\mathbf{q}^{-1} = (\\omega, -x, -y, -z)``.
"""
Base.conj(q::UnitQuaternion) = UnitQuaternion(q.ω, -q.x, -q.y, -q.z)

"""
    inv(q::UnitQuaternion{T}) -> UnitQuaternion{T}

For a unit quaternion ``\\mathbf{q} = (\\omega, x, y, z)``, the inverse is equal to its conjugate ``\\mathbf{q}^{-1} = (\\omega, -x, -y, -z)``.
"""
Base.inv(q::UnitQuaternion) = conj(q)

"""
    *(q1::UnitQuaternion{T}, q2::UnitQuaternion{T}) -> UnitQuaternion{T}

Overloads the multiplication operator to compose two unit quaternions 'q1' and 'q2'.

```jldoctest
julia> q_id = UnitQuaternion(1.0, 0.0, 0.0, 0.0)
UnitQuaternion{Float64}(1.0, 0.0, 0.0, 0.0)

julia> q_test = UnitQuaternion(0.5, 0.5, 0.5, 0.5)
UnitQuaternion{Float64}(0.5, 0.5, 0.5, 0.5)

julia> q_test * q_id == q_test
true
```
"""
function Base.:*(q1::UnitQuaternion, q2::UnitQuaternion)
    ω1, x1, y1, z1 = q1.ω, q1.x, q1.y, q1.z
    ω2, x2, y2, z2 = q2.ω, q2.x, q2.y, q2.z

    ω = (ω1 * ω2 - x1 * x2 - y1 * y2 - z1 * z2)
    x = (ω1 * x2 + x1 * ω2 + y1 * z2 - z1 * y2)
    y = (ω1 * y2 + y1 * ω2 + z1 * x2 - x1 * z2)
    z = (ω1 * z2 + x1 * y2 + z1 * ω2 - y1 * x2)

    @inbounds return UnitQuaternion(ω, x, y, z)
end


"""
    abs(q::UnitQuaternion{T}) -> T

Computes the absolute value operator to compute the norm of a unit quaternion.

```jldoctest
julia> q1 = UnitQuaternion(0.5, 0.5, 0.5, 0.5);

julia> q2 = UnitQuaternion(1.0, 0.0, 0.0, 0.0);

julia> abs(q1) ≈ 1.0
true

julia> abs(q2) ≈ 1.0
true

julia> q = q1 * q2
UnitQuaternion{Float64}(0.5, 0.5, 0.5, 0.5)

julia> abs(q) ≈ abs(q1) * abs(q2)
true
```
"""
function Base.abs(q::UnitQuaternion)
    a = max(abs(q.ω), abs(q.x), abs(q.y), abs(q.z))
    if isnan(a) && isinf(q)
        return typeof(a)(Inf)
    elseif iszero(a) || isinf(a)
        return a
    else
        ω = q.ω / a
        x = q.x / a
        y = q.y / a
        z = q.z / a
        return sqrt(ω * ω + x * x + y * y + z * z) * a
    end
end


"""
    rand([rng::AbstractRNG], ::Type{UnitQuaternion})

Generates a random `UnitQuaternion` uniformly distributed over ``\\mathbb{S}^3``.

```jldoctest
julia> using Random; Random.seed!(42);

julia> q = rand(UnitQuaternion);

julia> abs(q) ≈ 1.0
true
```
"""
function Random.rand(rng::Random.AbstractRNG, ::Random.SamplerType{UnitQuaternion})
    # Draw from standard normal and normalize
    ω, x, y, z = randn(rng), randn(rng), randn(rng), randn(rng)
    inv_norm = 1.0 / sqrt(ω^2 + x^2 + y^2 + z^2)
    return UnitQuaternion(ω * inv_norm, x * inv_norm, y * inv_norm, z * inv_norm)
end


"""
    rotate(q::UnitQuaternion{T}, v::NTuple{3, T}) where {T<:Real} -> NTuple{3, T}

Rotates a 3D vector `v` by the unit quaternion `q`, i.e., computes
```math
\\mathbf{R}_{\\mathbf{q}} \\mathbf{v} = \\mathbf{q} (0, \\mathbf{v}) \\mathbf{q}^{-1}.
```

```jldoctest
julia> using Random; Random.seed!(42);

julia> q = rand(UnitQuaternion);

julia> v = (1.0, 0.0, 0.0);

julia> v_rot = rotate(q, v);

julia> length(v_rot) == 3
true
```
"""
function rotate(q::UnitQuaternion{T}, v::NTuple{3,T}) where {T<:Real}
    ω, x, y, z = q.ω, q.x, q.y, q.z
    vx, vy, vz = v[1], v[2], v[3]

    # t = 2 * (q_vec x v)
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)

    # v_rot = v + ω * t + (q_vec x t)
    return (
        vx + ω * tx + (y * tz - z * ty),
        vy + ω * ty + (z * tx - x * tz),
        vz + ω * tz + (x * ty - y * tx)
    )
end

"""
    projection_axis(q::UnitQuaternion{T}) -> NTuple{3, T}

Extracts the projection axis ``\\mathbf{r}_{\\mathbf{q}} = \\mathbf{R}_{\\mathbf{q}}^{-1} \\mathbf{e}_3 \\in \\mathbb{S}^2``.

```jldoctest
julia> using Random; Random.seed!(42);

julia> q = rand(UnitQuaternion);

julia> e3 = (0.0, 0.0, 1.0);

julia> r3 = rotate(inv(q), e3);

julia> all(isapprox.(projection_axis(q), (r3.x, r3.y, r3.z), atol=1e-14))
true
```
"""
function projection_axis(q::UnitQuaternion)
    ω, x, y, z = q.ω, q.x, q.y, q.z
    return (
        2 * (x * z - ω * y),
        2 * (y * z + ω * x),
        1 - 2 * (x * x + y * y)
    )
end



"""
    shortest_arc(ux::Real, uy::Real, uz::Real) -> UnitQuaternion{T}

Computes the shortest-arc rotation ``\\mathbf{E}(\\mathbf{u}) \\in SO(3)`` mapping ``\\mathbf{e}_3`` to ``\\mathbf{u} \\in \\mathbb{S}^2``. Requires ``\\mathbf{u} \\in \\mathbb{S}^2 \\backslash \\{-\\mathbf{e}_3\\}`` for uniqueness.

```jldoctest
julia> using Random; Random.seed!(42);

julia> u = randn(3);

julia> u /= sqrt(sum(u .* u));

julia> q = shortest_arc(u...);

julia> e3 = rotate(q, Tuple(u));

julia> all(isapprox.(e3, (0.0, 0.0, 1.0), atol=1e-14))
true
```
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


"""
    planar_rotation(q::UnitQuaternion) -> NTuple{2, T}

Extracts the first column of the 2D in-plane rotation matrix, 
i.e., it returns a Tuple (c, s) representing ``\\begin{bmatrix} c & -s \\\\ s & c \\end{bmatrix} \\in SO(2)`` that rotates the detector plane around the projection axis.

```jldoctest
julia> using Random; Random.seed!(42);

julia> q = rand(UnitQuaternion);

julia> u = planar_rotation(q)
(0.07232925278690606, -0.997380809516249)

julia> u[1]^2 + u[2]^2 ≈ 1.0
true
```
"""
function planar_rotation(q::UnitQuaternion)
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
