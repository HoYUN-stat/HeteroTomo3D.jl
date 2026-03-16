"""
    KernelPhantom3D{T<:Real}

A 3D phantom represented functionally as a linear combination of Gaussian kernels.
This avoids heavy 3D voxel grid allocations and provides exact analytic X-ray transforms.
```math
f(\\mathbf{z}) = \\sum_{l=1}^L a_l \\exp(-\\gamma_l \\| \\mathbf{z} - \\mathbf{c}_l \\|^2)
```

# Fields
- `weights::Vector{T}`: The weight coefficients for each Gaussian kernel.
- `centers::Vector{NTuple{3,T}}`: The center coordinates of each Gaussian kernel in 3D space.
- `gammas::Vector{T}`: The bandwidth parameters of each 3D Gaussian kernel.
"""
struct KernelPhantom3D{T<:Real}
    weights::Vector{T}
    centers::Vector{NTuple{3,T}}
    gammas::Vector{T}
end

# Callable struct to evaluate the phantom at a 3D point z
function (phantom::KernelPhantom3D{T})(z::NTuple{3,T}) where {T<:Real}
    val = zero(T)
    @inbounds @simd for l in eachindex(phantom.centers)
        c = phantom.centers[l]
        d2 = (z[1] - c[1])^2 + (z[2] - c[2])^2 + (z[3] - c[3])^2
        val += phantom.weights[l] * exp(-phantom.gammas[l] * d2)
    end
    return val
end

"""
    xray_transform(phantom::KernelPhantom3D{T}, q::UnitQuaternion{T}, x::NTuple{2, T}) -> T

Analytically computes the X-ray transform of the `KernelPhantom3D` evaluated at a single 2D projection coordinate `x` and rotation `q` using `backproject`.
"""
function xray_transform(phantom::KernelPhantom3D{T}, q::UnitQuaternion{T}, x::NTuple{2,T}) where {T<:Real}
    val = zero(T)
    @inbounds for l in eachindex(phantom.centers)
        val += phantom.weights[l] * backproject(q, x, phantom.centers[l], phantom.gammas[l])
    end
    return val
end

"""
    xray_transform(phantom::KernelPhantom3D{T}, X::EvaluationGrid{I}, Q::QuaternionGrid{T}) where {T<:Real, I<:Integer}

Evaluates the analytic X-ray transform of a `KernelPhantom3D` over the specified evaluation and rotation grids.
Returns a 3D array of size `(s, r, n)`.
"""
function xray_transform(phantom::KernelPhantom3D{T}, X::EvaluationGrid{I}, Q::QuaternionGrid{T}) where {T<:Real,I<:Integer}
    s, r, n = size(X)
    m = X.m
    projections = Array{T,3}(undef, s, r, n)

    Threads.@threads for i in 1:n
        for j in 1:r
            @inbounds q = Q[j, i]
            for k in 1:s
                @inbounds x_int = X[k, j, i]
                x_real = grid_to_real(x_int, m, T)
                projections[k, j, i] = xray_transform(phantom, q, x_real)
            end
        end
    end

    return projections
end
