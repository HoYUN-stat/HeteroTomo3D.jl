"""
    KernelPhantom3D{T<:Real}

A 3D phantom represented functionally as a linear combination of Gaussian kernels.
This avoids heavy 3D voxel grid allocations and provides exact analytic X-ray transforms.
```math
f_i (\\mathbf{z}) = \\sum_{l=1}^L a_{l,i} \\exp(-\\gamma_l \\| \\mathbf{z} - \\mathbf{c}_l \\|^2)
```

# Fields
- `weights::Matrix{T}`: The weight coefficients of size `L × n` for `n` phantoms.
- `centers::Vector{NTuple{3,T}}`: The center coordinates of each Gaussian kernel in 3D space.
- `gammas::Vector{T}`: The bandwidth parameters of each 3D Gaussian kernel.
"""
struct KernelPhantom3D{T<:Real}
    weights::Matrix{T} # L × n matrix of weights for n phantoms
    centers::Vector{NTuple{3,T}}
    gammas::Vector{T}
end

"""
    rand_center_grid(L::Int; seed::Union{Nothing,Int}=nothing) -> Vector{NTuple{3,T}}

Create `L` random 3D centers within a unit sphere.

# Arguments
- `L::Int`: Number of random 3D centers to generate.

# Keyword Argument
- `seed::Union{Nothing, Int}`: Random seed for reproducibility. Defaults to `nothing`.

# Examples
```julia-repl
julia> using Random; Random.seed!(42);

julia> L = 2;

julia> centers = rand_center_grid(L; seed=42)
2-element Vector{Tuple{Float64, Float64, Float64}}:
 (0.6293451231426089, 0.4503389405961936, 0.47740714343281776)
 (0.7031298490032014, 0.6733461456394962, 0.16589443479313404)
```

See also [`EvaluationGrid`](@ref), [`rand_quaternion_grid`](@ref).
"""
function rand_center_grid(L::Int; seed::Union{Nothing,Int}=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end

    centers = Vector{NTuple{3,Float64}}(undef, L)
    for l in 1:L
        # Rejection sampling
        while true
            x = rand(Float64)
            y = rand(Float64)
            z = rand(Float64)
            if x^2 + y^2 + z^2 <= 1.0
                centers[l] = (x, y, z)
                break
            end
        end
    end
    return centers
end



"""
    xray_transform(phantom::KernelPhantom3D{T}, X::EvaluationGrid{I}, Q::QuaternionGrid{T}) where {T<:Real, I<:Integer}

Evaluates the analytic X-ray transform of a `KernelPhantom3D` over the specified evaluation and rotation grids.
Returns a 3D array of size `(s, r, n)`.
"""
function xray_transform(
    phantom::KernelPhantom3D{T},
    X::EvaluationGrid{I},
    Q::QuaternionGrid{T}
) where {T<:Real,I<:Integer}

    s = X.s
    r = X.r
    n = X.n
    m = X.m
    L = size(phantom.weights, 1)

    # Pre-allocate the continuous block of memory for the projections
    projections = Array{T,3}(undef, s, r, n)

    # Thread over the outermost grid dimension (functions).
    # Every thread owns a unique 'i' slice, guaranteeing 100% thread safety.
    Threads.@threads for i in 1:n
        for j in 1:r
            @inbounds q = Q[j, i]
            for k in 1:s
                @inbounds x_int = X[k, j, i]
                x_real = grid_to_real(x_int, m, T)
                pixel_val = zero(T)

                @simd for l in 1:L
                    @inbounds center = phantom.centers[l]
                    @inbounds γ = phantom.gammas[l]
                    @inbounds a = phantom.weights[l, i]

                    pixel_val += a * backproject(q, x_real, center, γ)
                end

                @inbounds projections[k, j, i] = pixel_val
            end
        end
    end

    return projections
end


