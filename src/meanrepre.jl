"""
    grid_to_real(x::NTuple{2, I}, m::Int, ::Type{T}) where {I<:Integer, T<:Real} -> NTuple{2, T}

Maps discrete voxel integer coordinates natively back to the continuous unit ball `[-1, 1]`.
"""
function grid_to_real(x::NTuple{2,I}, m::Int; T::Type{T}=Float64) where {I<:Integer,T<:Real}
    c = T(m + 1) / 2
    R = T(m - 1) / 2
    return ((T(x[1]) - c) / R, (T(x[2]) - c) / R)
end

s = 2
r = 2
n = 3
m = 50

x = (25, 25)
grid_to_real(x, m)

# Create the configuration vectors: [4, 4, 4] for rows and columns
block_sizes = repeat([s * r], n)

# Pre-allocate the BlockMatrix (uninitialized to save time)
K = BlockMatrix{Float64}(undef, block_sizes, block_sizes)
X = rand_evaluation_grid(s, r, n, m);
Q = rand_quaternion_grid(r, n);

X.blocks
Q.blocks
X.s
size(X)
size(Q)
size(K)
X[1, 1, 1]

"""
    build_mean_gram!(K::Matrix{Float64}, X::EvaluationGrid{Float64}, Q::QuaternionGrid{Float64}, γ::Float64)

Constructs the `(s * r * n) x (s * r * n)` Gram matrix `K` for the mean representer theorem.
Iterates over functions (i), quaternions (j), and evaluation points (k), adhering to the column-major indexing `k + (j - 1) s + (i - 1) s r`.

See also [`collinear_inner_product`](@ref), [`noncollinear_inner_product`](@ref).
"""
function build_gram_matrix!(
    K::BlockMatrix{T},
    X::EvaluationGrid{I},
    Q::QuaternionGrid{T},
    γ::T
) where {T<:Real,I<:Integer}
    @assert X.r == Q.r "Inconsistent dimensions"
    @assert X.n == Q.n "Inconsistent dimensions"

    N = size(K, 1)
    @assert N == X.s * X.r * X.n "Inconsistent dimensions"
    s = X.s
    r = X.r
    n = X.n
    m = X.m

    # Multithread over the outermost function index
    Threads.@threads for i2 in 1:n
        base_J_i2 = (i2 - 1) * s * r
        @inbounds for j2 in 1:r
            base_J = (j2 - 1) * s + base_J_i2
            q2 = Q[j2, i2]
            for k2 in 1:s
                J = k2 + base_J
                x2 = X[k2, j2, i2]

                for i1 in 1:n
                    base_I_i1 = (i1 - 1) * s * r
                    for j1 in 1:r
                        base_I = (j1 - 1) * s + base_I_i1
                        q1 = Q.block[j1, i1]
                        @simd for k1 in 1:s
                            I = k1 + base_I

                            # Compute only the upper triangle
                            if I <= J
                                x1_1 = X.block[1, k1, j1, i1]
                                x1_2 = X.block[2, k1, j1, i1]

                                val = inner_product(q1, x1_1, x1_2, q2, x2_1, x2_2, γ)

                                K[I, J] = val
                            end
                        end
                    end
                end
            end
        end
    end

    @inbounds for J in 1:N
        @simd for I in 1:(J-1)
            K[J, I] = K[I, J] # Fill the lower triangle by symmetry
        end
    end

    return K
end

"""
    solve_mean!(a::Vector{Float64}, K::Matrix{Float64}, y::Vector{Float64}, λ::Float64)

Solves the regularized system `(K + λI) a = y` for the mean representer theorem.
"""
function solve_mean!(a::Vector{Float64}, K::Matrix{Float64}, y::Vector{Float64}, λ::Float64)
    N = size(K, 1)

    # Allocate one copy so K is not destroyed by the factorization
    K_reg = copy(K)

    # Tikhonov regularization
    @inbounds @simd for i in 1:N
        K_reg[i, i] += λ
    end

    # Overwrite the copied matrix with its Cholesky factors
    # F = cholesky!(Symmetric(K_reg))

    # In-place Linear Solve: Overwrites data vector `y` with coefficient vector `c`
    # ldiv!(a, F, y)
    ldiv!(a, K_reg, y)

    return a
end