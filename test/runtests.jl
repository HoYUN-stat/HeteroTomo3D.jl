using HeteroTomo3D
using Test
using LinearAlgebra
using BlockArrays
using Krylov
using Random

# A clean list of all your test files
const TEST_FILES = [
    "test_quaternion.jl",
    "test_type.jl",
    "test_cdf.jl",
    "test_tomokernel.jl",
    "test_blocktensor.jl",
    "test_blockfpca.jl",
    "test_phantom.jl",
    "test_reconstruction.jl"
]

@testset "HeteroTomo3D.jl" begin
    for file in TEST_FILES
        include(file)
    end
end

# @testset "HeteroTomo3D.jl" begin



#     end
#     # @testset "tomokernel.jl (Math & Allocation)" begin
#     #     # 1. Math Verification (Expected values from docstrings)
#     #     # We access un-exported internal functions using HeteroTomo3D.function_name
#     #     @test HeteroTomo3D.antid_erf(2.0) ≈ 3.5466412019384204

#     #     # Test bvncdf against the 10^5 scaled example in the docstring
#     #     bvncdf_val = HeteroTomo3D.bvncdf(-2.0, -2.0, 0.0)
#     #     @test (10^5 * bvncdf_val) ≈ 51.75685036595643

#     #     # 2. Allocation Testing for 3D Kernel Operations
#     #     q_id = UnitQuaternion(1.0, 0.0, 0.0, 0.0)
#     #     q_test = shortest_arc(0.0, 1.0, 0.0)
#     #     γ = 10.0

#     #     # Backprojection
#     #     allocs_bp = @allocated backproject(q_test, (0.1, -0.2), (0.3, -0.4, 0.5), γ)
#     #     @test allocs_bp == 0

#     #     # Inner Product: Collinear Branch (rq ≈ 1.0)
#     #     allocs_ip_col = @allocated inner_product(q_id, (0.1, 0.2), (0.1, 0.2), γ)
#     #     @test allocs_ip_col == 0

#     #     # Inner Product: Non-Collinear Branch (rq < 1.0)
#     #     allocs_ip_ncol = @allocated inner_product(q_test, (0.1, 0.2), (0.1, 0.2), γ)
#     #     @test allocs_ip_ncol == 0
#     # end

#     # @testset "Mean Estimation and Reconstruction" begin
#     #     # Dimensions for a small test suite
#     #     s, r, n = 5, 3, 2
#     #     m = 20
#     #     γ = 10.0

#     #     # 1. Initialize Grids with random data
#     #     X = rand_evaluation_grid(s, r, n, m)
#     #     Q = rand_quaternion_grid(r, n)

#     #     # 2. Test Gram Matrix Assembly
#     #     block_sizes = repeat([s * r], n)
#     #     K = BlockMatrix{Float64}(undef, block_sizes, block_sizes)
#     #     build_gram_matrix!(K, X, Q, γ)
#     #     @test issymmetric(K)
#     #     @test K[1, 1] > 0

#     #     # 3. Test Reconstruction
#     #     a = zeros(s * r * n)
#     #     V = Array{Float64}(undef, m, m, m)
#     #     xray_recons!(V, a, X, Q, γ)
#     #     @test size(V) == (m, m, m)
#     #     @test !any(isnan.(V))
#     # end
# end