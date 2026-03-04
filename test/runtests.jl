using HeteroTomo3D
using Test
using LinearAlgebra

@testset "HeteroTomo3D.jl" begin

    @testset "quaternion.jl (Geometry & Allocation)" begin
        # 1. Setup
        q_id = UnitQuaternion(1.0, 0.0, 0.0, 0.0)
        q1 = UnitQuaternion(0.5, 0.5, 0.5, 0.5)

        # 2. Test Math Logic
        q_inv = inv(q1)
        @test q_inv.ω == 0.5
        @test q_inv.x == -0.5

        # Composition with identity should return itself
        q_comp = q_id * q1
        @test q_comp.ω == q1.ω && q_comp.x == q1.x

        # Shortest arc mapping e_3 (0,0,1) to e_2 (0,1,0)
        q_arc = shortest_arc(0.0, 1.0, 0.0)
        @test q_arc.ω ≈ sqrt(2) / 2
        @test q_arc.x ≈ sqrt(2) / 2
        @test q_arc.y ≈ 0
        @test q_arc.z ≈ 0

        # 3. Test Space Complexity (Zero Heap Allocation Guarantee)
        # We test allocations inside a local scope to avoid global REPL artifacts
        allocs_compose = @allocated (q1 * q_inv)
        @test allocs_compose == 0

        allocs_proj = @allocated projection_axis(q1)
        @test allocs_proj == 0

        allocs_arc = @allocated shortest_arc(0.0, 1.0, 0.0)
        @test allocs_arc == 0
    end

    @testset "tomokernel.jl (Math & Allocation)" begin
        # 1. Math Verification (Expected values from docstrings)
        # We access un-exported internal functions using HeteroTomo3D.function_name
        @test HeteroTomo3D.antid_erf(2.0) ≈ 3.5466412019384204

        # Test bvncdf against the 10^5 scaled example in the docstring
        bvncdf_val = HeteroTomo3D.bvncdf(-2.0, -2.0, 0.0)
        @test (10^5 * bvncdf_val) ≈ 51.75685036595643

        # 2. Allocation Testing for 3D Kernel Operations
        q_id = UnitQuaternion(1.0, 0.0, 0.0, 0.0)
        q_test = shortest_arc(0.0, 1.0, 0.0)
        γ = 10.0

        # Backprojection
        allocs_bp = @allocated backproject(q_test, 0.1, -0.2, 0.3, -0.4, 0.5, γ)
        @test allocs_bp == 0

        # Inner Product: Collinear Branch (rq ≈ 1.0)
        allocs_ip_col = @allocated inner_product(q_id, 0.1, 0.2, q_id, 0.1, 0.2, γ)
        @test allocs_ip_col == 0

        # Inner Product: Non-Collinear Branch (rq < 1.0)
        allocs_ip_ncol = @allocated inner_product(q_id, 0.1, 0.2, q_test, 0.1, 0.2, γ)
        @test allocs_ip_ncol == 0
    end

    @testset "Mean Estimation and Reconstruction" begin
        # Dimensions for a small test suite
        s, r, n = 5, 3, 2
        m = 20
        γ = 10.0
        λ = 1e-4

        # 1. Initialize Grids with random data
        X = EvaluationGrid(rand(2, s, r, n), s, r, n)
        Q = QuaternionGrid([shortest_arc(1.0, 0.0, 0.0) for _ in 1:r, _ in 1:n], r, n)

        y = rand(s * r * n)
        a = zeros(s * r * n)
        K = zeros(s * r * n, s * r * n)

        # 2. Test Gram Matrix Assembly
        build_mean_gram!(K, X, Q, γ)
        @test issymmetric(K)
        @test all(diag(K) .>= 0)

        # 3. Test Solver (In-place routes to a)
        solve_mean!(a, K, y, λ)
        @test !any(isnan.(a))
        @test length(a) == s * r * n

        # 4. Test Reconstruction
        V = reconstruct_mean(a, X, Q, m, γ)
        @test size(V) == (m, m, m)
        @test !any(isnan.(V))
    end
end