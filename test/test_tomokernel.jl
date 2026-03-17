@testset "tomokernel.jl (Feature Map & Gram Matrix)" begin

    @testset "1. Grid Mapping (grid_to_real)" begin
        @test HeteroTomo3D.grid_to_real((3, 3), 5) == (0.0, 0.0)
        @test HeteroTomo3D.grid_to_real((1, 1), 5) == (-1.0, -1.0)
        @test HeteroTomo3D.grid_to_real((5, 5), 5) == (1.0, 1.0)
    end

    @testset "2. Backprojection Math & Boundaries" begin
        q = shortest_arc(0.0, 1.0, 0.0)
        γ = 10.0

        # Test expected value from docstring
        val = backproject(q, (0.1, -0.2), (0.3, -0.4, 0.5), γ)
        @test val ≈ 0.15197718857623901

        # Boundary Logic: Kernel center outside unit disk must return 0.0
        @test backproject(q, (1.1, 0.0), (0.0, 0.0, 0.0), γ) == 0.0

        # Boundary Logic: Evaluation point outside unit ball must return 0.0
        @test backproject(q, (0.0, 0.0), (1.0, 0.0, 0.0), γ) == 0.0
    end

    @testset "3. Inner Product Symmetry & Allocations" begin
        q_parallel = UnitQuaternion(sqrt(2) / 2, 0.0, 0.0, sqrt(2) / 2)
        q_rand = UnitQuaternion(0.5, 0.5, 0.5, 0.5) # Arbitrary non-collinear
        x1 = (0.1, -0.2)
        x2 = (0.3, -0.4)
        γ = 5.0

        # Collinear Symmetry
        val1_c = collinear_inner_product(q_parallel, x1, x2, γ)
        val2_c = collinear_inner_product(inv(q_parallel), x2, x1, γ)
        @test val1_c == val2_c

        # Noncollinear Symmetry
        val1_nc = noncollinear_inner_product(q_rand, x1, x2, γ)
        val2_nc = noncollinear_inner_product(inv(q_rand), x2, x1, γ)
        @test val1_nc == val2_nc

        # General Wrapper Logic
        @test inner_product(q_parallel, x1, x2, γ) == val1_c
        @test inner_product(q_rand, x1, x2, γ) == val1_nc

        # Memory Constraint Verification (Zero Heap Allocation)
        # We test this inside a local function to avoid REPL global scope artifacts
        function test_allocs(q, pt1, pt2, gam)
            return @allocated inner_product(q, pt1, pt2, gam)
        end
        @test test_allocs(q_rand, x1, x2, γ) == 0
    end

    @testset "4. Multithreaded Gram Matrix Integrity" begin
        # Small dimensions to keep the test instantaneous
        s, r, n, m = 3, 3, 2, 5
        γ = 10.0

        X = rand_evaluation_grid(s, r, n, m; seed=123)
        Q = rand_quaternion_grid(r, n; seed=123)

        block_sizes = repeat([s * r], n)
        K = BlockMatrix{Float64}(undef, block_sizes, block_sizes)

        # Execute the multithreaded assembly
        build_gram_matrix!(K, X, Q, γ)

        # 1. Structural Integrity Check (Threads didn't collide and mirror logic works)
        @test issymmetric(K)

        # 2. Mathematical Integrity Check (Gram matrices must be Positive Semi-Definite)
        # We convert to a standard matrix to compute eigenvalues safely
        eigenvalues = eigvals(Symmetric(Matrix(K)))
        @test minimum(eigenvalues) > -1e-3 # Allow for microscopic floating-point noise
    end

end