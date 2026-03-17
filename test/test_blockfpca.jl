@testset "blockfpca.jl (Block FPCA & Lanczos)" begin

    # --------------------------------------------------------------------
    # Setup Shared Test Data
    # --------------------------------------------------------------------
    sizes = [2, 3]
    N = sum(sizes)

    # Create Block arrays
    A_diag = rand_block_diag(sizes)

    K_mat = rand(N, N)
    K_mat = K_mat' * K_mat + I # Ensure strictly PSD
    K = BlockMatrix(K_mat, sizes, sizes)

    v1 = BlockVector(rand(N), sizes)
    v2 = BlockVector(rand(N), sizes)

    @testset "1. Conjugate Lanczos Algorithm" begin
        # Run Lanczos
        Tri, Q, hist = HeteroTomo3D.conj_lanczos(v1, A_diag, K; itmax=5, tol=1e-10, history=true)

        # Check Return Types
        @test Tri isa SymTridiagonal
        @test Q isa AbstractMatrix
        @test length(hist) <= 5

        # Check K-Orthonormality (Q' * K * Q ≈ I)
        Q_dense = Matrix(Q)
        K_dense = Matrix(K)
        Identity_approx = Q_dense' * K_dense * Q_dense

        # The diagonal should be 1.0, off-diagonals should be 0.0
        @test Identity_approx ≈ I(size(Q_dense, 2)) atol = 1e-3
    end

    @testset "2. Functional PCA Wrapper" begin
        k_components = 2
        # Run FPCA
        pc_vals, pc_vecs = fpca(k_components, A_diag, K; itmax=5)

        # Verify Dimensions
        @test length(pc_vals) == k_components
        @test size(pc_vecs) == (N, k_components)

        # Verify eigenvalues are sorted in descending order
        if k_components > 1
            @test issorted(pc_vals, rev=true)
        end
    end

end