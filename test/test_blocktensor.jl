@testset "blocktensor.jl (Block Tensor Operations)" begin
    # --------------------------------------------------------------------
    # Setup Shared Test Data
    # --------------------------------------------------------------------
    sizes = [2, 3] # Two blocks: a 2x2 and a 3x3
    K_mat = rand(5, 5)
    K = BlockMatrix(K_mat, sizes, sizes)

    A = rand_block_diag(sizes)
    B = zero_block_diag(sizes)

    @testset "1. Tensor Initialization & Adjoints" begin
        # Test BlockOuter constructor and workspace allocation
        L = BlockOuter(K)
        @test L isa AbstractBlockTensor{Float64}
        @test size(L.workspace) == (3, 3) # max(sizes) x max(sizes)
        @test size(L) == (13, 13) # sum(2^2 + 3^2) = 4 + 9 = 13

        # Test the ⊙ alias
        L_alias = K ⊙ K
        @test L_alias isa BlockOuter

        # Test Adjoints
        L_adj = L'
        @test L_adj isa AdjointBlockOuter
        @test L_adj' === L # Double adjoint should return the exact original object

        # Test CovFwdTensor
        C = CovFwdTensor(K)
        @test C' isa AdjointCovFwdTensor
    end

    @testset "2. Mathematical Accuracy (In-Place Multiplication)" begin
        L = BlockOuter(K)

        # Run the highly optimized, in-place multiplication
        mul!(B, L, A)

        # Compute the naive, highly-allocating algebraic equivalent for Block 1
        K_11 = K[Block(1, 1)]
        K_12 = K[Block(1, 2)]
        naive_B1 = K_11 * A.blocks[1] * K_11' + K_12 * A.blocks[2] * K_12'

        # Compute the naive equivalent for Block 2
        K_21 = K[Block(2, 1)]
        K_22 = K[Block(2, 2)]
        naive_B2 = K_21 * A.blocks[1] * K_21' + K_22 * A.blocks[2] * K_22'

        # Assert strict mathematical equivalence
        @test B.blocks[1] ≈ naive_B1
        @test B.blocks[2] ≈ naive_B2
    end

    @testset "3. CovFwdTensor Math (Diagonal Elimination)" begin
        C_op = CovFwdTensor(K)
        A_cov = rand_block_diag(sizes)
        B_cov = zero_block_diag(sizes)

        mul!(B_cov, C_op, A_cov)

        # Output blocks have 0.0 on their diagonals
        @test B_cov.blocks[1][1, 1] == 0.0
        @test B_cov.blocks[1][2, 2] == 0.0
        @test B_cov.blocks[2][3, 3] == 0.0
    end

    @testset "4. Zero Heap Allocation Guarantee" begin
        L = BlockOuter(K)
        A_test = rand_block_diag(sizes)
        B_test = zero_block_diag(sizes)

        # Warm-up call to compile the mul! and sandwich! functions
        mul!(B_test, L, A_test)

        # The actual allocation test
        function test_tensor_allocs(out, op, in)
            return @allocated mul!(out, op, in)
        end

        # Memory-safe
        @test test_tensor_allocs(B_test, L, A_test) == 0
    end

end