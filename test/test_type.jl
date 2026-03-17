@testset "type.jl (Data Types)" begin
    # Setup
    s, r, n, m = 2, 3, 4, 5

    # 2. Test EvaluationGrid
    X = rand_evaluation_grid(s, r, n, m; seed=42)
    @test size(X) == (s, r, n)
    @test eltype(X) == Tuple{Int64,Int64}

    # Verify bounding box constraints (all points must be within 1:m)
    @test all(1 <= pt[1] <= m && 1 <= pt[2] <= m for pt in X.blocks)

    # 3. Test QuaternionGrid
    Q = rand_quaternion_grid(r, n; seed=42)
    @test size(Q) == (r, n)
    @test eltype(Q) == UnitQuaternion{Float64}

    # 4. Test BlockDiagonal
    block_sizes = [2, 3]
    D = zero_block_diag(Float64, block_sizes)
    @test length(D.blocks) == 2
    @test size(D.blocks[1]) == (2, 2)
    @test all(D.blocks[1] .== 0.0)

    #5. Test block_outer (⊙)
    y = BlockVector([1.0, 2.0, 3.0], [2, 1])
    outer_mat = block_outer(y)
    @test outer_mat isa BlockDiagonal
    @test size(outer_mat.blocks[1]) == (2, 2)
    @test outer_mat.blocks[1][1, 2] == 2.0 # 1.0 * 2.0

    #6. Krylov Compatibility
    sizes = [2, 2]
    # Create two identical BlockDiagonals filled with 1.0
    D1 = BlockDiagonal([fill(1.0, r, r) for r in sizes])
    D2 = BlockDiagonal([fill(1.0, r, r) for r in sizes])

    # Test knorm: length is 8 (two 2x2 blocks). sqrt(8 * 1.0^2) = sqrt(8)
    @test Krylov.knorm(1, D1) ≈ sqrt(8.0)

    # Test kdot: dot product of two vectors of ones of length 8 is 8.0
    @test Krylov.kdot(1, D1, D2) == 8.0

    # Test inplace scaling
    Krylov.kscal!(1, 5.0, D1)
    @test D1.blocks[1][1, 1] == 5.0
end