using HeteroTomo3D
using Test

@testset "HeteroTomo3D.jl" begin
    # Existing volume tests
    @test dummy_volume(2, 3, 4) == 24
    @test dummy_volume(1, 1, 1) == 1

    # New math tests
    @testset "Math Functions" begin
        my_matrix = [1.0 2.0; 3.0 4.0]

        # Use \approx + Tab to type the ≈ symbol!
        @test projection_norm(my_matrix) ≈ 5.477225575051661
    end
end