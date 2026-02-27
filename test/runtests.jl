using HeteroTomo3D
using Test

@testset "HeteroTomo3D.jl" begin
    # Test our dummy volume function
    @test dummy_volume(2, 3, 4) == 24

    # You can add more tests here later
    @test dummy_volume(1, 1, 1) == 1
end