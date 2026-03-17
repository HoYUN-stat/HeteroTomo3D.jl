@testset "phantom.jl (3D Phantoms & X-Ray Transform)" begin

    @testset "1. Phantom Initialization & Center Bounds" begin
        L = 5 # Generate a large number of centers to rigorously test the boundary
        centers = HeteroTomo3D.rand_center_grid(L; seed=42)

        @test length(centers) == L

        # Mathematically verify that every center is inside the unit sphere (radius <= 1.0)
        radii_squared = [x^2 + y^2 + z^2 for (x, y, z) in centers]
        @test all(radii_squared .<= 1.0)

        # Verify struct initialization
        n = 2
        weights = rand(L, n)
        gammas = fill(5.0, L)
        phantom = KernelPhantom3D(weights, centers, gammas)

        @test size(phantom.weights) == (L, n)
        @test length(phantom.gammas) == L
    end

    @testset "2. X-Ray Projection" begin
        # Setup a small grid
        s, r, n, m = 4, 3, 2, 10
        L = 3

        X = rand_evaluation_grid(s, r, n, m; seed=123)
        Q = rand_quaternion_grid(r, n; seed=123)

        weights = rand(L, n)
        centers = HeteroTomo3D.rand_center_grid(L; seed=123)
        gammas = fill(10.0, L)
        phantom = KernelPhantom3D(weights, centers, gammas)

        projections = xray_transform(phantom, X, Q)

        @test size(projections) == (s, r, n)
        @test all(projections .>= 0.0)
    end

end