@testset "reconstruction.jl (Reconstruction Engine)" begin

    @testset "1. Physical Boundaries & Cutoffs" begin
        s, r, n, m = 2, 2, 1, 5
        γ = 5.0
        X = rand_evaluation_grid(s, r, n, m; seed=1)
        Q = rand_quaternion_grid(r, n; seed=1)
        coeffs = rand(s * r * n)

        # Pre-allocate output volume
        F = zeros(m, m, m)
        xray_recons!(F, coeffs, X, Q, γ)

        @test F[1, 1, 1] == 0.0
        @test F[m, m, m] == 0.0
        @test F[m, 1, 1] == 0.0
    end

    @testset "2. Zero-Allocation" begin
        s, r, n, m = 2, 2, 1, 4
        γ = 5.0
        X = rand_evaluation_grid(s, r, n, m; seed=123)
        Q = rand_quaternion_grid(r, n; seed=123)
        coeffs = rand(s * r * n)
        F = zeros(m, m, m)

        # Warm-up call for JIT compilation and threading overhead
        xray_recons!(F, coeffs, X, Q, γ)

        function test_allocs(out, c, grid, quat, g)
            return @allocated xray_recons!(out, c, grid, quat, g)
        end

        allocs = test_allocs(F, coeffs, X, Q, γ)
        @test allocs == 0
    end
end