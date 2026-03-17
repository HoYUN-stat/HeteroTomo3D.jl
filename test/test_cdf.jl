@testset "cdf.jl (CDF and Error Functions)" begin
    #1. Univariate CDF and Antiderivatives" 
    @test unicdf(0.0) == 0.5
    @test antid_erf(0.0) == 1.0
    @test antid_erf(0.5) == antid_erf(-0.5)

    #2. Affine Error Function (affine_erf)
    w, c, γ = 0.5, 0.1, 10.0

    @test affine_erf(w, c, γ) > 0.0
    @test affine_erf(0.0, c, γ) == 0.0
    @test affine_erf(w, c, γ) == affine_erf(w, -c, γ)

    #3. Bivariate Normal CDF Approximation (bvncdf)
    p, q = 0.5, 0.5

    @test bvncdf(p, q, 0.0) ≈ unicdf(p) * unicdf(q)
    @test 0.0 <= bvncdf(p, q, 0.5) <= 1.0
    @test 0.0 <= bvncdf(p, q, -0.5) <= 1.0
    @test bvncdf(0.2, 0.8, 0.3) == bvncdf(0.8, 0.2, 0.3)

    # Verify the @assert guardrails block invalid correlations
    @test_throws AssertionError bvncdf(p, q, 1.0)
    @test_throws AssertionError bvncdf(p, q, -1.5)
end