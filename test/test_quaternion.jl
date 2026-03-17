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