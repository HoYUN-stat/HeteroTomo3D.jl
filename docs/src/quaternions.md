```@meta
CurrentModule = HeteroTomo3D
```

# 3D Rotations (Quaternions)

This section provides the rotation geometry for the tomographic pipeline using quaternions robust to Gimbal-lock.

```@docs
UnitQuaternion
Base.conj(::UnitQuaternion)
Base.inv(::UnitQuaternion)
Base.:*(::UnitQuaternion, ::UnitQuaternion)
Base.:abs(::UnitQuaternion)
Random.rand(::Random.AbstractRNG, ::Random.SamplerType{UnitQuaternion})
rotate
projection_axis
shortest_arc
planar_rotation
```

