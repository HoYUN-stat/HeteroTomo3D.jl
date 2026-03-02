# 3D Rotations (Quaternions)

This section provides the rotation geometry for the tomographic pipeline using quaternions robust to Gimbal-lock.

```@docs
UnitQuaternion
Base.:*(::UnitQuaternion, ::UnitQuaternion)
projection_axis
planar_rotation
shortest_arc
```