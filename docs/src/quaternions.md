# 3D Rotations (Quaternions)

This section provides the singularity-free rotation geometry for the tomographic pipeline.

```@docs
UnitQuaternion
Base.:*(::UnitQuaternion, ::UnitQuaternion)
projection_axis
planar_rotation
shortest_arc