module HeteroTomo3D

using LinearAlgebra
using SpecialFunctions

# Include the core files
include("quaternion.jl")
include("tomokernel.jl")

export UnitQuaternion, projection_axis, planar_rotation, shortest_arc
export backproject, inner_product

end