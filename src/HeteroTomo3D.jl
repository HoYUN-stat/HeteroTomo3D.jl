module HeteroTomo3D

using LinearAlgebra
using SpecialFunctions
using Random


include("quaternion.jl")
include("tomokernel.jl")
include("phantom.jl")
include("xray.jl")

export UnitQuaternion, projection_axis, planar_rotation, shortest_arc
export backproject, inner_product
export rand_shepp_logan_3d, TruncationType, HyperbolicTangent, ArcTangent, Gudermannian
export xray_transform, trilinear_interp


end