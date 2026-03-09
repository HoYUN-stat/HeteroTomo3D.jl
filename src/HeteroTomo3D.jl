module HeteroTomo3D

using LinearAlgebra
using SpecialFunctions
using Random
using BlockArrays
using Krylov

include("quaternion.jl")
# include("type.jl")
# include("meanrepre.jl")
# include("meanrecons.jl")
# include("blocktensor.jl")
# include("covrepre.jl")
# include("blockfpca.jl")
# include("phantom.jl")
# include("xray.jl")

export UnitQuaternion, projection_axis, planar_rotation, shortest_arc, rotate
# export backproject, inner_product
# export rand_shepp_logan_3d, TruncationType, HyperbolicTangent, ArcTangent, Gudermannian
# export xray_transform, trilinear_interp
# export EvaluationGrid, QuaternionGrid, BlockDiag, LazyKhatri
# export build_mean_gram!, solve_mean!
# export reconstruct_mean

end