# module HeteroTomo3D

using LinearAlgebra
using SpecialFunctions
using Random
using BlockArrays
using Krylov
import Krylov: FloatOrComplex

include("quaternion.jl")
include("type.jl")
include("tomokernel.jl")
include("cdf.jl")
# include("meanrepre.jl")
# include("meanrecons.jl")
# include("blocktensor.jl")
# include("covrepre.jl")
# include("blockfpca.jl")
# include("phantom.jl")
# include("xray.jl")

export UnitQuaternion, projection_axis, planar_rotation, shortest_arc, rotate
export EvaluationGrid, rand_evaluation_grid, QuaternionGrid, rand_quaternion_grid, BlockDiagonal, zero_block_diag, undef_block_diag, rand_block_diag, block_outer, ⊙, blocksizes
export antid_erf, affine_erf, bvncdf
# export backproject, inner_product
# export rand_shepp_logan_3d, TruncationType, HyperbolicTangent, ArcTangent, Gudermannian
# export xray_transform, trilinear_interp
# export EvaluationGrid, QuaternionGrid, BlockDiag, LazyKhatri
# export build_mean_gram!, solve_mean!
# export reconstruct_mean

# end