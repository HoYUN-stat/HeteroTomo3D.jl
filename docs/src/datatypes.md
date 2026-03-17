```@meta
CurrentModule = HeteroTomo3D
```

# Mean Estimation
This section covers the data type used for 3D tomography reconstruction.

## 3D Phantom
For 3D phantom consist of several gaussian blobs, we use a wrapper:
```@docs
KernelPhantom3D
```

```@example phantom_generation
using HeteroTomo3D, BlockArrays, LinearAlgebra, Random

centers = [
    (0.3, 0.3, 0.3),
    (-0.3, -0.3, 0.3),
    (-0.4, 0.4, -0.4),
    (0.3, -0.3, -0.3)
]
weights = reshape([1.0, 0.8, 0.6, 0.4], L, n)
gammas = [5.0, 4.0, 6.0, 4.0]

phantom = KernelPhantom3D(weights, centers, gammas)
```

## Observational Grid
The collection of 2D evaluation points and viewing angles are represented by an arry of `NTuples{2, I}` and `UnitQuaternion{T}`:
```@docs
EvaluationGrid
grid_to_real
QuaternionGrid
```


To generate random grids:
```@docs
rand_evaluation_grid
rand_quaternion_grid
```

## Estimation Coefficents
The Tikhonov solution of the representer theorem for mean estimation is represented by a vector:
```math
\hat{f} = \sum_{i=1}^{n} \sum_{j=1}^{r} \sum_{k=1}^{s} \mathbf{a}_{ijk} \varphi_{\gamma} (\mathbf{R}_{\mathbf{q}_{ij}}, \mathbf{x}_{ijk})
```
This coefficient of size `s * r * n` is stored in the column-major indexing `[k + (j-1) * s + (i-1) * s * r]`.

For covariance estimation, the solution is spanned as follow:
```math
\hat{f} = \sum_{i=1}^{n} \sum_{j, j'=1}^{r} \sum_{k, k'=1}^{s} \mathbf{A}_{i,jk, j'k'} \varphi_{ijk} \otimes \varphi_{ij'k'}, \quad \text{where} \varphi_{ijk} = \varphi_{\gamma} (\mathbf{R}_{\mathbf{q}_{ij}}, \mathbf{x}_{ijk}).
```

Hence, the data type for this covariance coeeficients:
```@docs
BlockDiagonal
block_outer
⊙
blocksizes
```
This custom type is compatible to create custom workspaces with `Krylov.jl`, such as `kdot, knorm`, etc. For in-place updates Kylov iterations, we can construct this data type:
```@docs
zero_block_diag
undef_block_diag
rand_block_diag
```