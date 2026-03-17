```@meta
CurrentModule = HeteroTomo3D
```

# Core Data Types
This section covers the fundamental data structures used for 3D tomography reconstruction, outlining the distinct representations needed for phantoms, grids, and the mathematical coefficients used in both mean and covariance estimations.

## 3D Phantom
For a 3D phantom consisting of several Gaussian blobs, we use a specialized wrapper:

```@docs
KernelPhantom3D
rand_center_grid
```

```@example phantom_generation
using HeteroTomo3D, BlockArrays, LinearAlgebra, Random

centers = [
    (0.3, 0.3, 0.3),
    (-0.3, -0.3, 0.3),
    (-0.4, 0.4, -0.4),
    (0.3, -0.3, -0.3)
]
weights = reshape([1.0, 0.8, 0.6, 0.4], 4, 1)
gammas = [5.0, 4.0, 6.0, 4.0]

phantom = KernelPhantom3D(weights, centers, gammas)
```

## Observational Grid
The collection of 2D evaluation points and viewing angles are represented by arrays of type `NTuples{2, I}` and `UnitQuaternion{T}`, respectively:
```@docs
EvaluationGrid
grid_to_real
QuaternionGrid
```


To generate random grids and perform the X-ray transform:
```@docs
rand_evaluation_grid
rand_quaternion_grid
xray_transform
```

## Mean Estimation
In mean estimation, we solve for a single global function. The Tikhonov solution of the representer theorem is defined by a standard vector of coefficients.

The continuous function is expanded as:
```math
\hat{f} = \sum_{i=1}^{n} \sum_{j=1}^{r} \sum_{k=1}^{s} \mathbf{a}_{ijk} \varphi_{\gamma} (\mathbf{R}_{\mathbf{q}_{ij}}, \mathbf{x}_{ijk})
```

Here, the coefficient is conceptualized as a vector of length `s * r * n` and is stored using column-major indexing at `[k + (j-1) * s + (i-1) * s * r]`.

## Covariance Estimation
Unlike mean estimation, which outputs a single volume, covariance estimation models the spatial relationship between points. This requires a much larger tensor-based expansion.

The covariance solution is spanned as follows:
```math
\hat{f} = \sum_{i=1}^{n} \sum_{j, j'=1}^{r} \sum_{k, k'=1}^{s} \mathbf{A}_{i,jk, j'k'} \varphi_{ijk} \otimes \varphi_{ij'k'},
```
where ``\varphi_{ijk} = \varphi_{\gamma} (\mathbf{R}_{\mathbf{q}_{ij}}, \mathbf{x}_{ijk})``.

### Coefficient Data Types
Because the covariance coefficients form a block-diagonal tensor rather than a simple vector, we provide custom types to manage them:

```@docs
BlockDiagonal
block_outer
⊙
blocksizes
```

These custom types are fully compatible with `Krylov.jl` (implementing methods like `kdot, knorm`, etc). For in-place updates during Krylov iterations, use:
```@docs
zero_block_diag
undef_block_diag
rand_block_diag
```

### Lazy Tensor Operations
To avoid heap memory allocations when working with the Khatri-Rao product of the mean Gram matrix, we wrap these operations in a lazy fashion. This ensures space complexity remains optimized during continuous solver iterations:

```@docs
AbstractBlockTensor
BlockOuter
AdjointBlockOuter
CovFwdTensor
AdjointCovFwdTensor
```

### Functional PCA
Once the covariance coefficients are found, you can extract the primary modes of variation (eigenfunctions) using the Conjugate Lanczos Algorithm:

```@docs
conj_lanczos
fpca
```