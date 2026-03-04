```@meta
CurrentModule = HeteroTomo3D
```

# Mean Estimation
This section covers the estimation of the 3D mean function using the RKHS representer theorem and direct linear solvers.

## Data Structures
```@docs
EvaluationGrid
QuaternionGrid
BlockDiag
LazyKhatri
```

## Representer Theorem Solver
The mean function is estimated by solving the system ``(\\mathbf{K} + \\lambda \\mathbf{I}) \\mathbf{a} = \\mathbf{y}``.

```@docs
build_mean_gram!
solve_mean!
```

## 3D Reconstruction
Once the coefficients ``\\mathbf{a}`` are found, the continuous 3D volume is reconstructed via the evaluation tensor action.

```@docs
reconstruct_mean
```
