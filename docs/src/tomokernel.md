```@meta
CurrentModule = HeteroTomo3D
```

# Tomographic Kernels

This section contains the analytical expressions for the 3D X-ray transform and inner products in the RKHS. 


## Special Functions
The tomographic feature maps and their inner products involve definite integrals of Gaussian density functions. Hence, we define:
```@docs
antid_erf
affine_erf
bvncdf
```

## Tomographic RKHS
```@docs
backproject
inner_product
build_mean_gram!
```

The calculation of the inner product between two tomographic feature maps depends on the collinearity of two viewing axes. This collinearity can be quantifies as
```math
\rho = \mathbf{e}_{3}^{\top} \mathbf{R}_{\mahtbf{q}_{2}} \mathbf{R}_{\mahtbf{q}_{1}}^{\top} \mathbf{e}_{3} \in [-1, +1]
```
For more details, see:
```@docs
collinear_inner_product
noncollinear_inner_product
```

