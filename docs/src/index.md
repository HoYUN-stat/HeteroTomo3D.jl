```@meta
CurrentModule = HeteroTomo3D
```

# HeteroTomo3D.jl

Welcome to the documentation for [HeteroTomo3D](https://github.com/HoYUN-stat/HeteroTomo3D.jl).

This package provides nonparametric estimation of conformational variability from 3D tomographic data using tensorized Krylov methods in a Reproducing Kernel Hilbert Space (RKHS).


## 🚀 Key Features

* **RKHS Framework**: Employs a functional estimation framework tailored for statistical inverse problems.
* **3D Heterogeneity**: Facilitates the estimation of mean and covariance structures for non-identical 3D objects.
* **Matrix-Free Operators**: Implicitly implements Khatri-Rao products, minimizing memory overhead for covariance estimation.
* **Singularity-Free Geometry**: Utilizes `UnitQuaternion` to ensure robust 3D rotations without gimbal lock.

## 📚 References

If you use this package in your research, please cite the following papers:

* Yun, H., & Panaretos, V. M. (2025). Computerized Tomography and Reproducing Kernels. *SIAM Review*, 67(2), 321–350. [https://doi.org/10.1137/23M1616716](https://doi.org/10.1137/23M1616716)
* Yun, H., Caponera, A., & Panaretos, V. M. (2025). Low-Dose Tomography of Random Fields and the Problem of Continuous Heterogeneity. *arXiv preprint arXiv:2507.10220*. [https://doi.org/10.48550/arXiv.2507.10220](https://doi.org/10.48550/arXiv.2507.10220)
* Yun, H., & Panaretos, V. M. (2026). Fast and Cheap Covariance Smoothing. *Journal of Computational and Graphical Statistics*. [https://doi.org/10.1080/10618600.2026.2615054](https://doi.org/10.1080/10618600.2026.2615054)
