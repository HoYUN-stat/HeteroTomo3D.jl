# HeteroTomo3D

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://HoYUN-stat.github.io/HeteroTomo3D.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://HoYUN-stat.github.io/HeteroTomo3D.jl/dev/)
[![Build Status](https://github.com/HoYUN-stat/HeteroTomo3D.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HoYUN-stat/HeteroTomo3D.jl/actions/workflows/CI.yml?query=branch%3Amain)


**HeteroTomo3D.jl** is a Julia package designed for 3D heterogeneous tomographic reconstruction using Reproducing Kernel Hilbert Space (RKHS) methods. 
Built for high-performance via advanced numerical linear algebra techniques, it allows for large-scale estimation by leveraging zero-heap allocation architectures, strict use of `@inbounds` and `@simd`, and multithreaded operations to heavily optimize space complexity.

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

## 📖 Documentation

For full documentation, function references, and examples, visit the [development documentation](https://hoyun-stat.github.io/HeteroTomo3D.jl/dev/).

## ⚙️ Installation

You can install the package directly from GitHub using the Julia REPL:

```julia
julia> ]
pkg> add(url="https://github.com/HoYUN-stat/HeteroTomo3D.jl")
```
