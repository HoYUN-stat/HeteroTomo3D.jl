```@meta
CurrentModule = HeteroTomo3D
```

# Forward Operations

This section covers the generation of 3D phantoms and the simulation of the 3D X-ray transform via ray casting.

## 3D Phantom Generation
```@docs
rand_shepp_logan_3d
TruncationType
HeteroTomo3D.truncation
```

## X-ray Transform
```@docs
xray_transform
HeteroTomo3D.trilinear_interp
```


## Simulation
Below is a demonstration of the 3D X-ray transform applied to a 100x100x100 random Shepp-Logan phantom across 60 projection regular angles.
