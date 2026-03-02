```@meta
CurrentModule = HeteroTomo3D
```

# Forward Operations

This section covers the generation of 3D phantoms and the simulation of the 3D X-ray transform via ray casting.

## Phantom Generation
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