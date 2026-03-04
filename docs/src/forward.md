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


## Example

This example demonstrates the 3D X-ray transform applied to a 100x100x100 random Shepp-Logan phantom across 60 projection regular angles. To run this example:

1.  Navigate to the `examples/` directory in your terminal.

2.  Launch Julia and activate the local environment:
```sh
julia --project=.
```

3.  From the Julia REPL, run the script:
```julia
include("test_fwd.jl")
```

This will generate the interactive 3D visualization and save the output image `forward_simulation.png` to your assets folder.
Here is the final layout showing the true 3D phantom density contours, the center sinogram, and the 2D integrated intensity shadows at 0° and 90°.

![3D Forward Simulation](assets/forward_simulation.png)
