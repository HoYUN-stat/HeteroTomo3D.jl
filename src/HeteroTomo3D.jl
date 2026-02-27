module HeteroTomo3D

using LinearAlgebra  # <-- Add this here!

# Include sub-files
include("math.jl")

# 2. Export the functions you want users to have access to
export dummy_volume          # From our previous step
export calculate_attenuation # From our new math.jl file
export projection_norm


# (You can leave your dummy_volume function definition down here for now)
"""
    dummy_volume(width, height, depth)

Calculates the volume of a simple 3D bounding box.
"""
function dummy_volume(width, height, depth)
    return width * height * depth
end

end