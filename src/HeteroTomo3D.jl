module HeteroTomo3D

# Exporting makes the function available when someone types `using HeteroTomo3D`
export dummy_volume

"""
    dummy_volume(width, height, depth)

Calculates the volume of a simple 3D bounding box.
"""
function dummy_volume(width, height, depth)
    return width * height * depth
end

end