"""
    calculate_attenuation(intensity, initial_intensity)

Calculates the basic X-ray attenuation given the initial and final intensities.
"""
function calculate_attenuation(intensity, initial_intensity)
    return -log(intensity / initial_intensity) + 1000
end


"""
    projection_norm(matrix)

Calculates the Frobenius norm of a projection matrix using LinearAlgebra.
"""
function projection_norm(matrix)
    return norm(matrix)
end