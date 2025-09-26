"""
    ======================================================================
                    Gaussian and Derivative Kernels Module
    ======================================================================
    Name:          __init__.py
    Author:        Luis Obed Vega Maisonet
    University:    University of Central Florida (UCF)
    Program:       M.S. in Systems Engineering
    Course:        CAP 5415 - Computer Vision
    Academic Year: 2025
    Semester:      Fall
    Section:       0V62
    Professor:     Dr. Yogesh Singh Rawat

    Description:
    Implements the mathematical foundation for image smoothing and
    derivative filtering. Functions validate and compute sigma and
    kernel radius. Provides get_gaussian_kernel for Gaussian smoothing
    and get_gaussian_derivative for first-order
    derivative-of-Gaussian filters.
"""
"""---------------------------------------------Imports and Globals--------------------------------------------------"""
import numpy as np
"""--------------------------------------------------Functions-------------------------------------------------------"""
'''
    Function Name: sigma_radius_check
    Description: Validates sigma and resolves the radius. Ensures sigma > 0 and if
                 radius is not provided, computes radius = ceil(3*sigma).
    Input: sigma (float), radius (int | None)
    Output: (sigma, radius) - tuple[float, int] with a concrete radius value
'''
def sigma_radius_check(sigma: float, radius: int | None = None) -> tuple[float, int]:
    if sigma <= 0:
        raise ValueError("Incorrect input: Sigma value must be > 0")

    if radius is None:
        radius = int(np.ceil(3 * sigma))
    return sigma, radius

'''
    Function Name: get_gaussian_kernel
    Description: Builds a 1D Gaussian kernel (row vector) with standard deviation sigma.
                 The kernel is L1-normalized so its coefficients sum to 1.
                 If radius is None, uses ceil(3*sigma) to capture ~99% energy.
    Input: sigma (float), radius (int | None)
    Output: G (np.ndarray) - 1D Gaussian kernel of length (2*radius + 1)
'''

def get_gaussian_kernel(sigma: float, radius: int | None = None) -> np.ndarray:
    sigma, radius = sigma_radius_check(sigma, radius)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    G = np.exp(-(x**2) / (2 * sigma**2))
    G /= G.sum()
    return G

'''
    Function Name: get_gaussian_derivative
    Description: Builds a 1D first-derivative-of-Gaussian (DoG) kernel using the
                 analytical derivative of the Gaussian. Uses the same radius rule
                 as the Gaussian. Scale is not normalized (kept physical).
    Input: sigma (float), radius (int | None)
    Output: dG (np.ndarray) - 1D derivative-of-Gaussian kernel
'''
def get_gaussian_derivative(sigma: float, radius: int | None = None) -> np.ndarray:
    sigma, radius = sigma_radius_check(sigma, radius)
    G = get_gaussian_kernel(sigma, radius)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    dG = -(x / (sigma**2)) * G
    return dG