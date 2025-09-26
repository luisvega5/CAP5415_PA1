"""
    ======================================================================
                               Computation Module
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
    Focuses on computing gradient magnitude and gradient orientation.
    gradient_magnitude: calculates √(Ix² + Iy²).
    gradient_orientation: uses atan2(Iy, Ix) to measure edge direction
    in degrees [0,180). These results drive the non-maximum suppression
    step.
"""
"""---------------------------------------------Imports and Globals--------------------------------------------------"""
import numpy as np
"""--------------------------------------------------Functions-------------------------------------------------------"""
'''
    Function Name: gradient_magnitude
    Description: Computes per-pixel gradient magnitude from I'x and I'y using
                 sqrt(Ix^2 + Iy^2). Ensures float precision for stable results.
    Input: Ix (np.ndarray), Iy (np.ndarray)
    Output: M (np.ndarray) - gradient magnitude image
'''
def gradient_magnitude(Ix: np.ndarray, Iy: np.ndarray) -> np.ndarray:
    Ix = Ix.astype(np.float64, copy=False)
    Iy = Iy.astype(np.float64, copy=False)
    return np.sqrt(Ix*Ix + Iy*Iy)

'''
    Function Name: gradient_orientation
    Description: Computes the gradient orientation (degrees) using atan2(I'y, I'x),
                 then maps angles into [0, 180) for edge direction semantics.
    Input: Ix (np.ndarray), Iy (np.ndarray)
    Output: theta (np.ndarray) - orientation image in degrees, range [0, 180)
'''
def gradient_orientation(Ix: np.ndarray, Iy: np.ndarray) -> np.ndarray:
    theta = np.degrees(np.arctan2(Iy, Ix))   # (-180, 180]
    theta = np.mod(theta, 180.0)             # [0, 180)
    return theta