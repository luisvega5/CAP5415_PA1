"""
    ======================================================================
            Orientation Quantization & Non-Maximum Suppression Module
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
    quantize_orientation_4: maps continuous gradient orientations into
    four canonical bins (0°, 45°, 90°, 135°).
    non_maximum_suppression_4dir: thins edges by keeping only local
    maxima along gradient directions. Ensures precision in edge
    localization by suppressing weaker, redundant pixels.
"""
"""---------------------------------------------Imports and Globals--------------------------------------------------"""
import numpy as np
"""--------------------------------------------------Functions-------------------------------------------------------"""
'''
    Function Name: quantize_orientation_4
    Description: Quantizes continuous orientations into four canonical bins:
                 {0°, 45°, 90°, 135°}. Bin edges are +/-22.5° around each bin center.
    Input: theta_deg (np.ndarray) - orientation in degrees [0, 180)
    Output: theta_q (np.ndarray, uint8) - quantized angles with values {0,45,90,135}
'''
def quantize_orientation_4(theta_deg: np.ndarray) -> np.ndarray:
    theta_q = np.zeros_like(theta_deg, dtype=np.uint8)
    theta_q[(theta_deg >= 22.5)  & (theta_deg < 67.5)]   = 45
    theta_q[(theta_deg >= 67.5)  & (theta_deg < 112.5)]  = 90
    theta_q[(theta_deg >= 112.5) & (theta_deg < 157.5)]  = 135
    return theta_q

'''
    Function Name: non_maximum_suppression_4dir
    Description: Performs non-maximum suppression using 4-direction quantized
                 orientations. For each interior pixel, compares the magnitude to
                 its two neighbors along the assigned direction and keeps the pixel
                 only if it is >= both neighbors (ties kept).
    Input: M (np.ndarray) - gradient magnitude, theta_q (np.ndarray) - quantized angles
    Output: output (np.ndarray) - thinned magnitude image (zeros where suppressed)
'''
def non_maximum_suppression_4dir(M: np.ndarray, theta_q: np.ndarray) -> np.ndarray:
    m_height, m_width = M.shape
    output = np.zeros_like(M, dtype=M.dtype)
    for y in range(1, m_height-1):
        for x in range(1, m_width-1):
            m0 = M[y, x]
            t  = theta_q[y, x]
            # Left and right pixels comparison
            if t == 0:
                m1 = M[y, x-1]; m2 = M[y, x+1]
            # Northeast and southwest pixel comparison
            elif t == 45:
                m1 = M[y-1, x+1]; m2 = M[y+1, x-1]
            # Up and down pixel comparison
            elif t == 90:
                m1 = M[y-1, x]; m2 = M[y+1, x]
            # Northwest and southeast pixel comparison
            else:
                m1 = M[y-1, x-1]; m2 = M[y+1, x+1]
            # Keeps the center if it's a tie else 0 (suppressed)
            if m0 >= m1 and m0 >= m2:
                output[y, x] = m0
    return output