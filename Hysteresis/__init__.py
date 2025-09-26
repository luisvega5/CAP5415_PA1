"""
    ======================================================================
                        Hysteresis Thresholding Module
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
    Implements double-threshold edge linking. Classifies edges as strong,
    weak, or suppressed. Uses OpenCV’s connectedComponents to ensure
    weak edges are retained only if connected to strong ones. Produces
    a clean binary edge map ({0,255}).
"""
"""---------------------------------------------Imports and Globals--------------------------------------------------"""
import cv2
import numpy as np
"""--------------------------------------------------Functions-------------------------------------------------------"""
'''
    Function Name: hysteresis
    Description: Applies double-threshold hysteresis to the NMS result. Pixels above
                 'high' are strong; pixels between 'low' and 'high' are weak. Keeps
                 all strong pixels and any weak pixel connected to a strong pixel
                 using the specified connectivity (4 or 8).
                 If relative=True, thresholds are fractions of max(NMS).
    Input: nms (np.ndarray), low (float), high (float),
           relative (bool, default=True), connectivity (int, default=8)
    Output: edges (np.ndarray, uint8) - binary edge map {0,255}
'''
def hysteresis(nms: np.ndarray, low: float, high: float, relative: bool = True, connectivity: int = 8) -> np.ndarray:
    # Convert to float 64 for precision
    nms = nms.astype(np.float64, copy=False)

    # Finding the global max gradient
    m_max = float(nms.max())

    # If the image has no edges return black image
    if m_max <= 0.0:
        return np.zeros_like(nms, dtype=np.uint8)

    # Threshold definition
    # If the relative flag is set to True then the t_low and t_high variables are
    # a factor of the maximum gradient and otherwise t_low and t_high are set by
    # as inputs
    if relative:
        t_low, t_high = low * m_max, high * m_max
    else:
        t_low, t_high = float(low), float(high)
    if t_low > t_high:
        raise ValueError("Hysteresis Threshold Error: 'low' variable must be <= 'high' variable.")

    # Candidate mask (weak or strong) and strong mask
    weak_or_strong = (nms >= t_low)
    strong = (nms >= t_high)

    # Label connected components among candidates
    # The function connectedComponents expects an uint8 image
    num_labels, labels = cv2.connectedComponents(weak_or_strong.astype(np.uint8), connectivity=connectivity)

    # Only retain/keep components that contain at least one strong pixel
    strong_labels = np.unique(labels[strong])
    keep = np.isin(labels, strong_labels)

    # Finally the edges retain will be the strong edges and the week_or_strong
    # edges connected to strong edges
    edges = (keep & weak_or_strong)
    return edges.astype(np.uint8) * 255