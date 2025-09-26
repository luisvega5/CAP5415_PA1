"""
    ======================================================================
                    Padding and Convolution Helpers Module
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
    Handles border conditions with reflecting padding, ensuring
    accurate convolution results near image edges. Provides dot_product
    to apply kernels and convolution_correlation for flexible filtering
    (both convolution and correlation). Includes separable filter support
    (get_image_subcomponents_by_axis) for efficient
    Gaussian + derivative application.
"""
"""---------------------------------------------Imports and Globals--------------------------------------------------"""
import numpy as np
"""--------------------------------------------------Functions-------------------------------------------------------"""
'''
    Function Name: apply_reflecting_padding
    Description: Pads a 2D image by reflecting pixels at the borders. For axis="x"
                 pads left/right; for axis="y" pads top/bottom. Validates that image
                 dimensions exceed the radius for the chosen axis.
    Input: I (np.ndarray) - HxW image, radius (int), axis (str: "x" or "y")
    Output: P (np.ndarray) - padded image
'''
def apply_reflecting_padding(I: np.ndarray, radius: int, axis: str) -> np.ndarray:
    image_height, image_width = I.shape
    if axis == "x" and image_width <= radius:
        raise ValueError(f"Image width ({image_width}) must be > radius ({radius}).")
    if axis == "y" and image_height <= radius:
        raise ValueError(f"Image height ({image_height}) must be > radius ({radius}).")

    if axis == "x":
        P = np.empty((image_height, image_width + 2*radius), dtype=I.dtype)
        # left reflect
        P[:, :radius] = I[:, 1:radius+1][:, ::-1]
        # center
        P[:, radius:radius+image_width] = I
        # right reflect
        P[:, radius+image_width:] = I[:, -radius-1:-1][:, ::-1]
        return P
    elif axis == "y":
        P = np.empty((image_height + 2*radius, image_width), dtype=I.dtype)
        # top reflect
        P[:radius, :] = I[1:radius+1, :][::-1, :]
        # center
        P[radius:radius+image_height, :] = I
        # bottom reflect
        P[radius+image_height:, :] = I[-radius-1:-1, :][::-1, :]
        return P
    else:
        raise ValueError("Applying Reflecting Padding Error: Axis must be 'x' or 'y'")
'''
    Function Name: dot_product
    Description: Performs the inner product between a sliding window of the padded
                 image and a 1D kernel for each output location. Operates along x
                 (columns) or y (rows) depending on 'axis'.
    Input: height (int), width (int), kernel_flipped (np.ndarray),
           padded_image (np.ndarray), axis (str: "x" or "y")
    Output: output (np.ndarray) - HxW filtered image
'''
def dot_product(height: int, width: int, kernel_flipped: np.ndarray, padded_image: np.ndarray, axis:str) -> np.ndarray:
    if axis not in ("x", "y"):
        raise ValueError("Dot Product Error: Axis must be 'x' or 'y'")
    output = np.zeros((height, width), dtype=np.float64)
    kernel_flipped_length = len(kernel_flipped)
    for y in range(height):
        for x in range(width):
            total_sum = 0.0
            for t in range(kernel_flipped_length):
                py, px = (y, x + t) if axis == "x" else (y + t, x)
                total_sum += float(padded_image[py, px]) * float(kernel_flipped[t])
            output[y, x] = total_sum
    return output

'''
    Function Name: convolution_correlation
    Description: Applies 1D convolution or correlation to image I along a single axis.
                 If convolution=True, the kernel is flipped (true convolution);
                 otherwise correlation is used. Uses reflect padding and dot_product.
    Input: I (np.ndarray), gaussian_kernel (np.ndarray), axis (str: "x" or "y"),
           convolution (bool)
    Output: output (np.ndarray) - filtered image along the chosen axis
'''
def convolution_correlation(I: np.ndarray, gaussian_kernel: np.ndarray, axis: str, convolution: bool) -> np.ndarray:
    if axis not in ("x","y"):
        raise ValueError("Convolution/Correlation: Axis must be 'x' or 'y'")
    if convolution:
        gaussian_kernel = gaussian_kernel[::-1]

    radius = len(gaussian_kernel) // 2
    image_height, image_width = I.shape

    # Convolution in x or y depending on axis, which computes the dot product by column if axis is x
    # or computes the dot product by row if axis is y
    padded_image = apply_reflecting_padding(I, radius, axis)
    output = dot_product(image_height, image_width, gaussian_kernel, padded_image, axis)
    return output
'''
    Function Name: get_image_subcomponents_by_axis
    Description: Computes a separable filter response by first applying a horizontal
                 pass (convolution with gaussian_kernel_x) and then a vertical pass
                 (correlation with gaussian_kernel_y). Used to produce I'x or I'y
                 depending on the provided kernels (dG with G or G with dG).
    Input: I (np.ndarray), gaussian_kernel_x (np.ndarray), gaussian_kernel_y (np.ndarray)
    Output: filtered (np.ndarray) - result after the two 1D passes
'''
def get_image_subcomponents_by_axis(I: np.ndarray, gaussian_kernel_x: np.ndarray, gaussian_kernel_y: np.ndarray) -> np.ndarray:
    convolution_result = convolution_correlation(I, gaussian_kernel_x, axis="x", convolution=True)
    correlation_result = convolution_correlation(convolution_result, gaussian_kernel_y, axis="y", convolution=False)
    return correlation_result
