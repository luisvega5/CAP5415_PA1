"""
    ======================================================================
                               Canny Edge Detection
    ======================================================================
    Name:          main.py
    Author:        Luis Obed Vega Maisonet
    University:    University of Central Florida (UCF)
    Program:       M.S. in Systems Engineering
    Course:        CAP 5415 - Computer Vision
    Academic Year: 2025
    Semester:      Fall
    Section:       0V62
    Professor:     Dr. Yogesh Singh Rawat

    Description:
        Implementation of the Canny Edge Detection algorithm as part of the
        Programming Assignment (PA1). This program sequentially performs:
            - Gaussian smoothing
            - Gradient computation (Ix, Iy)
            - Gradient magnitude and orientation
            - Non-maximum suppression (NMS)
            - Hysteresis thresholding
        The final output is a binarized edge map along with intermediate
        results (Gaussian smoothing, derivative responses, magnitude, and NMS).
"""
"""---------------------------------------------Imports and Globals--------------------------------------------------"""
import matplotlib.pyplot as plt
import GaussiansKernels as gkernel
import GradientComputation as gcomp
import Hysteresis as hys
import NMS as nms
import PaddingConvolutionHelpers as helper
import Utilities as utils
"""-----------------------------------------------Main Function------------------------------------------------------"""
'''
    Function Name: main
    Description: Primary function that runs Canny Edge Detection pipeline end-to-end.
                 1) Reads inputs and parameters.
                 2) Builds Gaussian and derivative kernels.
                 3) Computes I'x and I'y via separable filtering.
                 4) Computes gradient magnitude and orientation; quantizes orientation.
                 5) Runs 4-direction non-maximum suppression.
                 6) Applies hysteresis to obtain the final edge map.
                 7) Saves intermediate and final outputs to disk.
    Input: None
    Output: None
'''
def main():
    sigma, radius, low, high, I = utils.read_input()

    # Gaussian function calculation
    G = gkernel.get_gaussian_kernel(sigma, radius)

    # Gaussian derivative function calculation
    dG = gkernel.get_gaussian_derivative(sigma, radius)

    # Convolution calculations of I * G(x) and I * G(y) respectively
    Igx = helper.convolution_correlation(I, G, axis="x", convolution=True)
    Igy = helper.convolution_correlation(I, G, axis="y", convolution=True)

    # Matrix I subcomponent function calculation
    # Includes the convolution_correlation function call
    Ix = helper.get_image_subcomponents_by_axis(I, dG, G)
    Iy = helper.get_image_subcomponents_by_axis(I, G, dG)

    # Gradient magnitude function calculation M
    M = gcomp.gradient_magnitude(Ix, Iy)

    # Gradient orientation function calculation theta
    theta = gcomp.gradient_orientation(Ix, Iy)

    # Quantization orientation function calculation
    # It assigns a respective angle to each gradient orientation
    theta_q = nms.quantize_orientation_4(theta)

    # Non-Maximum suppression function calculation
    non_maximum_suppression = nms.non_maximum_suppression_4dir(M, theta_q)

    # Hysteresis Threshold function calculation
    canny_result = hys.hysteresis(non_maximum_suppression, low, high, relative=True)

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1,1,1], width_ratios=[1,1,1], hspace=0.15, wspace=0.05)

    # Plotting image (a) Gaussian along x
    plot_a = fig.add_subplot(gs[0, 0]); utils.prep_image(plot_a, Igx, "(a) Gaussian in x")

    # Plotting image (b) Gaussian along y
    plot_b = fig.add_subplot(gs[0, 1]); utils.prep_image(plot_b, Igy, "(b) Gaussian in y")

    # Plotting image (c) derivative along x (Ix)
    plot_c = fig.add_subplot(gs[0, 2]); utils.prep_image(plot_c, Ix,  "(c) Gaussian Derivative x (Ix')")

    # Plotting image (d) derivative along y (Iy)
    plot_d = fig.add_subplot(gs[1, 0]); utils.prep_image(plot_d, Iy,  "(d) Gaussian Derivative y (Iy')")

    # Plotting image (e) gradient magnitude
    plot_e = fig.add_subplot(gs[1, 1]); utils.prep_image(plot_e, M,   "(e) Gradient Magnitude")

    # Plotting image (f) NMS result
    plot_f = fig.add_subplot(gs[1, 2]); utils.prep_image(plot_f, non_maximum_suppression, "(f) NMS")

    # Plotting image (g) final hysteresis edges (optional output)
    plot_g = fig.add_subplot(gs[2, :]); utils.prep_image(plot_g, canny_result, "(g) Hysteresis")

    # Plot image layout
    plt.show()

# Runs the main function at the start of execution
if __name__ == "__main__":
    main()