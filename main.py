"""
    ======================================================================
                               Canny Edge Detection
    ======================================================================
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
import sys, cv2, numpy as np, matplotlib.pyplot as plt
from typing import Optional
"""--------------------------------------------Input/Utility Functions-----------------------------------------------"""
'''
    Function Name: parse_flag
    Description: Searches a tokenized command line ('user_input') for a flag and
                 returns the value after it, cast using 'caster'. If the flag is
                 absent returns None. Exits if the flag is present but no value is provided.
    Input: user_input (list[str]), flag (str), caster (callable)
    Output: value or None - casted parameter value if present
'''
def parse_flag(user_input: list[str], flag: str, caster):
    try:
        idx = user_input.index(flag)
        return caster(user_input[idx + 1])
    except ValueError:
        return None
    except (IndexError, Exception):
        print(f"Error: Please provide a valid value after {flag}.")
        sys.exit(1)

'''
    Function Name: read_input
    Description: Reads CLI-style input from the user. First token is the image path.
                 Optional flags: -s <sigma>, -r <radius>, and hysteresis -l <low> -h <high>.
                 Validates that either both -l and -h are given or neither. Loads the image.
                 Prints the resolved parameters and returns them alongside the normalized image.
    Input: None
    Output: (sigma, radius, low, high, I) - tuple[float, Optional[int], float, float, np.ndarray]
'''
def read_input() -> tuple[float, Optional[int], float, float, np.ndarray]:
    # Print user prompt so that user knows what are the correct inputs to enter
    print("Welcome to CAP5415 PA1: Canny Edge Detection")
    user_input = input(
        "Enter the image path, and optionally specify parameters:\n"
        "  -s <sigma>          (default = 1.0)\n"
        "  -r <radius>         (default = ceil(3*sigma))\n"
        "  -l <low> -h <high>  (defaults = 0.1, 0.2)\n"
        "Example: images/chessboard.png -s 2.0 -r 7 -l 0.05 -h 0.15\n> "
    ).strip().split()

    # Throw an error in the screen and exit
    if not user_input:
        print("Error: No input provided.")
        sys.exit(1)

    # Read the input image path
    image_path = user_input[0]

    # Definition of input defaults values
    sigma: float = 1.0
    radius: Optional[int] = None
    low, high = 0.1, 0.2

    # Parse flags using the parse_flag function
    # If successful values will be assigned elsewise
    # default values will be retained
    sigma_val = parse_flag(user_input, "-s", float)
    if sigma_val is not None:
        sigma = sigma_val

    radius_val = parse_flag(user_input, "-r", int)
    if radius_val is not None:
        radius = radius_val

    low_val = parse_flag(user_input, "-l", float)
    high_val = parse_flag(user_input, "-h", float)

    if (low_val is None) ^ (high_val is None):
        print("Error: You must specify both -l and -h if you desire to override the hysteresis thresholds.")
        sys.exit(1)

    if low_val is not None and high_val is not None:
        low, high = low_val, high_val

    # Try reading image path, throw error if not successful
    try:
        I, gray_image = read_image(image_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # Print parameters to be used in the Carry algorithm analysis
    print("Parameters in use:")
    print(f"  sigma   = {sigma} (default 1.0 if not specified)")
    print(f"  radius  = {radius if radius is not None else 'auto (ceil(3*sigma))'}")
    print(f"  low     = {low} (default 0.1 if not specified)")
    print(f"  high    = {high} (default 0.2 if not specified)")

    return sigma, radius, low, high, I

'''
    Function Name: read_image
    Description: Loads an image from disk, converts it to grayscale, and returns both
                 a normalized grayscale matrix I in [0,1] (float32) and the uint8
                 grayscale image suitable for saving/inspection.
    Input: path (str) - filesystem path to the image.
    Output: (I, gray_image) - tuple[np.ndarray, np.ndarray]
            I: float32 HxW normalized grayscale
            gray_image: uint8 HxW grayscale
'''
def read_image(path: str) -> tuple[np.ndarray, np.ndarray]:
    # Read image in color
    img = cv2.imread(path)

    # Manage Error if Image is not found throw an exception
    if img is None:
        raise FileNotFoundError(f"The specified image path {path} did not contain a valid image.")

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Return both normalized and raw grayscale
    I = gray_image.astype("float32") / 255.0

    return I, gray_image
'''
    Function Name: to_u8
    Description: Linearly normalizes an array to [0, 255] and converts to uint8.
                 Useful for saving intermediate floating-point images.
    Input: a (np.ndarray) - arbitrary numeric image
    Output: au8 (np.ndarray, uint8) - normalized 8-bit image
'''
def to_u8(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64, copy=False)
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-12:
        return np.zeros_like(a, dtype=np.uint8)
    a = (a - mn) / (mx - mn)
    return np.clip(a * 255.0, 0, 255).astype(np.uint8)
'''
    Function Name: show_gray
    Description: Utility function to visualize an image in grayscale.
                 Normalizes the input array into the range [0,1] for
                 consistent visualization regardless of its numeric scale,
                 applies a grayscale colormap, sets the plot title,
                 and hides axes for clarity.
    Input: ax    (matplotlib.axes.Axes) - axis object to plot on
           img   (np.ndarray) - image data of arbitrary numeric type
           title (str)        - title string for the subplot
    Output: None (renders the image on the provided axis)
'''
def show_gray(ax, img, title):
    img = img.astype(np.float64, copy=False)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-12:
        vis = np.zeros_like(img)
    else:
        vis = (img - mn) / (mx - mn)
    ax.imshow(vis, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")
"""-------------------------------------Gaussian and Derivative Kernels----------------------------------------------"""
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
"""-------------------------------------Padding and Convolution Helpers----------------------------------------------"""
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

"""-----------------------------------------Gradient Computation-----------------------------------------------------"""
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

"""----------------------------Orientation Quantization & Non-Maximum Suppression------------------------------------"""
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
"""-------------------------------------------Hysteresis Thresholding------------------------------------------------"""
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
    sigma, radius, low, high, I = read_input()

    # Gaussian function calculation
    G = get_gaussian_kernel(sigma, radius)

    # Gaussian derivative function calculation
    dG = get_gaussian_derivative(sigma, radius)

    # Convolution calculations of I * G(x) and I * G(y) respectively
    Igx = convolution_correlation(I, G, axis="x", convolution=True)
    Igy = convolution_correlation(I, G, axis="y", convolution=True)

    # Matrix I subcomponent function calculation
    # Includes the convolution_correlation function call
    Ix = get_image_subcomponents_by_axis(I, dG, G)
    Iy = get_image_subcomponents_by_axis(I, G, dG)

    # Gradient magnitude function calculation M
    M = gradient_magnitude(Ix, Iy)

    # Gradient orientation function calculation theta
    theta = gradient_orientation(Ix, Iy)

    # Quantization orientation function calculation
    # It assigns a respective angle to each gradient orientation
    theta_q = quantize_orientation_4(theta)

    # Non-Maximum suppression function calculation
    nms = non_maximum_suppression_4dir(M, theta_q)

    # Hysteresis Threshold function calculation
    canny_result = hysteresis(nms, low, high, relative=True)

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1,1,1], width_ratios=[1,1,1], hspace=0.15, wspace=0.05)

    # Plotting image (a) Gaussian along x
    plot_a = fig.add_subplot(gs[0, 0]); show_gray(plot_a, Igx, "(a) Gaussian in x")

    # Plotting image (b) Gaussian along y
    plot_b = fig.add_subplot(gs[0, 1]); show_gray(plot_b, Igy, "(b) Gaussian in y")

    # Plotting image (c) derivative along x (Ix)
    plot_c = fig.add_subplot(gs[0, 2]); show_gray(plot_c, Ix,  "(c) Gaussian Derivative x (Ix')")

    # Plotting image (d) derivative along y (Iy)
    plot_d = fig.add_subplot(gs[1, 0]); show_gray(plot_d, Iy,  "(d) Gaussian Derivative y (Iy')")

    # Plotting image (e) gradient magnitude
    plot_e = fig.add_subplot(gs[1, 1]); show_gray(plot_e, M,   "(e) Gradient Magnitude")

    # Plotting image (f) NMS result
    plot_f = fig.add_subplot(gs[1, 2]); show_gray(plot_f, nms, "(f) NMS")

    # Plotting image (g) final hysteresis edges (optional output)
    plot_g = fig.add_subplot(gs[2, :]); show_gray(plot_g, canny_result, "(g) Hysteresis")

    # Plot image layout
    plt.show()

# Runs the main function at the start of execution
if __name__ == "__main__":
    main()