"""
    ======================================================================
                        Input/Utility Functions Module
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
    Provides helper functions to read user input, parse command-line
    style flags, and load images. Includes error handling for missing
    inputs and invalid parameter combinations. Normalizes grayscale
    images into [0,1] and provides utility conversions (to_u8) and
    plotting helpers (prep_image). Ensures flexibility with default
    parameters (sigma, radius, thresholds) and user overrides.
"""
"""---------------------------------------------Imports and Globals--------------------------------------------------"""
import sys, cv2, numpy as np
from typing import Optional
"""--------------------------------------------------Functions-------------------------------------------------------"""
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
    Function Name: prep_image
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
def prep_image(ax, img, title):
    img = img.astype(np.float64, copy=False)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-12:
        vis = np.zeros_like(img)
    else:
        vis = (img - mn) / (mx - mn)
    ax.imshow(vis, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")