import sys
import cv2
import numpy as np
from pathlib import Path

"""
    Reads an image from disk, converts it to grayscale,
    and returns it as a NumPy array (matrix).
"""
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

def sigma_radius_check(sigma: float, radius: int | None = None) -> tuple[float, int]:
    if sigma <= 0:
        raise ValueError("Incorrect input: Sigma value must be > 0")

    if radius is None:
        radius = int(np.ceil(3 * sigma))
    return sigma, radius

"""
    Create a 1D Gaussian kernel with standard deviation sigma.
    If radius is not given, radius will be equal to ceil(3*sigma).
"""
def get_gaussian_kernel(sigma: float, radius: int | None = None) -> np.ndarray:
    sigma, radius = sigma_radius_check(sigma, radius)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    G = np.exp(-(x**2) / (2 * sigma**2))
    G /= G.sum()
    return G

"""
    Create a 1D first-derivative-of-Gaussian kernel.
"""
def get_gaussian_derivative(sigma: float, radius: int | None = None) -> np.ndarray:
    sigma, radius = sigma_radius_check(sigma, radius)
    G = get_gaussian_kernel(sigma, radius)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    dG = -(x / (sigma**2)) * G
    return dG

"""
    Apply reflecting padding to a 2D image I by r pixels along a given axis.
    axis=1 pads left/right; axis=0 pads top/bottom.
"""
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
    Dot Product Function:
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

"""
    Compute the gradient magnitude M of Ix prime and Iy prime.
"""
def gradient_magnitude(Ix: np.ndarray, Iy: np.ndarray) -> np.ndarray:
    Ix = Ix.astype(np.float64, copy=False)
    Iy = Iy.astype(np.float64, copy=False)
    return np.sqrt(Ix*Ix + Iy*Iy)

"""
    Computes Ix prime and Iy prime.
"""
def get_image_subcomponents_by_axis(I: np.ndarray, gaussian_kernel_x: np.ndarray, gaussian_kernel_y: np.ndarray) -> np.ndarray:
    convolution_result = convolution_correlation(I, gaussian_kernel_x, axis="x", convolution=True)
    correlation_result = convolution_correlation(convolution_result, gaussian_kernel_y, axis="y", convolution=False)
    return correlation_result

def test_read_image(I: np.ndarray, gray_image: np.ndarray, image_path: str) -> None:
    print("Grayscale matrix shape:", I.shape)
    print("Matrix dtype:", I.dtype)

    input_path = Path(image_path)
    output_path = input_path.with_name(input_path.stem + "_grayscale.png")
    cv2.imwrite(str(output_path), gray_image)
    print(f"Saved grayscale image as {output_path}")

def test_gaussian_kernel(sigma: float, radius: int | None = None):
    gaussian_result = get_gaussian_kernel(sigma, radius)
    radius_result = (len(gaussian_result) - 1) // 2
    print("sigma:", sigma, "| radius:", radius_result)
    print("kernel:", np.array2string(gaussian_result, precision=6))

"""
    Test the derivative-of-Gaussian kernel
"""
def test_gaussian_derivative_kernel(sigma: float, radius: int | None = None) -> None:
    dG = get_gaussian_derivative(sigma, radius)
    radius_result = (len(dG) - 1) // 2

    print("sigma:", sigma, "| radius:", radius_result)
    print("dG:", np.array2string(dG, precision=6))

def test_requirements(I: np.ndarray, gray_image: np.ndarray, image_path: str, sigma: float, radius: int | None = None) -> None:
    test_read_image(I, gray_image, image_path)
    test_gaussian_kernel(sigma, radius)
    test_gaussian_derivative_kernel(sigma, radius)


def main():
    sigma = 1.0
    # Prompt the user for the image path
    print("Welcome to CAP5415 PA1 Assignment: Canny Edge Detection Input Image Loader")
    image_path = input("Please enter the path to the image desired to be processed using Canny Edge Detection: ").strip()
    try:
        I, gray_image = read_image(image_path)

    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    G = get_gaussian_kernel(sigma)
    dG = get_gaussian_derivative(sigma)
    Ix = get_image_subcomponents_by_axis(I, dG, G)
    Iy = get_image_subcomponents_by_axis(I, G, dG)
    M = gradient_magnitude(Ix, Iy)
    test_requirements(I, gray_image, image_path, sigma)


if __name__ == "__main__":
    main()