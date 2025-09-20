import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple

"""
    Reads an image from disk, converts it to grayscale,
    and returns it as a NumPy array (matrix).
"""
def read_image(path: str) -> Tuple[np.ndarray, np.ndarray]:
    # Read image in color
    img = cv2.imread(path)

    # Manage Error if Image is not found throw an exception
    if img is None:
        raise FileNotFoundError(f"The specified image path {path} did not contain a valid image.")

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Return both normalized and raw grayscale
    I = gray_image.astype("float32") / 255.0

    return I, gray_image

"""
    Create a 1D Gaussian kernel with standard deviation sigma.
    If radius is not given, use ceil(3*sigma).
"""
def get_gaussian_kernel(sigma: float, radius: int | None = None) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("Incorrect input: Sigma value must be > 0")

    if radius is None:
        radius = int(np.ceil(3 * sigma))

    x = np.arange(-radius, radius + 1)
    G = np.exp(-(x**2) / (2 * sigma**2))
    G /= G.sum()
    return G

"""
    Create a 1D first-derivative-of-Gaussian kernel (DoG).
    If radius is not given, use ceil(3*sigma).
    By default, L1-normalize (sum of absolute values = 1) so scale is comparable across sigmas.
"""
def get_gaussian_derivative_kernel(sigma: float, radius: int | None = None) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("Incorrect input: sigma must be > 0")

    if radius is None:
        radius = int(np.ceil(3 * sigma))

    x = np.arange(-radius, radius + 1, dtype=np.float64)
    G = np.exp(-(x**2) / (2 * sigma**2))
    dG = -(x / (sigma**2)) * G
    return dG

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
    dG = get_gaussian_derivative_kernel(sigma)
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
    Gx = get_gaussian_derivative_kernel(sigma)
    Gy = get_gaussian_derivative_kernel(sigma)
    test_requirements(I, gray_image, image_path, sigma)


if __name__ == "__main__":
    main()