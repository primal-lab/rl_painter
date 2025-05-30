"""
This module provides functions to create, update, save, and load image canvases using numpy and openCV.

TODO:
- Add functionality for different types of strokes (e.g., circles, rectangles).
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import os


def init_canvas(image_shape: Tuple[int, int] | Tuple[int, int, int],
                color: Optional[Tuple[int, int, int] | int] = None) -> np.ndarray:
    """
    Initializes a canvas with the given image shape and optional background color.
    Defaults to black for grayscale and RGB canvases.
    Args:
        image_shape (Tuple[int, int] | Tuple[int, int, int]): A tuple representing the
            height and width (and optionally channels) of the canvas. For example:
            - (height, width) for grayscale
            - (height, width, 3) for RGB
        color (Optional[Tuple[int, int, int] | int], optional): The background color.
            For grayscale, an integer (0-255). For RGB, a tuple (R, G, B).
            Defaults to black for grayscale (0) and black for RGB (0, 0, 0).

    Returns:
        numpy.ndarray: A NumPy array representing the initialized canvas
            with dtype np.float32.
    """
    if color is None:
        return np.zeros(image_shape, dtype=np.float32)
    elif isinstance(color, int):
        return np.full(image_shape, color, dtype=np.float32)
    elif isinstance(color, tuple):
        if len(image_shape) == 2:  # Grayscale with tuple color - take average
            gray_color = int(0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
            return np.full(image_shape, gray_color, dtype=np.float32)
        elif len(image_shape) == 3 and image_shape[2] == 3:  # RGB
            # OpenCV uses BGR
            bgr_color = (color[2], color[1], color[0])
            return np.full(image_shape, bgr_color, dtype=np.float32)
        else:
            raise ValueError(
                "Color tuple provided for incompatible image shape.")
    else:
        raise ValueError(
            "Invalid color format. Use an integer for grayscale or a tuple for RGB.")


def update_canvas(canvas: np.ndarray, start_point: Tuple[float, float],
                  end_point: Tuple[float, float],
                  color: Optional[Tuple[int, int, int] | int] = None,
                  channels: int = 1, width: float = 1.0) -> np.ndarray:
    """
    Draws a line on the given canvas with the specified color and thickness.

    Args:
        canvas (numpy.ndarray): The image or canvas on which the line will be drawn.
        start_point (tuple): A tuple (x, y) representing the starting point of the line.
        end_point (tuple): A tuple (x, y) representing the ending point of the line.
        color (Optional[Tuple[int, int, int] | int], optional): The color of the line.
            For grayscale canvases (channels=1), this should be an integer (0-255).
            For RGB canvases (channels=3), this should be a tuple (R, G, B).
            If None, defaults to black for grayscale (0) and white for RGB (255, 255, 255).
            Defaults to None.
        channels (int, optional): The number of channels in the canvas (1 for grayscale,
            3 for RGB). Defaults to 1.
        width (float, optional): The width of the line stroke. Defaults to 1.0.

    Returns:
        numpy.ndarray: The updated canvas with the line drawn on it.
    """
    # Convert start and end points to integers
    start_point = (int(start_point[0]), int(start_point[1]))
    end_point = (int(end_point[0]), int(end_point[1]))

    if channels == 1:
        if color is None:
            line_color = 255  # White for grayscale
        elif isinstance(color, tuple):
            # Formula for converting RGB to grayscale (luminance)
            line_color = int(0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
        else:
            line_color = int(color)
    elif channels == 3:
        if color is None:
            line_color = (255, 255, 255)
        elif isinstance(color, tuple):
            # Convert RGB to BGR for OpenCV
            line_color = color
        else:
            raise ValueError(
                "Color for RGB canvas must be a tuple (R, G, B).")
    else:
        raise ValueError(
            "Unsupported number of channels. Only 1 (grayscale) or 3 (RGB) are supported.")

    # Ensure stroke width is a positive integer
    width = max(1, int(width))
    cv2.line(canvas, start_point, end_point, line_color, width)
    #print(canvas.shape)
    #cv2.imshow("Canvas here:",canvas)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return canvas


def save_canvas(canvas: np.ndarray, path: str):
    """
    Saves the given canvas to the specified file path.

    Args:
        canvas (numpy.ndarray): The canvas to be saved.
        path (str): The full path (including filename and extension) where the
            canvas will be saved. OpenCV determines the file format from the extension.
    Returns:
        bool: True if the saving was successful, False otherwise.
    """

    # update_canvas hardcodes black canvas and white stroke
    # invert colors before saving
    if isinstance(canvas, torch.Tensor): #!
        canvas = canvas.detach().cpu().numpy() #!

    inverted_canvas = cv2.bitwise_not(canvas)
    saved = cv2.imwrite(path, inverted_canvas)
    if not saved:
        print(f"Error: Could not save image to {path}")
    return saved

def load_canvas(path: str) -> Optional[np.ndarray]:
    """
    Loads a canvas from the specified image file path.

    Args:
        path (str): The full path to the image file to load.

    Returns:
        Optional[numpy.ndarray]: A NumPy array representing the loaded canvas,
            or None if the loading failed (e.g., file not found).
    """
    if not os.path.exists(path):
        print(f"Error: Image file not found at {path}")
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not load image from {path}")
        return None
    return img


if __name__ == '__main__':
    # Example usage of the canvas functions

    # Define canvas dimensions
    canvas_size_gray = (1080, 1080)
    canvas_size_rgb = (1080, 1080, 3)

    # Initialize grayscale and RGB canvases with specific background colors
    gray_canvas = init_canvas(canvas_size_gray, color=100)  # Light gray
    rgb_canvas = init_canvas(canvas_size_rgb, color=(255, 0, 0))  # Red

    # Draw lines on the canvases
    gray_canvas = update_canvas(gray_canvas, (100, 100), (500, 500), color=200, channels=1, width=2)
    rgb_canvas = update_canvas(rgb_canvas, (5, 5), (600, 300), color=(0, 255, 0), channels=3, width=3)

    # Save the canvases
    save_path_gray = 'gray_canvas.png'
    save_path_rgb = 'rgb_canvas.png'
    gray_saved = save_canvas(gray_canvas, save_path_gray)
    rgb_saved = save_canvas(rgb_canvas, save_path_rgb)

    if gray_saved:
        print(f"Grayscale canvas saved to: {save_path_gray}")
    if rgb_saved:
        print(f"RGB canvas saved to: {save_path_rgb}")

    # Load the saved canvases
    loaded_gray_canvas = load_canvas(save_path_gray)
    loaded_rgb_canvas = load_canvas(save_path_rgb)

    if loaded_gray_canvas is not None:
        print(f"Grayscale canvas loaded from: {save_path_gray} with shape: {loaded_gray_canvas.shape}")
    if loaded_rgb_canvas is not None:
        print(f"RGB canvas loaded from: {save_path_rgb} with shape: {loaded_rgb_canvas.shape}")

    # Example of initializing a black canvas
    black_canvas_rgb = init_canvas((1080, 1080, 3))
    save_canvas(black_canvas_rgb, 'black_canvas_rgb.png')
    print("Black RGB canvas saved to: black_canvas_rgb.png")

    # Example of loading a non-existent file
    non_existent_canvas = load_canvas('non_existent.png')
    if non_existent_canvas is None:
        print("Loading non-existent file handled correctly.")
