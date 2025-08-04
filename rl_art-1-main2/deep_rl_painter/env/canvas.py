"""
This module provides functions to create, update, save, and load image canvases using numpy and openCV.

TODO:
- Add functionality for different types of strokes (e.g., circles, rectangles).
"""

import torch
import numpy as np
import cv2
from typing import Tuple, Optional, Union
import os


def init_canvas(image_shape: Tuple[int, int] | Tuple[int, int, int],
                color: Optional[Tuple[int, int, int] | int] = None,
                device: str = "cuda") -> torch.Tensor:
    """
    Initializes a canvas on GPU in (H, W, C) format.
    Supports grayscale or RGB.
    """
    if len(image_shape) == 2:
        H, W = image_shape
        C = 1
    elif len(image_shape) == 3 and image_shape[2] in [1, 3]:
        H, W, C = image_shape
    else:
        raise ValueError("Invalid image shape")

    # Compute color value per channel
    if color is None:
        fill = [0.0] * C
    elif isinstance(color, int):
        fill = [float(color)] * C
    elif isinstance(color, tuple):
        if C == 1:
            # Convert RGB to grayscale using luminance
            gray = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            fill = [float(gray)]
        else:
            fill = list(color)
    else:
        raise ValueError("Invalid color format.")

    # Create and fill canvas
    canvas = torch.zeros((H, W, C), dtype=torch.float32, device=device)
    for i in range(C):
        canvas[:, :, i] = fill[i]

    return canvas

def update_canvas_torch(canvas: torch.Tensor,
                        start_point: Tuple[float, float],
                        end_point: Tuple[float, float],
                        color: Union[float, Tuple[float, float, float]] = 1.0,
                        width: int = 1,
                        channels: int = 1) -> torch.Tensor:
    """
    Draws a line on a (C, H, W) canvas using Bresenham’s algorithm (only works on integer grid coordinates)
    Returns modified canvas.
    """
    assert canvas.is_cuda # returns true if tensor is on gpu
    H, W = canvas.shape[-2:]
    device = canvas.device

    x0, y0 = int(start_point[0]), int(start_point[1])
    x1, y1 = int(end_point[0]), int(end_point[1])

    dx = abs(x1 - x0) #absolute difference between x coordinates
    dy = abs(y1 - y0)
    x, y = x0, y0 #initialise start point
    sx = 1 if x0 < x1 else -1 # direction to move in x coordinate (left/right)
    sy = 1 if y0 < y1 else -1 # (up/down)
    err = dx - dy # how far off the current pixel is from the ideal line path

    points = [] # collecting the (y, x) pixel coordinates the line will touch
    while True:
        if 0 <= x < W and 0 <= y < H: #point is within canvas boundaries, add it to the list
            points.append((y, x))
        if x == x1 and y == y1: #stop the loop once we’ve reached the destination point
            break
        e2 = 2 * err # consider both x and y directions
        if e2 > -dy: # if the error is leaning too far in the y-direction, step in x to correct it.
            err -= dy
            x += sx
        if e2 < dx: # if error is leaning too far in the x-direction, step in y to stay close to the ideal line.
            err += dx
            y += sy

    coords = torch.tensor(points, dtype=torch.long, device=device).T  # (2, N) - convert to tensor

    # 90% of the pixel (color) stays the same, and 10% is replaced with the new color
    alpha = 0.1  # 10% opacity 

    # Determine per-channel stroke color -> canvas (c, h, w)
    if isinstance(color, (int, float)): # single value like 255 or 0.5
        # if c=3 (rgb), if color = 255 and canvas is RGB, stroke_color = [255, 255, 255]
        stroke_color = [color] * canvas.shape[0] 
    elif isinstance(color, (tuple, list)):
        if len(color) == 1: # single-element tuple like (200,), expand it for all channels
            stroke_color = [color[0]] * canvas.shape[0]
        else:
            stroke_color = color # something like (255, 0, 0) and canvas has 3 channels, it's good to go
    else:
        raise ValueError("Unsupported color type for stroke")

    # Apply alpha blending per channel
    for c in range(canvas.shape[0]):
        # new_pixel_value = [(1 - alpha) * old_value] + [alpha * stroke_value]
        # so 90% old value + 10% new value of color
        canvas[c, coords[0], coords[1]] = ((1 - alpha) * canvas[c, coords[0], coords[1]] + alpha * stroke_color[c])

    return canvas

def update_canvas(canvas: Union[np.ndarray, torch.Tensor],
                  start_point: Tuple[float, float],
                  end_point: Tuple[float, float],
                  color: Optional[Union[int, Tuple[int, int, int]]] = None,
                  channels: int = 1,
                  width: float = 1.0) -> Union[np.ndarray, torch.Tensor]:
    """
    Updates a canvas using GPU (if torch.Tensor) or OpenCV (if np.ndarray).
    Canvas must be (H, W, C) format.
    """

    if isinstance(canvas, torch.Tensor) and canvas.is_cuda:
        canvas_chw = canvas.permute(2, 0, 1).contiguous()  # (C, H, W)
        if color is None:
            color = 255.0 if channels == 1 else (255.0, 255.0, 255.0)
        updated = update_canvas_torch(canvas_chw, start_point, end_point, color, width, channels)
        return updated.permute(1, 2, 0).contiguous()  # back to (H, W, C)

    # Fallback to CPU/OpenCV
    start = (int(start_point[0]), int(start_point[1]))
    end = (int(end_point[0]), int(end_point[1]))

    if channels == 1:
        if color is None:
            line_color = 255
        elif isinstance(color, tuple):
            line_color = int(0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
        else:
            line_color = int(color)
    elif channels == 3:
        line_color = color if color else (255, 255, 255)
    else:
        raise ValueError("Unsupported number of channels")

    width = max(1, int(width))
    cv2.line(canvas, start, end, line_color, width)
    return canvas

def save_canvas_torch(canvas: torch.Tensor, path: str) -> bool:
    """
    Converts (H, W, C) torch CUDA tensor to CPU NumPy and saves it.
    """
    if canvas.ndim not in [2, 3]:
        print(f"Error: Invalid tensor shape {canvas.shape} for saving.")
        return False

    img = canvas.detach().cpu().clamp(0, 255).to(torch.uint8).numpy()

    if img.ndim == 3 and img.shape[2] not in [1, 3]:
        print(f"Error: Invalid channel count in tensor shape {canvas.shape}")
        return False

    inverted = 255 - img
    success = cv2.imwrite(path, inverted)
    if not success:
        print(f"Error: Failed to save canvas to {path}")
    return success

def save_canvas(canvas: Union[np.ndarray, torch.Tensor], path: str) -> bool:
    """
    Saves a canvas (torch.Tensor or np.ndarray) to disk.
    Calls save_canvas_torch() if canvas is a GPU tensor.
    Calls legacy OpenCV save logic otherwise.
    """
    if isinstance(canvas, torch.Tensor) and canvas.is_cuda:
        return save_canvas_torch(canvas, path)

    # --- NumPy fallback ---
    if not isinstance(canvas, np.ndarray):
        print("Error: Canvas must be a NumPy array or GPU torch.Tensor.")
        return False

    if canvas.ndim == 3 and canvas.shape[2] in [1, 3]:
        inverted_canvas = 255 - canvas
    elif canvas.ndim == 2:
        inverted_canvas = 255 - canvas
    else:
        print(f"Error: Invalid NumPy canvas shape {canvas.shape} for saving.")
        return False

    saved = cv2.imwrite(path, inverted_canvas)
    if not saved:
        print(f"Error: Could not save image to {path}")
    return saved


def load_canvas(path: str, device: str = "cuda") -> Optional[Union[np.ndarray, torch.Tensor]]:
    """
    Loads a canvas from disk as a NumPy array or GPU tensor in (H, W, C) format.

    Args:
        path (str): Full path to image file.
        device (str): 'cuda' to return a torch.Tensor on GPU, 'cpu' for NumPy.

    Returns:
        torch.Tensor or np.ndarray: Loaded canvas image in (H, W, C), or None if failed.
    """
    if not os.path.exists(path):
        print(f"Error: Image file not found at {path}")
        return None

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not load image from {path}")
        return None

    img = img.astype(np.float32)

    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    if device == "cuda":
        return torch.from_numpy(img).float().to(device)  # (H, W, C)
    return img  # NumPy fallback



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
