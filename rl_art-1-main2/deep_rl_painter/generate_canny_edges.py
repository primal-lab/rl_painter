# not called anywhere in the project
# use this to generate an image once individually and just refer to the image directly in the code
# via config.py
import os
import cv2
import numpy as np
import importlib.util

# Dynamically load config.py as a module
def load_config():
    config_path = "config.py"
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def generate_canny_edges(image_path, save_folder, save_name="target_edges_1.png",
                         low_threshold=50, high_threshold=100):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 1.4)

    # Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # Save result
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, save_name)
    cv2.imwrite(save_path, edges)

    print(f"✅ Canny edges saved to: {save_path}")

if __name__ == "__main__":
    config = load_config()
    image_path = config.config["target_image_path"]  # Assuming your config uses config = {...}

    save_folder = "canny_edges"
    generate_canny_edges(image_path, save_folder)
