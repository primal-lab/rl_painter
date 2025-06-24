# called once at the beginning of the run 
# to generate simplified versions of the target image
# to compare with current_canvas as it progresses thro steps in an episode
import os
import cv2
import torch
import numpy as np


def generate_simplified_targets(target_tensor, save_dir=None):
    """
    Generates 5 progressively simplified versions of the target image.
    Saves them if save_dir is provided.
    Returns a list of 5 tensors [1, 1, H, W] scaled to [0, 1].
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    original_np = (target_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    #if save_dir:
    #    cv2.imwrite(os.path.join(save_dir, "0_check_original_input.png"), original_np)
    H, W = original_np.shape

    simplified = []

    for i in range(5):
        base = original_np.copy() 
        if i == 0:
            # pixelated down (8x), stretche it back to 256×256
            down = cv2.resize(base, (W // 8, H // 8), interpolation=cv2.INTER_LINEAR)
            up = cv2.resize(down, (W, H), interpolation=cv2.INTER_NEAREST)
            label = "1_pixelated_8x"
        elif i == 1:
            # applies Gaussian blur with a large kernel (9×9)
            blur = cv2.GaussianBlur(base, ksize=(9, 9), sigmaX=4)
            label = "2_blurred_strong"
            up = blur
        elif i == 2:
            # shrinks to half, then resizes to full
            down = cv2.resize(base, (W // 2, H // 2), interpolation=cv2.INTER_LINEAR)
            up = cv2.resize(down, (W, H), interpolation=cv2.INTER_LINEAR)
            label = "3_pixelated_2x"
        elif i == 3:
            # canny edges + grayscale blend
            edges = cv2.Canny(base, 50, 100)
            blended = 0.6 * base + 0.4 * edges  # blend edges with base image
            up = blended.astype(np.uint8)
            label = "4_edges_blended"
        elif i == 4:
            # full-resolution original target
            up = base
            label = "5_original"

        if save_dir:
            cv2.imwrite(os.path.join(save_dir, f"version_{i+1}_{label}.png"), up)

        tensor = torch.tensor(up / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        simplified.append(tensor)
        #print(f"[GenTarget] Appended version {i+1} - {label} | Tensor shape: {tensor.shape} | Shape[0]: {tensor.shape[0]}, Shape[1]: {tensor.shape[1]}")

    return simplified