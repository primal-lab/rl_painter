import os
import cv2
import torch
import numpy as np

def generate_simplified_targets(target_tensor, save_dir=None):
    """
    Generates 10 progressively simplified versions of the target image.
    From high-frequency edge maps to low-frequency blurred forms.
    Returns a list of tensors [1, 1, H, W] scaled to [0, 1].
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    original_np = (target_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    H, W = original_np.shape
    simplified = []

    for i in range(10):
        base = original_np.copy()

        if i == 0:
            # Strong Canny edges
            up = cv2.Canny(base, 100, 200)
            label = "1_edges_strong"
        
        elif i == 1:
            # Softer Canny edges
            up = cv2.Canny(base, 50, 100)
            label = "2_edges_soft"

        elif i == 2:
            # Blended edges + original base (structure + slight context)
            edges = cv2.Canny(base, 50, 100)
            blended = 0.6 * base + 0.4 * edges
            up = blended.astype(np.uint8)
            label = "3_edge_blend"

        elif i == 3:
            # Structure + detail preserving: Laplacian for highlights, then blur
            laplace = cv2.Laplacian(base, cv2.CV_8U)
            combined = cv2.addWeighted(base, 0.7, laplace, 0.3, 0)
            up = cv2.GaussianBlur(combined, (3, 3), sigmaX=1)
            label = "4_structure_detail"

        elif i == 4:
            # Silhouette based on strong blur + low threshold
            blurred = cv2.GaussianBlur(base, (15, 15), sigmaX=8)
            _, soft_thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
            up = soft_thresh
            label = "5_soft_fill_mask"

        elif i == 5:
            # Midtone blobs: 1/4 downsampling + upsample
            down = cv2.resize(base, (W // 4, H // 4), interpolation=cv2.INTER_LINEAR)
            up = cv2.resize(down, (W, H), interpolation=cv2.INTER_LINEAR)
            label = "6_midtone_blobs"

        elif i == 6:
            # Low-res color blockout: 1/8 down + nearest upsample
            down = cv2.resize(base, (W // 8, H // 8), interpolation=cv2.INTER_LINEAR)
            up = cv2.resize(down, (W, H), interpolation=cv2.INTER_NEAREST)
            label = "7_lowres_pixelated"

        elif i == 7:
            # Mild Gaussian blur for perceptual softening
            up = cv2.GaussianBlur(base, (9, 9), sigmaX=4)
            label = "8_soft_blur"

        elif i == 8:
            # Strong blur to simulate low-frequency perceptual mass
            up = cv2.GaussianBlur(base, (25, 25), sigmaX=12)
            label = "9_extreme_blur"

        elif i == 9:
            # Full-resolution original target image
            up = base
            label = "10_original"

        if save_dir:
            cv2.imwrite(os.path.join(save_dir, f"version_{i+1}_{label}.png"), up)

        tensor = torch.tensor(up / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        simplified.append(tensor)

    return simplified
