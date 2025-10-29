"""
# This code calculates the reward based on the cosine similarity of latent representations
# of images using CLIP. It includes functions to calculate the reward, extract latent representations,
# and compute cosine similarity. The code is designed to be run in a PyTorch environment.

TODO: 
1. Check the sign of reward values returned by the reward function.
2. Optimize the get_latent_representation function to avoid unnecessary preprocessing.
3. Add functionality to handle different image sizes and channels.
4. Add functionality for other reward functions if needed.
5. Fix the image dimension issue
"""

import torch
from torchvision import transforms
from torch.nn.functional import cosine_similarity
import clip
import time
import math
import os
import csv
import numpy as np
from skimage.draw import line as skimage_line  
from config import config
import torchvision.transforms as T
from torchvision.transforms.functional import resize, center_crop
from .clip_preprocess_gpu import build_gpu_preprocess
from pytorch_msssim import ms_ssim
import torch.nn.functional as F
import lpips
from .dithering import ordered_dither, _save_png01_nchw
import wandb
from torchvision.utils import save_image

# ----------------------- Config / Globals -----------------------

_IS_MAIN = int(os.environ.get("RANK", "0")) == 0

def wb_log(data: dict, step: int | None = None):
    # log only from rank0 and only if wandb is initialized
    if _IS_MAIN and getattr(wandb, "run", None) is not None:
        wandb.log(data, step=step)

# 10 radii for 1024×1024; first is DC-only, last ≈ 111
RADII_10 = [1, 4, 8, 13, 17, 22, 28, 42, 63, 111]

# caches / episode state
CACHED_NEGLOG_MSE_PREV = None
LAST_EPISODE_FOR_CACHE = -1

# per-episode 2DFT targets cache (list of 10 tensors (1,1,H,W) in [0,1])
TARGETS_2DFT = None
LAST_EPISODE_FOR_TARGETS = -1

SAVE_BASE = "/storage/axp4488/rl_painter/logs/2dft_targets"

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _to_gray_np_if_needed(img_uint8_hw_or_hwc: np.ndarray) -> np.ndarray:
    """
    If input is color (H,W,3), convert to grayscale (ITU-R BT.601).
    If already grayscale (H,W) or (H,W,1), return grayscale HxW.
    Output: (H,W) float32 in [0,255].
    """
    if img_uint8_hw_or_hwc.ndim == 3:
        H, W, C = img_uint8_hw_or_hwc.shape
        if C == 1:
            return img_uint8_hw_or_hwc[..., 0].astype(np.float32)
        elif C == 3:
            r = img_uint8_hw_or_hwc[..., 0].astype(np.float32)
            g = img_uint8_hw_or_hwc[..., 1].astype(np.float32)
            b = img_uint8_hw_or_hwc[..., 2].astype(np.float32)
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            return gray.astype(np.float32)
        else:
            # unexpected channels; take first
            return img_uint8_hw_or_hwc[..., 0].astype(np.float32)
    elif img_uint8_hw_or_hwc.ndim == 2:
        return img_uint8_hw_or_hwc.astype(np.float32)
    else:
        # fallback: squeeze last dim if present
        return np.squeeze(img_uint8_hw_or_hwc).astype(np.float32)

def _build_2dft_targets_from_gray(gray: np.ndarray, radii: list[int]) -> list[np.ndarray]:
    """
    gray: (H,W) float32 in [0,255]
    radii: list of integer radii (center-inclusive). r=0 => DC only.
    Returns list of (H,W) float32 images in [0,255] for each radius.
    """
    H, W = gray.shape
    F = np.fft.fft2(gray)
    F_shift = np.fft.fftshift(F)

    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    R2 = (Y - cy) ** 2 + (X - cx) ** 2

    outs = []
    for r in radii:
        mask = (R2 <= (r * r)).astype(np.float32)
        F_masked = F_shift * mask
        rec = np.fft.ifft2(np.fft.ifftshift(F_masked))
        rec = np.real(rec).astype(np.float32)
        mn, mx = rec.min(), rec.max()
        if mx > mn:
            rec = (rec - mn) / (mx - mn)
        else:
            rec = np.zeros_like(rec, dtype=np.float32)
        rec = (rec * 255.0).astype(np.float32)
        outs.append(rec)
    return outs

def _save_targets_to_disk(targets_hw_float255: list[np.ndarray]):
    """
    Saves each (H,W) float32 [0,255] as:
      /storage/.../2dft_targets/target{i}.png   (i=1..10)
    """
    _ensure_dir(SAVE_BASE)
    for i, img in enumerate(targets_hw_float255, start=1):
        img01 = torch.tensor(img / 255.0, dtype=torch.float32)[None, None, ...]  # (1,1,H,W)
        out_path = os.path.join(SAVE_BASE, f"target{i}.png")
        #_save_png01_nchw(out_path, img01.clamp(0, 1))
        _save_png01_nchw(img01, out_path)

def _prepare_episode_targets(target_canvas_bhwc_uint8: torch.Tensor):
    """
    Build and cache the 10 2DFT targets for this episode from the first item in the batch.
    Returns list of 10 tensors (1,1,H,W) in [0,1], on CPU (move to device on use).
    """
    first_np = target_canvas_bhwc_uint8[0].detach().cpu().numpy()  # (H,W,C) uint8 or (H,W,1)
    gray = _to_gray_np_if_needed(first_np)                         # (H,W) float32 [0,255]
    targets = _build_2dft_targets_from_gray(gray, RADII_10)       # list of (H,W) float32 [0,255]
    _save_targets_to_disk(targets)                                 # persist for inspection

    targets_torch = []
    for t in targets:
        t01 = torch.tensor(t / 255.0, dtype=torch.float32)[None, None, ...]  # (1,1,H,W)
        targets_torch.append(t01)
    return targets_torch


# ----------------------- Main Reward -----------------------

def calculate_reward(prev_canvas, current_canvas, target_canvas, device,
                     prev_prev_idx, prev_idx, current_idx, center,
                     edge_map=None, current_episode=None, current_step=None, segments_map=None):
    """
    Reward uses ONLY -log(MSE).
      IR = (-log MSE_curr) - (-log MSE_prev)
      GR = (-log MSE_curr)
      TOTAL = IR + GR

    Also:
      - Builds 10 frequency-limited target images (per-episode) via 2D FFT on the (grayscale) target.
      - Saves them to /storage/axp4488/rl_painter/logs/2dft_targets/number{i}/target{i}.png
      - Selects 1 of the 10 targets per 500 steps (5000 steps/episode).

    Assumes canvases are (B, H, W, C) uint8 in [0,255].
    current_canvas is grayscale in practice; we take the first channel.
    """
    global CACHED_NEGLOG_MSE_PREV, LAST_EPISODE_FOR_CACHE
    global TARGETS_2DFT, LAST_EPISODE_FOR_TARGETS

    with torch.no_grad():
        # Reset IR cache at new episode or at step 0
        if (LAST_EPISODE_FOR_CACHE != int(current_episode)) or (int(current_step) == 0):
            CACHED_NEGLOG_MSE_PREV = None
            LAST_EPISODE_FOR_CACHE = int(current_episode)

        # Build per-episode 2DFT targets (once per episode)
        if (TARGETS_2DFT is None) or (LAST_EPISODE_FOR_TARGETS != int(current_episode)):
            TARGETS_2DFT = _prepare_episode_targets(target_canvas)  # list of 10 (1,1,H,W) [0,1] on CPU
            LAST_EPISODE_FOR_TARGETS = int(current_episode)

        # ----------- Choose which of the 10 targets to use this step -----------
        # 5000 steps/ep, 10 targets => 500-step segments
        step_in_ep = int(current_step)
        seg_idx = min(step_in_ep // 500, 9)  # 0..9
        chosen_t = TARGETS_2DFT[seg_idx].to(device)  # (1,1,H,W)

        # ---------- Stage scheduling: 3 episodes per Fourier target ----------
        # Episode groups: (0–2)->stage0, (3–5)->stage1, ..., (27–29)->stage9
        """ep = int(current_episode)
        seg_idx = min(ep // 3, 9)  # 0..9, each index stays for 3 episodes
        chosen_t = TARGETS_2DFT[seg_idx].to(device)  # (1,1,H,W)"""

        # ----------- Current canvas grayscale NCHW [0,1] -----------
        # We assume grayscale always; just take first channel after BHWC->NCHW
        curr_nchw = current_canvas.permute(0, 3, 1, 2).contiguous().to(torch.float32) / 255.0  # (B,C,H,W)
        curr_gray = curr_nchw[:, :1, ...]  # (B,1,H,W) take first channel

        # Match target shape to batch
        #B = curr_gray.shape[0]
        #targ_gray = chosen_t.expand(B, -1, -1, -1)  # (B,1,H,W)

        # ----------- MSE and shaped scores -----------
        # Per-batch MSE (lower is better)
        mse_curr = F.mse_loss(curr_gray, chosen_t, reduction="none").flatten(1).mean(dim=1)  # (B,)
        # No epsilon: if mse == 0 -> +inf by design (your preference)
        neglog_mse_curr = -torch.log(mse_curr)  # (B,)

        # IR = Δ(-log MSE) relative to previous step in episode
        if CACHED_NEGLOG_MSE_PREV is None:
            ir_mse = torch.zeros_like(neglog_mse_curr)
        else:
            ir_mse = neglog_mse_curr - CACHED_NEGLOG_MSE_PREV

        gr_mse = neglog_mse_curr
        total = ir_mse + gr_mse  # active total

        total_reward = total.mean().item()

        # ----------- Update caches for next step -----------
        CACHED_NEGLOG_MSE_PREV = neglog_mse_curr.detach()

        # ----------- W&B logging -----------
        from config import config  # local import to avoid circular import at module load
        STEPS_PER_EP = int(config.get("max_strokes", 5000))
        global_step = int(current_episode) * STEPS_PER_EP + step_in_ep

        wb_log({
            #"reward_plots/episode": int(current_episode),
            #"reward_plots/step_in_episode": step_in_ep,
            #"reward_plots/segment_index_0_based": seg_idx,
            #"reward_plots/segment_radius": float(RADII_10[seg_idx]),

            # raw components
            "reward_plots/mse_curr": mse_curr.mean().item(),
            "reward_plots/neglog_mse_curr": neglog_mse_curr.mean().item(),

            # components
            "reward_plots/ir_mse": ir_mse.mean().item(),
            "reward_plots/gr_mse": gr_mse.mean().item(),

            # total
            "reward_plots/total_reward": total_reward,
        }, step=global_step)

    return total_reward

def _lpips_model(device):
    global LPIPS_MODEL
    if LPIPS_MODEL is None or next(LPIPS_MODEL.parameters()).device != torch.device(device):
        LPIPS_MODEL = lpips.LPIPS(net='alex').to(device).eval() #'vgg' or 'squeeze', alex is default & fast
        for p in LPIPS_MODEL.parameters():
            p.requires_grad_(False)
    return LPIPS_MODEL   

#def segments_reward(start, end, segments_map):
def segments_reward(start, end, target_canvas):
    """
    Computes a penalty for strokes based on how much they overlap with brighter regions
    in the segmentation map. Encourages filling darker (black) areas first by returning 
    low values for dark strokes and sharply increasing penalties as more white pixels are covered.
    """
    # read the start and end points
    x1, y1 = int(start[0]), int(start[1])
    x2, y2 = int(end[0]), int(end[1])
    
    # Ensure (H, W)
    if target_canvas.ndim == 4:
        target_canvas = target_canvas.squeeze(0).squeeze(-1)  # (H, W)
    elif target_canvas.ndim == 3 and target_canvas.shape[0] == 1:
        target_canvas = target_canvas.squeeze(0)  # (H, W)

    # read the shape of the canvas/segments_map    
    H, W = target_canvas.shape

    # line gets the pixels containing the stroke on the canvas image
    rr, cc = skimage_line(y1, x1, y2, x2)  #skimage uses (row, col) = (y, x)
    # just to make sure the pixels are on the canvas
    rr = np.clip(rr, 0, H - 1)
    cc = np.clip(cc, 0, W - 1)

    # read the values of all the selected pixels on the edges image
    values = target_canvas[rr, cc].detach().cpu().numpy()
    # normalise to [0.0, 1.0]
    if values.max() > 1.0:
        values = values / 255.0
    mean_val = np.mean(values) # range -> [0.0, 1.0]

    # if all pixels are black -> very good stroke => avg = 0 => penalty = 1 * 0.01
    # the above needs to be encouraged, so everything else will get high reward values
    # if almost all pixels are white -> bad stroke => avg = 0.9 => penalty = 10 * 0.01
    #penalty = (1 / (1 - mean_val + 1e-6)) * 100 #chnage to 100 or 10?
    # penalty = [(1 / (1 - mean_val + 1e-6)) * 0.01] - 0.01 # to get p = 0, when perfect stroke (avg=0)
    #return penalty

    reward = 1.0 - mean_val     # 0 → 1 (darker = high reward, brighter = low reward)
    return reward

#def stroke_intersects_edge(start, end, edge_map, threshold=0.8):
def stroke_intersects_edge(start, end, edge_map):
    """
    Checks if the line between start and end intersects any edge pixels.
    """
    # read the start and end points
    x1, y1 = int(start[0]), int(start[1])
    x2, y2 = int(end[0]), int(end[1])
    # read the shape of the canvas/edge_map
    H, W = edge_map.shape

    # line gets the pixels containing the stroke on the canvas image
    rr, cc = skimage_line(y1, x1, y2, x2)  #skimage uses (row, col) = (y, x)
    # just to make sure the pixels are on the canvas
    rr = np.clip(rr, 0, H - 1)
    cc = np.clip(cc, 0, W - 1)

    # read the values of all the selected pixels on the edges image
    values = edge_map[rr, cc].detach().cpu().numpy()
    # avg all the values of the selected pixels 
    # if the avg is > threshold, return true
    #return np.mean(values) > threshold

    mean_val = np.mean(values)
    stroke_penalty = ((1 / (mean_val + 1e-6)) - 1) * 0.01
    # if avg = 0 (no intersection w/edges) => 1/1e-6 - 1 = 1000000 - 1 = 999999 *0.01 = 9999.99
    return stroke_penalty

def calculate_auxiliary_reward(prev_prev_point, prev_point, current_point, center):
    """
    Computes the combined auxiliary reward:
    - Overlap penalty
    - Stroke length (alpha threshold) penalty

    Returns:
        float: Auxiliary reward value.
    """
    # if step 1 or 2, skip auxiliary reward
    if prev_prev_point is None or prev_point is None:
        return 0.0

    # penalty for overlapping strokes
    # current and previosu stroke vectors 
    v_stroke_prev = torch.tensor(prev_point - prev_prev_point, dtype=torch.float32)
    v_stroke_current = torch.tensor(current_point - prev_point, dtype=torch.float32)

    overlap_penalty = calculate_overlap_penalty(v_stroke_prev, v_stroke_current)

    # penalty for stroke length (smaller angles)
    #v_center_prev = torch.tensor(prev_point - center, dtype=torch.float32)
    #v_center_current = torch.tensor(current_point - center, dtype=torch.float32)

    #stroke_length_penalty = calculate_stroke_length_penalty(v_center_prev, v_center_current)

    # total auxiliary reward
    #aux_reward = overlap_penalty + stroke_length_penalty
    aux_reward = overlap_penalty

    return aux_reward

def calculate_overlap_penalty(v_stroke_prev, v_stroke_current):
    """
    Penalizes overlapping strokes using vector sum norm.

    Returns:
        float: Overlap penalty.
    """
    sum_vector = v_stroke_prev + v_stroke_current
    # norm captures direction
    # if same direction -> length = bigger
    # if opp direction -> length = smaller (values cancel out to a certain extent)
    norm_sum = torch.norm(sum_vector)

    # scale
    # TO-DO: adjust
    overlap_scale = 5.0

    # if norm_sum = smaller -> bigger penalty (for overlapping strokes)
    # if norm_sim = bigger -> smaller penalty
    overlap_penalty = (1.0 / (norm_sum + 1e-6)) * overlap_scale 

    # Instead do: cosine sim b/w the 2 strokes ??
    # theta = angle between their movement directions (to detect backtracking)
    # opp direction -> 180° -> cos theta ≈ -1 -> big penalty
    # same direction -> 0° -> cos theta ≈ +1 -> no penalty

    return overlap_penalty

def calculate_stroke_length_penalty(v_center_prev, v_center_current):
    """
    Penalizes short strokes based on angle alpha between center → prev and center → current vectors.

    Returns:
        float: Stroke length penalty.
    """
    angle_alpha = compute_angle_between_vectors(v_center_prev, v_center_current)

    # minimum threshold angle
    # TO-DO: adjust
    threshold_angle = math.radians(20) 

    # scale
    # TO-DO: adjust
    length_scale = 5.0

    # if angle >= threshold, reward = 0
    # if angle < threshold, 
    # the shorter the stroke (or smaller the angle), the more penalty
    if angle_alpha < threshold_angle:
        stroke_length_penalty = ((threshold_angle - angle_alpha) / threshold_angle) * length_scale
    else:
        stroke_length_penalty = 0.0

    return stroke_length_penalty

def compute_angle_between_vectors(v1, v2):
    """
    Computes the angle (in radians) between two vectors v1 and v2.

    Args:
        v1 (torch.Tensor): First vector (x, y)
        v2 (torch.Tensor): Second vector (x, y)

    Returns:
        torch.Tensor: Angle in radians.
    """
    # dot product between the vectors
    dot_product = torch.dot(v1, v2)

    # norm (length) of each vector
    norm_v1 = torch.norm(v1)
    norm_v2 = torch.norm(v2)

    # cos of the angle between vectors
    cos_theta = dot_product / (norm_v1 * norm_v2 + 1e-8)  # add epsilon for safety

    # clamp cos_theta to valid range [-1, 1] to avoid NaN due to floating point errors
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # angle in radians
    angle = torch.acos(cos_theta)

    return angle


def mse_loss(pred, target):
    """
    Calculates the Mean Squared Error (MSE) loss between two tensors.

    Args:
        pred (torch.Tensor): The predicted tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        torch.Tensor: The MSE loss value.
    """
    return torch.mean((pred.to(pred.device) - target.to(pred.device)) ** 2)

def get_latent_representation(image, device):
    """
    Extracts the latent representation of an image using a pre-trained model.
    The model should be a feature extractor (e.g., ResNet) with the last layer removed.

    Args:
        image (torch.Tensor): The input image tensor (shape: [channels, height, width]).
        device (torch.device): The device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: The latent representation of the image (a 1D tensor).
    """

    # Modify the model to output the features from the penultimate layer
    global CLIP_MODEL, PREPROCESS
    if CLIP_MODEL is None:
        CLIP_MODEL, _ = clip.load("ViT-B/32", device=device)
        PREPROCESS = build_gpu_preprocess(CLIP_MODEL.visual.input_resolution, device)
    # print(PREPROCESS) -> controls resizing to 224*224 already 

    # canvas shape (B, H, W, C)
    # clip needs (B, C, H, W) or (C, H, W)
    if len(image.shape) == 4 and (image.shape[1] != 1 or image.shape[1] != 3):
        image = image.permute(0, 3, 1, 2)
    try:
        if len(image.shape) == 4:
            image = image[0]
        image = image.detach().cpu() if image.is_cuda else image
        t2 = time.time()
        #image = PREPROCESS(transforms.ToPILImage()(image)).unsqueeze(0).to(device)
        image = PREPROCESS(image)  # returns [B=1, 3, n_px, n_px] on GPU, normalized
        t3 = time.time()
        total2 = t3-t2
        # preprocess Time: ", total2)

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        print("Image shape:", image.shape)
        return None

    with torch.no_grad():
        t4 = time.time()
        latent_representation = CLIP_MODEL.encode_image(image)
        t5 = time.time()
        total3 = t5-t4
        #print("(in reward.py) encode_image Time: ", total3)
        latent_representation = torch.flatten(
            latent_representation, 1)  # Flatten to a 1D tensor

    return latent_representation


def calculate_cosine_similarity(latent1, latent2):
    """
    Calculates the cosine similarity between two latent vectors.

    Args:
        latent1 (torch.Tensor): The first latent vector.
        latent2 (torch.Tensor): The second latent vector.

    Returns:
        torch.Tensor: The cosine similarity score (a scalar tensor). Returns None if
                      either latent vector is None.
    """
    if latent1 is None or latent2 is None:
        return None
    return cosine_similarity(latent1, latent2)


if __name__ == "__main__":
    # Determine the device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Example usage: Create dummy canvas tensors
    channels = 1
    height = 224
    width = 224

    prev_canvas = torch.randn(channels, height, width).to(device)
    current_canvas = torch.randn(channels, height, width).to(device)
    target_canvas = torch.randn(channels, height, width).to(device)

    # Calculate the reward
    reward = calculate_reward(
        prev_canvas, current_canvas, target_canvas, device)

    # Print the reward
    print("Reward:")
    print(reward)

    # Example of getting a single latent representation
    single_image = torch.randn(channels, height, width).to(device)
    latent_vector = get_latent_representation(single_image, device)
    print("\nSingle Latent Representation:")
    print(latent_vector.cpu().numpy())
    # should be [1, 512] for ViT-B/32
    print("Latent Vector Shape:", latent_vector.shape)

    # Example of calculating cosine similarity between two latent vectors
    # CLIP ViT-B/32 outputs a feature vector of size 512
    latent1_example = torch.randn(1, 512).to(device)
    latent2_example = torch.randn(1, 512).to(device)
    similarity = calculate_cosine_similarity(latent1_example, latent2_example)
    print("\nCosine Similarity Example:")
    print(similarity.item())
