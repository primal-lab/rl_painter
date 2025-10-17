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

_IS_MAIN = int(os.environ.get("RANK", "0")) == 0

def wb_log(data: dict, step: int | None = None):
    # log only from rank0 and only if wandb is initialized
    if _IS_MAIN and getattr(wandb, "run", None) is not None:
        wandb.log(data, step=step)

# Weights / knobs
W_SSIM, W_LPIPS, W_CLIP = 0.40, 0.40, 0.20   # combined similarity weights
LAMBDA_IR = 0.70                              # R = λ·IR + (1-λ)·GR

# Globals / caches 
CLIP_MODEL = None
PREPROCESS = None
LPIPS_MODEL = None
TARGET_SSIM_NCHW  = None   # (B,C,H,W) in [0,1]
TARGET_LPIPS_NCHW = None   # (B,3,H,W) in [-1,1]
TARGET_CLIP_UNIT  = None   # (1,D) unit-norm CLIP embedding of target

CACHED_S_PREV     = None   # (B,) previous step's combined similarity
LAST_EPISODE_FOR_CACHE = -1

def _get_lpips(device):
    return _lpips_model(device)

def _prep_ssim(bhwc_255: torch.Tensor) -> torch.Tensor:
    """(B,H,W,C)[0..255] -> (B,C,H,W)[0..1] for MS-SSIM."""
    return (bhwc_255.permute(0,3,1,2).contiguous()) / 255.0

def _prep_lpips(bhwc_255: torch.Tensor) -> torch.Tensor:
    """(B,H,W,C)[0..255] -> (B,3,H,W)[-1..1] for LPIPS."""
    x01 = (bhwc_255 / 255.0)
    x = x01.permute(0,3,1,2).contiguous()
    if x.size(1) == 1:
        x = x.repeat(1,3,1,2)
    return x * 2.0 - 1.0

def _clip_unit_from_bhwc(bhwc_255: torch.Tensor, device) -> torch.Tensor:
    """
    Uses YOUR get_latent_representation() to fetch CLIP features, then L2-normalizes.
    Your current implementation returns [1, D]; this wrapper keeps that interface.
    """
    z = get_latent_representation(bhwc_255, device)            # [1, D]
    z = z / (z.norm(dim=-1, keepdim=True) + 1e-12)             # unit-norm
    return z                                                   # [1, D]


def _combined_similarity_and_components(curr_bhwc_255: torch.Tensor,
                                        targ_bhwc_255: torch.Tensor,
                                        device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes:
      - ssim_gr    : MS-SSIM(C_t, T)          (B,), ↑ better
      - lpips_gr   : 1 - LPIPS(C_t, T)        (B,), ↑ better
      - clip_gr    : CLIP cosine(C_t, T)      (B,), ↑ better
      - S_curr     : weighted combo           (B,), ↑ better
    Returns (S_curr, ssim_gr, lpips_gr, clip_gr).
    """
    global TARGET_SSIM_NCHW, TARGET_LPIPS_NCHW, TARGET_CLIP_UNIT

    LP = _get_lpips(device)

    # --- SSIM ---
    curr_ssim = _prep_ssim(curr_bhwc_255).to(device)                     # (B,C,H,W) [0,1]
    if TARGET_SSIM_NCHW is None:
        TARGET_SSIM_NCHW = _prep_ssim(targ_bhwc_255).to(device)
    ssim_gr = ms_ssim(curr_ssim, TARGET_SSIM_NCHW, data_range=1.0, size_average=False)  # (B,)

    # --- LPIPS ---
    curr_lpips = _prep_lpips(curr_bhwc_255).to(device)
    if TARGET_LPIPS_NCHW is None:
        TARGET_LPIPS_NCHW = _prep_lpips(targ_bhwc_255).to(device)
    with torch.no_grad():
        lp = LP(curr_lpips, TARGET_LPIPS_NCHW).view(-1)  # lower = better
    lpips_gr = 1.0 - lp.clamp(0.0, 1.0)                  # convert to ↑ better

    # --- CLIP cosine ---
    if TARGET_CLIP_UNIT is None:
        TARGET_CLIP_UNIT = _clip_unit_from_bhwc(targ_bhwc_255, device)   # [1,D]
    curr_clip_unit = _clip_unit_from_bhwc(curr_bhwc_255, device)         # [1,D]
    with torch.no_grad():
        clip_gr = (curr_clip_unit * TARGET_CLIP_UNIT).sum(dim=-1).view(-1)  # (1,) -> (B=1,)
        clip_gr = clip_gr.clamp(-1.0, 1.0)

    # --- Weighted combo ---
    S_curr = W_SSIM*ssim_gr + W_LPIPS*lpips_gr + W_CLIP*clip_gr          # (B,)

    return S_curr, ssim_gr, lpips_gr, clip_gr

# Main reward
def calculate_reward(prev_canvas, current_canvas, target_canvas, device,
                     prev_prev_idx, prev_idx, current_idx, center,
                     edge_map=None, current_episode=None, current_step=None, segments_map=None):
    """
    HYBRID reward with CLIP+LPIPS+MS-SSIM.
    Logs:
      - reward_plots/IR_progress
      - reward_plots/GR_similarity
      - reward_plots/total_hybrid
      - reward_plots/gr_ssim        (RAW component curve)
      - reward_plots/gr_lpips_sim   (RAW component curve)
      - reward_plots/gr_clip_cos    (RAW component curve)
    Returns: float (mean over batch)
    """
    global CACHED_S_PREV, LAST_EPISODE_FOR_CACHE

    # Ensure float32 in [0,255]
    curr_bhwc = current_canvas.float()
    targ_bhwc = target_canvas.float()

    with torch.no_grad():
        # Reset caches at new episode or at step 0
        if (LAST_EPISODE_FOR_CACHE != int(current_episode)) or (int(current_step) == 0):
            CACHED_S_PREV = None
            LAST_EPISODE_FOR_CACHE = int(current_episode)

        # --- Combined similarity + components ---
        S_curr, ssim_gr, lpips_gr, clip_gr = _combined_similarity_and_components(curr_bhwc, targ_bhwc, device)  # (B,)*4

        # --- IR (progress) and GR (current) ---
        if CACHED_S_PREV is None:
            IR = torch.zeros_like(S_curr)
        else:
            IR = S_curr - CACHED_S_PREV
        GR = S_curr

        # --- Hybrid reward ---
        hybrid = LAMBDA_IR * IR + (1.0 - LAMBDA_IR) * GR

        # Optional: small penalty for repeating the same nail
        if prev_idx == current_idx:
            hybrid = hybrid - 0.5

        # Update cache
        CACHED_S_PREV = S_curr.detach()

        # --- Logging (global step across episodes) ---
        STEPS_PER_EP = int(config.get("max_strokes", 5000))
        global_step = int(current_episode) * STEPS_PER_EP + int(current_step)

        wb_log({
            # Totals
            "reward_plots/IR_progress":   IR.mean().item(),
            "reward_plots/GR_similarity": GR.mean().item(),
            "reward_plots/total_hybrid":  hybrid.mean().item(),

            # RAW component curves you asked for
            "reward_plots/gr_ssim":       ssim_gr.mean().item(),       # ~[0,1], ↑ better
            "reward_plots/gr_lpips_sim":  lpips_gr.mean().item(),      # ~[0,1], ↑ better
            "reward_plots/gr_clip_cos":   clip_gr.mean().item(),       # [-1,1], ↑ better
        }, step=global_step)

    return hybrid.mean().item()


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
