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
from torchvision.transforms.functional import to_pil_image
from dreamsim import dreamsim
from typing import Optional

# ----------------------- W&B helper -----------------------
_IS_MAIN = int(os.environ.get("RANK", "0")) == 0
def wb_log(data: dict, step: Optional[int] = None):
    import wandb
    if _IS_MAIN and getattr(wandb, "run", None) is not None:
        wandb.log(data, step=step)

# ----------------------- Caches / Globals -----------------------
# MSE caches
TARGET_NORM = None             # (1, C, H, W) float32 in [0,1] on device
CACHED_NEGLOG_MSE_PREV = None  # (B,) on device

# DreamSim caches
DS_MODEL = None                # dreamsim model (on device, eval, frozen)
TARGET_DS_EMB = None           # (1, D) on device (normalized)
CACHED_DS_SIM_PREV = None      # (B,) on device

# Book-keeping
LAST_EPISODE_FOR_CACHE = -1
EP_DS_START = None

# ----------------------- DreamSim loader -----------------------
def load_dreamsim_model(device, dreamsim_type: Optional[str] = None):
    """
    Loads DreamSim (once per rank), returns the global model.
    dreamsim_type: None (ensemble) or 'dino_vitb16' / 'openclip_vitb32' / 'clip_vitb32'
    """
    global DS_MODEL
    if DS_MODEL is not None:
        return DS_MODEL

    from dreamsim import dreamsim  # import lazily so your env doesn't pay cost unless needed
    cache_dir = os.environ.get("DREAMSIM_CACHE")
    if dreamsim_type:
        model, _preprocess = dreamsim(pretrained=True, device=device, dreamsim_type=dreamsim_type, cache_dir=cache_dir)
    else:
        model, _preprocess = dreamsim(pretrained=True, device=device, cache_dir=cache_dir)

    # freeze + eval
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    DS_MODEL = model
    return DS_MODEL


# ----------------------- Pure-tensor preprocess (GPU) -----------------------
@torch.no_grad()
def _to_bchw_01_from_bhwc(bhwc, device):
    """
    bhwc: (B, H, W, C) uint8 [0..255] or float [0..255]/[0..1], on any device
    return: (B, C, H, W) float32 in [0,1] on target device
    """
    x = bhwc.to(device)
    if x.dtype != torch.float32:
        x = x.float()
    # scale to [0,1] if it looks like 0..255
    if x.max() > 1.5:
        x = x / 255.0
    # BHWC -> BCHW
    x = x.permute(0, 3, 1, 2).contiguous()
    # 1-ch grayscale -> RGB
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {x.shape[1]}")
    return x.clamp_(0, 1)


@torch.no_grad()
def _ds_preprocess_tensor(bchw_01):
    """
    Resize to 224x224 on GPU for DreamSim. DreamSim’s own preprocess
    mainly resizes/normalizes images in [0,1]. We keep [0,1] and
    do only the resize here to avoid CPU hops.
    """
    return F.interpolate(bchw_01, size=(224, 224), mode="bilinear", align_corners=False)


# ----------------------- DreamSim embedding / distance (GPU) -----------------------
@torch.no_grad()
def _ds_embed_bhwc(bhwc, device):
    """Embed a batch (B,H,W,C) via DreamSim, entirely on GPU."""
    model = load_dreamsim_model(device)
    x = _to_bchw_01_from_bhwc(bhwc, device)
    x = _ds_preprocess_tensor(x)          # (B,3,224,224) on device
    emb = model.embed(x)                  # (B, D), normalized
    return emb


@torch.no_grad()
def _ds_distance_to_cached_target(curr_bhwc, device):
    """
    Fast path: use cached TARGET_DS_EMB (1,D).
    Returns cosine distance: (B,), lower is more similar.
    """
    global TARGET_DS_EMB
    assert TARGET_DS_EMB is not None, "TARGET_DS_EMB is None; call _ds_set_target_embed_once first."
    x_emb = _ds_embed_bhwc(curr_bhwc, device)                      # (B, D)
    # 1 - cosine_similarity((B,D),(1,D)) -> (B,)
    return (1.0 - F.cosine_similarity(x_emb, TARGET_DS_EMB.expand_as(x_emb), dim=-1))


@torch.no_grad()
def _ds_set_target_embed_once(target_bhwc, device):
    """Compute & cache target embedding once per episode. target_bhwc: (B,H,W,C), use sample 0."""
    global TARGET_DS_EMB
    if TARGET_DS_EMB is not None:
        return
    model = load_dreamsim_model(device)
    t = _to_bchw_01_from_bhwc(target_bhwc[:1], device)  # (1,C,H,W)
    t = _ds_preprocess_tensor(t)                        # (1,3,224,224)
    TARGET_DS_EMB = model.embed(t)                      # (1,D), normalized


# ----------------------- Reward function (GPU-only) -----------------------
@torch.no_grad()
def calculate_reward(prev_canvas, current_canvas, target_canvas, device,
                     prev_prev_idx, prev_idx, current_idx, center,
                     edge_map=None, current_episode=None, current_step=None, segments_map=None):
    """
    DreamSim + MSE reward (all on GPU, batched).

    Per-step reward (IR-only):
        ir_mse_coef * IR[-log(MSE)] + ir_ds_coef * IR[-DreamSimDistance]
    Terminal bonus (last step only):
        gr_ds_terminal_coef * max(0, DreamSimDist_start - DreamSimDist_final)

    Higher is better. No clipping.
    """
    # ---------- config ----------
    same_nail_penalty = float(config.get("same_nail_penalty", 0.05))
    steps_per_ep      = int(config.get("max_strokes", 5000))
    dreamsim_type     = config.get("dreamsim_type", None)

    # IR scalers (make signal large enough to matter)
    ir_mse_coef       = float(config.get("ir_mse_coef", 1.5e4))   # boosts tiny ~3e-4 deltas → ~4–5 reward
    ir_ds_coef        = float(config.get("ir_ds_coef",  2.0e2))   # DreamSim IR spikes (~1e-3) → ~0.2 reward

    # Terminal bonus scale (episode-level improvement)
    gr_ds_term        = float(config.get("gr_ds_terminal_coef", 50.0))

    global TARGET_NORM, CACHED_NEGLOG_MSE_PREV
    global DS_MODEL, TARGET_DS_EMB, CACHED_DS_SIM_PREV
    global LAST_EPISODE_FOR_CACHE, EP_DS_START

    ep = int(current_episode)
    st = int(current_step)

    # ---------- per-episode resets ----------
    if (LAST_EPISODE_FOR_CACHE != ep) or (st == 0):
        CACHED_NEGLOG_MSE_PREV = None
        CACHED_DS_SIM_PREV     = None
        TARGET_NORM            = None
        TARGET_DS_EMB          = None
        EP_DS_START            = None
        LAST_EPISODE_FOR_CACHE = ep

    # ---------- ensure DreamSim is loaded ----------
    load_dreamsim_model(device, dreamsim_type=dreamsim_type)

    # ---------- MSE path (BCHW in [0,1]) ----------
    curr_bchw = _to_bchw_01_from_bhwc(current_canvas, device)             # (B,C,H,W)
    if TARGET_NORM is None:
        TARGET_NORM = _to_bchw_01_from_bhwc(target_canvas[:1], device)    # (1,C,H,W)
    target_bchw = TARGET_NORM.expand(curr_bchw.shape[0], -1, -1, -1)

    mse_curr        = F.mse_loss(curr_bchw, target_bchw, reduction="none").flatten(1).mean(1)  # (B,)
    neglog_mse_curr = -torch.log(mse_curr)                                                     # (B,)

    if CACHED_NEGLOG_MSE_PREV is None:
        ir_mse_raw = torch.zeros_like(neglog_mse_curr, device=device)
    else:
        ir_mse_raw = neglog_mse_curr - CACHED_NEGLOG_MSE_PREV
    CACHED_NEGLOG_MSE_PREV = neglog_mse_curr.detach()

    # ---------- DreamSim path (cached target embedding) ----------
    _ds_set_target_embed_once(target_canvas, device)
    ds_dist_curr = _ds_distance_to_cached_target(current_canvas, device)   # (B,) lower=better
    ds_sim_curr  = -ds_dist_curr                                           # higher=better

    if CACHED_DS_SIM_PREV is None:
        ir_ds_raw = torch.zeros_like(ds_sim_curr, device=device)
    else:
        ir_ds_raw = ds_sim_curr - CACHED_DS_SIM_PREV
    CACHED_DS_SIM_PREV = ds_sim_curr.detach()

    # Record episode start distance (mean over batch) on step 0
    if st == 0:
        EP_DS_START = ds_dist_curr.mean().item()

    # ---------- per-step reward (IR only) ----------
    reward_step = (
        ir_mse_coef * ir_mse_raw.mean().item()
      + ir_ds_coef  * ir_ds_raw.mean().item()
    )

    # penalty for staying on the same nail
    penalty = same_nail_penalty if (int(prev_idx) == int(current_idx)) else 0.0
    reward_step -= penalty

    # ---------- terminal bonus (episode improvement in DreamSim distance) ----------
    terminal_bonus = 0.0
    if int(st) == (steps_per_ep - 1) and (EP_DS_START is not None):
        ds_improve    = max(0.0, EP_DS_START - ds_dist_curr.mean().item())  # positive if final is closer
        terminal_bonus = gr_ds_term * ds_improve
        reward_step   += terminal_bonus

    # ---------- logging ----------
    global_step = ep * steps_per_ep + st
    wb_log({
        "reward/mse_curr":               mse_curr.mean().item(),
        "reward/dreamsim_dist_curr":     ds_dist_curr.mean().item(),
        "reward/ir_mse_raw":             ir_mse_raw.mean().item(),
        "reward/ir_ds_raw":              ir_ds_raw.mean().item(),
        "reward/gr_ds_raw":              ds_sim_curr.mean().item(),   # = -distance
        "reward/penalty":                penalty,
        "reward/terminal_bonus":         terminal_bonus,
        "reward/ds_start":               0.0 if EP_DS_START is None else EP_DS_START,
        "reward/total_step":             reward_step,
        "reward/ir_mse_coef":            ir_mse_coef,
        "reward/ir_ds_coef":             ir_ds_coef,
        "reward/gr_ds_terminal_coef":    gr_ds_term,
    }, step=global_step)

    return reward_step


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
