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

# target_latent
TARGET_LATENT = None
# model
# preprocess
CLIP_MODEL = None
PREPROCESS = None
# model, preprocess = None, None


#def calculate_reward(current_canvas, target_canvas, device):
def calculate_reward(current_canvas, target_canvas, device,
                     prev_prev_point, prev_point, current_point, center, edge_map=None):
    """
    Calculates the reward based on the chosen reward function.

    Args:
        prev_canvas (torch.Tensor): The previous canvas state (shape: [batch_size, channels, height, width]).
        current_canvas (torch.Tensor): The current canvas state (shape: [batch_size, channels, height, width]).
        target_canvas (torch.Tensor): The target canvas state (shape: [batch_size, channels, height, width]).
    Returns:
        torch.Tensor: The calculated reward (shape: [batch_size, 1]).
    """
    # CLIP for calculating cosine similarity
    global TARGET_LATENT
    latent = get_latent_representation(current_canvas, device)
    if TARGET_LATENT is None:
        TARGET_LATENT = get_latent_representation(target_canvas, device)

    mse_score_current = mse_loss(latent, TARGET_LATENT) # current_canvas - this mse needs to be lower
    mse_reward = mse_score_current * 10

    # auxiliary reward - modified
    aux_reward = calculate_auxiliary_reward(prev_prev_point, prev_point, current_point, center)

    # combine rewards
    total_reward = -mse_reward - aux_reward  

    # logging MSE and Aux rewards ---
    os.makedirs("logs/debug", exist_ok=True)
    log_file = "logs/debug/reward_breakdown.csv"

    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if os.stat(log_file).st_size == 0:  # write header only if file is empty
            writer.writerow(["mse_reward", "aux_reward"])
        mse_val = -mse_reward.item()
        aux_val = aux_reward.item() if isinstance(aux_reward, torch.Tensor) else aux_reward
        writer.writerow([mse_val, aux_val])

    # check if current stroke intersects with target image canny edges
    # if so, reward higher
    if edge_map is not None and prev_point is not None and current_point is not None:
        if stroke_intersects_edge(prev_point, current_point, edge_map):
            edge_bonus = 0.5 # TO-DO: Adjust
            total_reward += edge_bonus
    
    total_reward /= 2.0
    return total_reward 

def stroke_intersects_edge(start, end, edge_map, threshold=0.8):
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
    return np.mean(values) > threshold


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
        CLIP_MODEL, PREPROCESS = clip.load("ViT-B/32", device=device)

    if len(image.shape) == 4 and (image.shape[1] != 1 or image.shape[1] != 3):
        image = image.permute(0, 3, 1, 2)
    try:
        if len(image.shape) == 4:
            image = image[0]
        image = image.detach().cpu() if image.is_cuda else image
        image = PREPROCESS(transforms.ToPILImage()(image)).unsqueeze(0).to(device)

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        print("Image shape:", image.shape)
        return None

    with torch.no_grad():
        latent_representation = CLIP_MODEL.encode_image(image)
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
