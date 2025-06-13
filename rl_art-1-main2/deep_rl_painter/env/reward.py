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

# target_latent
TARGET_LATENT = None
# model
# preprocess
CLIP_MODEL = None
PREPROCESS = None
# model, preprocess = None, None


#def calculate_reward(current_canvas, target_canvas, device):
def calculate_reward(current_canvas, target_canvas, device,
                     prev_prev_point, prev_point, current_point, center):
    """
    Calculates the reward based on the chosen reward function.

    Args:
        prev_canvas (torch.Tensor): The previous canvas state (shape: [batch_size, channels, height, width]).
        current_canvas (torch.Tensor): The current canvas state (shape: [batch_size, channels, height, width]).
        target_canvas (torch.Tensor): The target canvas state (shape: [batch_size, channels, height, width]).
    Returns:
        torch.Tensor: The calculated reward (shape: [batch_size, 1]).
    """
    # Using CLIP for calculating cosine similarity
    global TARGET_LATENT
    latent = get_latent_representation(current_canvas, device)
    if TARGET_LATENT is None:
        TARGET_LATENT = get_latent_representation(target_canvas, device)

    mse_score_current = mse_loss(latent, TARGET_LATENT) # current_canvas - this mse needs to be lower
    mse_reward = mse_score_current * 1000

    # To-do: implement aux reward based on stroke length?
    # Auxiliary reward based on angles
    aux_reward = calculate_auxiliary_reward(prev_prev_point, prev_point, current_point, center)

    # Combine rewards
    total_reward = -mse_reward + aux_reward    

    #return -mse_reward 
    return total_reward 

def calculate_auxiliary_reward(prev_prev_point, prev_point, current_point, center):
    """
    Computes an auxiliary reward based on angle change between consecutive vectors from the center (0,0).
    Penalizes if angle2 < angle1; rewards if angle2 > angle1.

    Args:
        prev_prev_point (np.array or list or None): previous previous point (x, y)
        prev_point (np.array or list or None): previous point (x, y)
        current_point (np.array or list): current point (x, y)

    Returns:
        float: Auxiliary reward value.
    """
    # if not enough points yet (steps 1 & 2) → no auxiliary reward
    if prev_prev_point is None or prev_point is None:
        return 0.0

    # convert points to torch vectors
    ## v1 = torch.tensor(prev_prev_point, dtype=torch.float32)
    ## the above represents vector from top left corner (canvas coordinates 0,0) to point on cirlce
    # vectors from center of circle (0,0) to point on circle 
    v1 = torch.tensor(prev_prev_point - center, dtype=torch.float32)
    v2 = torch.tensor(prev_point - center, dtype=torch.float32)
    v3 = torch.tensor(current_point - center, dtype=torch.float32)

    # compute angles
    angle1 = compute_angle_between_vectors(v1, v2)
    angle2 = compute_angle_between_vectors(v2, v3)

    # compute ratio of angles
    # To-do: what other ways to compare angle 1 and angle 2???????
    ratio = angle2 / (angle1 + 1e-6)  # add small epsilon to avoid division by 0

    # scale - how much of the aux_reward is to be added to main reward
    # To-do: figure out a proper value!!!!!
    scale = 5.0 

    #aux_reward = (ratio - 1.0) * scale
    # aux reward to encourage relatively larger angles b/w consecutive points
    # if ratio ≈ 1 → log(1) = 0 → no reward
    # if ratio > 1 → small positive reward
    # if ratio < 1 → small negative reward
    aux_reward = math.log(ratio + 1e-6) * scale # to stop the reward from exploding

    return aux_reward 


def compute_angle_between_vectors(v1, v2):
    """
    Computes the angle (in radians) between two vectors v1 and v2.

    Args:
        v1 (torch.Tensor): First vector (x, y)
        v2 (torch.Tensor): Second vector (x, y)

    Returns:
        torch.Tensor: Angle in radians.
    """
    # Dot product between the vectors
    dot_product = torch.dot(v1, v2)

    # Norm (length) of each vector
    norm_v1 = torch.norm(v1)
    norm_v2 = torch.norm(v2)

    # Cosine of the angle between vectors
    cos_theta = dot_product / (norm_v1 * norm_v2 + 1e-8)  # add epsilon for safety

    # Clamp cos_theta to valid range [-1, 1] to avoid NaN due to floating point errors
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # Angle in radians
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
