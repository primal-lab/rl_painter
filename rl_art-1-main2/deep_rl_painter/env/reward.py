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

global target_latent
target_latent = None
global model, preprocess
model, preprocess = None, None
def calculate_reward(prev_canvas, current_canvas, target_canvas, device):
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
    global target_latent
    #!latent1 = get_latent_representation(prev_canvas, device)
    #!latent2 = get_latent_representation(current_canvas, device)
    #!target_latent = get_latent_representation(target_canvas, device)
    #import pdb
    #pdb.set_trace()
    #latent1 = get_latent_representation(prev_canvas, device)
    latent2 = get_latent_representation(current_canvas, device)
    if target_latent is None:
        target_latent = get_latent_representation(target_canvas, device)

    # Calculate cosine similarity
    # cosine_similarity_score_prev = calculate_cosine_similarity(
    #     latent1, target_latent)

    # cosine_similarity_score_current = calculate_cosine_similarity(
    #     latent2, target_latent)

    # cosine_similarity_reward = cosine_similarity_score_current - cosine_similarity_score_prev


    mse_score_current = mse_loss(latent2, target_latent) # current_canvas - this mse needs to be lower
    #mse_score_prev = mse_loss(latent1, target_latent) # prev_canvas
    #mse_reward = mse_score_current - mse_score_prev
    mse_reward = mse_score_current * 1000

    #print("Reward tensor (before negation):", mse_reward) # reward
    #print("Reward scalar (before negation):", mse_reward.item()) # reward.item()
    return -mse_reward  


def mse_loss(pred, target):
    """
    Calculates the Mean Squared Error (MSE) loss between two tensors.

    Args:
        pred (torch.Tensor): The predicted tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        torch.Tensor: The MSE loss value.
    """
    #!return torch.mean((pred - target) ** 2)
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
    t0 = time.time()
    global model, preprocess
    if model is None:
        model, preprocess = clip.load("ViT-B/32", device=device)
    t1 = time.time()
    total = t1-t0
    print("Model Loading time: ", total)
    # This can be optimized further by removing some of the preprocessing steps

    if len(image.shape) == 4 and (image.shape[1] != 1 or image.shape[1] != 3):
        image = image.permute(0, 3, 1, 2)
    try:
        if len(image.shape) == 4:
            image = image[0]
        #!image = preprocess(transforms.ToPILImage()(image)).unsqueeze(0).to(device)
        #image = preprocess(transforms.ToPILImage()(image.cpu())).unsqueeze(0).to(device)
        image = image.detach().cpu() if image.is_cuda else image
        image = preprocess(transforms.ToPILImage()(image)).unsqueeze(0).to(device)

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        print("Image shape:", image.shape)
        return None

    with torch.no_grad():
        latent_representation = model.encode_image(image)
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
