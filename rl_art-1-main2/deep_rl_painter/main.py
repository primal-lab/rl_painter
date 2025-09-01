"""
Main Entry Point for Deep RL Painter

This script:
- Parses command-line arguments
- Updates config dictionary
- Delegates full training to `train(config)` from train.py

TODO:
- Add logging in the project
- Add production mode
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import argparse
import torch
from config import config
from train import train
import random
import numpy as np

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Train Deep RL Painter")

    # Painting environment
    parser.add_argument('--target_image_path', type=str, default=config["target_image_path"], help='Path to the target image')
    parser.add_argument('--max_strokes', type=int, default=config["max_strokes"], help='Maximum number of strokes per episode')

    # Model parameters
    parser.add_argument('--model_name', type=str, default=config["model_name"], help='Model architecture')
    parser.add_argument('--action_dim', type=int, default=config["action_dim"], help='Dimension of the action space')
    parser.add_argument('--reward_method', type=str, default=config["reward_method"], help='Reward function')
    return parser.parse_args()

def main():
    """
    Main function to initialize environment, agent, and run the training loop,
    accepting all configuration parameters via command line.
    """
    args = parse_args()

    # Update config with command-line arguments
    config["target_image_path"] = args.target_image_path
    config["max_strokes"] = args.max_strokes
    config["model_name"] = args.model_name
    config["action_dim"] = args.action_dim
    config["reward_method"] = args.reward_method

    set_seed(config["seed"])

    # for k, v in config.items():
    #     print(f"{k}: {v}")

    if config["train"]:
        print("Training mode")
        # Initialize the training process
        train(config)
    else:
        print("Production mode")
        # Initialize the production process
        pass

if __name__ == "__main__":
    main()
