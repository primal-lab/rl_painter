"""
Configuration file for the painting agent.
This file contains all the hyperparameters and settings for the training and evaluation of the agent.

TODO:
- Add more parameters for the neural networks
"""

import torch
config = {
    # training setup
    # True for training, False for production mode
    "train": True,
    "seed": 42,

    "episodes": 5000,                        # change to 500 or more later
    "nails": 180,                            # number of nails

    # painting environment
    "target_image_path": "target_images/target_image_1.jpg",
    #"target_image_path": "target_images/target_image_test.png",
    "target_edges_path": "reward_target_versions/target_edges_1.png",  # canny edges of the target image
    "target_segments_path": "reward_target_versions/segmentation_map.png",  # segmentation_map of the target image

    "max_strokes": 2000,                    # max number of strokes per episode

    "max_strokes_per_step": 1,               # max number of strokes per step
    "max_total_length": 10000,
    "error_threshold": 10000.0,

    # model parameters
    "model_name": "resnet18",                # check models/image_encoder.py for available models
    "actor_lr": 1e-4,                        # 1 * 10^(-4)  
    "critic_lr": 1e-4,
    #"buffer_size": 100000, same thing as below
    "replay_buffer_capacity": 100000,

    "batch_size": 64,                       # if training is unstable (reward jumps/loss spikes), =32
    "gamma": 0.99,
    "tau": 0.005,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # 2D action space (x, y, r, g, b, w) - not anymore
    "action_dim": 1, # previous point index only
    # reward function (ssim, mse, perceptual)
    "reward_method": "CLIP_cosine_similarity",

    # exploration noise
    "initial_noise_scale": 0.5, #0.2 initially
    "noise_decay": 0.99, #0.995

    # saving

    "save_every_step": 50,
    "save_every_episode": 100,            
    "save_model_dir": "models",
    "save_model_name": "model.pth",
    "save_best_model": True,
    "save_best_model_dir": "best_models",
    "save_best_model_name": "best_model.pth",

    # Canvas parameters
    #"canvas_size": (224, 224),  # (height, width)
    "canvas_size": (1024, 1024),
    "canvas_channels": 1,  # 1 for grayscale, 3 for RGB
    "canvas_color": (0, 0, 0),  # black background
    "canvas_stroke_color": 255,  # white stroke

    # logging
    "logging": True,    # whether to log training progress
    "log_dir": "logs",
    "log_every": 10,
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "log_file": "training.log",

    # profiling
    # incorporate this into train.py later
    "profile": False,  # whether to profile the training process

    # resume from last saved model
    "resume": False,

}
