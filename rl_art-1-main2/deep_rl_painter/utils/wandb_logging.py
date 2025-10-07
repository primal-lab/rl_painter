import wandb
import numpy as np
import os
import torch
import imageio

def log_canvas_video(episode, frames, fps=15, save_dir="logs/episode_videos"):
    """
    Creates and logs a GIF for an episode (grayscale-safe).
    Frames should be (H, W) or (H, W, 1).
    """
    os.makedirs(save_dir, exist_ok=True)
    gif_path = os.path.join(save_dir, f"episode_{episode + 1}.gif")

    processed_frames = []
    for img in frames:
        img = np.array(img, dtype=np.uint8)

        # ✅ Convert (H, W, 1) → (H, W)
        if img.ndim == 3 and img.shape[2] == 1:
            img = img.squeeze(axis=2)

        processed_frames.append(img)

    imageio.mimsave(gif_path, processed_frames, fps=fps)
    wandb.log({f"Episode_{episode + 1}_Video": wandb.Video(gif_path, fps=fps, format="gif")})


def log_step_to_table(episode_table, step, reward, canvas):
    """
    Adds a single step's data (step number, reward, and canvas image)
    to the W&B table.

    Args:
        episode_table (wandb.Table): The table object for the current episode.
        step (int): Current step number (1-based stroke count).
        reward (float): Reward for this step.
        canvas (torch.Tensor): (C, H, W) tensor of the current canvas.
    """
    img = canvas.detach().cpu().permute(1, 2, 0).numpy()  # (H, W, C)
    img = np.clip(img, 0, 255).astype(np.uint8)
    #img = 255 - img  # same inversion as save_canvas

    # ✅ Convert (H, W, 1) → (H, W)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(axis=2)

    episode_table.add_data(step, reward, wandb.Image(img))
