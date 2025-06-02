"""
Training Script

Handles the training loop for the DDPG painting agent. It:
- Initializes the environment and models
- Runs episodes where the agent paints using predicted actions
- Stores experiences (canvas, prev_action, action, next_canvas, reward, done)
- Periodically updates the Actor and Critic using replay buffer samples
- Applies soft target updates
- Logs training progress and saves checkpoints

Inputs:
    - Canvas (current canvas image)
    - Target image (fixed goal)
    - Previous action (used to generate the next one)

Outputs:
    - Trained Actor and Critic models saved to disk"""

import os
import torch
import numpy as np
from collections import deque
import csv
from env.environment import PaintingEnv
from models.actor import Actor
from models.critic import Critic
from models.ddpg import DDPGAgent
from utils.noise import OUNoise
from utils.replay_buffer import ReplayBuffer
import cv2
from env.canvas import save_canvas
import torch.profiler


def train(config):
    """
    Full training pipeline for the DDPG agent using canvas drawing environment.
    Initializes all components and trains the agent over multiple episodes.

    Args:
        config (dict): Configuration dictionary containing hyperparameters and paths.
    """

    # Initialize environment and load target image
    env = PaintingEnv(
        target_image_path=config["target_image_path"],
        canvas_size=config["canvas_size"],
        canvas_channels=config["canvas_channels"],
        max_strokes=config["max_strokes"],
        device=config["device"]
    )

    # Load target image
    # env.target_image = (H, W, C)
    # target_image = (B, C, H, W)
    target_image = torch.from_numpy(env.target_image).permute(2, 0, 1).unsqueeze(0).float().to(config["device"])

    # Initialize Actor & Critic networks (main and target)
    actor = Actor(
        image_encoder_model=config["model_name"],
        actor_network_input=config["action_dim"],
        in_channels=config["canvas_channels"],
        out_neurons=config["action_dim"]
    )
    critic = Critic(
        image_encoder_model=config["model_name"],
        actor_network_input=config["action_dim"]*2,
        in_channels=config["canvas_channels"],
        out_neurons=1
    )
    actor_target = Actor(
        image_encoder_model=config["model_name"],
        actor_network_input=config["action_dim"],
        out_neurons=config["action_dim"],
        in_channels=config["canvas_channels"]
    )
    critic_target = Critic(
        image_encoder_model=config["model_name"],
        actor_network_input=config["action_dim"]*2,
        in_channels=config["canvas_channels"],
        out_neurons=1
    )

    # Sync target networks
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    # Set optimizers
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=config["actor_lr"])
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config["critic_lr"])

    # Initialize replay buffer and noise process
    #replay_buffer = ReplayBuffer(config["buffer_size"])
    replay_buffer = ReplayBuffer(capacity=config["replay_buffer_capacity"], device=config["device"])
    noise = OUNoise(config["action_dim"])

    # Build the DDPG agent
    agent = DDPGAgent(
        actor=actor,
        critic=critic,
        actor_target=actor_target,
        critic_target=critic_target,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        replay_buffer=replay_buffer,
        noise=noise,
        config=config
    )

    # Setup logging
    os.makedirs("logs", exist_ok=True)

    scores_window = deque(maxlen=100) # keep track of last 100 episodes, more stable than 3 

    scores = []

    # Exploration noise control
    noise_scale = config["initial_noise_scale"]
    noise_decay = config["noise_decay"]

    # Profiling
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profiler'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for episode in range(config["episodes"]):
    # Main training loop
    #for episode in range(config["episodes"]):
            canvas = env.reset() # canvas = (H, W, C) tensor
            # random initial point is set above in env.reset
            prev_action = np.zeros(config["action_dim"], dtype=np.float32)
            # Initialize previous action with current point
            prev_action[0] = env.current_point[0]
            prev_action[1] = env.current_point[1]

            # Logs input tensor shapes at the start of each episode
            """with open("logs/input_shapes.log", "a") as f:
                f.write(f"Episode {episode + 1}\n")
                f.write(f"Canvas shape: {canvas.shape}\n")
                f.write(f"Target image shape: {target_image.shape}\n")
                f.write(f"Prev action shape: {prev_action.shape}\n\n")"""

            episode_reward = 0
            done = False

            # Episode step loop
            while not done:
                #print(f"in train.py- canvas shape 1: {canvas.shape}, type: {type(canvas)}")
                if isinstance(canvas, np.ndarray) and canvas.ndim == 3:
                    # NumPy: (H, W, C) → (C, H, W) → (1, C, H, W)
                    canvas = np.transpose(canvas, (2, 0, 1))
                    canvas = canvas[np.newaxis, :, :, :]
                elif isinstance(canvas, torch.Tensor) and canvas.ndim == 3:
                    # Torch: (H, W, C) → (C, H, W) → (1, C, H, W)
                    canvas = canvas.permute(2, 0, 1).unsqueeze(0).contiguous()
                #print(f"in train.py- canvas shape 2: {canvas.shape}, type: {type(canvas)}")

                # Convert to tensor and move to device
                # canvas_tensor = torch.from_numpy(canvas).float().to(config["device"])
                if isinstance(canvas, np.ndarray):
                    canvas_tensor = torch.from_numpy(canvas).float().to(config["device"])
                else:
                    canvas_tensor = canvas.float().to(config["device"])
                
                # Get action -> ddpg.py -> actor.py -> merge networks -> image encoder
                # prev_action = (6,) -> later converted to tensor and (6,1) in select_action()
                """print(f"in train.py- canvas_tensor shape: {canvas_tensor.shape}, type: {type(canvas_tensor)}")  # (1, C, H, W)
                print(f"in train.py- target_image shape: {target_image.shape}, type: {type(target_image)}")      # (1, C, H, W)
                print(f"in train.py- prev_action shape: {prev_action.shape}, type: {type(prev_action)}")         # (6,)"""
                action = agent.act(canvas_tensor, target_image, prev_action, noise_scale)

                # Logs action values per step
                """with open("logs/action_logs.csv", "a", newline="") as file:
                    writer = csv.writer(file)
                    if episode == 0 and env.used_strokes == 1:
                        writer.writerow(["episode", "step", "action_values"])
                    writer.writerow([episode + 1, env.used_strokes, action.tolist()])"""

                # Apply action in the environment
                next_canvas, reward, done = env.step(action)

                # Logs environment transitions including shapes and rewards
                """with open("logs/env_transitions.log", "a") as f:
                    f.write(f"Episode {episode + 1}, Step {env.used_strokes}:\n")
                    f.write(f"Action taken: {action.tolist()}\n")
                    f.write(f"Reward: {reward}, Done: {done}\n")
                    f.write(f"Canvas shape: {canvas.shape}, Next canvas shape: {next_canvas.shape}\n\n")"""

                # next_canvas from env.step = (C, H, W)
                # canvas_tensor = (B, C, H, W)
                if isinstance(next_canvas, np.ndarray):
                    canvas_tensor = torch.from_numpy(next_canvas).float().unsqueeze(0).to(config["device"])
                else:
                    canvas_tensor = next_canvas.float().unsqueeze(0).to(config["device"])

                # Store experience in replay buffer
                # replay_buffer.store(canvas, prev_action, action, next_canvas, reward, done)
                # replay buffer store gpu tensors directly now
                #def to_numpy(x):
                #    return x.detach().cpu().numpy() if torch.is_tensor(x) else x
                
                # canvas = (B, C, H, W)
                # canvas_for_buffer = (C, H, W) using this so that canvas sizes don't get messed up when doing canvas = next canvas
                canvas_for_buffer = canvas.squeeze(0) if isinstance(canvas, torch.Tensor) else np.squeeze(canvas, axis=0)

                # canvas_for_buffer = (C, H, W)
                # next_canvas = (C, H, W)
                replay_buffer.store(
                    canvas_for_buffer,
                    prev_action,
                    action,
                    next_canvas,
                    reward,
                    done
                )

                # keep track of epiosde and step numbers for update actor/critic
                agent.episode = episode + 1
                agent.step = env.used_strokes

                # Train the agent using sampled experiences
                agent.train(target_image)

                # log to make sure actor/critic are updated every step 
                """with open("logs/training_calls.log", "a") as f:
                    f.write(f"Episode {episode + 1}, Step {env.used_strokes}: agent.train() called\n")"""

                # Move to next state
                # canvas = (B, C, H, W)
                # canvas_tensor = (B, C, H, W)
                canvas = canvas_tensor
                
                # Save step frame every 50th stroke for select episodes
                if (episode + 1) in [1, 1000, 10000, 25000, 50000] and env.used_strokes % config["save_every_step"] == 0:
                    step_dir = f"step_outputs/episode_{episode + 1}"
                    os.makedirs(step_dir, exist_ok=True)
                    # sample path: step_outputs/episode_25000/episode_25000_step_00150.png
                    save_path = os.path.join(step_dir, f"step_{env.used_strokes}.png")
                    #save_canvas(canvas, save_path)
                    cv2.imwrite(save_path, canvas.astype("uint8"))
                prev_action = action
                episode_reward += reward

                # Log step reward
                """with open("logs/step_rewards.csv", mode="a", newline="") as file:
                    writer = csv.writer(file)
                    if episode == 0 and env.used_strokes == 1:  # write header only once
                        writer.writerow(["episode", "step", "reward"])
                    writer.writerow([episode + 1, env.used_strokes, reward])"""
                print(f"Episode {episode + 1} | Step {env.used_strokes} | Step Reward: {reward}")


            # Decay exploration noise
            noise_scale *= noise_decay
            scores.append(episode_reward)
            # Log episode reward
            with open("logs/episode_rewards.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                if episode == 0:  # Write header only once
                    writer.writerow(["episode", "total_reward"])
                writer.writerow([episode + 1, episode_reward])
            scores_window.append(episode_reward)

            # Logs the running average reward over last 100 episodes
            """with open("logs/running_avg.csv", "a", newline="") as file:
                writer = csv.writer(file)
                if episode == 0:
                    writer.writerow(["episode", "running_avg_100"])
                writer.writerow([episode + 1, np.mean(scores_window)])"""

            # Progress log
            print(
                f"Episode {episode + 1} | Reward: {episode_reward} | Running Avg(100): {np.mean(scores_window)}")

            # Save model checkpoints every 100 episodes
            if (episode + 1) % config["save_every_episode"] == 0:
                os.makedirs("trained_models", exist_ok=True)
                torch.save(actor.state_dict(),
                        f"trained_models/actor_{episode + 1}.pth")
                torch.save(critic.state_dict(),
                        f"trained_models/critic_{episode + 1}.pth")
                print(f"Saved model at episode {episode + 1}")

        print("Training complete.")
