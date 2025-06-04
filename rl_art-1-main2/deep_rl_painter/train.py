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
import time


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
        #actor_network_input=config["action_dim"],
        actor_network_input=2, # x,y
        in_channels=config["canvas_channels"],
        out_neurons=config["action_dim"]
    )
    critic = Critic(
        image_encoder_model=config["model_name"],
        #actor_network_input=config["action_dim"]*2,
        actor_network_input=4,
        in_channels=config["canvas_channels"],
        out_neurons=1
    )
    actor_target = Actor(
        image_encoder_model=config["model_name"],
        #actor_network_input=config["action_dim"],
        actor_network_input=2, # x,y
        out_neurons=config["action_dim"],
        in_channels=config["canvas_channels"]
    )
    critic_target = Critic(
        image_encoder_model=config["model_name"],
        #actor_network_input=config["action_dim"]*2,
        actor_network_input=4,
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

    #scores = []

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
            # Send only (x, y) point to the Actor
            prev_point = prev_action[:2].copy()
            first_step = True  # Flag to check for first loop

            # Logs input tensor shapes at the start of each episode
            """with open("logs/input_shapes.log", "a") as f:
                f.write(f"Episode {episode + 1}\n")
                f.write(f"Canvas shape: {canvas.shape}\n")
                f.write(f"Target image shape: {target_image.shape}\n")
                f.write(f"Prev action shape: {prev_action.shape}\n\n")"""

            episode_reward = 0
            done = False

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

            # Episode step loop
            while not done:

                # Prepare actor input from previous point
                if first_step:
                    actor_prev_input = prev_point  # already a point on circle (from env.current_point -> env.random_circle_point)
                    first_step = False  # flip the flag
                else: # to get x,y on the circle, instead of raw directional x,y
                    direction = prev_action[:2]
                    norm = np.linalg.norm(direction) + 1e-8
                    unit_vector = direction / norm
                    actor_prev_input = np.array([
                        env.center[0] + unit_vector[0] * env.radius,
                        env.center[1] + unit_vector[1] * env.radius], dtype=np.float32)

                #print(f"Prev Action: {prev_action}")
                #print(f"Actor Prev Input : {actor_prev_input}")

                # Get action -> ddpg.py -> actor.py -> merge networks -> image encoder
                # prev_action = (6,) -> later converted to tensor and (6,1) in select_action()
                #action = agent.act(canvas_tensor, target_image, prev_action, noise_scale)
                
                # actor_prev_input = (2,) -> later converted to tensor and (2,1) in select_action()
                # action is here numpy array (6,)
                t0 = time.time()
                action = agent.act(canvas_tensor, target_image, actor_prev_input, noise_scale)
                t1 = time.time()
                total = t1-t0
                print("Action Time: ", total)

                # Logs action values per step
                """with open("logs/action_logs.csv", "a", newline="") as file:
                    writer = csv.writer(file)
                    if episode == 0 and env.used_strokes == 1:
                        writer.writerow(["episode", "step", "action_values"])
                    writer.writerow([episode + 1, env.used_strokes, action.tolist()])"""

                # Apply action in the environment
                # actor_current_input contains the current action's normalised x,y values
                t2 = time.time()
                next_canvas, reward, done, actor_current_input = env.step(action)
                t3 = time.time()
                total1 = t3-t2
                print("Rendering Time: ", total1)
                #print(f"Action: {action}")
                #print(f"Actor Current Input (x, y): {actor_current_input}")

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
                actor_prev_input = np.round(actor_prev_input, 6).astype(np.float32)
                actor_current_input = np.round(actor_current_input, 6).astype(np.float32)
                replay_buffer.store(
                    canvas_for_buffer,
                    #prev_action,
                    actor_prev_input,
                    #action,
                    actor_current_input,
                    next_canvas,
                    reward,
                    done
                )

                """print(
                f"[ReplayBuffer Store] canvas.shape={canvas_for_buffer.shape}, "
                f"prev_action=({actor_prev_input[0]:.2f}, {actor_prev_input[1]:.2f}), "
                f"action=({actor_current_input[0]:.2f}, {actor_current_input[1]:.2f}), "
                f"next_canvas.shape={next_canvas.shape}, "
                f"reward={reward:.4f}, done={done}"
                )"""

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
                if (episode + 1) in [1, 2, 50, 100, 50000] and env.used_strokes % config["save_every_step"] == 0:
                    step_dir = f"step_outputs/episode_{episode + 1}"
                    os.makedirs(step_dir, exist_ok=True)
                    # sample path: step_outputs/episode_25000/episode_25000_step_00150.png
                    save_path = os.path.join(step_dir, f"step_{env.used_strokes}.png")
                    # canvas is (B, C, H, W)
                    """img_tensor = canvas[0].detach().cpu()  # Take the first image in the batch
                    if img_tensor.shape[0] == 1:
                        # Grayscale: (1, H, W) → (H, W)
                        img_np = img_tensor.squeeze(0).numpy().astype("uint8")
                    else:
                        # RGB: (3, H, W) → (H, W, 3)
                        img_np = img_tensor.permute(1, 2, 0).numpy().astype("uint8")
                    cv2.imwrite(save_path, img_np)"""
                    #cv2.imwrite(save_path, canvas.detach().cpu().numpy().astype("uint8"))
                    canvas_to_save = canvas[0]  # (C, H, W)
                    canvas_to_save = canvas_to_save.permute(1, 2, 0).contiguous()  # → (H, W, C)
                    save_canvas(canvas_to_save, save_path)
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
            #scores.append(episode_reward) - not being used anywhere
            # Log episode reward
            """with open("logs/episode_rewards.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                if episode == 0:  # Write header only once
                    writer.writerow(["episode", "total_reward"])
                writer.writerow([episode + 1, episode_reward])"""
            scores_window.append(episode_reward)
            running_avg = torch.stack(list(scores_window)).mean()

            # Logs the running average reward over last 100 episodes
            """with open("logs/running_avg.csv", "a", newline="") as file:
                writer = csv.writer(file)
                if episode == 0:
                    writer.writerow(["episode", "running_avg_100"])
                writer.writerow([episode + 1, np.mean(scores_window)])"""

            # Progress log
            print(
                f"Episode {episode + 1} | Reward: {episode_reward} | Running Avg(100): {running_avg.item():.2f}")

            # Save model checkpoints every 100 episodes
            if (episode + 1) % config["save_every_episode"] == 0:
                os.makedirs("trained_models", exist_ok=True)
                torch.save(actor.state_dict(),
                        f"trained_models/actor_{episode + 1}.pth")
                torch.save(critic.state_dict(),
                        f"trained_models/critic_{episode + 1}.pth")
                print(f"Saved model at episode {episode + 1}")

        print("Training complete.")
