"""
Training Script

Handles the training loop for the DDPG painting agent. It:
- Initializes the environment and models
- Runs episodes where the agent paints using predicted actions
- Stores experiences (canvas, prev_action, action, next_canvas, reward, done)
- Periodically updates the Actor and Critic using replay buffer samples
- Applies soft target updates
- Logs training progress and saves checkpoints
"""

import os
import torch
import numpy as np
from collections import deque
from env.environment import PaintingEnv
from models.actor import Actor
from models.critic import Critic
from models.ddpg import DDPGAgent
from utils.noise import OUNoise
from utils.replay_buffer import ReplayBuffer
from env.canvas import save_canvas
import time
import csv
import glob
import re
import wandb
from utils.wandb_logging import log_canvas_video, log_step_to_table
from utils.weights_viz import (snapshot_vector, flat_params, grad_flow_mean, update_ratio_gpu)

# >>> DDP EDIT: import DDP
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist  # only used for rank/world helpers


def train(config):
    """
    Full training pipeline for the DDP agent using canvas drawing environment.
    """
    # >>> DDP EDIT: derive rank / is_main and world size from config
    rank = int(config.get("local_rank", 0))
    world_size = int(config.get("world_size", 1))
    is_main = (rank == 0)

    # >>> DDP EDIT: set and record local batch size (read this in ddpg.py)
    # If you keep global batch_size=128 and use 2 GPUs, batch_size_local=64
    if "batch_size" in config:
        config["batch_size_local"] = max(1, config["batch_size"] // max(1, world_size))

    device = config["device"]
    LOG_EVERY_STEPS = int(config.get("log_every_steps", 200))
    VIDEO_EVERY = int(config.get("video_every", 20))

    # >>> DDP EDIT: wandb only on main process
    if is_main:
        wandb.init(
            project="ddpg-painter",
            name="actor_2dft_10targets_(r=-log(mse)ir+gr)",  
            #hack_actor_fixed_rotate(r=delta(-log1p(-s+E))*100)_penalty=-0.5
            config=config
        )

    # Initialize env (uses per-rank device)
    env = PaintingEnv(
        target_image_path=config["target_image_path"],
        target_edges_path=config["target_edges_path"],
        canvas_size=config["canvas_size"],
        canvas_channels=config["canvas_channels"],
        max_strokes=config["max_strokes"],
        device=config["device"],
        target_segments_path=config["target_segments_path"]
    )

    # Load target image → (1, C, H, W) on this rank's device
    target_image = torch.from_numpy(env.target_image).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # ----- Build models -----
    actor = Actor(
        image_encoder_model=config["model_name"],
        image_encoder_model_2=config["model_name"],
        actor_network_input=config["nails"],  # one-hot prev point
        in_channels=config["canvas_channels"],
        out_neurons=config["nails"] + 5
    ).to(device)

    critic = Critic(
        image_encoder_model=config["model_name"],
        image_encoder_model_2=config["model_name"],
        actor_network_input=config["nails"],  # one-hot current point
        in_channels=config["canvas_channels"],
        out_neurons=1
    ).to(device)

    actor_target = Actor(
        image_encoder_model=config["model_name"],
        image_encoder_model_2=config["model_name"],
        actor_network_input=config["nails"],
        in_channels=config["canvas_channels"],
        out_neurons=config["nails"] + 5
    ).to(device)

    critic_target = Critic(
        image_encoder_model=config["model_name"],
        image_encoder_model_2=config["model_name"],
        actor_network_input=config["nails"],
        in_channels=config["canvas_channels"],
        out_neurons=1
    ).to(device)

    # Sync target networks
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    # >>> DDP EDIT: wrap in DDP if distributed
    if config.get("is_ddp", False):
        # find_unused_parameters=True is handy if some branches don’t execute every step
        actor = DDP(actor, device_ids=[int(device.split(":")[-1])], output_device=int(device.split(":")[-1]), find_unused_parameters=False)
        critic = DDP(critic, device_ids=[int(device.split(":")[-1])], output_device=int(device.split(":")[-1]), find_unused_parameters=False)
        # Common pattern: keep targets as plain modules on each rank (no DDP)
        # You’ll copy weights from actor/critic into targets as you already do.

    # ----- Optimizers -----
    # >>> DDP NOTE: using .parameters() on the DDP-wrapped module is correct.
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config["actor_lr"])
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config["critic_lr"])

    # Replay buffer & noise
    replay_buffer = ReplayBuffer(capacity=config["replay_buffer_capacity"], 
    device=device,
    log_mode="main_only",   # or "main_only" or "off" or "per_rank"
    rank=rank,
    is_main=is_main)
    
    noise = OUNoise(config["action_dim"])

    # Build agent
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

    # ----- Weights viz hooks -----
    # comment this if visualisation (gradients flow) is to be skipped 
    def _viz_after_backward():
        agent.gf_actor_mean = grad_flow_mean(actor, device)
        agent.gf_critic_mean = grad_flow_mean(critic, device)
    agent.viz_after_backward = _viz_after_backward

    actor_prev_vec = flat_params(actor, device)
    critic_prev_vec = flat_params(critic, device)
    w0_actor = snapshot_vector(actor)
    w0_critic = snapshot_vector(critic)

    os.makedirs("logs", exist_ok=True)
    scores_window = deque(maxlen=100)

    noise_scale = config["initial_noise_scale"]
    noise_decay = config["noise_decay"]

    # ----- Resume (rank 0 decides) -----
    def find_last_checkpoint(model_folder, model_prefix):
        model_files = glob.glob(os.path.join(model_folder, f"{model_prefix}_*.pth"))
        if not model_files:
            return None, 0
        episode_numbers = []
        for file in model_files:
            m = re.search(rf"{model_prefix}_(\d+)\.pth", file)
            if m:
                episode_numbers.append(int(m.group(1)))
        if not episode_numbers:
            return None, 0
        last_episode = max(episode_numbers)
        last_model_path = os.path.join(model_folder, f"{model_prefix}_{last_episode}.pth")
        return last_model_path, last_episode

    start_episode = 0
    if config.get("resume", True):
        actor_path, actor_episode = find_last_checkpoint("trained_models", "actor")
        critic_path, critic_episode = find_last_checkpoint("trained_models", "critic")
        assert actor_episode == critic_episode, "Actor and Critic checkpoints out of sync!"
        if actor_path and critic_path:
            if is_main:
                print(f"\nResuming training from episode {actor_episode}\n")
            # >>> DDP EDIT: when loading into (possibly) DDP-wrapped models, load into .module if needed
            if isinstance(actor, DDP):
                actor.module.load_state_dict(torch.load(actor_path, map_location=device))
                critic.module.load_state_dict(torch.load(critic_path, map_location=device))
            else:
                actor.load_state_dict(torch.load(actor_path, map_location=device))
                critic.load_state_dict(torch.load(critic_path, map_location=device))
            start_episode = actor_episode
        else:
            if is_main:
                print("\nNo saved model found, starting fresh!\n")

    # Re-sync targets after possible resume
    if isinstance(actor, DDP):
        actor_target.load_state_dict(actor.module.state_dict())
        critic_target.load_state_dict(critic.module.state_dict())
    else:
        actor_target.load_state_dict(actor.state_dict())
        critic_target.load_state_dict(critic.state_dict())

    # ----- Training loop -----
    for episode in range(start_episode, start_episode + config["episodes"]):

        is_video_episode = ((episode + 1) == 1) or ((episode + 1) % VIDEO_EVERY == 0)
        episode_frames = []

        agent.noise.reset()
        episode_reward = 0.0
        done = False

        canvas, current_idx = env.reset()
        #print(f"(in train.py) raw canvas: type={type(canvas)}, shape={getattr(canvas, 'shape', None)}")

        if isinstance(canvas, np.ndarray) and canvas.ndim == 3:
            canvas = np.transpose(canvas, (2, 0, 1))
            canvas = canvas[np.newaxis, :, :, :]
        elif isinstance(canvas, torch.Tensor) and canvas.ndim == 3:
            canvas = canvas.permute(2, 0, 1).unsqueeze(0).contiguous()
            #print(f"[in train.py] after permute+unsqueeze: {canvas.shape}")
            

        canvas_tensor = torch.from_numpy(canvas).float().to(device) if isinstance(canvas, np.ndarray) else canvas.float().to(device)
        #print(f"[in train.py] canvas_tensor: {canvas_tensor.shape}, device={canvas_tensor.device}")
        #print(f"[in train.py] target_image: {target_image.shape}, device={target_image.device}")

        # ---- t = 0: build prev_soft from current_idx as one-hot ----
        """nails = config["nails"]
        if isinstance(current_idx, int):
            prev_soft = torch.zeros(nails, device=device, dtype=torch.float32)
            prev_soft[current_idx] = 1.0
        elif isinstance(current_idx, torch.Tensor):
            idx0 = int(current_idx.view(-1)[0].item())
            prev_soft = torch.zeros(nails, device=device, dtype=torch.float32)
            prev_soft[idx0] = 1.0
        else:
            # fallback if no index given by env.reset()
            # uniform probability distribution over all nails
            prev_soft = torch.full((nails,), 1.0 / nails, device=device, dtype=torch.float32)"""

        while not done:
            current_episode = episode

            t0 = time.time()
            action_idx = agent.act(canvas_tensor, target_image, current_idx, noise_scale, episode=current_episode, step=env.used_strokes)
            #action_idx, action_soft = agent.act(canvas_tensor, target_image, prev_soft, noise_scale, episode=current_episode, step=env.used_strokes)
            t1 = time.time()
            if is_main:
                print("(in train.py) Action Time: ", t1 - t0)

            t2 = time.time()
            next_canvas, reward, done = env.step(action_idx, current_episode=episode, current_step=env.used_strokes)
            t3 = time.time()
            if is_main:
                print("(in train.py) Rendering Time: ", t3 - t2)

            if isinstance(next_canvas, np.ndarray):
                canvas_tensor = torch.from_numpy(next_canvas).float().unsqueeze(0).to(device)
            else:
                canvas_tensor = next_canvas.float().unsqueeze(0).to(device)

            canvas_for_buffer = canvas.squeeze(0) if isinstance(canvas, torch.Tensor) else np.squeeze(canvas, axis=0)

            replay_buffer.store(
                canvas_for_buffer,
                current_idx,             
                action_idx,             
                next_canvas,
                reward,
                done,
                episode=current_episode,
                step=env.used_strokes
            )

            agent.episode = episode + 1
            agent.step = env.used_strokes

            t4 = time.time()
            agent.train(target_image)  
            t5 = time.time()
            if is_main:
                print("(in train.py) Training Time: ", t5 - t4) 

            #here    
            #print("(in train.py) env.used_strokes ", env.used_strokes)
            global_step = int(episode * config["max_strokes"] + env.used_strokes) 
            
            #step_i = env.used_strokes
            # ---- Periodic logs ----
            if ((env.used_strokes % LOG_EVERY_STEPS) == 0) or done:
                # every 200 steps
                if is_main:
                    gf_actor = getattr(agent, "gf_actor_mean", None)
                    gf_critic = getattr(agent, "gf_critic_mean", None)
                    actor_curr_vec = flat_params(actor, device)
                    critic_curr_vec = flat_params(critic, device)
                    actor_upd_ratio = update_ratio_gpu(actor_prev_vec, actor_curr_vec)
                    critic_upd_ratio = update_ratio_gpu(critic_prev_vec, critic_curr_vec)

                    wandb.log(
                        {
                            "grad_flow/actor_mean_abs_grad": float(gf_actor) if gf_actor is not None else 0.0,
                            "grad_flow/critic_mean_abs_grad": float(gf_critic) if gf_critic is not None else 0.0,
                            "update_ratio/actor_mean": float(actor_upd_ratio),
                            "update_ratio/critic_mean": float(critic_upd_ratio),
                            "loss/actor": getattr(agent, "last_actor_loss", None),
                            "loss/critic": getattr(agent, "last_critic_loss", None),
                            #"episode": int(episode + 1),
                            #"env_used_strokes": int(env.used_strokes),
                        },
                        step=global_step,  # use same step here too
                    )

                    actor_prev_vec = actor_curr_vec.detach().clone()
                    critic_prev_vec = critic_curr_vec.detach().clone()

            canvas = canvas_tensor

            if is_video_episode and is_main:
                img = canvas_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 255).astype(np.uint8)
                #img = 255 - img
                episode_frames.append(img)

            if ((episode + 1) == 1 or (episode + 1) % 10 == 0) and env.used_strokes == config["max_strokes"] - 1 and is_main:
                os.makedirs("/storage/axp4488/rl_painter/logs", exist_ok=True)
                step_dir = f"/storage/axp4488/rl_painter/logs/step_outputs/episode_{episode + 1}"
                os.makedirs(step_dir, exist_ok=True)
                save_path = os.path.join(step_dir, f"final_step_{config['max_strokes']}.png")
                canvas_to_save = canvas[0]  # (C, H, W)
                canvas_to_save = canvas_to_save.permute(1, 2, 0).contiguous()
                save_canvas(canvas_to_save, save_path)

            current_idx = action_idx
            #prev_soft = action_soft
            episode_reward += reward

            #here  

            if is_main:
                os.makedirs("/storage/axp4488/rl_painter/logs", exist_ok=True)
                with open("/storage/axp4488/rl_painter/logs/step_rewards.csv", mode="a", newline="") as file:
                    writer = csv.writer(file)
                    if episode == 0 and env.used_strokes == 1:
                        writer.writerow(["episode", "step", "reward"])
                    writer.writerow([episode + 1, env.used_strokes, reward])
                print(f"Episode {episode + 1} | Step {env.used_strokes} | Step Reward: {reward}")

                #wandb.log({"Step vs Reward (all episodes)": reward})
                wandb.log({"Step vs Reward (all episodes)": reward}, step=global_step)

        # ---- End of episode ----
        if is_video_episode and episode_frames and is_main:
            log_canvas_video(episode, episode_frames, fps=30)

        if (episode + 1) % 20 == 0:
            noise_scale *= noise_decay
        scores_window.append(episode_reward)
        running_avg = np.mean(list(scores_window))

        if is_main:
            os.makedirs("/storage/axp4488/rl_painter/logs", exist_ok=True)
            with open("/storage/axp4488/rl_painter/logs/episode_rewards.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                if episode == 0:
                    writer.writerow(["episode", "episode_reward"])
                writer.writerow([episode + 1, episode_reward])

            print(f"Episode {episode + 1} | Reward: {episode_reward} | Running Avg(100): {running_avg:.2f}")

            if (episode + 1) % config["save_every_episode"] == 0:
                os.makedirs("trained_models", exist_ok=True)
                # >>> DDP EDIT: save .module if wrapped
                if isinstance(actor, DDP):
                    torch.save(actor.module.state_dict(), f"trained_models/actor_{episode + 1}.pth")
                    torch.save(critic.module.state_dict(), f"trained_models/critic_{episode + 1}.pth")
                else:
                    torch.save(actor.state_dict(), f"trained_models/actor_{episode + 1}.pth")
                    torch.save(critic.state_dict(), f"trained_models/critic_{episode + 1}.pth")
                print(f"Saved model at episode {episode + 1}")

            wandb.log({
                "Episode vs Reward": episode_reward,
                "Episode": episode + 1
            })

        # ----- Rare CPU sanity delta -----
        if ((episode + 1) % 10 == 0) and is_main:
            w1_actor = snapshot_vector(actor)
            w1_critic = snapshot_vector(critic)
            wandb.log({
                "sanity/actor_delta_norm": (w1_actor - w0_actor).norm().item(),
                "sanity/critic_delta_norm": (w1_critic - w0_critic).norm().item()
            })
            w0_actor, w0_critic = w1_actor, w1_critic

    if is_main:
        print("Training complete.")
