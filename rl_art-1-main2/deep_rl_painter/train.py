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
from env.environment import PaintingEnv
from models.actor import Actor
from models.critic import Critic
from models.ddpg import DDPGAgent
from utils.noise import OUNoise
from utils.replay_buffer import ReplayBuffer
from env.canvas import save_canvas
import torch.profiler
import time
import csv
import glob
import re
from utils.simplified_targets import generate_simplified_targets
import cv2
import wandb
#from utils.wandb_logging import start_visual_logging, log_canvas_step, log_canvas_video_and_table
from utils.wandb_logging import log_canvas_video, log_step_to_table
from utils.weights_viz import (
    layer_stats, WeightChangeTracker, gradient_flow,
    conv_filter_grid, ActivationRecorder, actor_distribution,
    td_error_hist, snapshot_vector
)

# --- helper: find first Conv2d in a module tree ---
import torch.nn as nn
def first_conv(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            return m
    return None

def train(config):
    """
    Full training pipeline for the DDPG agent using canvas drawing environment.
    Initializes all components and trains the agent over multiple episodes.

    Args:
        config (dict): Configuration dictionary containing hyperparameters and paths.
    """

    wandb.init(
        project="ddpg-painter",             
        name="added_weights_viz",         
        config=config              
    )

    # create dummy env to access target_image 
    # to pass into generate_simplified_targets
    """dummy_env = PaintingEnv(
        target_image_path=config["target_image_path"],
        target_edges_path=config["target_edges_path"],
        canvas_size=config["canvas_size"],
        canvas_channels=config["canvas_channels"],
        max_strokes=config["max_strokes"],
        device=config["device"],
        target_segments_path=config["target_segments_path"]
    )

    # dummy_env.target_image = (H, W, C)
    # target_image = (B, C, H, W)
    target_image = torch.from_numpy(dummy_env.target_image / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(config["device"])

    # generate simplified versions once
    simplified_targets = generate_simplified_targets(
        target_image,
        save_dir="logs/target_versions"
    )"""

    # Initialize (actual/real) environment here and load target image
    env = PaintingEnv(
        target_image_path=config["target_image_path"],
        target_edges_path=config["target_edges_path"],
        canvas_size=config["canvas_size"],
        canvas_channels=config["canvas_channels"],
        max_strokes=config["max_strokes"],
        device=config["device"],
        #simplified_targets=simplified_targets, #pass in target_image versions
        target_segments_path=config["target_segments_path"]
    )

    # Load target image
    # env.target_image = (H, W, C)
    # target_image = (B, C, H, W)
    target_image = torch.from_numpy(env.target_image).permute(2, 0, 1).unsqueeze(0).float().to(config["device"])

    # Initialize Actor & Critic networks (main and target)
    actor = Actor(
        image_encoder_model=config["model_name"],
        #actor_network_input=config["action_dim"],
        actor_network_input=config["nails"], # one-hot vector of the prev point
        in_channels=config["canvas_channels"],
        out_neurons=config["nails"]+5  # next point probs (softmax) + r,g,b,w,o
    )
    critic = Critic(
        image_encoder_model=config["model_name"],
        #actor_network_input=config["action_dim"]*2,
        actor_network_input=config["nails"], # one-hot vector of the current point
        in_channels=config["canvas_channels"],
        out_neurons=1
    )
    actor_target = Actor(
        image_encoder_model=config["model_name"],
        #actor_network_input=config["action_dim"],
        actor_network_input=config["nails"], 
        in_channels=config["canvas_channels"],
        out_neurons=config["nails"]+5
    )
    critic_target = Critic(
        image_encoder_model=config["model_name"],
        #actor_network_input=config["action_dim"]*2,
        actor_network_input=config["nails"],
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

    # BEGIN WEIGHTS VIZ: setup for hooks, trackers, and initial snapshot
    modules_to_watch = {}
    # Try to hook FIRST conv blocks for canvas/target encoders in Actor
    # Adjust these lines if your model uses different attribute names.
    # hook first convs from both encoders (if found)
    try:
        conv1_a = first_conv(actor.model.image_encoder_1)
        modules_to_watch["actor_enc1_firstconv"] = conv1_a
    except Exception:
        pass

    try:
        conv1_b = first_conv(actor.model.image_encoder_2)
        modules_to_watch["actor_enc2_firstconv"] = conv1_b
    except Exception:
        pass

    # hook first activation in merged MLP (best for dead-ReLU % & feature snapshots)
    try:
        first_activation = None
        if hasattr(actor.model, "merge_network") and isinstance(actor.model.merge_network, nn.Sequential):
            for m in actor.model.merge_network:
                if isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.Tanh)):
                    first_activation = m
                    break
        if first_activation is not None:
            modules_to_watch["actor_merge_act0"] = first_activation
    except Exception:
        pass

    actrec = ActivationRecorder(modules_to_watch)
    wchg_actor  = WeightChangeTracker(actor)
    wchg_critic = WeightChangeTracker(critic)

    # Optional: gradient-flow hook out of the agent (requires tiny 1-liner in DDPGAgent)
    def _viz_grad():
        names_a, gf_a = gradient_flow(actor)
        names_c, gf_c = gradient_flow(critic)
        wandb.log({
            "grad_flow/actor_mean_abs_grad": sum(gf_a)/max(len(gf_a),1),
            "grad_flow/critic_mean_abs_grad": sum(gf_c)/max(len(gf_c),1),
        })
    agent.viz_after_backward = _viz_grad  

    # “before training” snapshot to confirm weights actually change later
    w0_actor  = snapshot_vector(actor)
    w0_critic = snapshot_vector(critic)
    # END WEIGHTS VIZ

    # Setup logging
    os.makedirs("logs", exist_ok=True)

    scores_window = deque(maxlen=100) # keep track of last 100 episodes, more stable than 3 

    # Exploration noise control
    noise_scale = config["initial_noise_scale"]
    noise_decay = config["noise_decay"]

    # Finding the last saved models
    def find_last_checkpoint(model_folder, model_prefix):
        # model_files will have a list of all files of the type "model_100.pth"
        # * represents any string of any character
        model_files = glob.glob(os.path.join(model_folder, f"{model_prefix}_*.pth"))
        if not model_files:
            return None, 0  # No saved model
        # extract episode numbers from file names
        episode_numbers = []
        for file in model_files:
            # (\d+) is the capture group which would contain the episode number here
            match = re.search(rf"{model_prefix}_(\d+)\.pth", file)
            if match:
                # for each file, if matched, the episode number would be 
                # appended to the list as an int (string -> int)
                episode_numbers.append(int(match.group(1)))
        if not episode_numbers:
            return None, 0
        # find max of the episode numbers list to get the last saved model   
        last_episode = max(episode_numbers)
        last_model_path = os.path.join(model_folder, f"{model_prefix}_{last_episode}.pth")
        # return last saved model path and episode number
        return last_model_path, last_episode

    # Loading last saved models
    # (currently - 700th)
    start_episode = 0
    # default to True if resume doesn't exist in config.py
    if config.get("resume", True):
        # get the last saved models
        actor_path, actor_episode = find_last_checkpoint("trained_models", "actor")
        critic_path, critic_episode = find_last_checkpoint("trained_models", "critic")

        # actor and critic should be aligned
        assert actor_episode == critic_episode, "Actor and Critic checkpoints out of sync!"

        # load
        if actor_path and critic_path:
            print(f"\nResuming training from episode {actor_episode} \n")
            actor.load_state_dict(torch.load(actor_path))
            critic.load_state_dict(torch.load(critic_path))
            start_episode = actor_episode
        else:
            print("\nNo saved model found, starting fresh! \n")

    # Sync target networks
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    # Profiling
    # incorporate "if config[profile]" later
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profiler'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # from last saved episode, run 50,000 more episodes
        for episode in range(start_episode, start_episode + config["episodes"]):

            # w and b
            global_step = 0
            episode_frames = []
            episode_table = wandb.Table(columns=["Step", "Reward", "Canvas"])
            #start_visual_logging(episode)

            # for replay buffer
            just_stuck_once = False

            #canvas = env.reset() 
            canvas, current_idx = env.reset() # canvas = (H, W, C) tensor
            
            """# random initial point is set above in env.reset
            prev_action = np.zeros(config["action_dim"], dtype=np.float32)
            # Initialize previous action with current point
            prev_action[0] = env.current_point[0]
            prev_action[1] = env.current_point[1]
            # Send only (x, y) point to the Actor
            prev_point = prev_action[:2].copy()
            first_step = True  # Flag to check for first loop"""

            # Logs input tensor shapes at the start of each episode
            """with open("logs/input_shapes.log", "a") as f:
                f.write(f"Episode {episode + 1}\n")
                f.write(f"Canvas shape: {canvas.shape}\n")
                f.write(f"Target image shape: {target_image.shape}\n")
                f.write(f"Prev action shape: {prev_action.shape}\n\n")"""

            # reset OUNoise every episode
            agent.noise.reset()
            episode_reward = 0.0
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
                """if first_step:
                    actor_prev_input = prev_point  # already a point on circle (from env.current_point -> env.random_circle_point)
                    first_step = False  # flip the flag
                else: # to get x,y on the circle, instead of raw directional x,y
                    direction = prev_action[:2]
                    norm = np.linalg.norm(direction) + 1e-8
                    unit_vector = direction / norm
                    actor_prev_input = np.array([
                        env.center[0] + unit_vector[0] * env.radius,
                        env.center[1] + unit_vector[1] * env.radius], dtype=np.float32)"""

                #print(f"Prev Action: {prev_action}")
                #print(f"Actor Prev Input : {actor_prev_input}")

                # Get action -> ddpg.py -> actor.py -> merge networks -> image encoder
                # prev_action = (6,) -> later converted to tensor and (6,1) in select_action()
                #action = agent.act(canvas_tensor, target_image, prev_action, noise_scale)
                
                current_episode = episode

                # actor_prev_input = (2,) -> later converted to tensor and (2,1) in select_action()
                # action is here numpy array (6,)
                t0 = time.time()
                #action = agent.act(canvas_tensor, target_image, actor_prev_input, noise_scale)
                action_idx = agent.act(canvas_tensor, target_image, current_idx, noise_scale, episode=current_episode, step=env.used_strokes)
                t1 = time.time()
                total = t1-t0
                print("(in train.py) Action Time: ", total)

                # Logs action values per step
                """with open("logs/action_logs.csv", "a", newline="") as file:
                    writer = csv.writer(file)
                    if episode == 0 and env.used_strokes == 1:
                        writer.writerow(["episode", "step", "action_values"])
                    writer.writerow([episode + 1, env.used_strokes, action.tolist()])"""

                # Apply action in the environment
                # actor_current_input contains the current action's normalised x,y values
                t2 = time.time()
                #current_episode = episode
                #next_canvas, reward, done, actor_current_input = env.step(action, current_episode=current_episode, current_step=env.used_strokes)
                next_canvas, reward, done = env.step(action_idx, current_episode=episode, current_step=env.used_strokes)
                t3 = time.time()
                total1 = t3-t2
                print("(in train.py) Rendering Time: ", total1)
                # #print(f"Action: {action}")
                #print(f"Actor Current Input (x, y): {actor_current_input}")

                # w and b
                #log_canvas_step(canvas, reward, env.used_strokes, episode)

                # next_canvas from env.step = (C, H, W)
                if isinstance(next_canvas, np.ndarray):
                    canvas_tensor = torch.from_numpy(next_canvas).float().unsqueeze(0).to(config["device"])
                else:
                    canvas_tensor = next_canvas.float().unsqueeze(0).to(config["device"])
                
                # canvas = (B, C, H, W)
                # canvas_for_buffer = (C, H, W) using this so that canvas sizes don't get messed up when doing canvas = next canvas
                canvas_for_buffer = canvas.squeeze(0) if isinstance(canvas, torch.Tensor) else np.squeeze(canvas, axis=0)

                # canvas_for_buffer = (C, H, W)
                # next_canvas = (C, H, W)
                #actor_prev_input = np.round(actor_prev_input, 6).astype(np.float32)
                #actor_current_input = np.round(actor_current_input, 6).astype(np.float32)
                # actor_prev_input, actor_current_input are being stored in absolute coordinates
                
                # Detect if agent is stuck
                #is_stuck = (current_idx == action_idx)
                # Store only if not stuck, or if stuck for the first time in a row
                # store "if (true)"
                #if not is_stuck or not just_stuck_once:
                replay_buffer.store(
                    canvas_for_buffer,
                    current_idx,
                    action_idx,
                    next_canvas,
                    reward,
                    done
                )
                # Update flag for next step
                #just_stuck_once = is_stuck

                """print(
                f"[ReplayBuffer Store] canvas.shape={canvas_for_buffer.shape}, "
                f"prev_action=({actor_prev_input[0]:.2f}, {actor_prev_input[1]:.2f}), "
                f"action=({actor_current_input[0]:.2f}, {actor_current_input[1]:.2f}), "
                f"next_canvas.shape={next_canvas.shape}, "
                f"reward={reward:.4f}, done={done}"
                )"""

                # keep track of epiosde and step numbers for update actor/critic
                # never used in ddpg.py (self.episode, self.step)
                agent.episode = episode + 1
                agent.step = env.used_strokes

                # Train the agent using sampled experiences
                t4 = time.time()
                agent.train(target_image)
                t5 = time.time()
                total2 = t5-t4
                print("(in train.py) Training Time: ", total2)

                # BEGIN WEIGHTS VIZ: lightweight, step-based logging
                step_i = env.used_strokes

                # 1) update ratios
                ratios_a = wchg_actor.step(actor)
                ratios_c = wchg_critic.step(critic)
                if ratios_a:
                    wandb.log({
                        "update_ratio/actor_mean": sum(ratios_a.values())/len(ratios_a),
                        "update_ratio/critic_mean": sum(ratios_c.values())/len(ratios_c),
                        "global_step": step_i,
                        "episode": episode + 1
                    })

                # 2) actor distributions
                if hasattr(actor, "last_logits") and actor.last_logits is not None and step_i % 50 == 0:
                    dist = actor_distribution(actor.last_logits, discrete=True)
                    wandb.log({
                        "actor/probs_hist": wandb.Histogram(dist["probs"]),
                        "global_step": step_i,
                        "episode": episode + 1
                    })
                if hasattr(actor, "last_actions") and isinstance(actor.last_actions, torch.Tensor) and step_i % 50 == 0:
                    acts = actor.last_actions.detach().cpu()
                    for i in range(acts.size(-1)):
                        wandb.log({
                            f"actor/actions_dim_{i}": wandb.Histogram(acts[..., i]),
                            "global_step": step_i,
                            "episode": episode + 1
                        })

                # 3) occasional layer stats
                if step_i % 200 == 0:
                    for d in layer_stats(actor)[:6]:
                        wandb.log({
                            f"weights/actor/{d['name']}_w_mean": d["w_mean"],
                            f"weights/actor/{d['name']}_w_std":  d["w_std"],
                            "global_step": step_i,
                            "episode": episode + 1
                        })
                    for d in layer_stats(critic)[:6]:
                        wandb.log({
                            f"weights/critic/{d['name']}_w_mean": d["w_mean"],
                            f"weights/critic/{d['name']}_w_std":  d["w_std"],
                            "global_step": step_i,
                            "episode": episode + 1
                        })

                # 4) filters & feature maps
                if step_i % 500 == 0:
                    try:
                        if conv1_a is not None and hasattr(conv1_a, "weight"):
                            grid = conv_filter_grid(conv1_a.weight, max_filters=64)
                            wandb.log({"filters/enc1_firstconv": wandb.Image(grid),
                                    "global_step": step_i, "episode": episode + 1})
                    except Exception:
                        pass
                    try:
                        if conv1_b is not None and hasattr(conv1_b, "weight"):
                            grid = conv_filter_grid(conv1_b.weight, max_filters=64)
                            wandb.log({"filters/enc2_firstconv": wandb.Image(grid),
                                    "global_step": step_i, "episode": episode + 1})
                    except Exception:
                        pass

                    for name in list(modules_to_watch.keys()):
                        grid = actrec.feature_map_grid(name, max_maps=64)
                        dead = actrec.dead_rate(name)
                        if grid is not None:
                            wandb.log({f"feats/{name}": wandb.Image(grid),
                                    "global_step": step_i, "episode": episode + 1})
                        if dead is not None:
                            wandb.log({f"dead_relu/{name}": dead,
                                    "global_step": step_i, "episode": episode + 1})
                # END WEIGHTS VIZ

                # Move to next state
                # canvas = (B, C, H, W)
                # canvas_tensor = (B, C, H, W)
                canvas = canvas_tensor

                # w and b
                #episode_frames.append(canvas[0].clone())
                #log_step_to_table(episode_table, env.used_strokes+1, reward, canvas[0])
                # Process frame once for both video & table
                img = canvas[0].detach().cpu().permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 255).astype(np.uint8)
                img = 255 - img  # same inversion as save_canvas
                # Squeeze to (H, W) for video (GIF-friendly)
                if img.ndim == 3 and img.shape[2] == 1:
                    video_frame = img.squeeze(axis=2)
                else:
                    video_frame = img
                # Append to episode_frames (used for video)
                episode_frames.append(video_frame)
                # Table still needs (C, H, W) torch tensor
                # If squeezed, re-add dummy channel for table consistency
                if img.ndim == 2:
                    img = img[:, :, np.newaxis]
                tensor_for_table = torch.from_numpy(img).permute(2, 0, 1)
                #print(tensor_for_table.shape)
                log_step_to_table(episode_table, env.used_strokes + 1, reward, tensor_for_table)

                # Save 1st episode's and every 10th episode's final step (2000th step)
                if ((episode + 1) == 1 or (episode + 1) % 100 == 0) and env.used_strokes == config["max_strokes"] - 1:
                    step_dir = f"step_outputs/episode_{episode + 1}"
                    os.makedirs(step_dir, exist_ok=True)
                    # sample path: step_outputs/episode_25000/episode_25000_step_00150.png
                    #save_path = os.path.join(step_dir, f"step_{env.used_strokes}.png")
                    save_path = os.path.join(step_dir, f"final_step_{config['max_strokes']}.png")
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
                #prev_action = action
                current_idx = action_idx
                #episode_reward += reward.item() #just assigning the float value
                episode_reward += reward

                # Log step reward
                with open("logs/step_rewards.csv", mode="a", newline="") as file:
                    writer = csv.writer(file)
                    if episode == 0 and env.used_strokes == 1:  # write header only once
                        writer.writerow(["episode", "step", "reward"])
                    #writer.writerow([episode + 1, env.used_strokes, reward.item()])
                    writer.writerow([episode + 1, env.used_strokes, reward])
                print(f"Episode {episode + 1} | Step {env.used_strokes} | Step Reward: {reward}")
                
                wandb.log({
                    "Step vs Reward (all episodes)": reward,
                    #"Global Step": global_step
                })
                global_step += 1


            # w and b
            #log_canvas_video_and_table(episode)
            # Log episode video only for the 1st and every 50th episode
            if (episode + 1) == 1 or (episode + 1) % 50 == 0:
                log_canvas_video(episode, episode_frames, fps=30)
            # table logging    
            #wandb.log({f"Episode_{episode + 1}_Step_Table": episode_table})

            # Decay exploration noise every 20 episodes
            if (episode + 1) % 20 == 0:
                noise_scale *= noise_decay
            #scores.append(episode_reward) - not being used anywhere
            # Log episode reward
            """with open("logs/episode_rewards.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                if episode == 0:  # Write header only once
                    writer.writerow(["episode", "total_reward"])
                writer.writerow([episode + 1, episode_reward])"""
            scores_window.append(episode_reward)
            #running_avg = torch.stack(list(scores_window)).mean()
            running_avg = np.mean(list(scores_window))

            # Logs the running average reward over last 100 episodes
            """with open("logs/running_avg.csv", "a", newline="") as file:
                writer = csv.writer(file)
                if episode == 0:
                    writer.writerow(["episode", "running_avg_100"])
                writer.writerow([episode + 1, np.mean(scores_window)])"""

            # Log episode reward
            with open("logs/episode_rewards.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                if episode == 0:
                    writer.writerow(["episode", "episode_reward"])  # Header once
                writer.writerow([episode + 1, episode_reward])
            # Progress log
            print(
                f"Episode {episode + 1} | Reward: {episode_reward} | Running Avg(100): {running_avg:.2f}")

            # Save model checkpoints every 100 episodes
            if (episode + 1) % config["save_every_episode"] == 0:
                os.makedirs("trained_models", exist_ok=True)
                torch.save(actor.state_dict(),
                        f"trained_models/actor_{episode + 1}.pth")
                torch.save(critic.state_dict(),
                        f"trained_models/critic_{episode + 1}.pth")
                print(f"Saved model at episode {episode + 1}")

            wandb.log({
                "Episode vs Reward": episode_reward,
                "Episode": episode + 1
            })
        
        # BEGIN WEIGHTS VIZ: final sanity deltas (did anything actually change?)
        w1_actor  = snapshot_vector(actor)
        w1_critic = snapshot_vector(critic)
        wandb.log({
            "sanity/actor_delta_norm": (w1_actor - w0_actor).norm().item(),
            "sanity/critic_delta_norm": (w1_critic - w0_critic).norm().item()
        })
        # END WEIGHTS VIZ
           
        print("Training complete.")
