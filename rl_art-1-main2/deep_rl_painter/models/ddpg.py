"""
DDPG Agent Module

Defines the DDPGAgent class for training and action inference in a goal-conditioned painting task.

The agent:
    - Uses dual CNN encoders to process canvas and target images.
    - Takes previous actions into account to maintain temporal coherence.
    - Trains Actor and Critic networks using transitions from a replay buffer.

Key Inputs:
    - canvas: current image
    - target_image: goal image to paint
    - prev_action: the last action applied to canvas
    - action: the action to evaluate or apply

Outputs:
    - Actor: next action to apply (x, y, r, g, b, width)
    - Critic: Q-value estimating expected return from a (state, action) pair
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import time
from torch.cuda.amp import autocast, GradScaler

class DDPGAgent:
    def __init__(self, actor, critic, actor_target, critic_target,
                 actor_optimizer, critic_optimizer, replay_buffer, noise, config):
        self.actor = actor
        self.critic = critic
        self.actor_target = actor_target
        self.critic_target = critic_target
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.replay_buffer = replay_buffer
        self.noise = noise
        self.config = config
        self.channels = config["canvas_channels"]
        self.device = config["device"]

        # Initialize target networks with the same weights as the main networks
        #self.actor_target.load_state_dict(actor.state_dict())
        #self.critic_target.load_state_dict(critic.state_dict())

        # handle both plain nn.Module and DDP-wrapped modules
        actor_sd  = actor.module.state_dict()  if hasattr(actor, "module")  else actor.state_dict()
        critic_sd = critic.module.state_dict() if hasattr(critic, "module") else critic.state_dict()

        self.actor_target.load_state_dict(actor_sd)
        self.critic_target.load_state_dict(critic_sd)


        # keep track of epiosde and step numbers - never used
        self.episode = 0
        self.step = 0

        # move networks to GPU
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.actor_target = actor_target.to(self.device)
        self.critic_target = critic_target.to(self.device)
        
        self.resized_target = None  # cache for resized target image

        # train.py may set this to a function that logs grad-flow to W&B
        self.viz_after_backward = None

        # decides gumbel or softmax in select_action
        self.use_gumbel_in_select = True       # runtime choice; doesn't affect training
        self.gumbel_tau = config["gumbel_tau"]

        self.scaler = GradScaler()  #  (for mixed precision)


    # resize canvas and target to 224x224 before passing into networks
    # runs on gpu because all inputs given later are tensors
    def resize_input(self, x, size=(224, 224)):
        # x: (B, C, H, W) or (1, C, H, W) 
        # interpolate = scales tensors up or down 
        # Bilinear interpolation: takes the 4 nearest pixels around a new location and averages them weighted by distance.`11111`
        # corners-false = Pixel centers are scaled proportionally inside the new size.
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def select_action(self, canvas, target_image, prev_soft):
        """
        Use the actor network to select the best action (nail index) given the current state.

        Args:
            canvas: (1, C, H, W)
            target_image: (1, C, H, W)
            prev_soft: (nails,) float distribution used as 'prev action' input
        Returns:
            action_idx (int)
            action_soft (Tensor, (nails,))
        """

        # canvas = target_image = (B, C, H, W) = already tensors from train.py

        # ensure shape (1, nails)
        if prev_soft.dim() == 1:
            prev_soft_in = prev_soft.unsqueeze(0)  # (1, nails)
        else:
            prev_soft_in = prev_soft  # assume already (1, nails)

        self.actor.eval()
        with torch.no_grad(), autocast(dtype=torch.float16):
            canvas_resized = self.resize_input(canvas)  # (1, C, 224, 224)
            target_resized = self.resize_input(target_image)
            nail_logits, stroke_params = self.actor(canvas_resized, target_resized, prev_soft_in)  # (1, nails)) 
            
            if self.use_gumbel_in_select:
                # Sample a soft vector (hard=False), then pick the argmax index
                a_soft = F.gumbel_softmax(nail_logits, tau=self.gumbel_tau, hard=False)  # (1, nails)
            # not being used, only applies gumbel as of now
            else:
                # Deterministic: softmax -> argmax
                a_soft = torch.softmax(nail_logits, dim=-1)

            action_idx = int(a_soft.argmax(dim=-1).item())
            action_soft = a_soft.squeeze(0).float().detach().to(self.device)    

        self.actor.train()
        #print ("Action idx(argmax)", action_idx)
        return action_idx, action_soft

    def act(self, canvas, target_image, prev_soft, noise_scale=0.01,episode=0, step=0):
        """"
        Behavior policy used in train.py.
        Choose action with warmup + epsilon exploration.
        Args:
            prev_soft: (nails,) Gumbel/softmax vector from previous step.
        Returns:
            action_idx (int), action_soft (nails,)
        """
        nails = self.config["nails"]
        # for the first episode, the first 500 steps are just exploration (p=1)
        if episode < 1 and step < 500:
            action_idx = int(torch.randint(0, nails, ()).item())
            action_soft = torch.zeros(nails, device=self.device, dtype=torch.float32)
            action_soft[action_idx] = 1.0
        else:
            p = np.random.rand()  # uniform in [0, 1]
            #p = 0
            # make it dynamic later (threshold=0.8 now)
            # more exploration in initial stages, more exploitation later on
            if p > 0.8:
                # exploration: pick a random nail index (20%)
                action_idx = int(torch.randint(0, nails, ()).item())
                action_soft = torch.zeros(nails, device=self.device, dtype=torch.float32)
                action_soft[action_idx] = 1.0
            else:
                #exploitation: use policy network (prev_soft is already a gumbel vector)
                action_idx, action_soft = self.select_action(canvas, target_image, prev_soft)
        return action_idx, action_soft

    def update_actor_critic(self, target_image):
        """
        One DDP-friendly update step.
        - Uses local batch size (split across ranks by train.py)
        - Caches target image resize
        - Mixed precision for big memory+speed wins
        """
        B = int(self.config.get("batch_size_local", self.config["batch_size"]))   # <<< local per-GPU batch
        if len(self.replay_buffer) < B:
            return

        nails = self.config["nails"]

         # ----- Sample replay buffer -----
        canvas, prev_soft, action_soft, next_canvas, rewards, dones = self.replay_buffer.sample(B)
        # shapes:
        # canvas, next_canvas: (B, C, H, W)
        # prev_soft, action_soft: (B, nails)
        # rewards, dones: (B, 1)

        # ----- Resize & target caching -----
        if self.resized_target is None:
            with torch.no_grad():
                self.resized_target = self.resize_input(target_image)   # (1, C, 224, 224)
        target_resized      = self.resized_target.expand(B, -1, -1, -1)  # <<< zero-copy expand along batch
        canvas_resized      = self.resize_input(canvas)
        next_canvas_resized = self.resize_input(next_canvas)

        # ===================== Critic Update =====================
        self.critic_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), autocast(dtype=torch.float16):
            next_logits, _ = self.actor_target(next_canvas_resized, target_resized, action_soft)  # (B, nails)
            next_a_soft = F.gumbel_softmax(next_logits, tau=self.gumbel_tau, hard=False)
            target_Q = self.critic_target(next_canvas_resized, target_resized, next_a_soft)
            target_Q = rewards + self.config["gamma"] * target_Q * (1 - dones)

        with autocast(dtype=torch.float16):
            current_Q   = self.critic(canvas_resized, target_resized, action_soft) 
            critic_loss = F.mse_loss(current_Q, target_Q)

        self.scaler.scale(critic_loss).backward()
        self.scaler.step(self.critic_optimizer)
        self.scaler.update()
        self.last_critic_loss = float(critic_loss.detach().cpu())

        if self.viz_after_backward is not None:
            try:
                self.viz_after_backward()
            except Exception:
                pass

        # ===================== Actor Update =====================
        self.actor_optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=torch.float16):
            nail_logits, _ = self.actor(canvas_resized, target_resized, prev_soft)  # (B, nails)
            a_soft = F.gumbel_softmax(nail_logits, tau=self.gumbel_tau, hard=False)   # (B, nails)
            Q_sa = self.critic(canvas_resized, target_resized, a_soft)                # (B,1)
            actor_loss = -Q_sa.mean()

        self.scaler.scale(actor_loss).backward()
        self.scaler.step(self.actor_optimizer)
        self.scaler.update()
        self.last_actor_loss = float(actor_loss.detach().cpu())

        if self.viz_after_backward is not None:
            try:
                self.viz_after_backward()
            except Exception:
                pass

        # ===================== Soft target updates =====================
        self.soft_update(self.critic, self.critic_target, self.config["tau"])
        self.soft_update(self.actor,  self.actor_target,  self.config["tau"])


    def train(self, target_image):
        """Wrapper to update actor and critic networks."""
        self.update_actor_critic(target_image)

    def soft_update(self, local_model, target_model, tau):
        """
        Perform soft target network update.

        Args:
            local_model: Actor or Critic main network.
            target_model: Corresponding target network.
            tau (float): Soft update coefficient (0 < tau << 1).
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

if __name__ == "__main__":
    """
    Quick diagnostic test for DDPGAgent setup and API:
    - Builds dummy Actor/Critic
    - Runs .act() and .train() with fake canvas data
    - Verifies integration works without crashing
    """
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    import numpy as np
    from .actor import Actor
    from .critic import Critic
    from ..utils.replay_buffer import ReplayBuffer
    from ..utils.noise import OUNoise

    print("Running DDPGAgent standalone integration test...")

    # Config setup
    config = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "image_size": (224, 224),
        "batch_size": 4,
        "gamma": 0.99,
        "tau": 0.005
    }

    B, C, H, W = config["batch_size"], 3, *config["image_size"]
    action_dim = 6

    # Models
    actor = Actor("resnet18", "resnet18", pretrained=False, out_neurons=action_dim, in_channels=C)
    critic = Critic("resnet18", "resnet18", pretrained=False, out_neurons=1, in_channels=C)
    actor_target = Actor("resnet18", "resnet18", pretrained=False, out_neurons=action_dim, in_channels=C)
    critic_target = Critic("resnet18", "resnet18", pretrained=False, out_neurons=1, in_channels=C)

    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # Agent and utils
    replay_buffer = ReplayBuffer(capacity=100)
    noise = OUNoise(action_dim)

    agent = DDPGAgent(
        actor=actor,
        critic=critic,
        actor_target=actor_target,
        critic_target=critic_target,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        replay_buffer=replay_buffer,
        noise=noise,
        config=config,
        channels=C
    )

    # Dummy tensors
    canvas = np.random.rand(C, H, W).astype(np.float32)
    target = np.random.rand(C, H, W).astype(np.float32)
    prev_action = np.zeros(action_dim, dtype=np.float32)

    # Test .act()
    try:
        action = agent.act(canvas, target, prev_action)
        print("DDPGAgent.act() output:", action)
    except Exception as e:
        print("DDPGAgent.act() failed:", e)

    # Populate buffer
    for _ in range(B):
        next_canvas = np.random.rand(C, H, W).astype(np.float32)
        reward = 1.0
        done = 0.0
        replay_buffer.store(canvas, prev_action, action, next_canvas, reward, done)

    # Test .train()
    try:
        agent.train(target)
        print("DDPGAgent.train() ran without error.")
    except Exception as e:
        print("DDPGAgent.train() failed:", e)
