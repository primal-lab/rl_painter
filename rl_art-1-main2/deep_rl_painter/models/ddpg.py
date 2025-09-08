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
        self.actor_target.load_state_dict(actor.state_dict())
        self.critic_target.load_state_dict(critic.state_dict())

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

    # resize canvas and target to 224x224 before passing into networks
    # runs on gpu because all inputs given later are tensors
    def resize_input(self, x, size=(224, 224)):
        # x: (B, C, H, W) or (1, C, H, W) 
        # interpolate = scales tensors up or down 
        # Bilinear interpolation: takes the 4 nearest pixels around a new location and averages them weighted by distance.`11111`
        # corners-false = Pixel centers are scaled proportionally inside the new size.
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def select_action(self, canvas, target_image, prev_action_onehot):
        """
        Use the actor network to select the best action (nail index) given the current state.

        Args:
            canvas (torch.Tensor): (1, C, H, W)
            target_image (torch.Tensor): (1, C, H, W)
            prev_action_onehot (torch.Tensor): (1, nails)

        Returns:
            action_idx (int): Index of the next nail to move to (0 to nails - 1)
        """

        # canvas = target_image = (B, C, H, W) = already tensors from train.py

        """if isinstance(prev_action, np.ndarray):
            prev_action = torch.from_numpy(prev_action).float() # convert to tensor
        if prev_action.ndim == 1:
            prev_action = prev_action.unsqueeze(0)  # (2,) → (1, 2)
        # prev_action = prev_action.to(self.device)"""

        self.actor.eval()
        with torch.no_grad():
            canvas_resized = self.resize_input(canvas)  # (1, C, 224, 224)
            target_resized = self.resize_input(target_image)
            nail_probs, stroke_params = self.actor(canvas_resized, target_resized, prev_action_onehot) 
            action_idx = torch.argmax(nail_probs, dim=-1).item()         # index of best nail 
            # Log actor output shape and values
            #with open("logs/model/actor_actions.log", "a") as f:
            #    f.write(f"Action shape: {out.shape}, Values: {out.tolist()}\n")
        self.actor.train()
        print ("Action idx(argmax)", action_idx)
        return action_idx

    def act(self, canvas, target_image, prev_action_idx, noise_scale=0.01,episode=0, step=0):
        """
        Select an action and apply Ornstein-Uhlenbeck exploration noise.
        Used in train.py

        Args:
            canvas (torch.tensor): Current canvas. Dimensions: (1, C, H, W)
            target_image (torch.tensor): Target image. Dimensions: (1, C, H, W)
            prev_action_idx (int): Index of previous nail (0 to nails-1)
            noise_scale (float): Exploration noise scale.

        Returns:
            action_idx (int): Selected nail index for next stroke.
        """
        # for the first episode, the first 500 steps are just exploration (p=1)
        if episode < 1 and step < 500:
            action_idx = np.random.randint(0, self.config["nails"])
        else:
            p = np.random.rand()  # uniform in [0, 1]
            #p = 0
            # make it dynamic later (threshold=0.8 now)
            # more exploration in initial stages, more exploitation later on
            if p > 0.8:
                # exploration: pick a random nail index (20%)
                action_idx = np.random.randint(0, self.config["nails"])
            else:
                # convert prev index to one-hot tensor
                nails = self.config["nails"]
                one_hot_prev = F.one_hot(torch.tensor([prev_action_idx]), num_classes=nails).float().to(self.device)
                # exploitation: use policy network
                action_idx = self.select_action(canvas, target_image, one_hot_prev)
        return action_idx

    def update_actor_critic(self, target_image):
        if len(self.replay_buffer) < self.config["batch_size"]:
            return

        B      = self.config["batch_size"]
        nails  = self.config["nails"]

        # ----- Sample replay buffer -----
        canvas, current_idx, action_idx, next_canvas, rewards, dones = self.replay_buffer.sample(B)
        current_idx = current_idx.view(-1)  # (B,)
        action_idx  = action_idx.view(-1)   # (B,)

        # One-hots
        prev_onehot   = F.one_hot(current_idx, nails).float().to(self.device)   # previous action at s_t
        action_onehot = F.one_hot(action_idx,  nails).float().to(self.device)   # taken action at s_t

        # ----- Resize & target caching -----
        if self.resized_target is None:
            self.resized_target = self.resize_input(target_image)
        target_resized      = self.resized_target.repeat(B, 1, 1, 1)
        canvas_resized      = self.resize_input(canvas)
        next_canvas_resized = self.resize_input(next_canvas)

        # ====== Critic Target ======
        with torch.no_grad():
            # For s_{t+1}, the "prev action" is the action taken at s_t
            next_probs, _ = self.actor_target(
                next_canvas_resized, target_resized, action_onehot
            )
            next_dist = torch.distributions.Categorical(probs=(next_probs + 1e-8))
            next_a    = next_dist.sample()  # (B,)
            next_onehot = F.one_hot(next_a, nails).float().to(self.device)  # (B, nails)

            target_Q = self.critic_target(next_canvas_resized, target_resized, next_onehot)  # (B,1)
            target_Q = rewards + self.config["gamma"] * target_Q * (1 - dones) # (B,1)

        # ====== Critic Update (use the *taken* action at s_t) ======
        current_Q   = self.critic(canvas_resized, target_resized, action_onehot)           # (B,1)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.viz_after_backward is not None: #log
            self.viz_after_backward()
        self.critic_optimizer.step()
        self.last_critic_loss = critic_loss.detach().item() #log

        # ====== Actor Update (stochastic policy gradient with Q) ======
        # π(a|s_t) using prev_onehot at s_t
        nail_probs, _ = self.actor(canvas_resized, target_resized, prev_onehot)  # (B, nails)
        dist   = torch.distributions.Categorical(probs=(nail_probs + 1e-8))
        a_samp = dist.sample()                                                   # (B,)
        a_onehot = F.one_hot(a_samp, nails).float().to(self.device)              # (B, nails)

        # Q(s_t, a_samp); detach so only actor updates here
        Q_sa = self.critic(canvas_resized, target_resized, a_onehot).detach()    # (B,1)
        logp = dist.log_prob(a_samp).unsqueeze(1)                                 # (B,1)

        actor_loss = -(logp * Q_sa).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.viz_after_backward is not None: #log
            self.viz_after_backward()
        self.actor_optimizer.step()
        self.last_actor_loss = actor_loss.detach().item()  #log

        # ====== Soft Updates ======
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
