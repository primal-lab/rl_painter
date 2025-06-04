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

        # keep track of epiosde and step numbers
        self.episode = 0
        self.step = 0

        # move networks to GPU
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.actor_target = actor_target.to(self.device)
        self.critic_target = critic_target.to(self.device)


    def select_action(self, canvas, target_image, prev_action):
        """
        Select an action from the actor network given the current state.
        Used in test.py

        Args:
            canvas (tensor): Current canvas.
            target_image (tensor): Target image.
            prev_action (np.ndarray): Action previously applied to canvas. 

        Returns:
            action (np.ndarray): Next action predicted by actor network. (6,)
        """
        os.makedirs("logs/model", exist_ok=True)

        device = self.config["device"]
        
        # canvas = target_image = (B, C, H, W) = already tensors from train.py

        if isinstance(prev_action, np.ndarray):
            prev_action = torch.from_numpy(prev_action).float() # convert to tensor
        if prev_action.ndim == 1:
            prev_action = prev_action.unsqueeze(0)  # (2,) → (1, 2)
        prev_action = prev_action.to(device)

        """print(f"When getting the action - ") 
        print(f"Canvas shape: {canvas.shape}")  # (B, H, W, C)
        print(f"Target image shape: {target_image.shape}")  # (B, H, W, C) 
        print(f"Previous action shape: {prev_action.shape}") # (B, action_dim)"""

        self.actor.eval()
        with torch.no_grad():
            out = self.actor(canvas, target_image, prev_action).cpu().numpy() # (B, action_dim)
            action = out[0]  # Remove batch dimension (take the first row - 6 params) = (6,)
            # Log actor output shape and values
            #os.makedirs("logs/model", exist_ok=True)
            with open("logs/model/actor_actions.log", "a") as f:
                #import pdb
                #pdb.set_trace()
                f.write(f"Action shape: {out.shape}, Values: {out.tolist()}\n")
        self.actor.train()
        return action

    def act(self, canvas, target_image, prev_action, noise_scale=0.01):
        """
        Select an action and apply Ornstein-Uhlenbeck exploration noise.
        Used in train.py
        Args:
            canvas (torch.tensor): Current canvas. Dimensions: (B, H, W, C)
            target_image (torch.tensor): Target image. Dimensions: (B, H, W, C)
            prev_action (np.ndarray): normalised (x,y).  Dimensions: (2,)
            noise_scale (float): Scale of the noise to be added. 
        Used to control exploration.


        Returns:
            action (np.ndarray): Noisy action for exploration.
        """
        action = self.select_action(canvas, target_image, prev_action)
        action += self.noise.sample() * noise_scale
        return action

    def update_actor_critic(self, target_image):
        """
        Perform one training update step for both the actor and critic networks using
        a mini-batch sampled from the replay buffer.

        The update includes:
            - Computing target Q-values using the target networks
            - Calculating critic loss (MSE between current Q and target Q)
            - Updating the critic network via backprop
            - Generating predicted actions from the actor
            - Calculating actor loss (negative mean Q-value)
            - Updating the actor network via backprop
            - Soft-updating the target networks (Polyak averaging)

        Args:
            target_image (np.ndarray): The fixed goal image for the episode.
                                    Used in both actor and critic inputs.
        
        Notes:
            - The state is represented as (canvas, prev_action)
            - target_image is passed separately (assumed fixed per episode)
            - next state's prev_action = action from the current transition
        """
        os.makedirs("logs/model", exist_ok=True)

        # skip training if RB doesn't have atleast 32 samples
        if len(self.replay_buffer) < self.config["batch_size"]:
            return

        B = self.config["batch_size"]
        device = self.config["device"]

        canvas, prev_actions, actions, next_canvas, rewards, dones = self.replay_buffer.sample(B)

        # Log sampled batch shapes for sanity check
        #with open("logs/model/batch_shapes.log", "a") as f:
         #   f.write(f"Canvas: {canvas.shape}, PrevActions: {prev_actions.shape}, Actions: {actions.shape}, NextCanvas: {next_canvas.shape}\n")

        # print("(in ddpg.py Original canvas shape:", canvas.shape)

        """canvas = torch.tensor(canvas, dtype=torch.float32).to(device)
        target = torch.tensor(target_image, dtype=torch.float32).to(device).repeat(canvas.shape[0], 1, 1, 1)
        prev_actions = torch.tensor(prev_actions, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        next_canvas = torch.tensor(next_canvas, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)"""

        # canvas, prev_actions, actions, next_canvas, rewards, dones are already on GPU
        if isinstance(target_image, np.ndarray):
            target = torch.from_numpy(target_image).float().to(self.device)
        else:
            target = target_image.to(self.device)

        target = target.repeat(canvas.shape[0], 1, 1, 1)

        """print("🟨 DEBUG BATCH SHAPES BEFORE ACTOR CALL")
        print("canvas shape:", canvas.shape) 
        print("next_canvas shape:", next_canvas.shape)              # Should be [B, 1, 224, 224]
        print("target shape:", target.shape)                        # Should be [B, 1, 224, 224]
        #print("next_prev_actions shape:", next_prev_actions.shape)  # Should be [B, 6]"""

        canvas_size = self.config["canvas_size"]  #[224, 224]
        center = torch.tensor([canvas_size[1] // 2, canvas_size[0] // 2], device=self.device).float()
        radius = min(canvas_size[0], canvas_size[1]) // 2

        with torch.no_grad():
            next_prev_actions = actions #2
            next_actions = self.actor_target(next_canvas, target, next_prev_actions) # next actions = 6
            
            # normalising the x,y values of next_actions
            direction = next_actions[:, :2]  # raw (x, y)
            norm = direction.norm(p=2, dim=1, keepdim=True) + 1e-8
            unit_vector = direction / norm
            current_action_xy = unit_vector * radius + center  # [B, 2]

            critic_actions = torch.cat((next_prev_actions, current_action_xy), dim=1) #2+2 = 4
            # target_Q = self.critic_target(next_canvas, target, next_prev_actions, next_actions) - not passing next_prev_action
            target_Q = self.critic_target(next_canvas, target, critic_actions)
            target_Q = rewards + self.config["gamma"] * target_Q * (1 - dones)

        # current_Q = self.critic(canvas, target, prev_actions, actions) - not passing prev_action
        current_Q = self.critic(canvas, target, critic_actions)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        """print("🟨 DEBUG BATCH SHAPES BEFORE ACTOR (PREDICTED) CALL")
        print("canvas shape:", canvas.shape)
        print("target shape:", target.shape)
        print("prev_actions shape:", prev_actions.shape)"""


        predicted_actions = self.actor(canvas, target, prev_actions) # 6

        # need to normalise again 
        direction = predicted_actions[:, :2]  # raw (x, y)
        norm = direction.norm(p=2, dim=1, keepdim=True) + 1e-8
        unit_vector = direction / norm
        pred_action_xy = unit_vector * radius + center  # [B, 2]

        predicted_actions = torch.cat((prev_actions, pred_action_xy), dim=1) # 2+ 2 = 4
        # actor_loss = -self.critic(canvas, target, prev_actions, predicted_actions).mean()
        actor_loss = -self.critic(canvas, target, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Log critic and actor loss to disk
        #with open("logs/model/losses.log", "a") as f:
        #    f.write(f"Critic Loss: {critic_loss.item():.6f}, Actor Loss: {actor_loss.item():.6f}\n")
        
        # error detection
        """if torch.isnan(critic_loss) or torch.isnan(actor_loss):
            os.makedirs("logs/model", exist_ok=True)
            with open("logs/model/nan_errors.log", "a") as f:
                f.write(f"NaN detected at Episode {self.episode}, Step {self.step}\n")
                f.write(f"Critic Loss: {critic_loss.item()}, Actor Loss: {actor_loss.item()}\n\n")"""

        self.soft_update(self.critic, self.critic_target, self.config["tau"])
        self.soft_update(self.actor, self.actor_target, self.config["tau"])

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
