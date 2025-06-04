import random
import torch
from collections import deque
import os

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device  # Make sure this is torch.device('cuda')

    def store(self, canvas, prev_action, action, next_canvas, reward, done):
        """
        Store GPU tensors directly. Convert NumPy → Tensor if needed.
        """

        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.detach().to(self.device)
            return torch.tensor(x, dtype=torch.float32, device=self.device)

        canvas = to_tensor(canvas)         # (C, H, W)
        next_canvas = to_tensor(next_canvas)  # (C, H, W)
        prev_action = to_tensor(prev_action).view(-1)  # (2,)
        action = to_tensor(action).view(-1)            # (2,)
        reward = to_tensor(reward).view(1)             # (1,)
        done = to_tensor(done).view(1)                 # (1,)

        self.buffer.append((canvas, prev_action, action, next_canvas, reward, done))

        # logging
        os.makedirs("logs/debug", exist_ok=True)
        with open("logs/debug/replay_buffer.log", "a") as f:
            f.write(
                #f"[ReplayBuffer] canvas.shape={tuple(canvas.shape)}, "
                f"prev_action=({prev_action[0].item():.6f}, {prev_action[1].item():.6f}), "
                f"action=({action[0].item():.6f}, {action[1].item():.6f}), "
                #f"next_canvas.shape={tuple(next_canvas.shape)}, "
                f"reward={reward.item():.4f}, done={bool(done.item())}\n"
            )

    def sample(self, batch_size):
        """
        Return a tuple of batches as GPU tensors: (B, C, H, W), (B, 6), ...
        """
        batch = random.sample(self.buffer, batch_size)
        canvas, prev_actions, actions, next_canvas, rewards, dones = zip(*batch)

        return (
            torch.stack(canvas),         # (B, C, H, W)
            torch.stack(prev_actions),   # (B, 2)
            torch.stack(actions),        # (B, 2)
            torch.stack(next_canvas),    # (B, C, H, W)
            torch.stack(rewards),        # (B, 1)
            torch.stack(dones)           # (B, 1)
        )

    def __len__(self):
        return len(self.buffer)
