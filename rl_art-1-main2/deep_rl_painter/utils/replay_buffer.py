import random
import torch
from collections import deque
import os

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)  # capacity = 500000
        self.device = device

    def store(self, canvas, prev_soft, action_soft, next_canvas, reward, done):
        """
        Store GPU tensors directly.
        - canvas, next_canvas: (C, H, W) float
        - prev_soft, action_soft: (nails,) float distributions
        - reward, done: scalars
        """
        def to_tensor(x, dtype=torch.float32):
            if isinstance(x, torch.Tensor):
                return x.detach().to(self.device).to(dtype)
            return torch.tensor(x, dtype=dtype, device=self.device)

        canvas      = to_tensor(canvas)                 # (C, H, W)
        next_canvas = to_tensor(next_canvas)            # (C, H, W)
        prev_soft   = to_tensor(prev_soft).view(-1)     # (nails,) 180
        action_soft = to_tensor(action_soft).view(-1)   # (nails,)
        reward      = to_tensor(reward).view(1)         # (1,)
        done        = to_tensor(done).view(1)           # (1,)

        self.buffer.append((
            canvas, prev_soft, action_soft, next_canvas, reward, done
        ))

        # Optional logging (argmax for readability only)
        try:
            os.makedirs("logs", exist_ok=True)
            with open("logs/replay_buffer.log", "a") as f:
                f.write(
                    f"prev_argmax={int(prev_soft.argmax().item())}, "
                    f"act_argmax={int(action_soft.argmax().item())}, "
                    f"reward={reward.item():.4f}, done={bool(done.item())}\n"
                )
        except Exception:
            pass  # don't crash training for logging issues

    def sample(self, batch_size):
        """
        Return batched GPU tensors:
        (B, C, H, W), (B, nails), (B, nails), (B, C, H, W), (B, 1), (B, 1)
        """
        batch = random.sample(self.buffer, batch_size)
        canvas, prev_soft, action_soft, next_canvas, rewards, dones = zip(*batch)

        return (
            torch.stack(canvas),        # (B, C, H, W)
            torch.stack(prev_soft),     # (B, nails)
            torch.stack(action_soft),   # (B, nails)
            torch.stack(next_canvas),   # (B, C, H, W)
            torch.stack(rewards),       # (B, 1)
            torch.stack(dones)          # (B, 1)
        )

    def __len__(self):
        return len(self.buffer)
