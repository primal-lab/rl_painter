import random
import torch
from collections import deque
import os

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)  # capacity = 500000
        self.device = device

    def store(self, canvas, current_idx, action_idx, next_canvas, reward, done):
        """
        Store GPU tensors directly.
        - canvas, next_canvas: (C, H, W) float
        - current_idx: int (previous action index at s_t)
        - action_idx: int (executed action index at s_t)
        - reward: scalar
        - done: scalar (0/1)
        """
        def to_tensor(x, dtype=torch.float32):
            if isinstance(x, torch.Tensor):
                return x.detach().to(self.device).to(dtype)
            return torch.tensor(x, dtype=dtype, device=self.device)

        canvas      = to_tensor(canvas)                 # (C,H,W)
        next_canvas = to_tensor(next_canvas)            # (C,H,W)
        current_idx = torch.tensor(current_idx, dtype=torch.long, device=self.device)  # scalar index
        action_idx  = torch.tensor(action_idx,  dtype=torch.long, device=self.device)  # scalar index
        reward      = to_tensor(reward).view(1)         # (1,)
        done        = to_tensor(done).view(1)           # (1,)

        self.buffer.append((
            canvas, current_idx, action_idx, next_canvas, reward, done
        ))

        # Optional logging
        try:
            os.makedirs("logs", exist_ok=True)
            with open("logs/replay_buffer.log", "a") as f:
                f.write(
                    f"prev_idx={int(current_idx.item())}, "
                    f"act_idx={int(action_idx.item())}, "
                    f"reward={reward.item():.4f}, done={bool(done.item())}\n"
                )
        except Exception:
            pass

    def sample(self, batch_size):
        """
        Return batched GPU tensors:
        (B,C,H,W), (B,), (B,), (B,C,H,W), (B,1), (B,1)
        """
        batch = random.sample(self.buffer, batch_size)
        canvas, current_idx, action_idx, next_canvas, rewards, dones = zip(*batch)

        return (
            torch.stack(canvas),          # (B,C,H,W)
            torch.stack(current_idx),     # (B,)
            torch.stack(action_idx),      # (B,)
            torch.stack(next_canvas),     # (B,C,H,W)
            torch.stack(rewards),         # (B,1)
            torch.stack(dones)            # (B,1)
        )

    def __len__(self):
        return len(self.buffer)
