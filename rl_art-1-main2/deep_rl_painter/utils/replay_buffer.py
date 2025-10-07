import random
import torch
from collections import deque
import os

class ReplayBuffer:
    def __init__(self, capacity, device, log_mode="main_only", rank=0, is_main=None):
        """
        log_mode: "off" | "main_only" | "per_rank"
        - "off":       no file logs
        - "main_only": only rank 0 writes to logs/replay_buffer.log
        - "per_rank":  each rank writes to logs/replay_buffer_rank{rank}.log
        """
        # current capacity = 500000
        self.buffer = deque(maxlen=capacity)
        self.device = device
        self.rank = int(rank)
        self.is_main = (self.rank == 0) if (is_main is None) else bool(is_main)
        self.log_mode = log_mode
        os.makedirs("logs", exist_ok=True)

    def store(self, canvas, current_idx, action_idx, next_canvas, reward, done,
              episode=None, step=None):
        """
        Store GPU tensors directly.
        - canvas, next_canvas: (C, H, W) float
        - current_idx: int (previous action index at s_t)
        - action_idx:  int (executed action index at s_t)
        - reward: scalar
        - done: scalar (0/1)
        """
        def to_tensor(x, dtype=torch.float32):
            if isinstance(x, torch.Tensor):
                return x.detach().to(self.device).to(dtype)
            return torch.tensor(x, dtype=dtype, device=self.device)

        canvas      = to_tensor(canvas)
        next_canvas = to_tensor(next_canvas)
        current_idx = torch.tensor(current_idx, dtype=torch.long,  device=self.device)
        action_idx  = torch.tensor(action_idx,  dtype=torch.long,  device=self.device)
        reward      = to_tensor(reward).view(1)
        done        = to_tensor(done).view(1)

        self.buffer.append((canvas, current_idx, action_idx, next_canvas, reward, done))

        # ---------- rank-aware logging ----------
        # currenlty. log_mode = per_rank (in train.py - RB initialisation)
        log_path = None
        if self.log_mode == "per_rank":
            log_path = f"logs/replay_buffer_rank{self.rank}.log"
        elif self.log_mode == "main_only" and self.is_main:
            log_path = "logs/replay_buffer.log"

        if log_path is not None:
            try:
                with open(log_path, "a") as f:
                    f.write(
                        f"ep={episode} step={step} rank={self.rank} "
                        f"prev_idx={int(current_idx.item())} "
                        f"act_idx={int(action_idx.item())} "
                        f"reward={float(reward.item()):.4f} "
                        f"done={bool(done.item())}\n"
                    )
                    f.flush()
            except Exception:
                pass
        # ---------------------------------------

    def sample(self, batch_size):
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
