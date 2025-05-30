"""
Replay Buffer:
Stores and samples transitions.

Each transition stored includes:
    - canvas (current state image)
    - prev_action (action taken to reach current canvas)
    - action (action to be applied - outputted by the Actor Network)
    - next_canvas (resulting canvas after applying `action`)
    - reward (scalar feedback)
    - done (episode completion flag)

Methods:
    - store: Save a transition
    - sample: Randomly retrieve a batch of transitions
    - __len__: Return current size of the buffer
"""

import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, canvas, prev_action, action, next_canvas, reward, done):
        """
        Store a single transition into the buffer.

        Args:
            canvas (np.ndarray): Current canvas image.
            prev_action (np.ndarray): Previous action taken before this state.
            action (np.ndarray): Action taken at this state.
            next_canvas (np.ndarray): Canvas image after action is applied.
            reward (float): Scalar reward.
            done (bool): Whether the episode ended after this transition.
        """
        # Ensure shape is (C, H, W) not (H, W, C)
        if canvas.ndim == 3 and canvas.shape[0] != 1 and canvas.shape[0] != 3:
            canvas = np.transpose(canvas, (2, 0, 1))  # from (H, W, C) → (C, H, W)
        if next_canvas.ndim == 3 and next_canvas.shape[0] != 1 and next_canvas.shape[0] != 3:
            next_canvas = np.transpose(next_canvas, (2, 0, 1))  # from (H, W, C) → (C, H, W)

        self.buffer.append((canvas, prev_action, action, next_canvas, reward, done))

    def sample(self, batch_size):
        """
        Sample a batch of transitions for training.

        Returns:
            Tuple of:
                canvas_batch: (Batch size, Channels, Height, Width)
                prev_actions: (B, action_dim)
                actions: (B, action_dim)
                next_canvas: (B, C, H, W)
                rewards: (B, 1)
                dones: (B, 1)
        """
        batch = random.sample(self.buffer, batch_size)
        canvas, prev_actions, actions, next_canvas, rewards, dones = map(np.array, zip(*batch))

        # Fix shapes: ensure all are (1, H, W)
        def fix_canvas_batch(batch):
            batch_fixed = []
            for img in batch:
                if img.ndim == 2:  # (H, W)
                    img = np.expand_dims(img, axis=0)  # (1, H, W)
                elif img.ndim == 3 and img.shape[0] != 1:
                    img = np.transpose(img, (2, 0, 1))  # (C, H, W)
                elif img.ndim == 3 and img.shape[0] == 1:
                    pass  # Already (1, H, W)
                batch_fixed.append(img)
            return np.stack(batch_fixed)

        canvas = fix_canvas_batch(canvas)
        next_canvas = fix_canvas_batch(next_canvas)

        return (
            canvas,
            np.stack(prev_actions),
            np.stack(actions),
            next_canvas,
            np.array(rewards).reshape(-1, 1),
            np.array(dones).reshape(-1, 1)
        )

    def __len__(self):
        """Return the current number of transitions in the buffer."""
        return len(self.buffer)

if __name__ == "__main__":
    """
    ReplayBuffer Check:
    - Fills buffer with dummy transitions
    - Samples a batch
    - Prints shapes of sampled components
    """

    import numpy as np

    print("Testing ReplayBuffer functionality...")

    buffer = ReplayBuffer(capacity=10)
    C, H, W = 3, 64, 64     # Image dimensions
    action_dim = 6
    batch_size = 4

    # Populate with dummy data
    for _ in range(8):
        canvas = np.random.rand(C, H, W).astype(np.float32)
        prev_action = np.random.rand(action_dim).astype(np.float32)
        action = np.random.rand(action_dim).astype(np.float32)
        next_canvas = np.random.rand(C, H, W).astype(np.float32)
        reward = np.random.rand()
        done = np.random.choice([0.0, 1.0])
        buffer.store(canvas, prev_action, action, next_canvas, reward, done)

    print(f"Buffer length: {len(buffer)}")

    if len(buffer) >= batch_size:
        canvas_b, prev_a_b, a_b, next_canvas_b, r_b, d_b = buffer.sample(batch_size)
        print("canvas batch:", canvas_b.shape)
        print("prev_action batch:", prev_a_b.shape)
        print("action batch:", a_b.shape)
        print("next_canvas batch:", next_canvas_b.shape)
        print("reward batch:", r_b.shape)
        print("done batch:", d_b.shape)
    else:
        print("Not enough samples in buffer to draw a batch.")

