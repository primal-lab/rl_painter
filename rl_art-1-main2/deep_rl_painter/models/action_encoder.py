import torch
import torch.nn as nn

class ActionEncoder(nn.Module):
    """
    Encodes a one-hot or Gumbel-softmax action vector into a compact embedding.
    """

    def __init__(self, in_dim: int, out_dim: int = 256, hidden: int = 256, act: str = "GELU"):
        super().__init__()

        if act == "ReLU":
            activation = nn.ReLU()
        elif act == "Tanh":
            activation = nn.Tanh()
        elif act == "LeakyReLU": # this being used rn
            activation = nn.LeakyReLU()
        else:
            activation = nn.GELU()  # default

        layers = [
            nn.Linear(in_dim, hidden),
            activation,
            nn.Linear(hidden, out_dim),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a: (B, in_dim) one-hot or Gumbel-soft action vector.
        Returns:
            (B, out_dim) encoded action embedding.
        """
        return self.net(a)
