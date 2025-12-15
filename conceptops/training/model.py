from __future__ import annotations

import torch
import torch.nn as nn


class TinyEventMLP(nn.Module):
    """
    Small MLP for baseline classification.

    Input: handcrafted feature vector (dim = 7)
    Output: logits over labels

    Keep it tiny to avoid overfitting on small Phase 3 data.
    """
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)