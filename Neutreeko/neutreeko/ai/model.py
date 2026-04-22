from __future__ import annotations

import torch
import torch.nn as nn

from neutreeko.ai.encoding import BOARD_PAD, MAX_MOVES


class NeutreekoPolicyNet(nn.Module):
    """Small CNN mapping padded board tensor to a fixed number of move logits."""

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        c = BOARD_PAD
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(embed_dim * c * c, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, MAX_MOVES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
