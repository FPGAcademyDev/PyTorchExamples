from __future__ import annotations

import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """Maps 9-d board encoding to 9 move logits."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        # Model 1 - Original model chosen by Cursor
        # self.net = nn.Sequential(
        #     nn.Linear(9, hidden),
        #     nn.ReLU(),
        #     nn.Linear(hidden, hidden),
        #     nn.ReLU(),
        #     nn.Linear(hidden, 9),
        # )

        # Model 2 - Original model but without ReLU
        # self.net = nn.Sequential(
        #     nn.Linear(9, hidden),
        #     nn.Linear(hidden, hidden),
        #     nn.Linear(hidden, 9),
        # )

        # Model 3 - Bigger model 2
        self.net = nn.Sequential(
            nn.Linear(9, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, 9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def best_move(self, board_input: torch.Tensor, legal_mask: torch.Tensor) -> int:
        logits = self.forward(board_input.unsqueeze(0)).squeeze(0)
        logits = logits.masked_fill(~legal_mask, float("-inf"))
        return int(torch.argmax(logits).item())
