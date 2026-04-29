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
        # self.net = nn.Sequential(
        #     nn.Linear(9, hidden),
        #     nn.Linear(hidden, hidden),
        #     nn.Linear(hidden, hidden),
        #     nn.Linear(hidden, hidden),
        #     nn.Linear(hidden, 9),
        # )

        # Model 4 - Even bigger
        # self.net = nn.Sequential(
        #     nn.Linear(9, hidden * 32),
        #     nn.Linear(hidden * 32, hidden * 32),
        #     nn.Linear(hidden * 32, 9),
        # )

        # Model 5 - Smaller model 1
        # self.net = nn.Sequential(
        #     nn.Linear(9, hidden),
        #     nn.ReLU(),
        #     nn.Linear(hidden, 9),
        # )

        # Model 6 - Bigger model 1
        # self.net = nn.Sequential(
        #     nn.Linear(9, hidden),
        #     nn.ReLU(),
        #     nn.Linear(hidden, hidden),
        #     nn.ReLU(),
        #     nn.Linear(hidden, hidden),
        #     nn.ReLU(),
        #     nn.Linear(hidden, hidden),
        #     nn.ReLU(),
        #     nn.Linear(hidden, 9),
        # )

        # Model 7 - Using Convolutional layers
        self.conv1 = nn.Conv2d(1, hidden, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(hidden * 9, 9)

    # Forward pass for linear models
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     return self.net(x)

    # Forward pass for convolutional models
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1, 3, 3)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.flatten(1)
        return self.fc(x)

    @torch.no_grad()
    def best_move(self, board_input: torch.Tensor, legal_mask: torch.Tensor) -> int:
        logits = self.forward(board_input.unsqueeze(0)).squeeze(0)
        logits = logits.masked_fill(~legal_mask, float("-inf"))
        return int(torch.argmax(logits).item())
