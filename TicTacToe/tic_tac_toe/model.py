from __future__ import annotations

import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """Maps 9-d board encoding to 9 move logits."""

    #: Highest supported index; valid ``model_num`` values are ``1 .. NUM_MODELS`` inclusive.
    NUM_MODELS = 7

    def __init__(self, hidden: int = 128, model_num: int = 1) -> None:
        super().__init__()
        mn = model_num
        if not isinstance(mn, int) or not (1 <= mn <= PolicyNet.NUM_MODELS):
            raise ValueError(
                f"Unsupported model_num={model_num!r}; expected an int in 1..{PolicyNet.NUM_MODELS}"
            )
        if mn == 1:
            # Model 1 - Original model chosen by Cursor
            self.net = nn.Sequential(
                nn.Linear(9, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 9),
            )
        elif mn == 2:
            # Model 2 - Original model but without ReLU
            self.net = nn.Sequential(
                nn.Linear(9, hidden),
                nn.Linear(hidden, hidden),
                nn.Linear(hidden, 9),
            )
        elif mn == 3:
            # Model 3 - Bigger model 2
            self.net = nn.Sequential(
                nn.Linear(9, hidden),
                nn.Linear(hidden, hidden),
                nn.Linear(hidden, hidden),
                nn.Linear(hidden, hidden),
                nn.Linear(hidden, 9),
            )
        elif mn == 4:
            # Model 4 - Even bigger
            self.net = nn.Sequential(
                nn.Linear(9, hidden * 32),
                nn.Linear(hidden * 32, hidden * 32),
                nn.Linear(hidden * 32, 9),
            )
        elif mn == 5:
            # Model 5 - Smaller model 1
            self.net = nn.Sequential(
                nn.Linear(9, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 9),
            )
        elif mn == 6:
            # Model 6 - Bigger model 1
            self.net = nn.Sequential(
                nn.Linear(9, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 9),
            )
        elif mn == 7:
            # Model 7 - Two convolutional layers over 3x3 board
            self.net = nn.Sequential(
                nn.Unflatten(1, (1, 3, 3)),
                nn.Conv2d(1, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(1),
                nn.Linear(hidden * 9, 9),
            )
        else:
            raise RuntimeError(
                f"PolicyNet.NUM_MODELS={PolicyNet.NUM_MODELS} includes model_num={mn}, "
                "but __init__ has no matching branch; add one or fix NUM_MODELS."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def best_move(self, board_input: torch.Tensor, legal_mask: torch.Tensor) -> int:
        logits = self.forward(board_input.unsqueeze(0)).squeeze(0)
        logits = logits.masked_fill(~legal_mask, float("-inf"))
        return int(torch.argmax(logits).item())
