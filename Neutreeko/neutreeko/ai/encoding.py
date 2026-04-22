from __future__ import annotations

import torch

# Fixed geometry for a small CNN (Neutreeko boards are at most 7×7).
BOARD_PAD = 7
# Upper bound on legal moves for 7×7 (three sliders, eight directions, long slides).
MAX_MOVES = 160


def encode_board_tensor(
    position_2col: list[list[int]],
    width: int,
    height: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """(4, BOARD_PAD, BOARD_PAD): mover, opponent, empty, on-board mask."""
    t = torch.zeros(4, BOARD_PAD, BOARD_PAD, device=device)
    for x in range(width):
        for y in range(height):
            v = position_2col[x][y]
            if v == 1:
                t[0, x, y] = 1.0
            elif v == -1:
                t[1, x, y] = 1.0
            else:
                t[2, x, y] = 1.0
            t[3, x, y] = 1.0
    return t
