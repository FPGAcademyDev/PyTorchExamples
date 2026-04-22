from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GameConfig:
    width: int
    height: int
    ruleset: int


@dataclass
class Move:
    """Linked list node for move history (retract / display)."""

    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    z1: int = 0  # previous x-index (position key) before this move
    prev: Optional[Move] = None
