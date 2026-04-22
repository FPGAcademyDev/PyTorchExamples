from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np


class Cell(IntEnum):
    EMPTY = 0
    X = 1
    O = 2


WIN_LINES: Tuple[Tuple[int, ...], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


@dataclass
class Board:
    """9 cells, row-major. X moves first."""

    cells: List[int]

    @staticmethod
    def empty() -> "Board":
        return Board([Cell.EMPTY] * 9)

    def copy(self) -> "Board":
        return Board(list(self.cells))

    def legal_moves(self) -> List[int]:
        return [i for i in range(9) if self.cells[i] == Cell.EMPTY]

    def current_player(self) -> int:
        nx = sum(1 for c in self.cells if c == Cell.X)
        no = sum(1 for c in self.cells if c == Cell.O)
        return Cell.X if nx == no else Cell.O

    def apply(self, index: int) -> None:
        p = self.current_player()
        if self.cells[index] != Cell.EMPTY:
            raise ValueError("Illegal move")
        self.cells[index] = p

    def winner(self) -> Optional[int]:
        for a, b, c in WIN_LINES:
            v = self.cells[a]
            if v != Cell.EMPTY and v == self.cells[b] == self.cells[c]:
                return int(v)
        if Cell.EMPTY not in self.cells:
            return 0
        return None

    def terminal(self) -> bool:
        return self.winner() is not None

    def to_model_input(self) -> np.ndarray:
        """Encode for the side to move: me=1, opp=-1, empty=0."""
        p = self.current_player()
        opp = Cell.O if p == Cell.X else Cell.X
        x = np.zeros(9, dtype=np.float32)
        for i, c in enumerate(self.cells):
            if c == p:
                x[i] = 1.0
            elif c == opp:
                x[i] = -1.0
        return x
