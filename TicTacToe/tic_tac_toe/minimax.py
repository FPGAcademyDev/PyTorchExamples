from __future__ import annotations

from typing import List, Optional, Tuple

from tic_tac_toe.game import Board, Cell


def _score_for_player(w: Optional[int], player: int) -> int:
    if w is None:
        return 0
    if w == 0:
        return 0
    if w == player:
        return 1
    return -1


def minimax(board: Board) -> Tuple[int, List[int]]:
    """
    Return (value from current player's view, list of optimal move indices).
    value in {-1, 0, 1}: loss, draw, win for side to move.
    """
    w = board.winner()
    if w is not None:
        return _score_for_player(w, board.current_player()), []

    player = board.current_player()
    best_val = -2
    best_moves: List[int] = []

    for m in board.legal_moves():
        b = board.copy()
        b.apply(m)
        child_w = b.winner()
        if child_w is not None:
            val = _score_for_player(child_w, player)
        else:
            opp_val, _ = minimax(b)
            val = -opp_val

        if val > best_val:
            best_val = val
            best_moves = [m]
        elif val == best_val:
            best_moves.append(m)

    return best_val, best_moves
