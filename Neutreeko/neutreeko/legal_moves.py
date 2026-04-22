"""Enumerate sliding moves from the oracle-backed game state."""

from __future__ import annotations

from dataclasses import dataclass

from neutreeko.constants import DRAW_POSITION, ILLEGAL_POSITION
from neutreeko.oracle import NeutreekoOracle


@dataclass(frozen=True)
class LegalMove:
    x1: int
    y1: int
    x2: int
    y2: int
    reply_idx: int
    raw_remaining: int

    def sort_key(self) -> float:
        r = self.raw_remaining
        if r % 2 == 0:
            return float(r - DRAW_POSITION)
        return float(DRAW_POSITION - r)


def diff_board(oracle: NeutreekoOracle, x_idx: int, o_idx: int) -> list[list[int]]:
    w, h = oracle.width, oracle.height
    return [
        [oracle.position_1col[x_idx][c][d] - oracle.position_1col[o_idx][c][d] for d in range(h)]
        for c in range(w)
    ]


def outcome_label(raw_remaining: int) -> str:
    if raw_remaining == DRAW_POSITION:
        return "Draw"
    if raw_remaining % 2 == 0:
        lab = "Win"
    else:
        lab = "Loss"
    if raw_remaining > 0:
        lab += " in " + str(raw_remaining + 1) + " moves"
    return lab


def enumerate_legal_moves(
    oracle: NeutreekoOracle, x_idx: int, o_idx: int, position_2col: list[list[int]]
) -> list[LegalMove]:
    w = oracle.width
    board_h = oracle.height
    rem = oracle.remaining_moves
    if rem[x_idx][o_idx] == ILLEGAL_POSITION:
        return []

    pfi_a = oracle.position_from_index[x_idx]
    idx_from_pos = oracle.index_from_position
    moves: list[LegalMove] = []

    for k in range(3):
        d = pfi_a[k] % w
        e = pfi_a[k] // w
        for f in range(-1, 2):
            for g in range(-1, 2):
                if (
                    f * f + g * g > 0
                    and 0 <= d + f < w
                    and 0 <= e + g < board_h
                    and position_2col[d + f][e + g] == 0
                ):
                    f1, g1 = f, g
                    while (
                        0 <= d + f1 + f < w
                        and 0 <= e + g1 + g < board_h
                        and position_2col[d + f1 + f][e + g1 + g] == 0
                    ):
                        f1 += f
                        g1 += g
                    perm = [pfi_a[0], pfi_a[1], pfi_a[2]]
                    perm[k] = d + f1 + w * (e + g1)
                    reply_idx = idx_from_pos[perm[0]][perm[1]][perm[2]]
                    raw = rem[o_idx][reply_idx]
                    moves.append(
                        LegalMove(
                            x1=d,
                            y1=e,
                            x2=d + f1,
                            y2=e + g1,
                            reply_idx=reply_idx,
                            raw_remaining=raw,
                        )
                    )
    return moves


def sorted_move_indices(moves: list[LegalMove]) -> list[int]:
    """Indices 0..n-1 sorted by oracle sort_key ascending, then stable tie-break."""
    return sorted(range(len(moves)), key=lambda i: (moves[i].sort_key(), i))


def teacher_move_index_unsorted(moves: list[LegalMove]) -> int:
    """Index into enumerate_legal_moves order for an oracle-optimal move (tie: smallest index)."""
    if not moves:
        return 0
    return min(range(len(moves)), key=lambda i: (moves[i].sort_key(), i))
