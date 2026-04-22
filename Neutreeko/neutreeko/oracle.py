"""Precompute symmetry tables, position indices, and perfect-play depths."""

from __future__ import annotations

from neutreeko.constants import DRAW_POSITION, ILLEGAL_POSITION
from neutreeko.models import GameConfig


class NeutreekoOracle:
    """Holds all precomputed tables for a given board size and ruleset."""

    def __init__(self, config: GameConfig) -> None:
        self.config = config
        w = config.width
        h = config.height
        self.width = w
        self.height = h
        self.ruleset = config.ruleset

        self.number_of_symmetries = 8 if w == h else 4
        self.number_of_cells = w * h
        self.number_of_positions = (
            self.number_of_cells * (self.number_of_cells - 1) * (self.number_of_cells - 2)
        ) // 6

        n_cells = self.number_of_cells
        n_pos = self.number_of_positions
        n_sym = self.number_of_symmetries

        self.index_from_position = [[[0 for _ in range(n_cells)] for _ in range(n_cells)] for _ in range(n_cells)]
        self.position_from_index = [[0 for _ in range(4)] for _ in range(n_pos)]
        self.position_1col = [[[0 for _ in range(h)] for _ in range(w)] for _ in range(n_pos)]
        self.completed_win = [0 for _ in range(n_pos)]
        self.remaining_moves = [[DRAW_POSITION for _ in range(n_pos)] for _ in range(n_pos)]
        self.symmetry = [[0 for _ in range(n_sym)] for _ in range(n_cells)]

        self._build_symmetry_table()
        self._enumerate_positions()
        self._reduce_canonical_symmetry_flags()
        initial_pending = self._label_terminal_remaining_moves()
        self.retrograde_depth = self._run_retrograde_analysis(initial_pending)
        self.draw_pair_count = self._count_draw_pairs()

    def _build_symmetry_table(self) -> None:
        w, h = self.width, self.height
        for a in range(self.number_of_cells):
            b, c = a % w, a // w
            self.symmetry[a][0] = a
            self.symmetry[a][1] = (w - 1 - b) + w * c
            self.symmetry[a][2] = b + w * (h - 1 - c)
            self.symmetry[a][3] = (w - 1 - b) + w * (h - 1 - c)
            if self.number_of_symmetries == 8:
                self.symmetry[a][4] = c + w * b
                self.symmetry[a][5] = (w - 1 - c) + w * b
                self.symmetry[a][6] = c + w * (w - 1 - b)
                self.symmetry[a][7] = (w - 1 - c) + w * (w - 1 - b)

    def _enumerate_positions(self) -> None:
        w, h = self.width, self.height
        ruleset = self.ruleset
        n_cells = self.number_of_cells
        idx = -1
        for a in range(n_cells - 2):
            for b in range(a + 1, n_cells - 1):
                for c in range(b + 1, n_cells):
                    idx += 1
                    d = idx
                    for e in range(w):
                        for f in range(h):
                            self.position_1col[d][e][f] = 0
                    self.position_1col[d][a % w][a // w] = 1
                    self.position_1col[d][b % w][b // w] = 1
                    self.position_1col[d][c % w][c // w] = 1
                    self.completed_win[d] = 0
                    if ruleset == 5:
                        centre = (n_cells - 1) // 2
                        if a == centre or b == centre or c == centre:
                            self.completed_win[d] = 1
                    else:
                        ax, ay = a % w, a // w
                        bx, by = b % w, b // w
                        cx, cy = c % w, c // w
                        if ax + cx == 2 * bx and ay + cy == 2 * by:
                            if abs(ax - cx) <= 2 and abs(ay - cy) <= 2:
                                self.completed_win[d] = 1
                        if ruleset == 2 and ax != cx and ay != cy:
                            self.completed_win[d] = 0
                        if ruleset == 3 and (ax == cx or ay == cy):
                            self.completed_win[d] = 0
                        if ax + cx == 2 * bx and ay + cy == 2 * by and ruleset == 4:
                            self.completed_win[d] = 1

                    self.index_from_position[a][b][c] = d
                    self.index_from_position[a][c][b] = d
                    self.index_from_position[b][a][c] = d
                    self.index_from_position[b][c][a] = d
                    self.index_from_position[c][a][b] = d
                    self.index_from_position[c][b][a] = d
                    self.position_from_index[d][0] = a
                    self.position_from_index[d][1] = b
                    self.position_from_index[d][2] = c
                    self.position_from_index[d][3] = 2

    def _reduce_canonical_symmetry_flags(self) -> None:
        w = self.width
        n_cells = self.number_of_cells
        n_pos = self.number_of_positions
        n_sym = self.number_of_symmetries
        for pos_idx in range(n_pos):
            for sym_idx in range(1, n_sym):
                if self.position_from_index[pos_idx][3] == 1:
                    self.position_from_index[pos_idx][3] = 2
                c = 0
                while c < n_cells and self.position_from_index[pos_idx][3] == 2:
                    s0 = self.symmetry[c][sym_idx]
                    if self.position_1col[pos_idx][c % w][c // w] != self.position_1col[pos_idx][s0 % w][s0 // w]:
                        if self.position_1col[pos_idx][c % w][c // w] == 1:
                            self.position_from_index[pos_idx][3] = 1
                        else:
                            self.position_from_index[pos_idx][3] = 0
                    c += 1
                if self.position_from_index[pos_idx][3] == 2:
                    self.position_from_index[pos_idx][3] = 1

    def _label_terminal_remaining_moves(self) -> int:
        """Count (a,b) with remaining_moves[a][b]==0 after terminal labeling; prints progress."""
        w = self.width
        n_pos = self.number_of_positions
        j = 0
        for a in range(n_pos):
            for b in range(n_pos):
                if self.completed_win[b] == 1:
                    self.remaining_moves[a][b] = 0
                if self.completed_win[a] == 1:
                    self.remaining_moves[a][b] = ILLEGAL_POSITION
                pfi_b = self.position_from_index[b]
                if (
                    self.position_1col[a][pfi_b[0] % w][pfi_b[0] // w] == 1
                    or self.position_1col[a][pfi_b[1] % w][pfi_b[1] // w] == 1
                    or self.position_1col[a][pfi_b[2] % w][pfi_b[2] // w] == 1
                ):
                    self.remaining_moves[a][b] = ILLEGAL_POSITION
                if self.remaining_moves[a][b] == 0:
                    j += 1
                    if j % 100_000 == 0:
                        print("-", end="")
        print("0 ", j)
        return j

    def _run_retrograde_analysis(self, initial_pending: int) -> int:
        """Return final depth c (one past last solved ply) when loop exits."""
        w = self.width
        board_h = self.height
        n_pos = self.number_of_positions
        n_sym = self.number_of_symmetries
        idx_from_pos = self.index_from_position
        pos_from_idx = self.position_from_index
        pos_1col = self.position_1col
        rem = self.remaining_moves
        sym = self.symmetry

        move_perm = [0 for _ in range(25)]
        j = initial_pending
        c = 1
        while j > 0:
            j = 0
            for a in range(n_pos):
                if pos_from_idx[a][3] == 1:
                    for b in range(n_pos):
                        if rem[a][b] == DRAW_POSITION:
                            stopped = False
                            for k in range(3):
                                d = pos_from_idx[a][k] % w
                                e = pos_from_idx[a][k] // w
                                for f in range(-1, 2):
                                    for g in range(-1, 2):
                                        if (
                                            f * f + g * g > 0
                                            and 0 <= d + f < w
                                            and 0 <= e + g < board_h
                                            and pos_1col[a][d + f][e + g] == 0
                                            and pos_1col[b][d + f][e + g] == 0
                                            and not stopped
                                        ):
                                            f1, g1 = f, g
                                            while (
                                                0 <= d + f1 + f < w
                                                and 0 <= e + g1 + g < board_h
                                                and pos_1col[a][d + f1 + f][e + g1 + g] == 0
                                                and pos_1col[b][d + f1 + f][e + g1 + g] == 0
                                            ):
                                                f1 += f
                                                g1 += g
                                            for h_idx in range(3):
                                                move_perm[h_idx] = pos_from_idx[a][h_idx]
                                            move_perm[k] = d + f1 + w * (e + g1)
                                            reply_idx = idx_from_pos[move_perm[0]][move_perm[1]][move_perm[2]]
                                            if c % 2 == 0:
                                                if rem[b][reply_idx] < c:
                                                    rem[a][b] = c
                                                else:
                                                    rem[a][b] = DRAW_POSITION
                                                    stopped = True
                                            else:
                                                if rem[b][reply_idx] == c - 1:
                                                    rem[a][b] = c
                                                    stopped = True
                            if rem[a][b] == c:
                                j += 1
                                if j % 100_000 == 0:
                                    print("-", end="")
                                for d_sym in range(1, n_sym):
                                    e_idx = idx_from_pos[
                                        sym[pos_from_idx[a][0]][d_sym]
                                    ][sym[pos_from_idx[a][1]][d_sym]][sym[pos_from_idx[a][2]][d_sym]]
                                    f_idx = idx_from_pos[
                                        sym[pos_from_idx[b][0]][d_sym]
                                    ][sym[pos_from_idx[b][1]][d_sym]][sym[pos_from_idx[b][2]][d_sym]]
                                    if rem[e_idx][f_idx] != c:
                                        rem[e_idx][f_idx] = c
                                        j += 1
                                        if j % 100_000 == 0:
                                            print("-", end="")
            print(str(c) + "  " + str(j))
            c += 1
        return c

    def _count_draw_pairs(self) -> int:
        n_pos = self.number_of_positions
        j = 0
        for a in range(n_pos):
            for b in range(n_pos):
                if self.remaining_moves[a][b] == DRAW_POSITION:
                    j += 1
        return j

    def default_opening_indices(self) -> tuple[int, int]:
        """Return (x_index, o_index) for the standard suggested start when applicable."""
        w, h = self.width, self.height
        ruleset = self.ruleset
        n_cells = self.number_of_cells
        idx = self.index_from_position

        if (w == 5 or w == 7) and h == w and (ruleset == 1 or ruleset == 5):
            if ruleset == 1:
                a = idx[(w - 3) // 2][(w + 1) // 2][(n_cells - 1) // 2 + w]
                b = idx[(n_cells - 1) // 2 - w][n_cells - (w + 3) // 2][n_cells - (w - 1) // 2]
            else:
                a = idx[(n_cells - 1) // 2 - 2 * w][(n_cells - 1) // 2 + w - 2][(n_cells - 1) // 2 + w + 2]
                b = idx[(n_cells - 1) // 2 - w - 2][(n_cells - 1) // 2 - w + 2][(n_cells - 1) // 2 + 2 * w]
            return a, b

        print("\n(This may not be a suitable opening position)")
        a = idx[0][2][n_cells - 2]
        b = idx[1][n_cells - 3][n_cells - 1]
        return a, b

    def print_deepest_positions(self) -> None:
        """Print all canonical x vs o positions at deepest win depth."""
        w = self.width
        n_pos = self.number_of_positions
        c = self.retrograde_depth
        print("\nDeepest position(s):")
        for a in range(n_pos):
            if self.position_from_index[a][3] == 1:
                for b in range(n_pos):
                    if self.remaining_moves[a][b] == c - 2:
                        line = "x -"
                        for d in range(3):
                            cell = self.position_from_index[a][d]
                            line += " " + chr(cell % w + 65) + str(1 + cell // w)
                        line += "   o -"
                        for d in range(3):
                            cell = self.position_from_index[b][d]
                            line += " " + chr(cell % w + 65) + str(1 + cell // w)
                        print(line)
