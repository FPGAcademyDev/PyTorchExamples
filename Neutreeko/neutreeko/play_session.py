"""Interactive oracle explorer after tables are built."""

from __future__ import annotations

from typing import Any

from neutreeko.constants import ILLEGAL_POSITION
from neutreeko.io_utils import read_int
from neutreeko.legal_moves import diff_board, enumerate_legal_moves, outcome_label, sorted_move_indices
from neutreeko.models import Move
from neutreeko.oracle import NeutreekoOracle


class OraclePlaySession:
    def __init__(
        self,
        oracle: NeutreekoOracle,
        *,
        ai_player: Any | None = None,
        ai_autoplay: bool = False,
        ai_vs_human: bool = False,
    ) -> None:
        self.oracle = oracle
        w, h = oracle.width, oracle.height
        self.width = w
        self.height = h
        self.ruleset = oracle.ruleset
        self.number_of_cells = oracle.number_of_cells
        self.number_of_positions = oracle.number_of_positions
        self.number_of_symmetries = oracle.number_of_symmetries

        self.position_2col = [[0 for _ in range(h)] for _ in range(w)]
        self.piece = ["o", " ", "x"]
        self.choice = ["" for _ in range(25)]
        self.move_information = [[0 for _ in range(6)] for _ in range(25)]
        self.move_permutation = [0 for _ in range(25)]

        self._x_pos: int = 0
        self._o_pos: int = 0
        self._move_head: Move = Move()
        self._ai_player = ai_player
        self._ai_autoplay = ai_autoplay
        self._ai_vs_human = ai_vs_human
        self._game_ply = 0

    def run(self) -> None:
        self._x_pos, self._o_pos = self.oracle.default_opening_indices()
        self._refresh_board_diff()
        self._move_head = Move()
        self._game_ply = 0

        if self._ai_vs_human:
            print("\nHuman vs AI: you choose first; the AI plays every other turn.\n")

        stripe = "\n +"
        for c in range(self.width):
            stripe += "---+"

        while self.oracle.remaining_moves[self._x_pos][self._o_pos] > 0:
            self._print_board(stripe)
            num_choices = self._enumerate_legal_moves()
            self._sort_move_choices(num_choices)
            self._print_move_menu(num_choices)
            ai_turn = False
            if self._ai_autoplay and self._ai_player is not None and num_choices > 0:
                ai_turn = True
            elif self._ai_vs_human and self._ai_player is not None and num_choices > 0:
                ai_turn = self._game_ply % 2 == 1

            if ai_turn:
                choice = self._ai_player.choose_menu_index(
                    self.oracle, self._x_pos, self._o_pos, self.position_2col, num_choices
                )
                print(f"Enter alternative: {choice} (AI)")
            else:
                choice = self._read_menu_choice(num_choices)
            self._apply_menu_choice(choice, num_choices)

        self._print_board(stripe)
        self._print_file_labels()
        print("\n\nGAME OVER\n")

    def _refresh_board_diff(self) -> None:
        self.position_2col = diff_board(self.oracle, self._x_pos, self._o_pos)

    def _print_board(self, stripe: str) -> None:
        w, h = self.width, self.height
        print(stripe)
        for d in range(h):
            row = str(h - d) + "|"
            for c in range(w):
                row += " " + self.piece[1 + self.position_2col[c][h - 1 - d]] + " |"
            print(row + stripe)
        labels = ""
        for c in range(w):
            labels += "   " + chr(c + 65)
        print(labels + "\n")

    def _print_file_labels(self) -> None:
        w, h = self.width, self.height
        stripe = "\n +"
        for c in range(w):
            stripe += "---+"
        print(stripe)
        for d in range(h):
            row = str(h - d) + "|"
            for c in range(w):
                row += " " + self.piece[1 + self.position_2col[c][h - 1 - d]] + " |"
            print(row + stripe)
        labels = ""
        for c in range(w):
            labels += "   " + chr(c + 65)
        print(labels)

    def _enumerate_legal_moves(self) -> int:
        o = self.oracle
        a, b = self._x_pos, self._o_pos

        if o.remaining_moves[a][b] == ILLEGAL_POSITION:
            print("Illegal position")
            return 0

        raw_moves = enumerate_legal_moves(o, a, b, self.position_2col)
        j = 0
        order = sorted_move_indices(raw_moves)
        for ui_idx, mi in enumerate(order, start=1):
            m = raw_moves[mi]
            j = ui_idx
            d, e = m.x1, m.y1
            self.choice[j] = (
                ". "
                + chr(d + 65)
                + str(e + 1)
                + " -> "
                + chr(m.x2 + 65)
                + str(m.y2 + 1)
                + " : "
                + outcome_label(m.raw_remaining)
            )
            sk = m.sort_key()
            self.move_information[j][0] = m.x1
            self.move_information[j][1] = m.y1
            self.move_information[j][2] = m.x2
            self.move_information[j][3] = m.y2
            self.move_information[j][4] = m.reply_idx
            self.move_information[j][5] = sk
        return j

    def _sort_move_choices(self, num_choices: int) -> None:
        for c in range(1, num_choices + 1):
            self.move_permutation[c] = c
        for c in range(1, num_choices):
            for d in range(c + 1, num_choices + 1):
                if (
                    self.move_information[self.move_permutation[c]][5]
                    > self.move_information[self.move_permutation[d]][5]
                ):
                    self.move_permutation[c], self.move_permutation[d] = (
                        self.move_permutation[d],
                        self.move_permutation[c],
                    )

    def _print_move_menu(self, num_choices: int) -> None:
        for c in range(1, num_choices + 1):
            print(str(c) + self.choice[self.move_permutation[c]])
        if self._move_head.prev is not None:
            print("77. Retract last move")
        if (self.width == 5 or self.width == 7) and self.height == self.width and (
            self.ruleset == 1 or self.ruleset == 5
        ):
            print("88. Back to start")
        print("99. Enter new position")

    def _read_menu_choice(self, num_choices: int) -> int:
        w, h = self.width, self.height
        r = self.ruleset
        d = 98
        while (
            d != 99
            and (d < 1 or d > num_choices)
            and (d != 88 or (w != 5 and w != 7) or h != w or (r != 1 and r != 5))
            and (d != 77 or self._move_head.prev is None)
        ):
            d = read_int("Enter alternative: ")
        return d

    def _apply_menu_choice(self, d: int, num_choices: int) -> None:
        if d == 77:
            self._retract_move()
            return
        if d == 88:
            self._reset_to_standard_opening()
            return
        if d == 99:
            self._enter_custom_position()
            return
        self._apply_move_choice(d, num_choices)

    def _retract_move(self) -> None:
        prev_x_index = self._x_pos
        self._x_pos = self._move_head.z1
        for c in range(self.width):
            for d in range(self.height):
                self.position_2col[c][d] = -self.position_2col[c][d]
        self.piece[1] = self.piece[0]
        self.piece[0] = self.piece[2]
        self.piece[2] = self.piece[1]
        self.piece[1] = " "
        self.position_2col[self._move_head.x1][self._move_head.y1] = 1
        self.position_2col[self._move_head.x2][self._move_head.y2] = 0
        self._o_pos = prev_x_index
        prev = self._move_head.prev
        assert prev is not None
        self._move_head = prev
        if self._ai_vs_human:
            self._game_ply = max(0, self._game_ply - 1)

    def _reset_to_standard_opening(self) -> None:
        o = self.oracle
        w = self.width
        n_cells = self.number_of_cells
        idx = o.index_from_position
        if self.ruleset == 1:
            self._x_pos = idx[(w - 3) // 2][(w + 1) // 2][(n_cells - 1) // 2 + w]
            self._o_pos = idx[(n_cells - 1) // 2 - w][n_cells - (w + 3) // 2][n_cells - (w - 1) // 2]
        else:
            self._x_pos = idx[(n_cells - 1) // 2 - 2 * w][(n_cells - 1) // 2 + w - 2][(n_cells - 1) // 2 + w + 2]
            self._o_pos = idx[(n_cells - 1) // 2 - w - 2][(n_cells - 1) // 2 - w + 2][(n_cells - 1) // 2 + 2 * w]
        self._refresh_board_diff()
        self.piece[0] = "o"
        self.piece[1] = " "
        self.piece[2] = "x"
        self._move_head = Move()
        self._game_ply = 0

    def _enter_custom_position(self) -> None:
        o = self.oracle
        w, h = self.width, self.height
        n_cells = self.number_of_cells
        idx = o.index_from_position

        h1 = 0
        h2 = 0
        h3 = 0
        while h1 != n_cells - 6 or h2 != 3 or h3 != 3:
            print("Enter one " + str(w) + "-digit number for each row")
            print("0 = blank; 1 = 'x' (plays first); 2 = 'o'")
            print("(The input is read as an integer, and starting zeros can be omitted)")
            for e in range(w):
                for f in range(h):
                    self.position_2col[e][f] = 0
            for e in range(h):
                f = read_int("")
                for g in range(w):
                    self.position_2col[w - 1 - g][h - 1 - e] = f % 10
                    f = f // 10
                for f in range(w):
                    if self.position_2col[f][h - 1 - e] == 2:
                        self.position_2col[f][h - 1 - e] = -1
            h1 = 0
            h2 = 0
            h3 = 0
            for e in range(w):
                for f in range(h):
                    if self.position_2col[e][f] == 0:
                        h1 += 1
                    if self.position_2col[e][f] == 1:
                        h2 += 1
                    if self.position_2col[e][f] == -1:
                        h3 += 1
            if h1 != n_cells - 6:
                print("Error: " + str(h1) + " blanks (should be " + str(n_cells - 6) + ")")
            if h2 != 3:
                print("Error: " + str(h2) + " x's (should be 3)")
            if h3 != 3:
                print("Error: " + str(h3) + " o's (should be 3)")

        h1 = 0
        while self.position_2col[h1 % w][h1 // w] <= 0:
            h1 += 1
        h2 = h1 + 1
        while self.position_2col[h2 % w][h2 // w] <= 0:
            h2 += 1
        h3 = h2 + 1
        while self.position_2col[h3 % w][h3 // w] <= 0:
            h3 += 1
        self._x_pos = idx[h1][h2][h3]

        h1 = 0
        while self.position_2col[h1 % w][h1 // w] >= 0:
            h1 += 1
        h2 = h1 + 1
        while self.position_2col[h2 % w][h2 // w] >= 0:
            h2 += 1
        h3 = h2 + 1
        while self.position_2col[h3 % w][h3 // w] >= 0:
            h3 += 1
        self._o_pos = idx[h1][h2][h3]

        self.piece[0] = "o"
        self.piece[1] = " "
        self.piece[2] = "x"
        self._move_head = Move()
        self._game_ply = 0

    def _apply_move_choice(self, d: int, num_choices: int) -> None:
        if d < 1 or d > num_choices:
            return
        c_perm = self.move_permutation[d]
        mi = self.move_information[c_perm]
        prev_head = self._move_head
        new_move = Move()
        new_move.prev = prev_head
        new_move.x1 = mi[0]
        new_move.y1 = mi[1]
        new_move.x2 = mi[2]
        new_move.y2 = mi[3]
        new_move.z1 = self._x_pos
        self._move_head = new_move

        self._x_pos = self._o_pos
        self._o_pos = mi[4]
        self.position_2col[mi[0]][mi[1]] = 0
        self.position_2col[mi[2]][mi[3]] = 1
        for c in range(self.width):
            for dcol in range(self.height):
                self.position_2col[c][dcol] = -self.position_2col[c][dcol]
        self.piece[1] = self.piece[0]
        self.piece[0] = self.piece[2]
        self.piece[2] = self.piece[1]
        self.piece[1] = " "
        if self._ai_vs_human:
            self._game_ply += 1
