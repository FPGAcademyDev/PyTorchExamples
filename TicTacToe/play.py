#!/usr/bin/env python3
"""Play Tic Tac Toe in the terminal against the trained PyTorch policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tic_tac_toe.game import Board, Cell
from tic_tac_toe.minimax import minimax
from tic_tac_toe.model import PolicyNet


def print_board(b: Board) -> None:
    def ch(c: int) -> str:
        if c == Cell.EMPTY:
            return "."
        if c == Cell.X:
            return "X"
        return "O"

    row = lambda r: " ".join(ch(b.cells[r * 3 + c]) for c in range(3))
    print()
    for r in range(3):
        print(row(r))
    print()


def load_model(path: Path, device: torch.device) -> PolicyNet:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    hidden = int(ckpt.get("hidden", 128))
    m = PolicyNet(hidden=hidden).to(device)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m


def legal_mask(b: Board, device: torch.device) -> torch.Tensor:
    m = torch.zeros(9, dtype=torch.bool, device=device)
    for i in b.legal_moves():
        m[i] = True
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("policy.pt"))
    ap.add_argument("--human", choices=("X", "O"), default="X", help="Human plays this side (X goes first)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if not args.weights.is_file():
        raise SystemExit(f"Missing {args.weights}. Run: python train.py")

    net = load_model(args.weights, device)
    board = Board.empty()

    human_mark = Cell.X if args.human == "X" else Cell.O
    print("You are", args.human, ". Enter moves as row col (0-2), e.g. 1 2")
    print_board(board)

    while board.winner() is None:
        side = board.current_player()
        if side == human_mark:
            line = input("Your move (r c): ").strip()
            parts = line.split()
            if len(parts) != 2:
                print("Need two numbers.")
                continue
            try:
                r, c = int(parts[0]), int(parts[1])
            except ValueError:
                print("Invalid input.")
                continue
            if not (0 <= r < 3 and 0 <= c < 3):
                print("Out of range.")
                continue
            idx = r * 3 + c
            if idx not in board.legal_moves():
                print("Illegal move.")
                continue
            board.apply(idx)
        else:
            x = torch.from_numpy(board.to_model_input()).to(device)
            mask = legal_mask(board, device)
            idx = net.best_move(x, mask)
            _, optimal = minimax(board)
            if idx not in optimal:
                print("(warning: model chose a non-minimax move; retrain or check weights)")
            board.apply(idx)
            print(f"AI plays {idx // 3} {idx % 3}")
        print_board(board)

    w = board.winner()
    if w == 0:
        print("Draw.")
    elif w == human_mark:
        print("You win.")
    else:
        print("AI wins.")


if __name__ == "__main__":
    main()
