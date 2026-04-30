#!/usr/bin/env python3
"""Play Tic Tac Toe in the terminal against the trained PyTorch policy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

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


def load_model(path: Path, device: torch.device, model_num: int | None) -> PolicyNet:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    hidden = int(ckpt.get("hidden", 128))
    chosen_model = int(ckpt.get("model_num", 1)) if model_num is None else model_num
    m = PolicyNet(hidden=hidden, model_num=chosen_model).to(device)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m


def legal_mask(b: Board, device: torch.device) -> torch.Tensor:
    m = torch.zeros(9, dtype=torch.bool, device=device)
    for i in b.legal_moves():
        m[i] = True
    return m


def idx_to_rc(move: int) -> str:
    return f"{move // 3} {move % 3}"


def format_moves(moves: Sequence[int]) -> str:
    if not moves:
        return "(none)"
    return " -> ".join(idx_to_rc(m) for m in moves)


def ai_pick_move(net: PolicyNet, board: Board, device: torch.device) -> int:
    x = torch.from_numpy(board.to_model_input()).to(device)
    mask = legal_mask(board, device)
    return net.best_move(x, mask)


def run_ai_test_mode(net: PolicyNet, device: torch.device, ai_side: int) -> Tuple[int, int, List[int]]:
    """
    Play all possible games against the AI.
    Opponent enumerates all legal moves, AI uses policy move.
    Returns:
      - total_games
      - games_with_suboptimal_ai_move
      - one example move list where AI played sub-optimally
    """

    total_games = 0
    games_with_suboptimal = 0
    example_suboptimal_game: List[int] = []

    def dfs(board: Board, moves: List[int], has_suboptimal: bool) -> None:
        nonlocal total_games, games_with_suboptimal, example_suboptimal_game

        if board.winner() is not None:
            total_games += 1
            if has_suboptimal:
                games_with_suboptimal += 1
                if not example_suboptimal_game:
                    example_suboptimal_game = list(moves)
            return

        side = board.current_player()
        if side == ai_side:
            ai_move = ai_pick_move(net, board, device)
            _, optimal_moves = minimax(board)
            next_has_suboptimal = has_suboptimal or (ai_move not in optimal_moves)
            next_board = board.copy()
            next_board.apply(ai_move)
            dfs(next_board, [*moves, ai_move], next_has_suboptimal)
            return

        for opp_move in board.legal_moves():
            next_board = board.copy()
            next_board.apply(opp_move)
            dfs(next_board, [*moves, opp_move], has_suboptimal)

    dfs(Board.empty(), [], False)
    return total_games, games_with_suboptimal, example_suboptimal_game


def print_ai_test_results(net: PolicyNet, device: torch.device) -> None:
    total_games = 0
    games_with_suboptimal = 0
    example_suboptimal_game: List[int] = []

    for side, side_name in ((Cell.X, "X"), (Cell.O, "O")):
        side_total, side_suboptimal, side_example = run_ai_test_mode(net, device, side)
        total_games += side_total
        games_with_suboptimal += side_suboptimal
        if not example_suboptimal_game and side_example:
            example_suboptimal_game = side_example
        print(f"[AI as {side_name}] games={side_total}, suboptimal_games={side_suboptimal}")

    print()
    print("AI Test Mode Results")
    print(f"Total games played: {total_games}")
    print(f"Games where AI played sub-optimally: {games_with_suboptimal}")
    if example_suboptimal_game:
        print(f"Moves of one sub-optimal game: {format_moves(example_suboptimal_game)}")
    else:
        print("Moves of one sub-optimal game: none (AI always chose minimax-optimal moves)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("policy.pt"))
    ap.add_argument("--human", choices=("X", "O"), default="X", help="Human plays this side (X goes first)")
    ap.add_argument(
        "--model",
        type=int,
        choices=range(1, PolicyNet.NUM_MODELS + 1),
        default=None,
        help=f"Policy model number (1..{PolicyNet.NUM_MODELS}); default: checkpoint value",
    )
    ap.add_argument(
        "--test-ai",
        action="store_true",
        help="Run exhaustive AI test mode (all opponent move sequences) and print weakness stats",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if not args.weights.is_file():
        raise SystemExit(f"Missing {args.weights}. Run: python train.py")

    net = load_model(args.weights, device, args.model)
    if args.test_ai:
        print_ai_test_results(net, device)
        return

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
            idx = ai_pick_move(net, board, device)
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
