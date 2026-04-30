#!/usr/bin/env python3
"""Train PolicyNet on minimax-optimal moves for all reachable non-terminal states."""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tic_tac_toe.game import Board, Cell
from tic_tac_toe.minimax import minimax
from tic_tac_toe.model import PolicyNet


def reachable_non_terminal_boards() -> list[Board]:
    seen: set[tuple[int, ...]] = set()
    out: list[Board] = []
    q: deque[Board] = deque([Board.empty()])
    while q:
        b = q.popleft()
        key = tuple(b.cells)
        if key in seen:
            continue
        seen.add(key)
        if b.winner() is not None:
            continue
        out.append(b.copy())
        for m in b.legal_moves():
            nb = b.copy()
            nb.apply(m)
            q.append(nb)
    return out


def build_dataset() -> tuple[np.ndarray, np.ndarray]:
    boards = reachable_non_terminal_boards()
    xs = np.stack([b.to_model_input() for b in boards], axis=0)
    ys = np.zeros(len(boards), dtype=np.int64)
    for i, b in enumerate(boards):
        _, moves = minimax(b)
        ys[i] = min(moves)
    return xs, ys


def checkpoint_name(out: Path, epoch: int, hidden: int) -> Path:
    """$P.h$H.e$E.pt where P is --out with its file suffix removed."""
    p = str(out.with_suffix(""))
    return Path(f"{p}.h{hidden}.e{epoch}.pt")


def save_weights(path: Path, model: nn.Module, hidden: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden": hidden,
        },
        path,
    )
    print(f"saved {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--out", type=Path, default=Path("policy.pt"))
    args = p.parse_args()

    xs, ys = build_dataset()
    print(f"x shape: {xs.shape}:")
    print(f"y shape {ys.shape}:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    x_t = torch.from_numpy(xs).to(device)
    y_t = torch.from_numpy(ys).to(device)
    loader = DataLoader(TensorDataset(x_t, y_t), batch_size=args.batch, shuffle=True)

    model = PolicyNet(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    train_start = time.perf_counter()
    for epoch in range(args.epochs):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
        avg = total / len(x_t)
        e = epoch + 1
        if e % 50 == 0 or epoch == 0:
            elapsed = time.perf_counter() - train_start
            print(f"epoch {e}/{args.epochs}  loss={avg:.6f}  training_time_seconds={elapsed:.3f}")
        if e % 50 == 0:
            save_weights(checkpoint_name(args.out, e, args.hidden), model, args.hidden)
    train_seconds = time.perf_counter() - train_start
    print(f"training_time_seconds={train_seconds:.3f}")

    print(f"  ({len(xs)} positions)")


if __name__ == "__main__":
    main()
