"""Supervised training: imitate oracle-best moves (unsorted enumeration index)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn

from neutreeko.ai.encoding import MAX_MOVES, encode_board_tensor
from neutreeko.ai.model import NeutreekoPolicyNet
from neutreeko.constants import ILLEGAL_POSITION
from neutreeko.legal_moves import diff_board, enumerate_legal_moves, teacher_move_index_unsorted
from neutreeko.models import GameConfig
from neutreeko.oracle import NeutreekoOracle


def collect_samples(oracle: NeutreekoOracle, max_samples: int | None = None) -> list[tuple[int, int, int]]:
    """List of (x_idx, o_idx, teacher_unsorted) for non-terminal, legal positions."""
    n = oracle.number_of_positions
    samples: list[tuple[int, int, int]] = []
    for a in range(n):
        for b in range(n):
            if max_samples is not None and len(samples) >= max_samples:
                return samples
            rem = oracle.remaining_moves[a][b]
            if rem == ILLEGAL_POSITION or rem <= 0:
                continue
            grid = diff_board(oracle, a, b)
            moves = enumerate_legal_moves(oracle, a, b, grid)
            if not moves:
                continue
            t = teacher_move_index_unsorted(moves)
            samples.append((a, b, t))
    return samples


def train(
    oracle: NeutreekoOracle,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    max_samples: int | None = None,
) -> NeutreekoPolicyNet:
    samples = collect_samples(oracle, max_samples=max_samples)
    if not samples:
        raise RuntimeError("No training samples (check oracle / board configuration).")

    model = NeutreekoPolicyNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        random.shuffle(samples)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            boards: list[torch.Tensor] = []
            targets: list[int] = []
            for a, b, t in batch:
                grid = diff_board(oracle, a, b)
                boards.append(
                    encode_board_tensor(grid, oracle.width, oracle.height, device=device)
                )
                targets.append(t)
            x = torch.stack(boards, dim=0)
            y = torch.tensor(targets, dtype=torch.long, device=device)
            logits = model(x)
            # Mask unused logits so padding slots never win argmax during training.
            for i, (a, b, _) in enumerate(batch):
                k = len(enumerate_legal_moves(oracle, a, b, diff_board(oracle, a, b)))
                if k < MAX_MOVES:
                    logits[i, k:] = -1e9
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        acc = _eval_accuracy(model, samples[: min(2000, len(samples))], oracle, device)
        print(f"epoch {epoch + 1}/{epochs}  loss={avg:.4f}  acc(sample)={acc:.3f}")
    return model


def _eval_accuracy(
    model: NeutreekoPolicyNet,
    subset: list[tuple[int, int, int]],
    oracle: NeutreekoOracle,
    device: torch.device,
) -> float:
    model.eval()
    ok = 0
    with torch.no_grad():
        for a, b, t in subset:
            grid = diff_board(oracle, a, b)
            moves = enumerate_legal_moves(oracle, a, b, grid)
            k = len(moves)
            if k == 0:
                continue
            x = encode_board_tensor(grid, oracle.width, oracle.height, device=device).unsqueeze(0)
            logits = model(x).squeeze(0)
            if k < MAX_MOVES:
                logits[k:] = -1e9
            pred = int(torch.argmax(logits[:k]).item())
            if pred == t:
                ok += 1
    return ok / max(len(subset), 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Train Neutreeko policy net on oracle labels.")
    p.add_argument("--out", type=Path, default=Path("neutreeko_ai.pt"))
    p.add_argument("--width", type=int, default=5)
    p.add_argument("--height", type=int, default=5)
    p.add_argument("--ruleset", type=int, default=1)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-samples", type=int, default=None, help="Cap dataset size for quick tests")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = GameConfig(width=args.width, height=args.height, ruleset=args.ruleset)
    print("Building oracle (this can take a while)...")
    oracle = NeutreekoOracle(cfg)
    print("Collecting supervised positions...")
    model = train(
        oracle,
        args.epochs,
        args.batch_size,
        args.lr,
        device,
        max_samples=args.max_samples,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "width": oracle.width,
            "height": oracle.height,
            "ruleset": oracle.ruleset,
        },
        args.out,
    )
    print("Saved", args.out)


if __name__ == "__main__":
    main()
