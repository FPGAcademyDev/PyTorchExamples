#!/usr/bin/env python3
"""Summarize training/play sweep text logs into a CSV."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Tuple

TRAIN_RE = re.compile(r"^train\.(?:e(\d+)\.)?h(\d+)\.txt$")
PLAY_EH_RE = re.compile(r"^play\.e(\d+)\.h(\d+)\.txt$")
PLAY_HE_RE = re.compile(r"^play\.h(\d+)\.e(\d+)\.txt$")
LOSS_RE = re.compile(r"loss=([0-9]*\.?[0-9]+)")
TRAIN_TIME_RE = re.compile(r"training_time_seconds=([0-9]*\.?[0-9]+)")
EPOCH_TIME_RE = re.compile(
    r"epoch\s+(\d+)/\d+\s+loss=[0-9]*\.?[0-9]+\s+training_time_seconds=([0-9]*\.?[0-9]+)"
)
EPOCH_LOSS_RE = re.compile(
    r"epoch\s+(\d+)/\d+\s+loss=([0-9]*\.?[0-9]+)\s+training_time_seconds=[0-9]*\.?[0-9]+"
)
GAMES_RE = re.compile(r"Total games played:\s*(\d+)")
SUBOPT_RE = re.compile(r"Games where AI played sub-optimally:\s*(\d+)")


def parse_train_key(path: Path) -> Tuple[int | None, int] | None:
    m = TRAIN_RE.match(path.name)
    if not m:
        return None
    epoch_text, hidden_text = m.group(1), m.group(2)
    epoch = int(epoch_text) if epoch_text is not None else None
    hidden = int(hidden_text)
    return (epoch, hidden)


def parse_play_key(path: Path) -> Tuple[int, int] | None:
    m = PLAY_EH_RE.match(path.name)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = PLAY_HE_RE.match(path.name)
    if m:
        return (int(m.group(2)), int(m.group(1)))
    return None


def parse_final_loss(text: str) -> float | None:
    matches = LOSS_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def parse_games_played(text: str) -> int | None:
    m = GAMES_RE.search(text)
    if not m:
        return None
    return int(m.group(1))


def parse_training_time_seconds(text: str) -> float | None:
    matches = TRAIN_TIME_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def parse_epoch_training_times(text: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for epoch_text, t_text in EPOCH_TIME_RE.findall(text):
        out[int(epoch_text)] = float(t_text)
    return out


def parse_epoch_losses(text: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for epoch_text, loss_text in EPOCH_LOSS_RE.findall(text):
        out[int(epoch_text)] = float(loss_text)
    return out


def parse_suboptimal_games(text: str) -> int | None:
    m = SUBOPT_RE.search(text)
    if not m:
        return None
    return int(m.group(1))


def summarize_logs(input_dir: Path) -> Dict[Tuple[str, int, int], dict]:
    # Row key is (run, hidden, epochs) so CSV groups epochs within each hidden.
    rows: Dict[Tuple[str, int, int], dict] = {}
    train_by_run_hidden: Dict[Tuple[str, int], dict] = {}
    train_times_by_run_hidden_epoch: Dict[Tuple[str, int, int], float] = {}
    train_losses_by_run_hidden_epoch: Dict[Tuple[str, int, int], float] = {}

    for path in input_dir.rglob("*.txt"):
        run = str(path.parent.relative_to(input_dir))
        if run == ".":
            run = "root"
        text = path.read_text(encoding="utf-8", errors="replace")

        play_key = parse_play_key(path)
        if play_key is not None:
            epoch, hidden = play_key
            key = (run, hidden, epoch)
            row = rows.setdefault(
                key,
                {
                    "run": run,
                    "hidden": hidden,
                    "epochs": epoch,
                    "epoch_loss": None,
                    "training_time_seconds": None,
                    "games_played": None,
                    "sub_optimal_games": None,
                    "train_file": "",
                    "play_file": "",
                },
            )
            row["games_played"] = parse_games_played(text)
            row["sub_optimal_games"] = parse_suboptimal_games(text)
            row["play_file"] = path.name
            continue

        train_key = parse_train_key(path)
        if train_key is None:
            continue

        epoch, hidden = train_key
        final_loss = parse_final_loss(text)
        training_time = parse_training_time_seconds(text)
        epoch_times = parse_epoch_training_times(text)
        epoch_losses = parse_epoch_losses(text)
        for epoch_key, t_value in epoch_times.items():
            train_times_by_run_hidden_epoch[(run, hidden, epoch_key)] = t_value
        for epoch_key, loss_value in epoch_losses.items():
            train_losses_by_run_hidden_epoch[(run, hidden, epoch_key)] = loss_value
        if epoch is None:
            train_by_run_hidden[(run, hidden)] = {
                "final_loss": final_loss,
                "training_time_seconds": training_time,
                "train_file": path.name,
            }
            continue

        key = (run, hidden, epoch)
        row = rows.setdefault(
            key,
            {
                "run": run,
                "hidden": hidden,
                "epochs": epoch,
                "epoch_loss": None,
                "training_time_seconds": None,
                "games_played": None,
                "sub_optimal_games": None,
                "train_file": "",
                "play_file": "",
            },
        )
        row["epoch_loss"] = final_loss
        row["training_time_seconds"] = training_time
        row["train_file"] = path.name

    for key, row in rows.items():
        run, hidden, _ = key
        train_summary = train_by_run_hidden.get((run, hidden))
        if train_summary is None:
            continue
        if row["epoch_loss"] is None:
            row["epoch_loss"] = train_summary["final_loss"]
        if not row["train_file"]:
            row["train_file"] = train_summary["train_file"]

    for key, row in rows.items():
        run, hidden, epoch = key
        epoch_loss = train_losses_by_run_hidden_epoch.get((run, hidden, epoch))
        epoch_time = train_times_by_run_hidden_epoch.get((run, hidden, epoch))
        if epoch_loss is not None:
            row["epoch_loss"] = epoch_loss
        if epoch_time is not None:
            row["training_time_seconds"] = epoch_time
        elif row["training_time_seconds"] is None:
            train_summary = train_by_run_hidden.get((run, hidden))
            if train_summary is not None:
                row["training_time_seconds"] = train_summary["training_time_seconds"]

    return rows


def write_csv(rows: Dict[Tuple[str, int, int], dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "run",
        "hidden",
        "epochs",
        "epoch_loss",
        "training_time_seconds",
        "games_played",
        "sub_optimal_games",
        "train_file",
        "play_file",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(rows[key])


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract final loss, games played, and sub-optimal games from sweep .txt logs into CSV."
    )
    p.add_argument("--input-dir", type=Path, default=Path("ai_sweeps"), help="Directory containing train/play .txt logs")
    p.add_argument("--out", type=Path, default=Path("ai_sweeps/sweep_results.csv"), help="Output CSV path")
    args = p.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")

    rows = summarize_logs(args.input_dir)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
