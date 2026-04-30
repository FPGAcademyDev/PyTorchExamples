#!/usr/bin/env python3
"""Summarize training/play sweep text logs into a CSV."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

TRAIN_RE = re.compile(r"^train\.h(\d+)\.txt$")
PLAY_RE = re.compile(r"^play\.h(\d+)\.e(\d+)\.txt$")
EPOCH_METRICS_RE = re.compile(
    r"epoch\s+(\d+)/\d+\s+loss=([0-9]*\.?[0-9]+)\s+training_time_seconds=([0-9]*\.?[0-9]+)"
)
GAMES_RE = re.compile(r"Total games played:\s*(\d+)")
SUBOPT_RE = re.compile(r"Games where AI played sub-optimally:\s*(\d+)")


def parse_train_key(path: Path) -> int | None:
    m = TRAIN_RE.match(path.name)
    if not m:
        return None
    return int(m.group(1))


def parse_play_key(path: Path) -> tuple[int, int] | None:
    # Always play.h$H.e$E.txt
    m = PLAY_RE.match(path.name)
    if not m:
        return None
    hidden = int(m.group(1))
    epoch = int(m.group(2))
    return (epoch, hidden)


def parse_games_played(text: str) -> int | None:
    m = GAMES_RE.search(text)
    if not m:
        return None
    return int(m.group(1))


def parse_epoch_metrics(text: str) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    for epoch_text, loss_text, t_text in EPOCH_METRICS_RE.findall(text):
        epoch = int(epoch_text)
        out[epoch] = (float(loss_text), float(t_text))
    return out


def parse_suboptimal_games(text: str) -> int | None:
    m = SUBOPT_RE.search(text)
    if not m:
        return None
    return int(m.group(1))


def summarize_logs(input_dir: Path) -> dict[tuple[str, int, int], dict]:
    # Row key is (run, hidden, epochs) so CSV groups epochs within each hidden.
    rows: dict[tuple[str, int, int], dict] = {}
    train_by_run_hidden: dict[tuple[str, int], dict] = {}
    train_metrics_by_run_hidden_epoch: dict[tuple[str, int, int], tuple[float, float]] = {}

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

        hidden = parse_train_key(path)
        if hidden is None:
            continue

        epoch_metrics = parse_epoch_metrics(text)
        for epoch_key, metrics in epoch_metrics.items():
            train_metrics_by_run_hidden_epoch[(run, hidden, epoch_key)] = metrics

        final_loss = None
        training_time = None
        if epoch_metrics:
            last_epoch = max(epoch_metrics)
            final_loss, training_time = epoch_metrics[last_epoch]

        train_by_run_hidden[(run, hidden)] = {
            "final_loss": final_loss,
            "training_time_seconds": training_time,
            "train_file": path.name,
        }

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
        epoch_metrics = train_metrics_by_run_hidden_epoch.get((run, hidden, epoch))
        if epoch_metrics is not None:
            epoch_loss, epoch_time = epoch_metrics
            row["epoch_loss"] = epoch_loss
            row["training_time_seconds"] = epoch_time
        elif row["training_time_seconds"] is None:
            train_summary = train_by_run_hidden.get((run, hidden))
            if train_summary is not None:
                row["training_time_seconds"] = train_summary["training_time_seconds"]

    return rows


def write_csv(rows: dict[tuple[str, int, int], dict], output_csv: Path) -> None:
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
        description="Extract per-epoch loss/time and AI test metrics from sweep logs into CSV."
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
