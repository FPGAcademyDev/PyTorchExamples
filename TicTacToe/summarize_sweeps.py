#!/usr/bin/env python3
"""Summarize training/play sweep text logs into a CSV."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Tuple

KEY_RE = re.compile(r"^(train|play)\.e?(\d+)\.h(\d+)\.txt$")
LOSS_RE = re.compile(r"loss=([0-9]*\.?[0-9]+)")
GAMES_RE = re.compile(r"Total games played:\s*(\d+)")
SUBOPT_RE = re.compile(r"Games where AI played sub-optimally:\s*(\d+)")


def parse_key(path: Path) -> Tuple[int, int] | None:
    m = KEY_RE.match(path.name)
    if not m:
        return None
    epochs = int(m.group(2))
    hidden = int(m.group(3))
    return (epochs, hidden)


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


def parse_suboptimal_games(text: str) -> int | None:
    m = SUBOPT_RE.search(text)
    if not m:
        return None
    return int(m.group(1))


def summarize_logs(input_dir: Path) -> Dict[Tuple[int, int], dict]:
    rows: Dict[Tuple[int, int], dict] = {}
    for path in input_dir.glob("*.txt"):
        key = parse_key(path)
        if key is None:
            continue

        row = rows.setdefault(
            key,
            {
                "epochs": key[0],
                "hidden": key[1],
                "final_loss": None,
                "games_played": None,
                "sub_optimal_games": None,
                "train_file": "",
                "play_file": "",
            },
        )

        text = path.read_text(encoding="utf-8", errors="replace")
        if path.name.startswith("train."):
            row["final_loss"] = parse_final_loss(text)
            row["train_file"] = path.name
        elif path.name.startswith("play."):
            row["games_played"] = parse_games_played(text)
            row["sub_optimal_games"] = parse_suboptimal_games(text)
            row["play_file"] = path.name
    return rows


def write_csv(rows: Dict[Tuple[int, int], dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "epochs",
        "hidden",
        "final_loss",
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
