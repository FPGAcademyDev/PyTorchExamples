#!/usr/bin/env python3
"""CLI entry: configure board, precompute the oracle, then explore interactively."""

from __future__ import annotations

import argparse
from pathlib import Path

from neutreeko.config_prompt import prompt_game_config
from neutreeko.oracle import NeutreekoOracle
from neutreeko.play_session import OraclePlaySession


def main() -> None:
    parser = argparse.ArgumentParser(description="Neutreeko oracle explorer (J K Haugland).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--ai",
        type=Path,
        default=None,
        metavar="CHECKPOINT.pt",
        help="AI plays every move (watch / demo).",
    )
    mode.add_argument(
        "--vs-ai",
        type=Path,
        default=None,
        metavar="CHECKPOINT.pt",
        help="You play first; the AI plays every other turn.",
    )
    args = parser.parse_args()

    config = prompt_game_config()
    print("\nCounting positions...")
    oracle = NeutreekoOracle(config)
    print("\nNumber of draws:", oracle.draw_pair_count)
    oracle.print_deepest_positions()

    ai_player = None
    checkpoint = args.ai or args.vs_ai
    if checkpoint is not None:
        try:
            from neutreeko.ai.player import PolicyAgent
        except ImportError as err:
            raise SystemExit(
                "PyTorch is required for --ai / --vs-ai. Install with: pip install -r requirements-ai.txt"
            ) from err

        ai_player = PolicyAgent(checkpoint)

    OraclePlaySession(
        oracle,
        ai_player=ai_player,
        ai_autoplay=args.ai is not None,
        ai_vs_human=args.vs_ai is not None,
    ).run()


if __name__ == "__main__":
    main()
