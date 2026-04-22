from __future__ import annotations

from pathlib import Path

import torch

from neutreeko.ai.encoding import encode_board_tensor
from neutreeko.ai.model import NeutreekoPolicyNet
from neutreeko.legal_moves import enumerate_legal_moves, sorted_move_indices
from neutreeko.oracle import NeutreekoOracle


class PolicyAgent:
    """Loads a trained policy and maps board states to menu indices (1-based, oracle-sorted UI)."""

    def __init__(self, checkpoint_path: str | Path, device: str | torch.device | None = None) -> None:
        path = Path(checkpoint_path)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        self.width: int = int(payload["width"])
        self.height: int = int(payload["height"])
        self.ruleset: int = int(payload.get("ruleset", 1))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = NeutreekoPolicyNet()
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _assert_board(self, oracle: NeutreekoOracle) -> None:
        if (oracle.width, oracle.height) != (self.width, self.height):
            raise ValueError(
                f"Checkpoint is for {self.width}x{self.height} but oracle is "
                f"{oracle.width}x{oracle.height}. Train a matching checkpoint."
            )
        if oracle.ruleset != self.ruleset:
            raise ValueError(
                f"Checkpoint ruleset {self.ruleset} does not match oracle ruleset {oracle.ruleset}."
            )

    def choose_menu_index(
        self,
        oracle: NeutreekoOracle,
        x_idx: int,
        o_idx: int,
        position_2col: list[list[int]],
        num_choices: int,
    ) -> int:
        self._assert_board(oracle)
        if num_choices <= 0:
            return 1
        raw_moves = enumerate_legal_moves(oracle, x_idx, o_idx, position_2col)
        if len(raw_moves) != num_choices:
            raise RuntimeError(
                f"AI move list length {len(raw_moves)} != UI choices {num_choices}"
            )

        board = encode_board_tensor(
            position_2col, oracle.width, oracle.height, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(board).squeeze(0)
        k = len(raw_moves)
        u = int(torch.argmax(logits[:k]).item())
        perm = sorted_move_indices(raw_moves)
        return int(perm.index(u)) + 1
