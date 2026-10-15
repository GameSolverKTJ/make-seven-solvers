"""Regression checks for the public unlimited-tile release."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from make7.game import Make7State, THREE_ONLY, move_to_action, new_game
from make7.model import Make7Net, ModelConfig, NetEvaluator
from make7.mcts import BatchedMCTS, MCTSConfig
import play
import train
import final_opening_evaluator


def main():
    # Isolate supply legality: the player has used four 3s, with a fifth
    # red square available at (row=4, column=1). This is a synthetic fixture,
    # not a claim about the reachability or value of this position.
    board = np.zeros((7, 7), dtype=np.int8)
    heights = np.zeros(7, dtype=np.int8)
    for row, col in [(2, 0), (2, 6), (3, 3), (4, 5)]:
        board[:row, col] = 1
        board[row, col] = 3
        heights[col] = row + 1
    board[:4, 1] = -1
    heights[1] = 4
    used = (int((board == 1).sum()), 0, 4, 4, 0, 0)
    args = dict(board=board, heights=heights, player=1,
                ply=int(np.count_nonzero(board)), used=used)
    unlimited = Make7State(**args, limits=None)
    limited = Make7State(**args, limits=THREE_ONLY)
    fifth_three = move_to_action(3, 1)
    assert unlimited.legal_actions()[fifth_three]
    assert not limited.legal_actions()[fifth_three]
    assert not unlimited.legal_actions()[move_to_action(3, 2)]
    assert unlimited.step(fifth_three).board[4, 1] == 3
    assert np.all(unlimited.encode()[8:10] == 1.0)

    for module, argv in [
        (play, ["play.py", "--checkpoint", "weights/best.pt"]),
        (train, ["train.py"]),
        (final_opening_evaluator,
         ["final_opening_evaluator.py", "--checkpoint", "weights/best.pt"]),
    ]:
        with patch.object(sys, "argv", argv):
            assert module.parse_args().tile_limits == "unlimited"
        with patch.object(sys, "argv", argv + ["--tile-limits", "three_only"]):
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    module.parse_args()
                except SystemExit as exc:
                    assert exc.code == 2
                else:
                    raise AssertionError("limited ruleset was accepted")

    weights = ROOT / "weights/best.pt"
    info = json.loads((ROOT / "MODEL_INFO.json").read_text())
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == info["checkpoint_sha256"]
    payload = torch.load(weights, map_location="cpu", weights_only=True)
    net = Make7Net(ModelConfig(**payload["model_config"]))
    net.load_state_dict(payload["state_dict"], strict=True)
    torch.set_num_threads(2)
    evaluator = NetEvaluator(net, "cpu", use_amp=False, symmetrise=True)
    states = [new_game(), unlimited]
    logits, values = evaluator(np.stack([s.encode() for s in states]))
    assert np.isfinite(logits).all() and np.isfinite(values).all()
    for mode in ["puct", "gumbel"]:
        search = BatchedMCTS(MCTSConfig(simulations=32, search_mode=mode),
                             np.random.default_rng(42))
        visits = search.search(states, evaluator, add_noise=False)
        for state, counts in zip(states, visits):
            assert counts.sum() > 0
            assert not np.any(counts[~state.legal_actions()] > 0)
    print("PASS: fifth 3, red-square restriction, unlimited encoding and CLI,")
    print("      unchanged checkpoint, CPU inference, PUCT and Gumbel searches")


if __name__ == "__main__":
    main()
