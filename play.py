#!/usr/bin/env python3
"""Play a game against a trained network.

    python play.py --checkpoint runs/make7/best.pt --simulations 800

Enter moves as tile+column, e.g. `2D` or `3G`. `#` marks a red square, on
which a 3 is allowed to land. GREEN tiles print bare, YELLOW tiles in
parentheses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from make7.game import action_to_str, new_game, str_to_action
from make7.mcts import BatchedMCTS, MCTSConfig
from make7.model import NetEvaluator, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--simulations", type=int, default=800)
    p.add_argument("--tile-limits", choices=["unlimited"], default="unlimited",
                   help="Unlimited supply of 1, 2 and 3 tiles.")
    p.add_argument("--human", default="green", choices=["green", "yellow"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--show-thinking", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net, generation = load_checkpoint(args.checkpoint, device)
    evaluator = NetEvaluator(net, device, use_amp=(device.type == "cuda"))
    mcts = BatchedMCTS(MCTSConfig(simulations=args.simulations),
                       np.random.default_rng())
    human = 1 if args.human == "green" else -1

    print(f"generation {generation}, {args.simulations} simulations per move; tiles unlimited")
    state = new_game(limits=None)

    while not state.is_terminal:
        print()
        print(state)
        if state.player == human:
            legal = state.legal_action_list()
            while True:
                raw = input("your move: ").strip().upper()
                if raw in ("Q", "QUIT", "EXIT"):
                    return
                try:
                    action = str_to_action(raw)
                except (ValueError, IndexError):
                    print("  format is tile+column, e.g. 2D")
                    continue
                if action not in legal:
                    print("  not legal here. options: "
                          + " ".join(action_to_str(a) for a in legal))
                    continue
                break
        else:
            visits = mcts.search([state], evaluator, add_noise=False)[0]
            action = int(np.argmax(visits))
            if args.show_thinking:
                order = np.argsort(-visits)[:5]
                detail = "  ".join(
                    f"{action_to_str(int(a))} {visits[a] / visits.sum():.0%}"
                    for a in order if visits[a] > 0)
                print(f"  thinking: {detail}")
            print(f"  engine plays {action_to_str(action)}")
        state = state.step(action)

    print()
    print(state)
    if state.winner == 0:
        print("draw")
    elif state.winner == human:
        print("you win")
    else:
        print("the engine wins")


if __name__ == "__main__":
    main()
