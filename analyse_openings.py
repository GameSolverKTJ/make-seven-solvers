#!/usr/bin/env python3
"""Measure how good each opening move is according to a trained network.

    python analyse_openings.py --checkpoint runs/make7/best.pt \
        --games 400 --simulations 400

Three independent readings are produced for every legal first move:

  raw value      the value head's opinion, no search
  search value   the root Q after `--simulations` MCTS simulations
  empirical      GREEN's score over `--games` games in which GREEN is
                 *forced* to open with that move and both sides then play
                 normally, reported with a 95% Wilson interval

The empirical column is the one to quote.  The first two are cheap and can
disagree with it when the network is under-trained, which is itself a useful
diagnostic.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from make7.arena import wilson_interval
from make7.game import ACTION_SIZE, action_to_str, new_game
from make7.mcts import BatchedMCTS, MCTSConfig, visits_to_policy
from make7.model import NetEvaluator, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--games", type=int, default=400,
                   help="games played per opening move")
    p.add_argument("--simulations", type=int, default=400)
    p.add_argument("--search-mode", default="gumbel", choices=["puct", "gumbel"])
    p.add_argument("--gumbel-root-candidates", type=int, default=0)
    p.add_argument("--gumbel-c-visit", type=float, default=8.0)
    p.add_argument("--gumbel-c-scale", type=float, default=0.10)
    p.add_argument("--symmetrise", dest="symmetrise", action="store_true", default=True)
    p.add_argument("--no-symmetrise", dest="symmetrise", action="store_false")
    p.add_argument("--parallel-games", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.3,
                   help="sampling temperature during the games; a little "
                        "randomness is needed or every game is identical")
    p.add_argument("--temperature-moves", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("opening_analysis.csv"))
    return p.parse_args()


def play_from(start_states, evaluator, cfg, rng, temperature, temperature_moves,
              parallel_games):
    """Play out a list of positions to the end; return winners (+1/-1/0)."""
    mcts = BatchedMCTS(cfg, rng)
    winners = []
    for chunk_start in range(0, len(start_states), parallel_games):
        states = list(start_states[chunk_start:chunk_start + parallel_games])
        move_number = 0
        while any(not s.is_terminal for s in states):
            if cfg.search_mode == "gumbel":
                visits, improved = mcts.search(states, evaluator, add_noise=False,
                                               with_policy=True)
            else:
                visits = mcts.search(states, evaluator, add_noise=False)
                improved = None
            t = temperature if move_number < temperature_moves else 0.0
            for i, s in enumerate(states):
                if s.is_terminal:
                    continue
                source = improved[i] if improved is not None else visits[i]
                probs = visits_to_policy(source, t)
                if probs.sum() <= 0:
                    continue
                states[i] = s.step(int(rng.choice(ACTION_SIZE, p=probs)))
            move_number += 1
        winners.extend(s.winner for s in states)
    return winners


@torch.inference_mode()
def raw_opening_values(net, device, openings):
    root = new_game()
    children = [root.step(a) for a in openings]
    batch = torch.from_numpy(np.stack([c.encode() for c in children])).to(device)
    net.eval()
    _, values = net(batch)
    # value is for the player to move (YELLOW); negate for GREEN's view
    return (-values.float().cpu().numpy())


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net, generation = load_checkpoint(args.checkpoint, device)
    net.eval()
    print(f"loaded generation {generation} "
          f"({net.num_parameters():,} parameters) on {device}")

    rng = np.random.default_rng(args.seed)
    evaluator = NetEvaluator(net, device, use_amp=(device.type == "cuda"),
                             symmetrise=args.symmetrise)

    root = new_game()
    openings = np.flatnonzero(root.legal_actions()).tolist()

    # --- search evaluation at the root ----------------------------------- #
    search_cfg = MCTSConfig(
        simulations=max(args.simulations * 4, 1600),
        search_mode=args.search_mode,
        gumbel_root_candidates=args.gumbel_root_candidates,
        gumbel_c_visit=args.gumbel_c_visit, gumbel_c_scale=args.gumbel_c_scale)
    mcts = BatchedMCTS(search_cfg, rng)
    if args.search_mode == "gumbel":
        root_node_visits, root_policy = mcts.search(
            [root], evaluator, add_noise=False, with_policy=True)
        root_node_visits = root_node_visits[0]
        root_share = root_policy[0]
    else:
        root_node_visits = mcts.search([root], evaluator, add_noise=False)[0]
        root_share = root_node_visits / max(root_node_visits.sum(), 1)

    raw = raw_opening_values(net, device, openings)

    # --- forced-opening tournament --------------------------------------- #
    cfg = MCTSConfig(
        simulations=args.simulations, temperature_moves=args.temperature_moves,
        search_mode=args.search_mode,
        gumbel_root_candidates=args.gumbel_root_candidates,
        gumbel_c_visit=args.gumbel_c_visit, gumbel_c_scale=args.gumbel_c_scale)
    results = {}
    for a, raw_value in zip(openings, raw):
        starts = [root.step(a) for _ in range(args.games)]
        winners = play_from(starts, evaluator, cfg, rng, args.temperature,
                            args.temperature_moves, args.parallel_games)
        g = len(winners)
        wins = sum(1 for w in winners if w == 1)
        losses = sum(1 for w in winners if w == -1)
        draws = g - wins - losses
        score = (wins + 0.5 * draws) / g
        lo, hi = wilson_interval(wins + 0.5 * draws, g)
        results[a] = {
            "action": action_to_str(a),
            "raw_value": float(raw_value),
            "root_visit_share": float(root_share[a]),
            "games": g, "wins": wins, "losses": losses, "draws": draws,
            "green_score": score, "ci_low": lo, "ci_high": hi,
        }
        print(f"  {action_to_str(a):>3}  green score {score:.3f} "
              f"[{lo:.3f}, {hi:.3f}]  W/D/L {wins}/{draws}/{losses}  "
              f"raw {raw_value:+.3f}  root visits {root_share[a]:.1%}")

    ranked = sorted(results.values(), key=lambda r: -r["green_score"])

    print("\nranking by empirical GREEN score")
    print(f"{'move':>5} {'score':>7} {'95% CI':>18} {'raw':>7} {'visits':>8}")
    for r in ranked:
        ci = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        print(f"{r['action']:>5} {r['green_score']:7.3f} {ci:>18} "
              f"{r['raw_value']:+7.3f} {r['root_visit_share']:7.1%}")

    best, second = ranked[0], ranked[1]
    separated = best["ci_low"] > second["ci_high"]
    print(f"\nbest opening: {best['action']} "
          f"({best['green_score']:.3f}, 95% CI "
          f"[{best['ci_low']:.3f}, {best['ci_high']:.3f}])")
    print(f"runner-up:    {second['action']} ({second['green_score']:.3f})")
    print("the top two are statistically separated at 95%"
          if separated else
          "the top two OVERLAP at 95% — more games are needed to rank them")

    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ranked[0].keys()))
        writer.writeheader()
        writer.writerows(ranked)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
