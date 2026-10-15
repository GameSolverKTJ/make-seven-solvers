#!/usr/bin/env python3
"""Test whether a network improved as GREEN and as YELLOW by the same amount.

    python colour_split_match.py --a runs/make7b/best.pt \
        --b runs/make7b/anchor_gen55.pt --games 400

Why this is worth measuring
---------------------------
In self-play both sides share one network, so a falling first-player win
rate has two very different explanations that the self-play log cannot
separate:

  * the game really does favour the second player, or
  * the network happens to be getting better at playing second than at
    playing first.

Under the second explanation the arena score still rises — the new network
genuinely beats the old one — while the self-play first-player rate keeps
sliding, which is exactly the pattern that prompted this script.

Fixing the colours breaks the symmetry.  Network A plays network B twice,
once as GREEN and once as YELLOW.  But those two scores cannot be compared
against 0.5, because whichever side the game itself favours will score above
0.5 even between equally strong players.  Comparing to 0.5 would report a
one-sided gain purely because the game is one-sided.

So a third match is played first: B against itself with the colours fixed.
That measures `p`, the first-player score between two equally strong players
at B's level, and gives the baseline each colour should be judged against:

    gain as GREEN  = (A's score as GREEN)  -  p
    gain as YELLOW = (A's score as YELLOW) - (1 - p)

Both gains are then on the same footing, and the game's own bias has been
divided out.  Their sum is A's overall improvement over B — a figure that
happens to be free of `p` either way, which is a useful check that the three
measurements hang together.

The openings are randomised for a few plies so the games are not all
identical, and every match uses the same seed, so all three see the same set
of opening positions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from make7.arena import score_to_elo, wilson_interval
from make7.game import ACTION_SIZE, new_game
from make7.mcts import BatchedMCTS, MCTSConfig, visits_to_policy
from make7.model import NetEvaluator, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, type=Path, help="network under test")
    p.add_argument("--b", required=True, type=Path, help="reference network")
    p.add_argument("--games", type=int, default=400,
                   help="games per colour assignment")
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--search-mode", default="gumbel", choices=["puct", "gumbel"])
    p.add_argument("--gumbel-root-candidates", type=int, default=0)
    p.add_argument("--parallel-games", type=int, default=256)
    p.add_argument("--random-opening-plies", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def play_fixed_colours(green_eval, yellow_eval, games, cfg, seed,
                       random_opening_plies, temperature, parallel_games):
    """Play `games` with the colour assignment fixed. Returns winners."""
    rng = np.random.default_rng(seed)
    mcts_g = BatchedMCTS(cfg, rng)
    mcts_y = BatchedMCTS(cfg, rng)
    winners = []

    for start in range(0, games, parallel_games):
        n = min(parallel_games, games - start)
        states = [new_game() for _ in range(n)]
        for _ in range(random_opening_plies):
            for i, s in enumerate(states):
                if not s.is_terminal:
                    states[i] = s.step(int(rng.choice(s.legal_action_list())))

        while any(not s.is_terminal for s in states):
            for player, engine, evaluator in (
                    (1, mcts_g, green_eval), (-1, mcts_y, yellow_eval)):
                idx = [i for i, s in enumerate(states)
                       if not s.is_terminal and s.player == player]
                if not idx:
                    continue
                subset = [states[i] for i in idx]
                if cfg.search_mode == "gumbel":
                    visits, improved = engine.search(subset, evaluator,
                                                     add_noise=False, with_policy=True)
                else:
                    visits = engine.search(subset, evaluator, add_noise=False)
                    improved = None
                for k, i in enumerate(idx):
                    source = improved[k] if improved is not None else visits[k]
                    probs = visits_to_policy(source, temperature)
                    if probs.sum() <= 0:
                        continue
                    states[i] = states[i].step(
                        int(rng.choice(ACTION_SIZE, p=probs)))

        winners.extend(s.winner for s in states)
    return winners


def summarise(winners, a_is_green: bool):
    n = len(winners)
    draws = sum(1 for w in winners if w == 0)
    a_wins = sum(1 for w in winners
                 if (w == 1) == a_is_green and w != 0)
    b_wins = n - a_wins - draws
    score = (a_wins + 0.5 * draws) / n
    lo, hi = wilson_interval(a_wins + 0.5 * draws, n)
    return {"games": n, "a_wins": a_wins, "b_wins": b_wins, "draws": draws,
            "score": score, "lo": lo, "hi": hi, "elo": score_to_elo(score)}


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    net_a, gen_a = load_checkpoint(args.a, device)
    net_b, gen_b = load_checkpoint(args.b, device)
    eval_a = NetEvaluator(net_a, device, use_amp=(device.type == "cuda"))
    eval_b = NetEvaluator(net_b, device, use_amp=(device.type == "cuda"))
    print(f"A = {args.a.name} (generation {gen_a})")
    print(f"B = {args.b.name} (generation {gen_b})")
    print(f"{args.games} games per colour, {args.simulations} simulations\n")

    cfg = MCTSConfig(simulations=args.simulations, search_mode=args.search_mode,
                     gumbel_root_candidates=args.gumbel_root_candidates)

    print("B against itself (measuring the first-player baseline) ...")
    w0 = play_fixed_colours(eval_b, eval_b, args.games, cfg, args.seed,
                            args.random_opening_plies, args.temperature,
                            args.parallel_games)
    b_green = sum(1 for w in w0 if w == 1)
    b_draw = sum(1 for w in w0 if w == 0)
    p_first = (b_green + 0.5 * b_draw) / len(w0)
    plo, phi = wilson_interval(b_green + 0.5 * b_draw, len(w0))
    print(f"  first-player score at B's level: {p_first:.3f} "
          f"[95% CI {plo:.3f}, {phi:.3f}]\n")

    print("A as GREEN ...")
    w1 = play_fixed_colours(eval_a, eval_b, args.games, cfg, args.seed,
                            args.random_opening_plies, args.temperature,
                            args.parallel_games)
    as_green = summarise(w1, a_is_green=True)

    print("A as YELLOW ...")
    w2 = play_fixed_colours(eval_b, eval_a, args.games, cfg, args.seed,
                            args.random_opening_plies, args.temperature,
                            args.parallel_games)
    as_yellow = summarise(w2, a_is_green=False)

    print("\n" + "=" * 64)
    print(f"{'':12} {'score':>7} {'95% CI':>18} {'Elo':>7}  W/D/L")
    for name, r in (("A as GREEN", as_green), ("A as YELLOW", as_yellow)):
        ci = f"[{r['lo']:.3f}, {r['hi']:.3f}]"
        print(f"{name:12} {r['score']:7.3f} {ci:>18} {r['elo']:+7.0f}  "
              f"{r['a_wins']}/{r['draws']}/{r['b_wins']}")
    print("=" * 64)

    # Also report the raw first-player rate in these games, which is a
    # colour-balanced estimate rather than a self-play artefact.
    green_wins = sum(1 for w in w1 + w2 if w == 1)
    total = len(w1) + len(w2)
    draws = sum(1 for w in w1 + w2 if w == 0)
    fp = (green_wins + 0.5 * draws) / total
    flo, fhi = wilson_interval(green_wins + 0.5 * draws, total)
    print(f"\nfirst-player score across all {total} games "
          f"(A and B each play GREEN half the time): "
          f"{fp:.3f} [95% CI {flo:.3f}, {fhi:.3f}]")

    gain_green = as_green["score"] - p_first
    gain_yellow = as_yellow["score"] - (1.0 - p_first)

    print(f"\nbaseline: between two copies of B, GREEN scores {p_first:.3f}")
    print(f"  A's gain as GREEN : {gain_green:+.3f}  "
          f"({as_green['score']:.3f} against a baseline of {p_first:.3f})")
    print(f"  A's gain as YELLOW: {gain_yellow:+.3f}  "
          f"({as_yellow['score']:.3f} against a baseline of {1 - p_first:.3f})")
    print(f"  overall           : {gain_green + gain_yellow:+.3f}")

    # A gain counts as real only if the colour score clears its own baseline.
    green_up = as_green["lo"] > p_first
    yellow_up = as_yellow["lo"] > (1.0 - p_first)
    asymmetry = gain_yellow - gain_green

    print()
    if green_up and yellow_up:
        print("A is stronger than B at BOTH colours once the game's own bias\n"
              "is divided out. The improvement is symmetric, so a falling\n"
              "self-play first-player rate is not explained by one-sided\n"
              "skill growth.")
    elif yellow_up and not green_up:
        print("A clears the baseline as YELLOW but not as GREEN. The training\n"
              "gains are one-sided: the network has been learning to play\n"
              "second. A self-play first-player rate measured under these\n"
              "conditions reflects that asymmetry and cannot be read as the\n"
              "value of the game.")
    elif green_up and not yellow_up:
        print("A clears the baseline as GREEN but not as YELLOW — one-sided in\n"
              "the other direction.")
    else:
        print("A does not clearly clear the baseline at either colour. The\n"
              "arena gating may be promoting on noise; consider more games or\n"
              "a higher promotion threshold.")
    print(f"\nasymmetry (YELLOW gain minus GREEN gain): {asymmetry:+.3f}")
    print("0 would mean the two colours improved by the same amount.")


if __name__ == "__main__":
    main()
