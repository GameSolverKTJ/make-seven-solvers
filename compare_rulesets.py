#!/usr/bin/env python3
"""Play the same positions under both rulesets and compare the outcomes.

    python compare_rulesets.py --checkpoint runs/make7/best.pt \
        --database exact_propositions_campaign.txt --parents 30 --games 128

What this settles, and what it does not
---------------------------------------
The network disagrees with the DF-PN database about positions the campaign
proved to be GREEN wins.  Playing them out with the network on both sides
showed GREEN losing, but that result on its own is ambiguous: the same
network supplies both players, so "GREEN loses" is equally consistent with

  (a) the unlimited-supply variant genuinely being worse for GREEN, and
  (c) the network simply failing to find GREEN's forcing line.

Under (c) the value head and the play share one blind spot, so they cannot
corroborate each other.  The only way to separate the two is to change the
rules and nothing else.

This script therefore plays each position twice from the identical starting
state, with the identical network and the identical random seed, differing
only in whether the 3-tile supply is capped:

    GREEN's score improves under the cap  ->  supports (a)
    GREEN's score is unchanged            ->  supports (c)

Choice of cap
-------------
`THREE_ONLY` caps 3s at four per player and leaves 1s and 2s unlimited.
That is what the database supports: across 88,172 placements neither player
ever uses a fifth 3, while one player does use twelve 2s, which contradicts
the usually quoted cap of eleven and leaves the 1/2 supply unknown.  Capping
only the 3s also keeps this a single-variable experiment.

Known limitation
----------------
The network was trained without limits and its input planes carry no hand
counts, so under the cap it is playing a game it cannot fully see: the
search will never make an illegal move, because legality is enforced by the
engine, but the priors and values are those of the unlimited game.  This
biases the comparison *against* finding an effect, so a clear improvement
under the cap is meaningful while a null result is weaker evidence than it
looks.  State that caveat wherever this number is reported.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from make7.arena import wilson_interval
from make7.game import (
    ACTION_SIZE, THREE_ONLY, Make7State, new_game, str_to_action,
)
from make7.mcts import BatchedMCTS, MCTSConfig, visits_to_policy
from make7.model import NetEvaluator, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--database", required=True, type=Path)
    p.add_argument("--parents", type=int, default=30)
    p.add_argument("--prefix-plies", type=int, default=19)
    p.add_argument("--games", type=int, default=128)
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--parallel-games", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--temperature-moves", type=int, default=6)
    p.add_argument("--verdict", default="GREEN_WIN")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("ruleset_comparison.csv"))
    return p.parse_args()


def load_grouped(path: Path, verdict: str, prefix_plies: int):
    groups = defaultdict(list)
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3 or parts[2] != verdict:
                continue
            toks = re.findall(r"[123][A-G]", parts[1])
            if "".join(toks) != parts[1] or len(toks) <= prefix_plies:
                continue
            groups["".join(toks[:prefix_plies])].append(toks)
    return groups


def replay(tokens, limits):
    s = new_game(limits)
    for tok in tokens:
        if s.is_terminal:
            return None
        s = s.step(str_to_action(tok))
    return None if s.is_terminal else s


def play_out(state: Make7State, games: int, evaluator, cfg, seed: int,
             temperature: float, temperature_moves: int, parallel_games: int):
    """Play `games` copies of one position. The seed is passed in so the two
    rulesets get the identical random stream."""
    rng = np.random.default_rng(seed)
    mcts = BatchedMCTS(cfg, rng)
    winners = []
    stuck = 0
    for start in range(0, games, parallel_games):
        batch = [state] * min(parallel_games, games - start)
        move = 0
        while any(not s.is_terminal for s in batch):
            visits = mcts.search(batch, evaluator, add_noise=False)
            t = temperature if move < temperature_moves else 0.0
            for i, s in enumerate(batch):
                if s.is_terminal:
                    continue
                probs = visits_to_policy(visits[i], t)
                if probs.sum() <= 0:
                    continue
                batch[i] = s.step(int(rng.choice(ACTION_SIZE, p=probs)))
            move += 1
        for s in batch:
            winners.append(s.winner)
            if s.winner == 0 and s.ply < 49:
                stuck += 1
    return winners, stuck


def score_of(winners):
    g = len(winners)
    wins = sum(1 for w in winners if w == 1)
    draws = sum(1 for w in winners if w == 0)
    return (wins + 0.5 * draws) / g, wins, draws, g - wins - draws


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net, generation = load_checkpoint(args.checkpoint, device)
    net.eval()
    print(f"loaded generation {generation} on {device}")

    rng = np.random.default_rng(args.seed)
    evaluator = NetEvaluator(net, device, use_amp=(device.type == "cuda"))
    cfg = MCTSConfig(simulations=args.simulations,
                     temperature_moves=args.temperature_moves)

    groups = load_grouped(args.database, args.verdict, args.prefix_plies)
    keys = sorted(groups)
    rng.shuffle(keys)
    keys = keys[:args.parents]
    print(f"{len(keys)} distinct {args.prefix_plies}-ply groups, "
          f"{args.games} games per position per ruleset")

    rows = []
    tot_un = tot_lim = tot_games = 0.0
    stuck_total = 0

    for n, key in enumerate(keys):
        tokens = groups[key][int(rng.integers(len(groups[key])))]
        s_un = replay(tokens, None)
        s_lim = replay(tokens, THREE_ONLY)
        if s_un is None or s_lim is None:
            continue

        seed = int(rng.integers(1 << 30))
        w_un, _ = play_out(s_un, args.games, evaluator, cfg, seed,
                           args.temperature, args.temperature_moves,
                           args.parallel_games)
        w_lim, stuck = play_out(s_lim, args.games, evaluator, cfg, seed,
                                args.temperature, args.temperature_moves,
                                args.parallel_games)
        stuck_total += stuck

        su, wu, du, lu = score_of(w_un)
        sl, wl, dl, ll = score_of(w_lim)
        tot_un += su * args.games
        tot_lim += sl * args.games
        tot_games += args.games

        left = s_lim.remaining(1)[2], s_lim.remaining(-1)[2]
        rows.append({
            "group": key, "ply": s_un.ply,
            "threes_left_green": left[0], "threes_left_yellow": left[1],
            "green_score_unlimited": round(su, 4),
            "green_score_capped": round(sl, 4),
            "delta": round(sl - su, 4),
            "wdl_unlimited": f"{wu}/{du}/{lu}",
            "wdl_capped": f"{wl}/{dl}/{ll}",
        })
        print(f"  [{n + 1:2d}/{len(keys)}] ply {s_un.ply:2d}  "
              f"unlimited {su:.3f}  capped {sl:.3f}  "
              f"delta {sl - su:+.3f}  (3s left G{left[0]}/Y{left[1]})")

    if not rows:
        print("no usable positions")
        return

    mean_un = tot_un / tot_games
    mean_lim = tot_lim / tot_games
    lo_u, hi_u = wilson_interval(tot_un, int(tot_games))
    lo_l, hi_l = wilson_interval(tot_lim, int(tot_games))
    deltas = np.array([r["delta"] for r in rows])
    improved = int((deltas > 0).sum())

    print("\n" + "=" * 66)
    print(f"positions           : {len(rows)} "
          f"(database verdict: {args.verdict} under the standard rules)")
    print(f"games per ruleset   : {int(tot_games)}")
    print(f"GREEN, unlimited 3s : {mean_un:.3f}  [95% CI {lo_u:.3f}, {hi_u:.3f}]")
    print(f"GREEN, 3s capped @4 : {mean_lim:.3f}  [95% CI {lo_l:.3f}, {hi_l:.3f}]")
    print(f"mean delta          : {deltas.mean():+.3f} "
          f"(SD {deltas.std(ddof=1):.3f} over positions)")
    print(f"positions improved  : {improved}/{len(rows)}")
    if stuck_total:
        print(f"games ending with no legal move under the cap: {stuck_total}")
    print("=" * 66)

    separated = lo_l > hi_u
    if separated:
        print("\nGREEN does measurably better once the 3-tile supply is capped,\n"
              "on the same positions with the same seeds. The ruleset, not the\n"
              "evaluation, accounts for at least part of the disagreement with\n"
              "the exact database.")
    elif abs(deltas.mean()) < 0.02:
        print("\nCapping the 3s changes almost nothing. The disagreement with the\n"
              "exact database is then unlikely to be a ruleset effect, which\n"
              "points instead at the network failing to find GREEN's winning\n"
              "line. Note the caveat above: the network cannot see hand counts,\n"
              "so this null result is weaker than it appears.")
    else:
        print("\nThere is a shift but the intervals overlap. More positions or\n"
              "more games per position are needed before reading anything into\n"
              "the direction.")

    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
