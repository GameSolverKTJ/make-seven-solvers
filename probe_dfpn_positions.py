#!/usr/bin/env python3
"""Play out positions the DF-PN campaign proved, instead of just asking the
value head about them.

    python probe_dfpn_positions.py --checkpoint runs/make7/best.pt \
        --database exact_propositions_campaign.txt --parents 40 --games 64

Why this exists
---------------
A low agreement rate between the network and the exact database has at least
three explanations, and the raw value head cannot tell them apart:

  (a) the unlimited-supply variant genuinely favours YELLOW where the
      standard game favours GREEN;
  (b) the network carries a global bias towards YELLOW;
  (c) the database positions are off-distribution — proof-number search
      deliberately explores proof-critical lines that self-play never
      visits, and a value head can be confidently wrong out there.

Playing the positions out separates them.  If GREEN actually wins these
positions when the network plays both sides, the value head is simply wrong
about them and (c) is in play.  If YELLOW really does win, the disagreement
is about the position rather than the evaluation, which points at (a).

Sampling
--------
The database is not a random sample of Make 7 positions: the ply-25 records
descend from only a few dozen ply-19 parents, and one parent alone accounts
for over a quarter of them.  Treating them as independent would massively
overstate the evidence.  This script therefore groups records by their
shared prefix and draws at most `--per-parent` positions from each, so the
reported numbers rest on distinct regions of the tree.
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
from make7.game import ACTION_SIZE, new_game, str_to_action
from make7.mcts import BatchedMCTS, MCTSConfig, visits_to_policy
from make7.model import NetEvaluator, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--database", required=True, type=Path)
    p.add_argument("--parents", type=int, default=40,
                   help="distinct prefix groups to sample from")
    p.add_argument("--per-parent", type=int, default=1,
                   help="positions drawn from each group")
    p.add_argument("--prefix-plies", type=int, default=19,
                   help="how many plies define a group")
    p.add_argument("--games", type=int, default=64,
                   help="playouts per sampled position")
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--parallel-games", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--temperature-moves", type=int, default=6)
    p.add_argument("--verdict", default="GREEN_WIN",
                   help="which database verdict to probe")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("dfpn_playout_probe.csv"))
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


def replay(tokens):
    s = new_game()
    for tok in tokens:
        if s.is_terminal:
            return None
        s = s.step(str_to_action(tok))
    return None if s.is_terminal else s


def play_out(states, evaluator, cfg, rng, temperature, temperature_moves,
             parallel_games):
    mcts = BatchedMCTS(cfg, rng)
    winners = []
    for start in range(0, len(states), parallel_games):
        batch = list(states[start:start + parallel_games])
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
        winners.extend(s.winner for s in batch)
    return winners


@torch.inference_mode()
def raw_values(net, device, states):
    net.eval()
    out = []
    for i in range(0, len(states), 256):
        chunk = states[i:i + 256]
        x = torch.from_numpy(np.stack([s.encode() for s in chunk])).to(device)
        _, v = net(x)
        signs = np.array([s.player for s in chunk], dtype=np.float32)
        out.extend((v.float().cpu().numpy() * signs).tolist())
    return out


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net, generation = load_checkpoint(args.checkpoint, device)
    print(f"loaded generation {generation} on {device}")

    rng = np.random.default_rng(args.seed)
    evaluator = NetEvaluator(net, device, use_amp=(device.type == "cuda"))

    groups = load_grouped(args.database, args.verdict, args.prefix_plies)
    print(f"{len(groups)} distinct {args.prefix_plies}-ply groups carry "
          f"verdict {args.verdict}")
    if not groups:
        print("nothing to probe")
        return

    keys = sorted(groups)
    rng.shuffle(keys)
    keys = keys[:args.parents]

    sampled = []
    for key in keys:
        options = groups[key]
        idx = rng.choice(len(options), size=min(args.per_parent, len(options)),
                         replace=False)
        for i in np.atleast_1d(idx):
            state = replay(options[int(i)])
            if state is not None:
                sampled.append((key, "".join(options[int(i)]), state))

    print(f"probing {len(sampled)} positions from {len(keys)} groups, "
          f"{args.games} playouts each "
          f"({len(sampled) * args.games} games total)")

    values = raw_values(net, device, [s for _, _, s in sampled])
    cfg = MCTSConfig(simulations=args.simulations,
                     temperature_moves=args.temperature_moves)

    rows = []
    agree_value = agree_play = 0
    green_total = games_total = 0.0

    for (key, pos, state), value in zip(sampled, values):
        starts = [state] * args.games
        winners = play_out(starts, evaluator, cfg, rng, args.temperature,
                           args.temperature_moves, args.parallel_games)
        g = len(winners)
        wins = sum(1 for w in winners if w == 1)
        draws = sum(1 for w in winners if w == 0)
        score = (wins + 0.5 * draws) / g
        green_total += wins + 0.5 * draws
        games_total += g
        if value > 0:
            agree_value += 1
        if score > 0.5:
            agree_play += 1
        rows.append({
            "group": key, "position": pos, "ply": state.ply,
            "net_green_value": round(float(value), 4),
            "green_score": round(score, 4),
            "wins": wins, "draws": draws, "losses": g - wins - draws,
        })
        print(f"  ply {state.ply:2d}  net {value:+.3f}  played {score:.3f}  "
              f"({wins}W/{draws}D/{g - wins - draws}L)  {key[:24]}...")

    n = len(rows)
    overall = green_total / games_total
    lo, hi = wilson_interval(green_total, int(games_total))

    print("\n" + "=" * 62)
    print(f"database verdict for every position: {args.verdict}")
    print(f"positions probed                   : {n} "
          f"(from {len(keys)} distinct groups)")
    print(f"value head says GREEN is better in : {agree_value}/{n} "
          f"= {agree_value / n:.1%}")
    print(f"GREEN actually scores > 0.5 in     : {agree_play}/{n} "
          f"= {agree_play / n:.1%}")
    print(f"overall GREEN score across games   : {overall:.3f} "
          f"[95% CI {lo:.3f}, {hi:.3f}]")
    print("=" * 62)

    if agree_play > agree_value + max(2, n // 10):
        print("\nGREEN wins these positions far more often than the value head\n"
              "expects. That points at the evaluation being wrong on\n"
              "off-distribution positions rather than at a real difference\n"
              "between the two rulesets.")
    elif agree_play < n * 0.5:
        print("\nGREEN loses these positions in play as well, not just in the\n"
              "value head. The disagreement is about the positions rather than\n"
              "the evaluation, which is consistent with the unlimited-supply\n"
              "variant genuinely differing from the standard game here.")
    else:
        print("\nThe two measures broadly agree; no clear separation.")

    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
