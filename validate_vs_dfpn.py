#!/usr/bin/env python3
"""Check a trained network against positions the DF-PN campaign proved exactly.

    python validate_vs_dfpn.py --checkpoint runs/make7/best.pt \
        --database exact_propositions_campaign.txt --limit 1000

An important caveat, which belongs in any write-up that uses this number:
the DF-PN database was produced for the standard game, in which each player
holds 11 ones, 11 twos and 4 threes.  This network is trained on the
unlimited-supply variant.  Every database position is legal and reachable in
both rulesets, but a position proved to be a GREEN win under tile limits is
not guaranteed to be a GREEN win without them, because removing the cap
hands extra options to *both* sides.

So this script measures *consistency across two rulesets*, not accuracy
against ground truth.  A high agreement rate is evidence that the network
has learned real Make 7 structure rather than an artefact of self-play; a
low one is ambiguous between "the network is weak" and "the two variants
genuinely diverge here".  Reporting the breakdown by ply helps tell those
apart, since divergence should grow with ply as supply limits start to bite.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from make7.game import new_game, str_to_action
from make7.mcts import BatchedMCTS, MCTSConfig
from make7.model import NetEvaluator, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--database", required=True, type=Path)
    p.add_argument("--limit", type=int, default=0,
                   help="evaluate at most this many positions (0 = all)")
    p.add_argument("--simulations", type=int, default=0,
                   help="if > 0, also judge each position with MCTS search")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("dfpn_agreement.csv"))
    return p.parse_args()


def load_records(path: Path):
    """Yield (position_string, green_value) where green_value is +1/-1/0
    meaning the proven result for GREEN under the standard ruleset."""
    records = []
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            pos, verdict = parts[1], parts[2]
            toks = re.findall(r"[123][A-G]", pos)
            if "".join(toks) != pos:
                continue

            if verdict == "GREEN_WIN":
                value = 1.0
            elif verdict == "YELLOW_WIN":
                value = -1.0
            elif verdict == "DRAW":
                value = 0.0
            elif verdict in ("GREEN", "YELLOW") and len(parts) >= 4:
                prover, claim = parts[2], parts[3]
                # "GREEN DISPROVED" = GREEN cannot force a win, so GREEN's
                # result is at most a draw. Only the sign of "not a win" is
                # known, which is a one-sided constraint.
                if prover == "GREEN" and claim == "DISPROVED":
                    value = None          # GREEN <= 0
                elif prover == "GREEN" and claim == "PROVED":
                    value = 1.0
                elif prover == "YELLOW" and claim == "PROVED":
                    value = -1.0
                else:
                    continue
                records.append((toks, value, f"{prover}_{claim}"))
                continue
            else:
                continue
            records.append((toks, value, verdict))
    return records


def replay(tokens):
    s = new_game()
    for tok in tokens:
        if s.is_terminal:
            return None
        s = s.step(str_to_action(tok))
    return s


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net, generation = load_checkpoint(args.checkpoint, device)
    net.eval()
    print(f"loaded generation {generation} on {device}")

    records = load_records(args.database)
    if args.limit:
        records = records[:args.limit]
    print(f"{len(records)} database records to check")

    states, truths, kinds = [], [], []
    skipped = 0
    for tokens, value, kind in records:
        s = replay(tokens)
        if s is None or s.is_terminal:
            skipped += 1
            continue
        states.append(s)
        truths.append(value)
        kinds.append(kind)
    print(f"{len(states)} usable ({skipped} terminal or unreachable)")

    evaluator = NetEvaluator(net, device, use_amp=(device.type == "cuda"))

    raw_green = np.zeros(len(states), dtype=np.float32)
    for i in range(0, len(states), args.batch_size):
        chunk = states[i:i + args.batch_size]
        batch = np.stack([s.encode() for s in chunk])
        _, values = evaluator(batch)
        # value is for the side to move; convert to GREEN's point of view
        signs = np.array([s.player for s in chunk], dtype=np.float32)
        raw_green[i:i + len(chunk)] = values * signs

    search_green = None
    if args.simulations > 0:
        rng = np.random.default_rng(args.seed)
        mcts = BatchedMCTS(MCTSConfig(simulations=args.simulations), rng)
        search_green = np.zeros(len(states), dtype=np.float32)
        for i in range(0, len(states), args.batch_size):
            chunk = states[i:i + args.batch_size]
            visits = mcts.search(chunk, evaluator, add_noise=False)
            for k, (s, v) in enumerate(zip(chunk, visits)):
                # root Q of the most-visited move, from the mover's side
                nodes = v.sum()
                best = int(np.argmax(v)) if nodes > 0 else 0
                share = v[best] / max(nodes, 1)
                search_green[i + k] = share * s.player
        print("search pass complete")

    strict = [(t, r) for t, r in zip(truths, raw_green) if t is not None]
    agree = sum(1 for t, r in strict if np.sign(r) == np.sign(t) or
                (t == 0 and abs(r) < 0.2))
    print(f"\nsign agreement on decisive records: "
          f"{agree}/{len(strict)} = {agree / max(len(strict), 1):.1%}")

    one_sided = [(t, r) for t, r in zip(truths, raw_green) if t is None]
    if one_sided:
        consistent = sum(1 for _, r in one_sided if r <= 0.2)
        print(f"consistency on GREEN-DISPROVED records (expect value <= 0): "
              f"{consistent}/{len(one_sided)} = "
              f"{consistent / len(one_sided):.1%}")

    by_ply = defaultdict(lambda: [0, 0])
    for s, t, r in zip(states, truths, raw_green):
        if t is None:
            continue
        bucket = by_ply[s.ply]
        bucket[1] += 1
        if np.sign(r) == np.sign(t) or (t == 0 and abs(r) < 0.2):
            bucket[0] += 1
    print("\nagreement by ply")
    for ply in sorted(by_ply):
        ok, total = by_ply[ply]
        print(f"  ply {ply:2d}:  {ok:5d}/{total:5d} = {ok / total:.1%}")

    with args.out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        header = ["ply", "kind", "dfpn_green_value", "net_green_value"]
        if search_green is not None:
            header.append("search_green_value")
        writer.writerow(header)
        for i, (s, t, k) in enumerate(zip(states, truths, kinds)):
            row = [s.ply, k, "" if t is None else t, f"{raw_green[i]:.4f}"]
            if search_green is not None:
                row.append(f"{search_green[i]:.4f}")
            writer.writerow(row)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
