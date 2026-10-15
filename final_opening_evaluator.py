#!/usr/bin/env python3
"""Final opening evaluator for Make Seven.

This script is deliberately separate from training. It freezes one or more
checkpoints and evaluates the forced openings (default: 2A, 2B, 2C, 2D) with
three complementary measurements:

1. Post-opening high-budget search (PRIMARY)
   Apply the opening to the board first, then search from the resulting
   YELLOW-to-move position. The principal estimate is GREEN's value of the
   reply selected by high-budget search, i.e. mover-relative searched Q
   converted back to GREEN's point of view. A budget sweep checks convergence.

2. Root-value repeat / uncertainty analysis
   At the primary budget, repeat the post-opening search with independent
   Gumbel root perturbations. Report mean/SD/95% CI for opening value, reply
   stability, and paired opening-value differences (for example 2C - 2D).
   This measures uncertainty at the opening itself and does not accumulate
   search errors over an entire stochastic game.

3. Deterministic principal variation (PV)
   Force the opening, then both sides use temperature 0, no Dirichlet/Gumbel
   perturbation, and fixed high-budget search until terminal.

When multiple checkpoints are supplied, the evaluator also reports opening
rank/value stability and pairwise-difference sign consistency across
checkpoints. The evaluator NEVER ranks openings by empty-board policy
probability and does NOT use stochastic full-game robustness matches in the
final analysis.

Example
-------
python final_opening_evaluator.py \
  --checkpoint gen52=/path/to/gen52_best.pt \
  --checkpoint gen73=/path/to/gen73_best.pt \
  --budgets 500 2000 5000 10000 \
  --value-repeats 24 \
  --pv-simulations 2000 \
  --out-dir final_opening_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from make7.game import (
    ACTION_SIZE,
    Make7State,
    action_to_str,
    new_game,
    str_to_action,
)
from make7.mcts import BatchedMCTS, MCTSConfig
from make7.model import NetEvaluator, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoint", action="append", required=True,
        help="checkpoint path, or LABEL=PATH; repeat for multiple checkpoints")
    p.add_argument("--openings", nargs="+", default=["2A", "2B", "2C", "2D"])
    p.add_argument("--tile-limits", choices=["unlimited"],
                   default="unlimited",
                   help="Public 7x7 release: unlimited supply of all tile values.")

    p.add_argument("--budgets", nargs="+", type=int,
                   default=[500, 2000, 5000, 10000],
                   help="deterministic post-opening MCTS budgets")
    p.add_argument("--primary-budget", type=int, default=None,
                   help="budget for repeated root-value analysis and final summary; default=max(budgets)")
    p.add_argument("--value-repeats", type=int, default=24,
                   help="independent root-search repeats at primary budget")

    p.add_argument("--pv-simulations", type=int, default=2000,
                   help="MCTS simulations per move in deterministic PV")
    p.add_argument("--skip-pv", action="store_true")

    p.add_argument("--search-mode", choices=["gumbel", "puct"], default="gumbel")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--gumbel-root-candidates", type=int, default=0)
    p.add_argument("--gumbel-c-visit", type=float, default=50.0)
    p.add_argument("--gumbel-c-scale", type=float, default=0.10)
    p.add_argument("--no-gumbel-rescale-q", dest="gumbel_rescale_q",
                   action="store_false", default=True)
    p.add_argument("--symmetrise", dest="symmetrise", action="store_true",
                   default=True)
    p.add_argument("--no-symmetrise", dest="symmetrise", action="store_false")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", dest="use_amp", action="store_false", default=True)
    p.add_argument("--seed", type=int, default=1701)
    p.add_argument("--out-dir", type=Path, default=Path("final_opening_eval"))
    return p.parse_args()


def parse_checkpoint_specs(specs: Iterable[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for spec in specs:
        if "=" in spec:
            label, raw = spec.split("=", 1)
            label = label.strip()
            path = Path(raw).expanduser()
        else:
            path = Path(spec).expanduser()
            label = path.stem
        if not label:
            raise ValueError(f"empty checkpoint label in {spec!r}")
        if label in seen:
            raise ValueError(f"duplicate checkpoint label: {label}")
        if not path.exists():
            raise FileNotFoundError(path)
        seen.add(label)
        out.append((label, path))
    return out


def mcts_config(args: argparse.Namespace, simulations: int) -> MCTSConfig:
    return MCTSConfig(
        simulations=int(simulations),
        c_puct=float(args.c_puct),
        search_mode=args.search_mode,
        gumbel_root_candidates=int(args.gumbel_root_candidates),
        gumbel_c_visit=float(args.gumbel_c_visit),
        gumbel_c_scale=float(args.gumbel_c_scale),
        gumbel_rescale_q=bool(args.gumbel_rescale_q),
        temperature=0.0,
        temperature_moves=0,
    )


def _seed(base: int, *parts: int) -> int:
    """Stable uint32 seed derived from integer components."""
    ss = np.random.SeedSequence([int(base), *[int(x) for x in parts]])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def _legal_softmax(logits: np.ndarray, legal: np.ndarray) -> np.ndarray:
    out = np.zeros(ACTION_SIZE, dtype=np.float64)
    idx = np.flatnonzero(legal)
    if idx.size == 0:
        return out
    x = np.asarray(logits, dtype=np.float64)[idx]
    x -= np.max(x)
    ex = np.exp(x)
    out[idx] = ex / max(float(ex.sum()), 1e-300)
    return out


def _selected_source(search_mode: str, visits: np.ndarray,
                     improved: np.ndarray | None) -> np.ndarray:
    return improved if (search_mode == "gumbel" and improved is not None) else visits


def _argmax_legal(values: np.ndarray, legal: np.ndarray,
                  visited: np.ndarray | None = None) -> int:
    mask = np.asarray(legal, dtype=bool).copy()
    if visited is not None:
        vmask = np.asarray(visited) > 0
        if np.any(mask & vmask):
            mask &= vmask
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return -1
    return int(idx[np.argmax(np.asarray(values)[idx])])


def evaluate_post_opening_once(
    states: list[Make7State], evaluator: NetEvaluator, args: argparse.Namespace,
    simulations: int, seed: int, stochastic_gumbel: bool,
) -> list[dict]:
    """Search all forced-opening child states as one batch."""
    cfg = mcts_config(args, simulations)
    rng = np.random.default_rng(seed)
    mcts = BatchedMCTS(cfg, rng)
    if args.search_mode == "gumbel":
        visits, improved, root_values, action_q = mcts.search(
            states, evaluator, add_noise=stochastic_gumbel,
            with_policy=True, with_values=True, with_action_values=True)
    else:
        visits, root_values, action_q = mcts.search(
            states, evaluator, add_noise=stochastic_gumbel,
            with_values=True, with_action_values=True)
        improved = [None] * len(states)

    enc = np.stack([s.encode() for s in states])
    logits, net_values, q_head = evaluator.evaluate_with_q(enc)

    rows: list[dict] = []
    for i, state in enumerate(states):
        legal = state.legal_actions()
        source = _selected_source(args.search_mode, visits[i], improved[i])
        selected = _argmax_legal(source, legal)
        max_q_action = _argmax_legal(action_q[i], legal, visits[i])
        prior = _legal_softmax(logits[i], legal)
        selected_q = float(action_q[i][selected]) if selected >= 0 else float("nan")
        max_q = float(action_q[i][max_q_action]) if max_q_action >= 0 else float("nan")
        # Every value/Q here is mover-relative. Multiplication by state.player
        # converts it to GREEN's fixed point of view (+1 GREEN, -1 YELLOW).
        to_green = float(state.player)
        qh_max = _argmax_legal(q_head[i], legal)
        rows.append({
            "simulations": int(simulations),
            "stochastic_gumbel": int(bool(stochastic_gumbel)),
            "search_seed": int(seed),
            "root_player": "GREEN" if state.player == 1 else "YELLOW",
            "selected_reply": action_to_str(selected) if selected >= 0 else "",
            "selected_reply_policy": float(source[selected]) if selected >= 0 else float("nan"),
            "selected_reply_prior": float(prior[selected]) if selected >= 0 else float("nan"),
            "selected_reply_visits": float(visits[i][selected]) if selected >= 0 else 0.0,
            "selected_reply_visit_share": (
                float(visits[i][selected] / max(float(np.sum(visits[i])), 1.0))
                if selected >= 0 else 0.0),
            "selected_reply_q_mover": selected_q,
            "opening_value_green": to_green * selected_q,
            "max_searched_q_action": action_to_str(max_q_action) if max_q_action >= 0 else "",
            "max_searched_q_mover": max_q,
            "max_searched_q_green": to_green * max_q,
            "root_mean_value_mover": float(root_values[i]),
            "root_mean_value_green": to_green * float(root_values[i]),
            "network_value_mover": float(net_values[i]),
            "network_value_green": to_green * float(net_values[i]),
            "q_head_selected_reply_mover": (
                float(q_head[i][selected]) if selected >= 0 else float("nan")),
            "q_head_selected_reply_green": (
                to_green * float(q_head[i][selected]) if selected >= 0 else float("nan")),
            "q_head_max_legal_action": action_to_str(qh_max) if qh_max >= 0 else "",
            "q_head_max_legal_mover": float(q_head[i][qh_max]) if qh_max >= 0 else float("nan"),
            "total_root_visits": float(np.sum(visits[i])),
        })
    return rows


def deterministic_pv(
    start_state: Make7State, opening_token: str, evaluator: NetEvaluator,
    args: argparse.Namespace, seed: int,
) -> tuple[dict, list[dict]]:
    """Temperature-0/no-noise principal variation until terminal."""
    state = start_state
    moves = [opening_token]
    steps: list[dict] = []
    cfg = mcts_config(args, args.pv_simulations)
    mcts = BatchedMCTS(cfg, np.random.default_rng(seed))

    while not state.is_terminal:
        if args.search_mode == "gumbel":
            visits, improved, root_values, action_q = mcts.search(
                [state], evaluator, add_noise=False, with_policy=True,
                with_values=True, with_action_values=True)
            source = improved[0]
        else:
            visits, root_values, action_q = mcts.search(
                [state], evaluator, add_noise=False, with_values=True,
                with_action_values=True)
            source = visits[0]
        legal = state.legal_actions()
        action = _argmax_legal(source, legal)
        if action < 0:
            raise RuntimeError("non-terminal PV state has no selected legal action")
        q = float(action_q[0][action])
        steps.append({
            "ply_before": int(state.ply),
            "player": "GREEN" if state.player == 1 else "YELLOW",
            "action": action_to_str(action),
            "selected_q_mover": q,
            "selected_q_green": float(state.player) * q,
            "root_mean_value_mover": float(root_values[0]),
            "root_mean_value_green": float(state.player) * float(root_values[0]),
            "selected_policy": float(source[action]),
            "selected_visits": float(visits[0][action]),
            "total_visits": float(np.sum(visits[0])),
        })
        moves.append(action_to_str(action))
        state = state.step(action)

    summary = {
        "winner": "GREEN" if state.winner == 1 else "YELLOW" if state.winner == -1 else "DRAW",
        "green_result": float(state.result_for(1)),
        "terminal_ply": int(state.ply),
        "pv_length_after_opening": int(state.ply - start_state.ply),
        "moves": " ".join(moves),
    }
    return summary, steps


def _mean_ci(values: np.ndarray) -> tuple[float, float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return (float("nan"),) * 4
    mean = float(x.mean())
    if x.size == 1:
        return mean, 0.0, mean, mean
    sd = float(x.std(ddof=1))
    se = sd / math.sqrt(x.size)
    return mean, sd, mean - 1.96 * se, mean + 1.96 * se


def paired_repeat_differences(rows: list[dict], openings: list[str]) -> list[dict]:
    """Pairwise opening-value differences using matched repeat indices."""
    by_key = {(r["opening"], int(r["repeat"])): r for r in rows}
    repeats = sorted({int(r["repeat"]) for r in rows})
    out: list[dict] = []
    checkpoint = rows[0]["checkpoint"] if rows else ""
    generation = int(rows[0]["generation"]) if rows else -1
    simulations = int(rows[0]["simulations"]) if rows else 0
    for i, a in enumerate(openings):
        for b in openings[i + 1:]:
            vals = []
            for rep in repeats:
                ra = by_key.get((a, rep))
                rb = by_key.get((b, rep))
                if ra is not None and rb is not None:
                    vals.append(float(ra["opening_value_green"]) - float(rb["opening_value_green"]))
            x = np.asarray(vals, dtype=np.float64)
            mean, sd, lo, hi = _mean_ci(x)
            out.append({
                "checkpoint": checkpoint,
                "generation": generation,
                "opening_a": a,
                "opening_b": b,
                "simulations": simulations,
                "paired_repeats": int(x.size),
                "mean_difference_a_minus_b": mean,
                "sd_difference": sd,
                "ci95_low": lo,
                "ci95_high": hi,
                "fraction_a_greater_b": float(np.mean(x > 0)) if x.size else float("nan"),
                "fraction_equal": float(np.mean(x == 0)) if x.size else float("nan"),
            })
    return out


def cross_checkpoint_summaries(
    final_rows: list[dict], opening_order: list[str]
) -> tuple[list[dict], list[dict]]:
    """Summarise primary-value and rank consistency across checkpoints."""
    if not final_rows:
        return [], []

    labels = []
    for r in final_rows:
        if r["checkpoint"] not in labels:
            labels.append(r["checkpoint"])

    ranks: dict[tuple[str, str], float] = {}
    for label in labels:
        rr = [r for r in final_rows if r["checkpoint"] == label]
        ordered = sorted(rr, key=lambda r: -float(r["deterministic_opening_value_green"]))
        # Dense ordinal ranks; deterministic values almost never tie exactly,
        # but equal values receive the same midpoint rank for completeness.
        values = np.asarray([float(r["deterministic_opening_value_green"]) for r in ordered])
        for idx, r in enumerate(ordered):
            ties = np.flatnonzero(np.isclose(values, values[idx], atol=1e-12, rtol=0.0))
            ranks[(label, r["opening"])] = float(np.mean(ties + 1))

    opening_rows: list[dict] = []
    for opening in opening_order:
        rr = [r for r in final_rows if r["opening"] == opening]
        vals = np.asarray([float(r["deterministic_opening_value_green"]) for r in rr], dtype=np.float64)
        rank_vals = np.asarray([ranks[(r["checkpoint"], opening)] for r in rr], dtype=np.float64)
        opening_rows.append({
            "opening": opening,
            "checkpoints": len(rr),
            "mean_primary_value_green": float(vals.mean()) if vals.size else float("nan"),
            "sd_primary_value_green": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "min_primary_value_green": float(vals.min()) if vals.size else float("nan"),
            "max_primary_value_green": float(vals.max()) if vals.size else float("nan"),
            "mean_rank": float(rank_vals.mean()) if rank_vals.size else float("nan"),
            "best_rank_count": int(np.sum(np.isclose(rank_vals, 1.0))) if rank_vals.size else 0,
            "values_by_checkpoint": ";".join(
                f"{r['checkpoint']}:{float(r['deterministic_opening_value_green']):+.6f}" for r in rr),
            "ranks_by_checkpoint": ";".join(
                f"{r['checkpoint']}:{ranks[(r['checkpoint'], opening)]:.3f}" for r in rr),
        })

    pair_rows: list[dict] = []
    for i, a in enumerate(opening_order):
        for b in opening_order[i + 1:]:
            diffs = []
            diff_labels = []
            for label in labels:
                ma = next((r for r in final_rows if r["checkpoint"] == label and r["opening"] == a), None)
                mb = next((r for r in final_rows if r["checkpoint"] == label and r["opening"] == b), None)
                if ma is None or mb is None:
                    continue
                d = float(ma["deterministic_opening_value_green"]) - float(mb["deterministic_opening_value_green"])
                diffs.append(d)
                diff_labels.append((label, d))
            x = np.asarray(diffs, dtype=np.float64)
            pair_rows.append({
                "opening_a": a,
                "opening_b": b,
                "checkpoints": int(x.size),
                "mean_difference_a_minus_b": float(x.mean()) if x.size else float("nan"),
                "sd_difference": float(x.std(ddof=1)) if x.size > 1 else 0.0,
                "min_difference": float(x.min()) if x.size else float("nan"),
                "max_difference": float(x.max()) if x.size else float("nan"),
                "fraction_a_greater_b": float(np.mean(x > 0)) if x.size else float("nan"),
                "same_nonzero_sign_across_checkpoints": int(
                    x.size >= 2 and (np.all(x > 0) or np.all(x < 0))),
                "differences_by_checkpoint": ";".join(
                    f"{label}:{d:+.6f}" for label, d in diff_labels),
            })

    return opening_rows, pair_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=names, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    if any(b < 1 for b in args.budgets):
        raise ValueError("all budgets must be >= 1")
    args.budgets = sorted(set(int(b) for b in args.budgets))
    primary_budget = int(args.primary_budget or max(args.budgets))
    if primary_budget not in args.budgets:
        args.budgets.append(primary_budget)
        args.budgets.sort()
    if args.value_repeats < 0:
        raise ValueError("--value-repeats must be non-negative")

    checkpoint_specs = parse_checkpoint_specs(args.checkpoint)
    openings = [(tok.upper(), str_to_action(tok.upper())) for tok in args.openings]
    if len({a for _, a in openings}) != len(openings):
        raise ValueError("duplicate opening action")
    opening_tokens = [t for t, _ in openings]
    limits = None
    root = new_game(limits)
    for tok, action in openings:
        if not root.legal_actions()[action]:
            raise ValueError(f"opening {tok} is not legal under {args.tile_limits}")
    post_states = [root.step(a) for _, a in openings]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    budget_rows: list[dict] = []
    repeat_rows: list[dict] = []
    repeat_summary_rows: list[dict] = []
    pairwise_repeat_rows: list[dict] = []
    pv_rows: list[dict] = []
    pv_move_rows: list[dict] = []
    final_rows: list[dict] = []

    metadata = {
        "openings": opening_tokens,
        "tile_limits": args.tile_limits,
        "budgets": args.budgets,
        "primary_budget": primary_budget,
        "value_repeats": int(args.value_repeats),
        "pv_simulations": int(args.pv_simulations),
        "search_mode": args.search_mode,
        "c_puct": float(args.c_puct),
        "gumbel_root_candidates": int(args.gumbel_root_candidates),
        "gumbel_c_visit": float(args.gumbel_c_visit),
        "gumbel_c_scale": float(args.gumbel_c_scale),
        "gumbel_rescale_q": bool(args.gumbel_rescale_q),
        "symmetrise": bool(args.symmetrise),
        "seed": int(args.seed),
        "device": str(device),
        "checkpoints": [{"label": l, "path": str(p)} for l, p in checkpoint_specs],
        "full_game_robustness_matches": False,
    }

    for ckpt_index, (label, path) in enumerate(checkpoint_specs):
        net, generation = load_checkpoint(path, device)
        net.eval()
        evaluator = NetEvaluator(
            net, device, use_amp=(args.use_amp and device.type == "cuda"),
            symmetrise=args.symmetrise)
        print(f"\n[{label}] generation {generation} | {path}")

        # 1) deterministic budget sweep from states AFTER each forced opening.
        for budget in args.budgets:
            rows = evaluate_post_opening_once(
                post_states, evaluator, args, budget,
                seed=_seed(args.seed, ckpt_index, budget, 0),
                stochastic_gumbel=False)
            print(f"  deterministic post-opening search: {budget:,} simulations")
            for (opening, _), row in zip(openings, rows):
                row.update({
                    "checkpoint": label,
                    "checkpoint_path": str(path),
                    "generation": int(generation),
                    "opening": opening,
                })
                budget_rows.append(row)
                print(
                    f"    {opening}: V_G={row['opening_value_green']:+.4f} "
                    f"reply={row['selected_reply']} "
                    f"Q_mover={row['selected_reply_q_mover']:+.4f} "
                    f"visits={row['selected_reply_visit_share']:.1%}")

        # 2) repeat the primary-budget search with independent root perturbations.
        ckpt_repeat_rows: list[dict] = []
        for rep in range(int(args.value_repeats)):
            rows = evaluate_post_opening_once(
                post_states, evaluator, args, primary_budget,
                seed=_seed(args.seed, ckpt_index, 900001, rep),
                stochastic_gumbel=True)
            for (opening, _), row in zip(openings, rows):
                row.update({
                    "checkpoint": label,
                    "checkpoint_path": str(path),
                    "generation": int(generation),
                    "opening": opening,
                    "repeat": int(rep),
                })
                repeat_rows.append(row)
                ckpt_repeat_rows.append(row)

        for opening, _ in openings:
            rr = [r for r in ckpt_repeat_rows if r["opening"] == opening]
            vals = np.asarray([r["opening_value_green"] for r in rr], dtype=np.float64)
            mean, sd, lo, hi = _mean_ci(vals)
            replies = Counter(r["selected_reply"] for r in rr)
            mode_reply, mode_count = replies.most_common(1)[0] if replies else ("", 0)
            repeat_summary_rows.append({
                "checkpoint": label,
                "generation": int(generation),
                "opening": opening,
                "simulations": primary_budget,
                "repeats": len(rr),
                "mean_opening_value_green": mean,
                "sd_opening_value_green": sd,
                "ci95_low": lo,
                "ci95_high": hi,
                "modal_reply": mode_reply,
                "modal_reply_fraction": mode_count / len(rr) if rr else float("nan"),
            })

        pairwise_repeat_rows.extend(
            paired_repeat_differences(ckpt_repeat_rows, opening_tokens))

        # 3) deterministic principal variation.
        if not args.skip_pv:
            print(f"  deterministic PV: {args.pv_simulations:,} simulations/move")
            for oi, ((opening, _), start) in enumerate(zip(openings, post_states)):
                summary, steps = deterministic_pv(
                    start, opening, evaluator, args,
                    seed=_seed(args.seed, ckpt_index, 910001, oi))
                summary.update({
                    "checkpoint": label,
                    "generation": int(generation),
                    "opening": opening,
                    "simulations_per_move": int(args.pv_simulations),
                })
                pv_rows.append(summary)
                for step in steps:
                    step.update({
                        "checkpoint": label,
                        "generation": int(generation),
                        "opening": opening,
                    })
                    pv_move_rows.append(step)
                print(f"    {opening}: {summary['winner']} in {summary['terminal_ply']} plies")

        # Compact final table for this checkpoint.
        det_primary = {
            r["opening"]: r for r in budget_rows
            if r["checkpoint"] == label and r["simulations"] == primary_budget
            and not r["stochastic_gumbel"]
        }
        rep_summary = {
            r["opening"]: r for r in repeat_summary_rows if r["checkpoint"] == label
        }
        pv_summary = {r["opening"]: r for r in pv_rows if r["checkpoint"] == label}
        for opening, _ in openings:
            d = det_primary[opening]
            rs = rep_summary.get(opening, {})
            pv = pv_summary.get(opening, {})
            final_rows.append({
                "checkpoint": label,
                "generation": int(generation),
                "opening": opening,
                "primary_simulations": primary_budget,
                "deterministic_opening_value_green": d["opening_value_green"],
                "deterministic_best_reply": d["selected_reply"],
                "deterministic_reply_visit_share": d["selected_reply_visit_share"],
                "deterministic_max_searched_q_action": d["max_searched_q_action"],
                "deterministic_max_searched_q_green": d["max_searched_q_green"],
                "network_value_green": d["network_value_green"],
                "q_head_selected_reply_green": d["q_head_selected_reply_green"],
                "repeat_q_mean_green": rs.get("mean_opening_value_green", float("nan")),
                "repeat_q_sd_green": rs.get("sd_opening_value_green", float("nan")),
                "repeat_q_ci95_low": rs.get("ci95_low", float("nan")),
                "repeat_q_ci95_high": rs.get("ci95_high", float("nan")),
                "repeat_modal_reply": rs.get("modal_reply", ""),
                "repeat_modal_reply_fraction": rs.get("modal_reply_fraction", float("nan")),
                "pv_winner": pv.get("winner", ""),
                "pv_terminal_ply": pv.get("terminal_ply", ""),
            })

        del evaluator, net
        if device.type == "cuda":
            torch.cuda.empty_cache()

    checkpoint_opening_rows, checkpoint_pairwise_rows = cross_checkpoint_summaries(
        final_rows, opening_tokens)

    write_csv(out_dir / "budget_sweep.csv", budget_rows)
    write_csv(out_dir / "search_repeat_details.csv", repeat_rows)
    write_csv(out_dir / "search_repeat_summary.csv", repeat_summary_rows)
    write_csv(out_dir / "pairwise_root_value.csv", pairwise_repeat_rows)
    write_csv(out_dir / "pv_summary.csv", pv_rows)
    write_csv(out_dir / "pv_moves.csv", pv_move_rows)
    write_csv(out_dir / "final_summary.csv", final_rows)
    write_csv(out_dir / "checkpoint_opening_summary.csv", checkpoint_opening_rows)
    write_csv(out_dir / "checkpoint_pairwise_consistency.csv", checkpoint_pairwise_rows)
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print("\nFinal summary (primary metric = deterministic post-opening searched Q)")
    for label, _ in checkpoint_specs:
        rows = [r for r in final_rows if r["checkpoint"] == label]
        rows.sort(key=lambda r: -float(r["deterministic_opening_value_green"]))
        print(f"  [{label}]")
        for rank, r in enumerate(rows, 1):
            lo = r["repeat_q_ci95_low"]
            hi = r["repeat_q_ci95_high"]
            ci = "n/a" if not (np.isfinite(lo) and np.isfinite(hi)) else f"[{lo:+.4f},{hi:+.4f}]"
            print(
                f"    {rank}. {r['opening']}  "
                f"V_G={r['deterministic_opening_value_green']:+.4f}  "
                f"reply={r['deterministic_best_reply']}  repeatCI={ci}")

    if len(checkpoint_specs) > 1:
        print("\nCross-checkpoint primary-value stability")
        for r in sorted(checkpoint_opening_rows, key=lambda x: float(x["mean_rank"])):
            print(
                f"  {r['opening']}: mean V_G={r['mean_primary_value_green']:+.4f}, "
                f"mean rank={r['mean_rank']:.2f}, best in {r['best_rank_count']}/{r['checkpoints']}")
        print("  pairwise sign consistency written to checkpoint_pairwise_consistency.csv")

    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
