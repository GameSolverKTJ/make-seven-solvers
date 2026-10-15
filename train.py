#!/usr/bin/env python3
"""AlphaZero training loop for 7x7 Make 7.

    python train.py --generations 100 --preset base

Every generation does three things: play a batch of self-play games with
the current best network, take gradient steps on a replay buffer of recent
positions, and play a gating match against the previous best.  A new network
is promoted only if it scores above `--promote-threshold`.

Two logs are written to the run directory:

  metrics.csv    one row per generation — losses, arena score, Elo, result
                 split, mean game length.  This is the file to plot when
                 arguing that training has converged.
  openings.csv   one row per generation — how often each of the 21 opening
                 moves was chosen in self-play, plus the network's raw
                 evaluation of every opening.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from make7.arena import play_match, random_evaluator, score_to_elo
from make7.game import ACTION_SIZE, action_to_str, new_game, str_to_action
from make7.mcts import BatchedMCTS, MCTSConfig
from make7.parallel import ParallelSelfPlay
from make7.validation import (
    calibration_report, describe, evaluate as evaluate_exact, green_values,
    load_exact_records, sample_exact_decisive_batch,
    stratified_split_by_group, write_proposition_predictions,
)
from make7.model import (
    Make7Net, ModelConfig, NetEvaluator, load_checkpoint, save_checkpoint,
)
from make7.selfplay import ReplayBuffer, SelfPlayConfig, generate_games


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="runs/make7", type=Path)
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--preset", default="base", choices=["small", "base", "large", "shallow",
                            "shallow_wide", "coord_base"])
    p.add_argument("--tile-limits", default="unlimited", choices=["none", "unlimited"],
                   help="Unlimited supply of 1, 2 and 3 tiles; 'none' is an alias. "
                        "Use exact labels proved under the same ruleset.")
    p.add_argument("--validation-database", type=Path, default=None,
                   help="exact_propositions_campaign.txt. When given, every "
                        "generation is scored against it and the result is "
                        "logged to validation.csv. This is a fixed yardstick, "
                        "unlike arena or the self-play win rate, so it is the "
                        "metric to plot when arguing that training converged.")
    p.add_argument("--validation-every", type=int, default=1)
    p.add_argument("--validation-draw-band", type=float, default=0.2,
                   help="absolute GREEN-value band treated as draw/uncertain "
                        "for exact-proposition diagnostics and one-sided bounds")
    p.add_argument("--validation-select", default="holdout_auc",
                   choices=["holdout_auc", "holdout_balanced_accuracy",
                            "holdout_mcc", "calibrated_balanced_accuracy",
                            "calibrated_mcc", "auc", "balanced_accuracy",
                            "macro_f1", "mcc"],
                   help="which validation figure decides best_validation.pt. "
                        "holdout_auc is the v2.1.4 default: it is computed only "
                        "on proof groups that are never used by the optional "
                        "exact auxiliary loss. The legacy all-data metrics are "
                        "kept for diagnostics/backward compatibility but can "
                        "leak exact-training data when --exact-loss-weight > 0.")
    p.add_argument("--validation-group-plies", type=int, default=12)
    p.add_argument("--validation-calibration-fraction", type=float, default=0.5,
                   help="fraction of shallow exact-proof groups reserved for calibration/"
                        "optional exact auxiliary training; the remainder is untouched test")
    p.add_argument("--validation-calibration-seed", type=int, default=1701,
                   help="fixed group-stratified exact-data split seed")
    p.add_argument("--exact-loss-weight", type=float, default=0.0,
                   help="optional auxiliary MSE weight from decisive exact positions in "
                        "the calibration/train split only. 0 disables it; try 0.05 for "
                        "short exact-guided fine-tuning")
    p.add_argument("--exact-batch-size", type=int, default=128,
                   help="class- and group-balanced exact positions per gradient step")
    p.add_argument("--init-checkpoint", type=Path, default=None,
                   help="start a new run from an existing checkpoint such as "
                        "best_validation.pt; ignored when --resume loads run-dir/best.pt")
    p.add_argument("--init-buffer", type=Path, default=None,
                   help="inherit a replay buffer when starting a new run. This is "
                        "recommended when fine-tuning a mature checkpoint; otherwise "
                        "the first generation can overfit one thin self-play batch")
    p.add_argument("--init-run-dir", type=Path, default=None,
                   help="convenience shortcut for a new run: load <dir>/best.pt and "
                        "<dir>/buffer.npz together. Cannot be combined with --resume, "
                        "--init-checkpoint or --init-buffer")
    p.add_argument("--blocks", type=int, default=None,
                   help="override the preset's block count")
    p.add_argument("--channels", type=int, default=None,
                   help="override the preset's channel width")
    p.add_argument("--coord-conv", dest="coord_conv", action="store_true",
                   default=None, help="append row/column coordinate planes")
    p.add_argument("--no-coord-conv", dest="coord_conv", action="store_false",
                   help="force coordinate planes off")

    p.add_argument("--games-per-generation", type=int, default=256)
    p.add_argument("--parallel-games", type=int, default=128)
    p.add_argument("--gather-timeout", type=float, default=0.0,
                   help="seconds the inference server waits for straggling "
                        "workers after draining the queue. 0 serves whatever "
                        "has arrived; a large value (0.5) makes it a barrier. "
                        "Small non-zero values are usually the worst choice.")
    p.add_argument("--workers", type=int, default=0,
                   help="self-play worker processes. 0 runs everything in "
                        "this process. Inference always stays here, so VRAM "
                        "does not grow with the worker count; leave 2-4 cores "
                        "free for the parent and the data loader.")
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--search-mode", default="gumbel", choices=["puct", "gumbel"],
                   help="root search. 'gumbel' (v2 default) uses Gumbel "
                        "sequential halving at the root and PUCT below it; "
                        "'puct' reproduces the original AlphaZero search")
    p.add_argument("--gumbel-root-candidates", type=int, default=0,
                   help="number of legal root actions admitted to sequential "
                        "halving; 0 = all legal actions (recommended for Make 7)")
    p.add_argument("--gumbel-c-visit", type=float, default=50.0,
                   help="visit offset in the Gumbel completed-Q transform")
    p.add_argument("--gumbel-c-scale", type=float, default=0.10,
                   help="scale in the Gumbel completed-Q transform")
    p.add_argument("--gumbel-rescale-q", dest="gumbel_rescale_q",
                   action="store_true", default=True,
                   help="min-max normalise legal completed Q values before "
                        "the Gumbel policy-improvement transform (v2.1.2 default)")
    p.add_argument("--no-gumbel-rescale-q", dest="gumbel_rescale_q",
                   action="store_false",
                   help="use raw completed Q values as in v2.1.1")
    p.add_argument("--temperature-moves", type=int, default=12)
    p.add_argument("--dirichlet-alpha", type=float, default=0.8)
    p.add_argument("--dirichlet-epsilon", type=float, default=0.25,
                   help="weight of the root exploration noise")
    p.add_argument("--forced-opening-fraction", type=float, default=0.25,
                   help="fraction of self-play games whose first move is forced. "
                        "The empty-root policy target is masked while value/Q "
                        "targets are retained. 0 disables forced-opening self-play.")
    p.add_argument("--forced-openings", nargs="+", default=["2A", "2B", "2C", "2D"],
                   help="opening actions balanced across forced games. Mirror "
                        "augmentation supplies symmetric 2G/2F/2E branches.")
    p.add_argument("--policy-target-smoothing", type=float, default=0.05,
                   help="uniform-legal smoothing applied to every non-masked policy "
                        "training target. Search/action selection remain unsmoothed; "
                        "forced roots stay policy-masked. 0 disables.")

    p.add_argument("--buffer-size", type=int, default=400_000)
    p.add_argument("--train-steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr-final", type=float, default=None,
                   help="if set, decay the learning rate geometrically from "
                        "--lr to this value over --lr-decay-generations. A "
                        "constant high rate lets each generation shift the "
                        "policy a long way, which is a major source of "
                        "generation-to-generation swing in self-play results.")
    p.add_argument("--lr-decay-generations", type=int, default=40)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--value-loss-weight", type=float, default=1.0)
    p.add_argument("--q-loss-weight", type=float, default=0.25,
                   help="weight of the auxiliary root Q(s,a) Huber loss. "
                        "The Q head is NOT used by search in v2.1; 0 disables "
                        "its training while keeping checkpoint compatibility.")
    p.add_argument("--q-min-visits", type=int, default=3,
                   help="minimum root visits before an action's searched Q "
                        "is allowed to supervise the auxiliary Q head")
    p.add_argument("--td-lambda", type=float, default=1.0,
                   help="TD(lambda) for the value target. 1.0 labels every "
                        "position with the final result, as in AlphaZero. "
                        "Lower values bootstrap from the search values of "
                        "later positions, which cuts the variance an "
                        "exploratory opening injects into early-game labels. "
                        "Try 0.7 first.")
    p.add_argument("--value-mix", type=float, default=1.0,
                   help="weight on the game result z in the value target; the "
                        "remainder comes from the MCTS root value. 1.0 is "
                        "plain AlphaZero, 0.5 blends the two. Lower values cut "
                        "target noise without touching exploration, so they "
                        "do not risk the policy collapse that narrowing the "
                        "search causes.")
    p.add_argument("--balanced-sampling", action="store_true",
                   help="draw training batches equally across "
                        "(side to move) x (win/loss). Because the board is "
                        "encoded from the mover's point of view, the value "
                        "target in a decided game is a perfect function of "
                        "the ply parity, so lopsided self-play lets the value "
                        "head predict from the tile count instead of the "
                        "position. Balancing makes parity uninformative and "
                        "forces it to read the board.")
    p.add_argument("--min-buffer", type=int, default=8_000,
                   help="skip gradient steps until the buffer holds this many")

    p.add_argument("--arena-games", type=int, default=80)
    p.add_argument("--arena-simulations", type=int, default=100)
    p.add_argument("--promote-threshold", type=float, default=0.55)
    p.add_argument("--promotion-holdout-gate", dest="promotion_holdout_gate",
                   action="store_true", default=True,
                   help="when an exact validation DB is available, require an arena-"
                        "passing candidate to keep holdout AUC within tolerance of the "
                        "incumbent before promotion (v2.1.5 default)")
    p.add_argument("--no-promotion-holdout-gate", dest="promotion_holdout_gate",
                   action="store_false",
                   help="restore arena-only promotion")
    p.add_argument("--promotion-holdout-tolerance", type=float, default=0.01,
                   help="maximum allowed holdout-AUC drop versus the incumbent for "
                        "promotion; 0.01 permits small validation noise without letting "
                        "arena cycling silently degrade the value estimator")
    p.add_argument("--baseline-every", type=int, default=5,
                   help="also measure against the random baseline every N gens")

    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--symmetrise", dest="symmetrise", action="store_true",
                   default=True,
                   help="average each evaluation with its mirror image (v2 default)")
    p.add_argument("--no-symmetrise", dest="symmetrise", action="store_false",
                   help="disable mirror-averaged inference")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--buffer-save-every", type=int, default=1,
                   help="write the replay buffer to the run directory every "
                        "N generations (0 disables). Without this a restart "
                        "throws away several generations of self-play and the "
                        "next few generations train on a thin, skewed buffer.")
    return p.parse_args()


def build_configs(args):
    mcts_cfg = MCTSConfig(
        simulations=args.simulations,
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        temperature_moves=args.temperature_moves,
        search_mode=args.search_mode,
        gumbel_root_candidates=args.gumbel_root_candidates,
        gumbel_c_visit=args.gumbel_c_visit,
        gumbel_c_scale=args.gumbel_c_scale,
        gumbel_rescale_q=args.gumbel_rescale_q,
    )
    forced_openings = tuple(str_to_action(tok.upper()) for tok in args.forced_openings)
    root = new_game(None)
    legal_root = root.legal_actions()
    for action in forced_openings:
        if action < 0 or action >= ACTION_SIZE or not legal_root[action]:
            raise ValueError(f"illegal forced opening: {action_to_str(action)}")
    if not (0.0 <= args.forced_opening_fraction <= 1.0):
        raise ValueError("--forced-opening-fraction must be in [0, 1]")
    if not (0.0 <= args.policy_target_smoothing <= 1.0):
        raise ValueError("--policy-target-smoothing must be in [0, 1]")

    sp_cfg = SelfPlayConfig(
        games_per_generation=args.games_per_generation,
        parallel_games=args.parallel_games,
        mcts=mcts_cfg,
        value_mix=args.value_mix,
        td_lambda=args.td_lambda,
        limits=None,
        q_min_visits=args.q_min_visits,
        forced_opening_fraction=args.forced_opening_fraction,
        forced_openings=forced_openings,
        policy_target_smoothing=args.policy_target_smoothing,
    )
    return mcts_cfg, sp_cfg


def train_steps(net, optimizer, buffer, args, device, scaler, rng,
                exact_train_records=None):
    net.train()
    totals = np.zeros(5)
    steps = 0
    for _ in range(args.train_steps):
        s, p, v, q_t, q_w = buffer.sample(
            args.batch_size, rng, balanced=args.balanced_sampling, include_q=True)
        s = torch.from_numpy(s).to(device, non_blocking=True)
        p = torch.from_numpy(p).to(device, non_blocking=True)
        v = torch.from_numpy(v).to(device, non_blocking=True)
        q_t = torch.from_numpy(q_t).to(device, non_blocking=True)
        q_w = torch.from_numpy(q_w).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=scaler is not None):
            logits, value, q_pred = net.forward_with_q(s)
            # A zero-mass policy row marks a forced-opening root. It still
            # supervises value and Q, but must not teach the externally forced
            # action as a policy preference. With no forcing every row sums
            # to one, reproducing the legacy loss exactly.
            policy_mass = p.sum(dim=1)
            policy_rows = -(p * F.log_softmax(logits.float(), dim=1)).sum(1)
            policy_mask = (policy_mass > 1e-8).to(policy_rows.dtype)
            policy_loss = ((policy_rows * policy_mask).sum() /
                           policy_mask.sum().clamp_min(1.0))
            value_loss = F.mse_loss(value.float(), v)
            q_element = F.smooth_l1_loss(
                q_pred.float(), q_t, reduction="none")
            q_denom = q_w.sum().clamp_min(1.0)
            q_loss = (q_element * q_w).sum() / q_denom
            exact_loss = torch.zeros((), device=device, dtype=torch.float32)
            if args.exact_loss_weight > 0 and exact_train_records:
                exact_batch = sample_exact_decisive_batch(
                    exact_train_records, args.exact_batch_size, rng)
                if exact_batch is not None:
                    ex_s, ex_v = exact_batch
                    ex_s = torch.from_numpy(ex_s).to(device, non_blocking=True)
                    ex_v = torch.from_numpy(ex_v).to(device, non_blocking=True)
                    _ex_logits, ex_pred, _ex_q = net.forward_with_q(ex_s)
                    exact_loss = F.mse_loss(ex_pred.float(), ex_v)
            loss = (policy_loss + args.value_loss_weight * value_loss
                    + args.q_loss_weight * q_loss
                    + args.exact_loss_weight * exact_loss)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()

        totals += [loss.item(), policy_loss.item(), value_loss.item(), q_loss.item(), exact_loss.item()]
        steps += 1
    return totals / max(steps, 1)


@torch.inference_mode()
def evaluate_openings(evaluator, limits=None):
    """Network value for every legal opening move, from GREEN's side.

    The caller supplies the same ``NetEvaluator`` used by the production
    solver, so mirror symmetrisation and AMP settings cannot silently differ
    between training/search and the opening report.
    """
    root = new_game(limits)
    legal = np.flatnonzero(root.legal_actions())
    children = [root.step(int(a)) for a in legal]
    batch = np.stack([c.encode() for c in children]).astype(np.float32)
    _logits, values = evaluator(batch)
    out = {}
    for a, v in zip(legal, values):
        out[action_to_str(int(a))] = float(-v)   # GREEN's point of view
    return out


def evaluate_opening_diagnostics(evaluator, mcts_cfg, limits=None, rng_seed=12345):
    """Return P, Q-head, searched MCTS-Q, improved policy and visits at root.

    The search runs without Gumbel/Dirichlet noise so generation-to-generation
    changes reflect the model rather than diagnostic sampling noise.
    """
    root = new_game(limits)
    encoded = root.encode()[None, ...]
    logits, _value, q_head = evaluator.evaluate_with_q(encoded)
    logits = logits[0].astype(np.float64)
    legal = root.legal_actions()
    logits[~legal] = -np.inf
    prior = np.zeros(ACTION_SIZE, dtype=np.float32)
    if legal.any():
        x = logits[legal]
        x = x - np.max(x)
        ex = np.exp(x)
        prior[legal] = (ex / ex.sum()).astype(np.float32)

    diag_mcts = BatchedMCTS(mcts_cfg, np.random.default_rng(rng_seed))
    visits, improved, action_q = diag_mcts.search(
        [root], evaluator, add_noise=False, with_policy=True,
        with_action_values=True)
    qh = q_head[0].astype(np.float32)
    mq = action_q[0].astype(np.float32)
    qh = qh.copy()
    mq = mq.copy()
    qh[~legal] = np.nan
    mq[~legal] = np.nan
    return {
        "policy": prior,
        "q_head": qh,
        "mcts_q": mq,
        "visits": visits[0].astype(np.float32),
        "improved": improved[0].astype(np.float32),
    }


def _policy_entropy(policies: np.ndarray) -> float:
    """Mean normalised entropy of policy targets, in [0, 1].

    A deterministic target is 0 and a uniform distribution over all 21 actions
    is 1. This is a direct collapse diagnostic that does not depend on value
    calibration.
    """
    p = np.asarray(policies, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    row_sum = p.sum(axis=1, keepdims=True)
    valid = row_sum[:, 0] > 1e-12
    if not np.any(valid):
        return float("nan")
    p = np.divide(p, row_sum, out=np.zeros_like(p), where=row_sum > 0)
    ent = -(np.where(p > 0, p * np.log(np.clip(p, 1e-12, 1.0)), 0.0)).sum(axis=1)
    return float(ent[valid].mean() / np.log(ACTION_SIZE))


def _opening_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return float("nan")
    p = counts / total
    ent = -np.sum(np.where(p > 0, p * np.log(np.clip(p, 1e-12, 1.0)), 0.0))
    return float(ent / np.log(ACTION_SIZE))


def _td_parity_diagnostics(states: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """TD(lambda)-safe parity diagnostics.

    ``sign_gap`` measures whether positive/negative value *sign* remains
    predictable from ply parity. This is the actual shortcut balanced sampling
    is meant to remove. ``abs_gap`` separately reports a parity difference in
    target magnitude, which is expected under TD(lambda) and must not be
    mistaken for the old sign shortcut. Near-zero targets are ignored for the
    sign calculation.
    """
    states = np.asarray(states)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        return float("nan"), float("nan")
    ply = np.rint(states[:, :6].sum(axis=(1, 2, 3))).astype(np.int64)
    parity = ply & 1
    sign_rates = []
    abs_means = []
    for k in (0, 1):
        vk = values[parity == k]
        if len(vk) == 0:
            return float("nan"), float("nan")
        decisive = vk[np.abs(vk) > 1e-6]
        sign_rates.append(float(np.mean(decisive > 0)) if len(decisive) else float("nan"))
        abs_means.append(float(np.mean(np.abs(vk))))
    sign_gap = (abs(sign_rates[0] - sign_rates[1])
                if np.all(np.isfinite(sign_rates)) else float("nan"))
    abs_gap = abs(abs_means[0] - abs_means[1])
    return sign_gap, abs_gap


def _promotion_gate(arena_score: float, arena_threshold: float,
                    candidate_auc: float, incumbent_auc: float,
                    tolerance: float, enabled: bool) -> tuple[bool, bool]:
    """Return (promote, holdout_gate_pass)."""
    arena_pass = np.isfinite(arena_score) and arena_score >= arena_threshold
    if not arena_pass:
        return False, False
    if not enabled:
        return True, True
    auc_ok = (np.isfinite(candidate_auc) and
              (not np.isfinite(incumbent_auc) or
               candidate_auc >= incumbent_auc - tolerance))
    return bool(auc_ok), bool(auc_ok)


def _open_csv_append_compatible(path: Path, header: list[str]):
    """Open a CSV for append, migrating older headers when necessary.

    v2.1 adds Q-loss diagnostics to metrics.csv. Resuming a v2 run should not
    silently append wider rows under the old header, so existing rows are
    rewritten by column name and new fields are left blank.
    """
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            old_header = reader.fieldnames or []
            rows = list(reader)
        if old_header != header:
            tmp = path.with_name(path.name + ".migrate.tmp")
            with tmp.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: row.get(k, "") for k in header})
            tmp.replace(path)
    fresh = not path.exists() or path.stat().st_size == 0
    f = path.open("a", newline="")
    writer = csv.writer(f)
    if fresh:
        writer.writerow(header)
        f.flush()
    return f, writer


def main():
    args = parse_args()
    # 7x7 is an unusual shape; letting cuDNN pick its algorithm once is worth
    # a few percent and costs one warm-up iteration.
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")
    if device.type != "cuda":
        print("warning: CUDA not available, running on CPU (this will be slow)")

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.init_run_dir is not None:
        if args.resume or args.init_checkpoint is not None or args.init_buffer is not None:
            raise SystemExit("--init-run-dir cannot be combined with --resume, "
                             "--init-checkpoint or --init-buffer")
        args.init_checkpoint = args.init_run_dir / "best.pt"
        args.init_buffer = args.init_run_dir / "buffer.npz"
    if args.resume and (args.init_checkpoint is not None or args.init_buffer is not None):
        raise SystemExit("--resume cannot be combined with --init-checkpoint/--init-buffer")

    (run_dir / "config.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    if (args.validation_database is not None and args.exact_loss_weight > 0
            and args.validation_select in {"auc", "balanced_accuracy", "macro_f1", "mcc"}):
        print("warning: --validation-select uses the full exact database while "
              "--exact-loss-weight > 0. This leaks exact-training groups into "
              "checkpoint selection. Prefer --validation-select holdout_auc.")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    mcts_cfg, sp_cfg = build_configs(args)

    best_path = run_dir / "best.pt"
    start_gen = 0
    if args.resume and best_path.exists():
        net, start_gen = load_checkpoint(best_path, device)
        print(f"resumed from generation {start_gen}")
    elif args.init_checkpoint is not None:
        net, start_gen = load_checkpoint(args.init_checkpoint, device)
        print(f"initialised from {args.init_checkpoint} (generation {start_gen})")
    else:
        model_cfg = ModelConfig.preset(args.preset)
        if args.blocks is not None:
            model_cfg.blocks = args.blocks
        if args.channels is not None:
            model_cfg.channels = args.channels
        if args.coord_conv is not None:
            model_cfg.coord_conv = args.coord_conv
        net = Make7Net(model_cfg).to(device)
    print(f"network: {args.preset} "
          f"({net.cfg.blocks} blocks x {net.cfg.channels} ch, "
          f"{'residual' if net.cfg.residual else 'plain'}"
          f"{', coordconv' if net.cfg.coord_conv else ''}), "
          f"{net.num_parameters():,} parameters")

    best_net = Make7Net(net.cfg).to(device)
    best_net.load_state_dict(net.state_dict())
    if not best_path.exists():
        # Write an initial best so that --resume and the analysis scripts
        # have something to load even if generation 1 never promotes.
        save_checkpoint(best_path, best_net, generation=start_gen)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    use_amp = (device.type == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    buffer = ReplayBuffer(args.buffer_size)
    buffer_path = run_dir / "buffer.npz"
    if args.resume and buffer.load(buffer_path):
        print(f"restored replay buffer: {len(buffer):,} samples")
    elif not args.resume and args.init_buffer is not None:
        if buffer.load(args.init_buffer):
            print(f"inherited replay buffer from {args.init_buffer}: "
                  f"{len(buffer):,} samples")
            # Persist immediately in the new run so an interruption before the
            # first generation still leaves the inherited buffer recoverable.
            buffer.save(buffer_path)
        else:
            raise SystemExit(f"could not load --init-buffer {args.init_buffer}")

    limits = None
    validation_records = []
    exact_train_records = []
    exact_test_records = []
    validation_writer = None
    validation_file = None
    exact_predictions_path = None
    if args.validation_database is not None:
        validation_records = load_exact_records(
            args.validation_database, limits=limits,
            group_plies=args.validation_group_plies)
        exact_train_records, exact_test_records = stratified_split_by_group(
            validation_records, args.validation_calibration_fraction,
            args.validation_calibration_seed)
        print(describe(validation_records, "validation all"))
        print(describe(exact_train_records, "calibration/exact-train"))
        print(describe(exact_test_records, "untouched exact-test"))
        if limits is None:
            print("  note: training without tile limits while validating "
                  "against a database proved with them; treat the score as a "
                  "cross-ruleset consistency check, not as accuracy")
        best_validation_path = run_dir / "best_validation.pt"
        vpath = run_dir / "validation.csv"
        validation_header = [
            "generation", "balanced_accuracy", "macro_f1", "auc", "mcc",
            "green_win_accuracy", "yellow_win_accuracy",
            "le0_consistency", "ge0_consistency", "mean_green_value",
            "raw_auc", "raw_balanced_accuracy", "sym_auc", "sym_balanced_accuracy",
            "holdout_auc", "holdout_balanced_accuracy", "holdout_mcc",
            "holdout_raw_auc", "holdout_raw_balanced_accuracy", "holdout_raw_mcc",
            "holdout_sym_auc", "holdout_sym_balanced_accuracy", "holdout_sym_mcc",
            "optimal_threshold", "calibration_balanced_accuracy",
            "test_balanced_accuracy_at_zero", "calibrated_balanced_accuracy",
            "test_mcc_at_zero", "calibrated_mcc",
            "calibrated_green_win_accuracy", "calibrated_yellow_win_accuracy",
            "calibration_records", "calibration_groups", "test_records", "test_groups",
            "records", "groups", "proposition_support_rate",
            "decisive_support_rate", "bound_support_rate",
            "proposition_supported", "proposition_total",
        ]
        validation_file, validation_writer = _open_csv_append_compatible(
            vpath, validation_header)
        exact_predictions_path = run_dir / "exact_proposition_predictions.csv"

    def lr_at(generation: int) -> float:
        if args.lr_final is None:
            return args.lr
        span = max(1, args.lr_decay_generations)
        frac = min(1.0, max(0, generation - start_gen) / span)
        return args.lr * (args.lr_final / args.lr) ** frac

    metrics_path = run_dir / "metrics.csv"
    openings_path = run_dir / "openings.csv"
    metrics_header = [
        "generation", "loss", "policy_loss", "value_loss", "q_loss", "exact_loss",
        "arena_score", "arena_elo", "candidate_holdout_auc",
        "incumbent_holdout_auc", "holdout_gate_pass", "promoted",
        "baseline_score", "baseline_elo", "green_win_rate", "yellow_win_rate",
        "draw_rate", "organic_green_win_rate", "forced_green_win_rate",
        "mean_plies", "buffer",
        "parity_gap", "parity_raw_sign_gap", "parity_abs_gap",
        "policy_target_entropy", "policy_target_entropy_raw",
        "policy_target_entropy_train", "opening_entropy", "forced_opening_entropy",
        "forced_games", "forced_opening_fraction_actual",
        "q_actions_per_position", "selfplay_seconds", "train_seconds",
    ]
    metrics_file, metrics_writer = _open_csv_append_compatible(
        metrics_path, metrics_header)
    new_openings = not openings_path.exists()
    openings_file = openings_path.open("a", newline="")
    openings_writer = csv.writer(openings_file)
    if new_openings:
        header = ["generation"]
        header += [f"visits_{action_to_str(a)}" for a in range(ACTION_SIZE)]
        header += [f"value_{action_to_str(a)}" for a in range(ACTION_SIZE)]
        openings_writer.writerow(header)

    opening_sources_path = run_dir / "opening_sources.csv"
    opening_sources_header = ["generation", "forced_games"]
    opening_sources_header += [f"organic_{action_to_str(a)}" for a in range(ACTION_SIZE)]
    opening_sources_header += [f"forced_{action_to_str(a)}" for a in range(ACTION_SIZE)]
    opening_sources_file, opening_sources_writer = _open_csv_append_compatible(
        opening_sources_path, opening_sources_header)

    opening_outcomes_path = run_dir / "opening_outcomes.csv"
    opening_outcomes_header = [
        "generation", "organic_games", "organic_green_wins", "organic_draws",
        "organic_yellow_wins", "organic_green_win_rate", "organic_green_score",
        "forced_games", "forced_green_wins", "forced_draws",
        "forced_yellow_wins", "forced_green_win_rate", "forced_green_score",
    ]
    for tok in args.forced_openings:
        t = tok.upper()
        opening_outcomes_header += [
            f"forced_{t}_games", f"forced_{t}_green_wins", f"forced_{t}_draws",
            f"forced_{t}_yellow_wins", f"forced_{t}_green_win_rate",
            f"forced_{t}_green_score",
        ]
    opening_outcomes_file, opening_outcomes_writer = _open_csv_append_compatible(
        opening_outcomes_path, opening_outcomes_header)

    root_policy_path = run_dir / "organic_root_policy.csv"
    root_policy_header = ["generation", "organic_roots", "smoothing"]
    root_policy_header += [f"raw_{action_to_str(a)}" for a in range(ACTION_SIZE)]
    root_policy_header += [f"train_{action_to_str(a)}" for a in range(ACTION_SIZE)]
    root_policy_file, root_policy_writer = _open_csv_append_compatible(
        root_policy_path, root_policy_header)

    qdiag_path = run_dir / "opening_q_diagnostics.csv"
    new_qdiag = not qdiag_path.exists()
    qdiag_file = qdiag_path.open("a", newline="")
    qdiag_writer = csv.writer(qdiag_file)
    if new_qdiag:
        header = ["generation"]
        for prefix in ("policy", "qhead", "mctsq", "visits", "improved"):
            header += [f"{prefix}_{action_to_str(a)}" for a in range(ACTION_SIZE)]
        qdiag_writer.writerow(header)

    cumulative_elo = 0.0
    # Arena gating promotes whatever beats the previous network, which is not
    # the same thing as the network that evaluates positions best: in the
    # previous run the validation score peaked around generation 40 and then
    # fell away for another forty generations while promotions carried on.
    # Tracking the best validation score separately keeps that checkpoint.
    best_validation_score = float("-inf")
    best_validation_generation = 0
    if args.resume and args.validation_database is not None:
        meta = run_dir / "best_validation.json"
        if meta.exists():
            try:
                saved = json.loads(meta.read_text())
                saved_metric = saved.get("metric")
                if saved_metric == args.validation_select:
                    best_validation_score = float(saved["score"])
                    best_validation_generation = int(saved["generation"])
                    print(f"previous best validation "
                          f"{args.validation_select}={best_validation_score:.4f} "
                          f"at generation {best_validation_generation}")
                else:
                    print(f"previous best_validation.json used {saved_metric!r}; "
                          f"resetting best score for {args.validation_select!r}")
            except Exception:
                pass

    promotion_gate_active = bool(
        args.promotion_holdout_gate and len(exact_test_records) > 0)
    if args.promotion_holdout_gate and not promotion_gate_active:
        print("note: promotion holdout gate requested but no exact holdout is "
              "available; using arena-only promotion")
    incumbent_holdout_auc = float("nan")
    if promotion_gate_active:
        incumbent_holdout_auc = evaluate_exact(
            exact_test_records,
            NetEvaluator(best_net, device, use_amp, symmetrise=args.symmetrise),
            draw_band=args.validation_draw_band)["auc"]
        print(f"promotion gate incumbent holdout AUC: "
              f"{incumbent_holdout_auc:.4f} "
              f"(tolerance {args.promotion_holdout_tolerance:.3f})")

    pool = None
    if args.workers > 0:
        pool = ParallelSelfPlay(
            n_workers=args.workers,
            max_batch=max(1, -(-args.games_per_generation // args.workers)),
        )
        print(f"self-play pool: {args.workers} workers, "
              f"inference stays in the parent process")

    try:
      for gen in range(start_gen, start_gen + args.generations):
          t0 = time.perf_counter()
          evaluator = NetEvaluator(best_net, device, use_amp,
                                   symmetrise=args.symmetrise)
          if pool is None:
              samples, stats = generate_games(evaluator, sp_cfg, rng)
              buffer.add(samples)
          else:
              (s_arr, p_arr, v_arr, q_arr, qw_arr), stats = pool.run(
                  evaluator, sp_cfg, seed=gen,
                  gather_timeout=args.gather_timeout)
              if s_arr is not None and len(s_arr):
                  buffer.add_arrays(s_arr, p_arr, v_arr, q_arr, qw_arr)
          selfplay_seconds = time.perf_counter() - t0

          t1 = time.perf_counter()
          current_lr = lr_at(gen)
          for group in optimizer.param_groups:
              group["lr"] = current_lr
          if len(buffer) >= args.min_buffer:
              loss, ploss, vloss, qloss, exactloss = train_steps(
                  net, optimizer, buffer, args, device, scaler, rng,
                  exact_train_records=exact_train_records)
          else:
              loss = ploss = vloss = qloss = exactloss = float("nan")
          train_seconds = time.perf_counter() - t1

          promoted = 0
          arena = {"score": float("nan"), "elo_diff": float("nan")}
          candidate_holdout_auc = float("nan")
          incumbent_auc_before = incumbent_holdout_auc
          holdout_gate_pass = 0
          if len(buffer) >= args.min_buffer:
              candidate_eval = NetEvaluator(
                  net, device, use_amp, symmetrise=args.symmetrise)
              arena = play_match(
                  candidate_eval,
                  NetEvaluator(best_net, device, use_amp, symmetrise=args.symmetrise),
                  n_games=args.arena_games,
                  mcts_config=MCTSConfig(
                      simulations=args.arena_simulations,
                      search_mode=args.search_mode,
                      gumbel_root_candidates=args.gumbel_root_candidates,
                      gumbel_c_visit=args.gumbel_c_visit,
                      gumbel_c_scale=args.gumbel_c_scale,
                      gumbel_rescale_q=args.gumbel_rescale_q),
                  rng=rng,
                  parallel_games=min(args.parallel_games, args.arena_games),
              )
              arena_pass = arena["score"] >= args.promote_threshold
              # Only pay for an extra exact evaluation when the candidate has
              # already cleared the arena gate. This keeps the guard cheap.
              if arena_pass and promotion_gate_active:
                  candidate_holdout_auc = evaluate_exact(
                      exact_test_records, candidate_eval,
                      draw_band=args.validation_draw_band)["auc"]
              promote, gate_ok = _promotion_gate(
                  arena["score"], args.promote_threshold,
                  candidate_holdout_auc, incumbent_holdout_auc,
                  args.promotion_holdout_tolerance, promotion_gate_active)
              holdout_gate_pass = int(gate_ok)
              if promote:
                  best_net.load_state_dict(net.state_dict())
                  promoted = 1
                  cumulative_elo += arena["elo_diff"]
                  if promotion_gate_active and np.isfinite(candidate_holdout_auc):
                      incumbent_holdout_auc = candidate_holdout_auc
                  save_checkpoint(best_path, best_net, generation=gen + 1,
                                  extra={"cumulative_elo": cumulative_elo,
                                         "holdout_auc": incumbent_holdout_auc})

          baseline_score = baseline_elo = float("nan")
          if args.baseline_every and (gen + 1) % args.baseline_every == 0:
              base = play_match(
                  NetEvaluator(best_net, device, use_amp, symmetrise=args.symmetrise),
                  random_evaluator,
                  n_games=args.arena_games,
                  mcts_config=MCTSConfig(
                      simulations=args.arena_simulations,
                      search_mode=args.search_mode,
                      gumbel_root_candidates=args.gumbel_root_candidates,
                      gumbel_c_visit=args.gumbel_c_visit,
                      gumbel_c_scale=args.gumbel_c_scale,
                      gumbel_rescale_q=args.gumbel_rescale_q),
                  rng=rng,
                  parallel_games=min(args.parallel_games, args.arena_games),
              )
              baseline_score = base["score"]
              baseline_elo = score_to_elo(baseline_score)

          # TD(lambda)-safe shortcut diagnostics. The old mean-value parity
          # gap mixed up sign leakage with a harmless difference in target
          # magnitude. We now measure sign leakage separately, both in the raw
          # replay distribution and in the actually sampled training batches.
          if len(buffer) >= args.min_buffer:
              raw_s, _raw_p, raw_v = buffer.sample(8192, rng, balanced=False)
              train_s, _train_p, train_v = buffer.sample(
                  8192, rng, balanced=args.balanced_sampling)
              parity_raw_sign_gap, parity_abs_gap = _td_parity_diagnostics(
                  raw_s, raw_v)
              parity_gap, _train_abs_gap = _td_parity_diagnostics(
                  train_s, train_v)
          else:
              parity_gap = parity_raw_sign_gap = parity_abs_gap = float("nan")

          hpi_n = int(stats.get("policy_entropy_count", 0))
          if hpi_n > 0:
              policy_target_entropy_raw = (
                  float(stats.get("policy_entropy_raw_sum", 0.0)) / hpi_n)
              policy_target_entropy_train = (
                  float(stats.get("policy_entropy_train_sum", 0.0)) / hpi_n)
          else:
              policy_target_entropy_raw = policy_target_entropy_train = float("nan")
          # Keep the legacy column as the actual teacher entropy for backwards
          # compatibility with existing plotting scripts.
          policy_target_entropy = policy_target_entropy_train
          # Keep Hopen as a collapse diagnostic: forced openings are excluded
          # so the intervention cannot manufacture artificial entropy.
          opening_entropy = _opening_entropy(stats["organic_opening_counts"])
          forced_opening_entropy = _opening_entropy(stats["forced_opening_counts"])

          g = max(stats["games"], 1)
          forced_fraction_actual = stats.get("forced_games", 0) / g
          q_actions_per_position = (
              stats.get("q_supervised_actions", 0)
              / max(stats.get("q_positions", 0), 1))
          metrics_writer.writerow([
              gen + 1, f"{loss:.4f}", f"{ploss:.4f}", f"{vloss:.4f}",
              f"{qloss:.4f}", f"{exactloss:.4f}", f"{arena['score']:.4f}",
              f"{arena['elo_diff']:.1f}",
              f"{candidate_holdout_auc:.4f}", f"{incumbent_auc_before:.4f}",
              holdout_gate_pass, promoted,
              f"{baseline_score:.4f}", f"{baseline_elo:.1f}",
              f"{stats['green_wins'] / g:.4f}",
              f"{stats['yellow_wins'] / g:.4f}",
              f"{stats['draws'] / g:.4f}",
              f"{stats.get('organic_green_wins', 0) / max(stats.get('organic_games', 0), 1):.4f}",
              f"{stats.get('forced_green_wins', 0) / max(stats.get('forced_games', 0), 1):.4f}",
              f"{stats['total_plies'] / g:.2f}",
              len(buffer), f"{parity_gap:.4f}",
              f"{parity_raw_sign_gap:.4f}", f"{parity_abs_gap:.4f}",
              f"{policy_target_entropy:.4f}",
              f"{policy_target_entropy_raw:.4f}",
              f"{policy_target_entropy_train:.4f}", f"{opening_entropy:.4f}",
              f"{forced_opening_entropy:.4f}", stats.get("forced_games", 0),
              f"{forced_fraction_actual:.4f}", f"{q_actions_per_position:.3f}",
              f"{selfplay_seconds:.1f}", f"{train_seconds:.1f}",
          ])
          metrics_file.flush()

          if (validation_records and args.validation_every
                  and (gen + 1) % args.validation_every == 0):
              raw_eval = NetEvaluator(best_net, device, use_amp, symmetrise=False)
              sym_eval = NetEvaluator(best_net, device, use_amp, symmetrise=True)
              raw_metrics, raw_values = evaluate_exact(
                  validation_records, raw_eval,
                  draw_band=args.validation_draw_band, return_values=True)
              sym_metrics, sym_values = evaluate_exact(
                  validation_records, sym_eval,
                  draw_band=args.validation_draw_band, return_values=True)
              holdout_raw_metrics = evaluate_exact(
                  exact_test_records, raw_eval,
                  draw_band=args.validation_draw_band)
              holdout_sym_metrics = evaluate_exact(
                  exact_test_records, sym_eval,
                  draw_band=args.validation_draw_band)

              # The production solver normally uses mirror-symmetrised
              # inference; make that the primary validation view when enabled,
              # while always logging both raw and symmetrised metrics.
              vmetrics = dict(sym_metrics if args.symmetrise else raw_metrics)
              primary_holdout = (holdout_sym_metrics if args.symmetrise
                                 else holdout_raw_metrics)
              vmetrics.update({
                  "raw_auc": raw_metrics["auc"],
                  "raw_balanced_accuracy": raw_metrics["balanced_accuracy"],
                  "sym_auc": sym_metrics["auc"],
                  "sym_balanced_accuracy": sym_metrics["balanced_accuracy"],
                  "holdout_auc": primary_holdout["auc"],
                  "holdout_balanced_accuracy": primary_holdout["balanced_accuracy"],
                  "holdout_mcc": primary_holdout["mcc"],
                  "holdout_raw_auc": holdout_raw_metrics["auc"],
                  "holdout_raw_balanced_accuracy": holdout_raw_metrics["balanced_accuracy"],
                  "holdout_raw_mcc": holdout_raw_metrics["mcc"],
                  "holdout_sym_auc": holdout_sym_metrics["auc"],
                  "holdout_sym_balanced_accuracy": holdout_sym_metrics["balanced_accuracy"],
                  "holdout_sym_mcc": holdout_sym_metrics["mcc"],
              })

              # Fit the classification threshold on calibration groups only and
              # evaluate it on untouched groups. This is diagnostic only: MCTS
              # continues to consume continuous values, not thresholded labels.
              cal_values = green_values(exact_train_records,
                                        sym_eval if args.symmetrise else raw_eval)
              test_values = green_values(exact_test_records,
                                         sym_eval if args.symmetrise else raw_eval)
              creport = calibration_report(
                  exact_train_records, cal_values, exact_test_records, test_values)
              vmetrics.update(creport)
              vmetrics.update({
                  "calibration_records": len(exact_train_records),
                  "calibration_groups": len({r.group for r in exact_train_records}),
                  "test_records": len(exact_test_records),
                  "test_groups": len({r.group for r in exact_test_records}),
              })

              primary_values = sym_values if args.symmetrise else raw_values
              pmetrics = write_proposition_predictions(
                  exact_predictions_path, gen + 1, validation_records,
                  primary_values, draw_band=args.validation_draw_band)
              vmetrics.update(pmetrics)
              validation_writer.writerow([
                  gen + 1,
                  f"{vmetrics['balanced_accuracy']:.4f}",
                  f"{vmetrics['macro_f1']:.4f}",
                  f"{vmetrics['auc']:.4f}",
                  f"{vmetrics['mcc']:.4f}",
                  f"{vmetrics['green_win_accuracy']:.4f}",
                  f"{vmetrics['yellow_win_accuracy']:.4f}",
                  f"{vmetrics['le0_consistency']:.4f}",
                  f"{vmetrics['ge0_consistency']:.4f}",
                  f"{vmetrics['mean_green_value']:.4f}",
                  f"{vmetrics['raw_auc']:.4f}",
                  f"{vmetrics['raw_balanced_accuracy']:.4f}",
                  f"{vmetrics['sym_auc']:.4f}",
                  f"{vmetrics['sym_balanced_accuracy']:.4f}",
                  f"{vmetrics['holdout_auc']:.4f}",
                  f"{vmetrics['holdout_balanced_accuracy']:.4f}",
                  f"{vmetrics['holdout_mcc']:.4f}",
                  f"{vmetrics['holdout_raw_auc']:.4f}",
                  f"{vmetrics['holdout_raw_balanced_accuracy']:.4f}",
                  f"{vmetrics['holdout_raw_mcc']:.4f}",
                  f"{vmetrics['holdout_sym_auc']:.4f}",
                  f"{vmetrics['holdout_sym_balanced_accuracy']:.4f}",
                  f"{vmetrics['holdout_sym_mcc']:.4f}",
                  f"{vmetrics['optimal_threshold']:.4f}",
                  f"{vmetrics['calibration_balanced_accuracy']:.4f}",
                  f"{vmetrics['test_balanced_accuracy_at_zero']:.4f}",
                  f"{vmetrics['calibrated_balanced_accuracy']:.4f}",
                  f"{vmetrics['test_mcc_at_zero']:.4f}",
                  f"{vmetrics['calibrated_mcc']:.4f}",
                  f"{vmetrics['calibrated_green_win_accuracy']:.4f}",
                  f"{vmetrics['calibrated_yellow_win_accuracy']:.4f}",
                  vmetrics["calibration_records"], vmetrics["calibration_groups"],
                  vmetrics["test_records"], vmetrics["test_groups"],
                  vmetrics["records"], vmetrics["groups"],
                  f"{vmetrics['proposition_support_rate']:.4f}",
                  f"{vmetrics['decisive_support_rate']:.4f}",
                  f"{vmetrics['bound_support_rate']:.4f}",
                  vmetrics["proposition_supported"],
                  vmetrics["proposition_total"],
              ])
              validation_file.flush()

              score = vmetrics.get(args.validation_select, float("nan"))
              if np.isfinite(score) and score > best_validation_score:
                  best_validation_score = float(score)
                  best_validation_generation = gen + 1
                  save_checkpoint(
                      best_validation_path, best_net, generation=gen + 1,
                      extra={"validation": vmetrics,
                             "selected_by": args.validation_select})
                  (run_dir / "best_validation.json").write_text(json.dumps({
                      "generation": gen + 1,
                      "metric": args.validation_select,
                      "score": best_validation_score,
                      "metrics": {k: (None if isinstance(v, float)
                                      and not np.isfinite(v) else v)
                                  for k, v in vmetrics.items()},
                  }, indent=2))
          else:
              vmetrics = None

          opening_eval = NetEvaluator(best_net, device, use_amp,
                                      symmetrise=args.symmetrise)
          opening_values = evaluate_openings(opening_eval, limits)
          row = [gen + 1] + stats["opening_counts"].tolist()
          row += [f"{opening_values.get(action_to_str(a), float('nan')):.4f}"
                  for a in range(ACTION_SIZE)]
          openings_writer.writerow(row)
          openings_file.flush()
          opening_sources_writer.writerow(
              [gen + 1, stats.get("forced_games", 0)]
              + stats["organic_opening_counts"].tolist()
              + stats["forced_opening_counts"].tolist())
          opening_sources_file.flush()

          org_n = int(stats.get("organic_games", 0))
          org_w = int(stats.get("organic_green_wins", 0))
          org_d = int(stats.get("organic_draws", 0))
          org_l = int(stats.get("organic_yellow_wins", 0))
          frc_n = int(stats.get("forced_games", 0))
          frc_w = int(stats.get("forced_green_wins", 0))
          frc_d = int(stats.get("forced_draws", 0))
          frc_l = int(stats.get("forced_yellow_wins", 0))
          outcome_row = [
              gen + 1, org_n, org_w, org_d, org_l,
              f"{org_w / max(org_n, 1):.6f}",
              f"{(org_w + 0.5 * org_d) / max(org_n, 1):.6f}",
              frc_n, frc_w, frc_d, frc_l,
              f"{frc_w / max(frc_n, 1):.6f}",
              f"{(frc_w + 0.5 * frc_d) / max(frc_n, 1):.6f}",
          ]
          forced_rate_note = []
          for tok in args.forced_openings:
              a0 = str_to_action(tok.upper())
              n0 = int(stats["forced_result_games"][a0])
              w0 = int(stats["forced_green_wins_by_opening"][a0])
              d0 = int(stats["forced_draws_by_opening"][a0])
              l0 = n0 - w0 - d0
              wr0 = w0 / max(n0, 1)
              sc0 = (w0 + 0.5 * d0) / max(n0, 1)
              outcome_row += [n0, w0, d0, l0, f"{wr0:.6f}", f"{sc0:.6f}"]
              forced_rate_note.append(f"{tok.upper()} {wr0:.2f}")
          opening_outcomes_writer.writerow(outcome_row)
          opening_outcomes_file.flush()

          root_n = int(stats.get("organic_root_policy_count", 0))
          if root_n > 0:
              raw_root_policy = stats["organic_root_policy_raw_sum"] / root_n
              train_root_policy = stats["organic_root_policy_train_sum"] / root_n
          else:
              raw_root_policy = np.full(ACTION_SIZE, np.nan)
              train_root_policy = np.full(ACTION_SIZE, np.nan)
          root_policy_writer.writerow(
              [gen + 1, root_n, f"{args.policy_target_smoothing:.4f}"]
              + [f"{x:.8f}" for x in raw_root_policy]
              + [f"{x:.8f}" for x in train_root_policy])
          root_policy_file.flush()

          diag_eval = NetEvaluator(best_net, device, use_amp,
                                   symmetrise=args.symmetrise)
          diag_cfg = MCTSConfig(
              simulations=args.simulations,
              c_puct=args.c_puct,
              search_mode=args.search_mode,
              gumbel_root_candidates=args.gumbel_root_candidates,
              gumbel_c_visit=args.gumbel_c_visit,
              gumbel_c_scale=args.gumbel_c_scale,
              gumbel_rescale_q=args.gumbel_rescale_q,
          )
          qdiag = evaluate_opening_diagnostics(
              diag_eval, diag_cfg, limits=limits, rng_seed=args.seed)
          qrow = [gen + 1]
          for key in ("policy", "q_head", "mcts_q", "visits", "improved"):
              qrow += [f"{float(x):.6f}" for x in qdiag[key]]
          qdiag_writer.writerow(qrow)
          qdiag_file.flush()

          save_checkpoint(run_dir / "latest.pt", net, optimizer, gen + 1,
                          extra={"cumulative_elo": cumulative_elo})
          if args.buffer_save_every and (gen + 1) % args.buffer_save_every == 0:
              buffer.save(buffer_path)

          top = sorted(opening_values.items(), key=lambda kv: -kv[1])[:3]
          top_str = ", ".join(f"{k} {v:+.3f}" for k, v in top)
          val_note = ("" if vmetrics is None else
                      f"| exact holdout bal {vmetrics['holdout_balanced_accuracy']:.3f} "
                      f"auc {vmetrics['holdout_auc']:.3f} "
                      f"mcc {vmetrics['holdout_mcc']:+.3f} "
                      f"| all auc {vmetrics['auc']:.3f} "
                      f"prop {vmetrics['proposition_support_rate']:.3f}"
                      + (" *BEST*" if best_validation_generation == gen + 1
                         else "") + " ")
          batch_note = (
              f"| batch {stats.get('mean_batch', 0):.0f} "
              f"fwd {stats.get('forward_passes', 0)} "
              f"gpu {stats.get('gpu_seconds', 0):.0f}s "
              if pool is not None else "")
          print(
              f"gen {gen + 1:4d} | loss {loss:6.3f} "
              f"(p {ploss:5.3f} v {vloss:5.3f} q {qloss:5.3f}) "
              f"| arena {arena['score']:.3f}{' PROMOTED' if promoted else ''}"
              + (f" gate {candidate_holdout_auc:.3f}/{incumbent_auc_before:.3f}"
                 if np.isfinite(candidate_holdout_auc) else "") + " "
              f"| green {stats['green_wins'] / g:.2f} draw {stats['draws'] / g:.2f} "
              f"(org {stats.get('organic_green_wins', 0) / max(stats.get('organic_games', 0), 1):.2f} "
              f"forced {stats.get('forced_green_wins', 0) / max(stats.get('forced_games', 0), 1):.2f}; "
              + ",".join(forced_rate_note) + ") "
              f"| plies {stats['total_plies'] / g:4.1f} "
              f"| buf {len(buffer):6d} "
              f"| lr {current_lr:.1e} "
              f"| {selfplay_seconds:5.1f}s+{train_seconds:4.1f}s "
              f"{batch_note}{val_note}"
              f"| parity sign {parity_gap:.2f}/{parity_raw_sign_gap:.2f} "
              f"abs {parity_abs_gap:.2f} "
              f"Hpi {policy_target_entropy_raw:.2f}/{policy_target_entropy_train:.2f} "
              f"Hopen {opening_entropy:.2f} "
              f"forced {forced_fraction_actual:.2f} "
              f"q/a {q_actions_per_position:.1f} "
              f"| best openings: {top_str}"
          )

    finally:
        if pool is not None:
            pool.close()

    metrics_file.close()
    openings_file.close()
    opening_sources_file.close()
    opening_outcomes_file.close()
    root_policy_file.close()
    qdiag_file.close()
    if validation_file is not None:
        validation_file.close()
    if best_validation_generation:
        print(f"\nbest validation {args.validation_select}="
              f"{best_validation_score:.4f} at generation "
              f"{best_validation_generation} -> best_validation.pt")
    print(f"\ndone. checkpoints and logs in {run_dir}")


if __name__ == "__main__":
    main()
