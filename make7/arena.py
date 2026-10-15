"""Head-to-head evaluation between two players.

Used for two things: gating a new generation against the current best
network, and producing the strength-versus-generation curve that a write-up
needs in order to claim that training has converged.
"""

from __future__ import annotations

import math

import numpy as np

from .game import ACTION_SIZE, new_game
from .mcts import BatchedMCTS, MCTSConfig, visits_to_policy


def random_evaluator(batch: np.ndarray):
    """Zero-information baseline: uniform policy, value 0."""
    b = batch.shape[0]
    return np.zeros((b, ACTION_SIZE), dtype=np.float32), np.zeros(b, dtype=np.float32)


def play_match(evaluator_a, evaluator_b, n_games: int,
               mcts_config: MCTSConfig | None = None,
               rng: np.random.Generator | None = None,
               parallel_games: int = 64,
               random_opening_plies: int = 2,
               temperature: float = 0.0) -> dict:
    """Play `n_games` between A and B, alternating who moves first.

    Returns wins/losses/draws from A's point of view plus the score rate.
    A short random opening keeps the games from being identical without
    changing which player is stronger.
    """
    cfg = mcts_config or MCTSConfig(simulations=100)
    rng = rng or np.random.default_rng()
    mcts_a = BatchedMCTS(cfg, rng)
    mcts_b = BatchedMCTS(cfg, rng)

    a_wins = b_wins = draws = 0
    played = 0

    while played < n_games:
        n = min(parallel_games, n_games - played)
        # a_is_green[i] decides the seating for game i
        a_is_green = np.array([(played + i) % 2 == 0 for i in range(n)])
        played += n

        states = [new_game() for _ in range(n)]
        for _ in range(random_opening_plies):
            for i, s in enumerate(states):
                if not s.is_terminal:
                    states[i] = s.step(int(rng.choice(s.legal_action_list())))

        while any(not s.is_terminal for s in states):
            # Split by which engine is to move, so each network still sees a
            # batch rather than one position at a time.
            for is_a_turn in (True, False):
                idx = [i for i, s in enumerate(states)
                       if not s.is_terminal
                       and ((s.player == 1) == a_is_green[i]) == is_a_turn]
                if not idx:
                    continue
                engine = mcts_a if is_a_turn else mcts_b
                evaluator = evaluator_a if is_a_turn else evaluator_b
                subset = [states[i] for i in idx]
                if cfg.search_mode == "gumbel":
                    visits, improved = engine.search(
                        subset, evaluator, add_noise=False, with_policy=True)
                else:
                    visits = engine.search(subset, evaluator, add_noise=False)
                    improved = None
                for k, i in enumerate(idx):
                    source = improved[k] if improved is not None else visits[k]
                    probs = visits_to_policy(source, temperature)
                    if probs.sum() <= 0:
                        continue
                    action = int(rng.choice(ACTION_SIZE, p=probs))
                    states[i] = states[i].step(action)

        for i, s in enumerate(states):
            if s.winner == 0:
                draws += 1
            else:
                a_won = (s.winner == 1) == bool(a_is_green[i])
                if a_won:
                    a_wins += 1
                else:
                    b_wins += 1

    total = a_wins + b_wins + draws
    score = (a_wins + 0.5 * draws) / total if total else 0.0
    return {
        "wins": a_wins, "losses": b_wins, "draws": draws,
        "games": total, "score": score,
        "elo_diff": score_to_elo(score),
    }


def score_to_elo(score: float, cap: float = 800.0) -> float:
    """Convert a score rate in [0,1] to an Elo difference."""
    eps = 1e-4
    s = min(max(score, eps), 1 - eps)
    return max(-cap, min(cap, -400.0 * math.log10(1.0 / s - 1.0)))


def wilson_interval(wins: float, games: int, z: float = 1.96
                    ) -> tuple[float, float]:
    """Wilson score interval for a win rate — the interval to quote in a
    write-up, since normal-approximation intervals misbehave near 0 and 1."""
    if games == 0:
        return (0.0, 1.0)
    p = wins / games
    denom = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denom
    margin = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))
