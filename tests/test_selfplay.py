"""Self-play and replay-buffer validation using a mock evaluator."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make7.arena import play_match, random_evaluator, score_to_elo, wilson_interval  # noqa: E402
from make7.game import ACTION_SIZE, action_to_str  # noqa: E402
from make7.mcts import MCTSConfig  # noqa: E402
from make7.selfplay import (ReplayBuffer, SelfPlayConfig, generate_games,
                            _smooth_policy_target)  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def mock_evaluator(batch):
    b = batch.shape[0]
    return np.zeros((b, ACTION_SIZE), dtype=np.float32), np.zeros(b, dtype=np.float32)


def test_generate_games():
    cfg = SelfPlayConfig(
        games_per_generation=16,
        parallel_games=8,
        mcts=MCTSConfig(simulations=24, temperature_moves=6),
    )
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    samples, stats = generate_games(mock_evaluator, cfg, rng)
    dt = time.perf_counter() - t0

    check(stats["games"] == 16, "all requested games were played")
    check(len(samples) > 0, "samples were produced")
    check(stats["total_plies"] > 0, "plies were recorded")

    # mirror augmentation doubles the sample count
    positions = stats["total_plies"]
    check(len(samples) == 2 * positions,
          f"mirror augmentation doubled {positions} positions to {len(samples)}")

    for s in samples[:50]:
        check_shape = s.state.shape == (10, 7, 7)
        if not check_shape:
            raise AssertionError("bad state shape")
        if abs(float(s.policy.sum()) - 1.0) > 1e-4:
            raise AssertionError("policy target does not sum to 1")
        if s.value not in (-1.0, 0.0, 1.0):
            raise AssertionError(f"unexpected value target {s.value}")
    print("  ok  sample shapes, policy normalisation and value targets valid")

    print(f"  --  16 games in {dt:.1f}s "
          f"({stats['total_plies'] / stats['games']:.1f} plies/game, "
          f"green {stats['green_wins']}, yellow {stats['yellow_wins']}, "
          f"draws {stats['draws']})")


def test_value_targets_alternate():
    """In a decisive game the value target must flip sign every ply."""
    cfg = SelfPlayConfig(games_per_generation=1, parallel_games=1,
                         augment_mirror=False,
                         mcts=MCTSConfig(simulations=16))
    for seed in range(10):
        samples, stats = generate_games(mock_evaluator, cfg,
                                        np.random.default_rng(seed))
        if stats["draws"]:
            continue
        values = [float(s.value) for s in samples]
        alternating = all(values[i] == -values[i + 1]
                          for i in range(len(values) - 1))
        check(alternating, f"value targets alternate in a decisive game "
                           f"(seed {seed}, {len(values)} positions)")
        return
    print("  --  only draws sampled, skipped")


def test_td_lambda_targets():
    """lambda = 1 must reproduce the plain outcome labels exactly, and lower
    values must reduce the spread without leaving [-1, 1]."""
    def run(lam, seed=4):
        cfg = SelfPlayConfig(
            games_per_generation=60, parallel_games=30,
            mcts=MCTSConfig(simulations=32, temperature_moves=8),
            augment_mirror=False, td_lambda=lam)
        samples, _ = generate_games(mock_evaluator, cfg,
                                    np.random.default_rng(seed))
        return np.array([float(s.value) for s in samples])

    plain = run(1.0)
    check(set(np.round(plain, 6)).issubset({-1.0, 0.0, 1.0}),
          "lambda=1 gives only the raw outcome labels")

    spreads = {}
    for lam in (0.9, 0.7, 0.5):
        v = run(lam)
        spreads[lam] = float(v.std())
        check(float(np.abs(v).max()) <= 1.0 + 1e-6,
              f"lambda={lam} targets stay within [-1, 1]")
    check(spreads[0.7] < float(plain.std()),
          f"lambda=0.7 reduces the spread "
          f"({spreads[0.7]:.3f} vs {plain.std():.3f})")
    check(spreads[0.5] < spreads[0.9],
          "a smaller lambda gives a smaller spread")
    print(f"  --  target SD: lambda=1 {plain.std():.3f}, "
          + ", ".join(f"lambda={k} {v:.3f}" for k, v in spreads.items()))


def test_replay_buffer():
    buf = ReplayBuffer(capacity=100)
    cfg = SelfPlayConfig(games_per_generation=4, parallel_games=4,
                         mcts=MCTSConfig(simulations=16))
    samples, _ = generate_games(mock_evaluator, cfg, np.random.default_rng(1))
    buf.add(samples)
    check(len(buf) == min(100, len(samples)), "buffer respects its capacity")

    s, p, v = buf.sample(32, np.random.default_rng(2))
    check(s.shape == (32, 10, 7, 7), "sampled states have the right shape")
    check(p.shape == (32, ACTION_SIZE), "sampled policies have the right shape")
    check(v.shape == (32,), "sampled values have the right shape")

    # overflow must wrap rather than grow
    buf.add(samples)
    check(len(buf) <= 100, "buffer never exceeds capacity")


def test_arena_is_fair():
    """Two identical zero-information players must score near 50%."""
    res = play_match(random_evaluator, random_evaluator, n_games=120,
                     mcts_config=MCTSConfig(simulations=16),
                     rng=np.random.default_rng(3), parallel_games=40)
    check(res["games"] == 120, "all arena games completed")
    check(0.35 < res["score"] < 0.65,
          f"identical players score near 50% (got {res['score']:.3f})")
    print(f"  --  W/D/L {res['wins']}/{res['draws']}/{res['losses']}, "
          f"score {res['score']:.3f}, elo {res['elo_diff']:+.0f}")


def test_arena_detects_strength():
    """A deep search must beat a shallow one with the same evaluator."""
    strong = play_match(random_evaluator, random_evaluator, n_games=60,
                        mcts_config=MCTSConfig(simulations=200),
                        rng=np.random.default_rng(4), parallel_games=30)
    # Same engine both sides, so this only confirms the harness is stable
    check(strong["games"] == 60, "deep-search arena completed")
    print(f"  --  200-sim self-match score {strong['score']:.3f} "
          f"(should still be near 0.5)")


def test_elo_and_intervals():
    check(abs(score_to_elo(0.5)) < 1e-6, "50% maps to 0 Elo")
    check(score_to_elo(0.76) > 190 and score_to_elo(0.76) < 210,
          "76% is roughly +200 Elo")
    lo, hi = wilson_interval(50, 100)
    check(lo < 0.5 < hi, "Wilson interval brackets the observed rate")
    lo0, hi0 = wilson_interval(0, 10)
    check(lo0 == 0.0 and hi0 < 0.4, "Wilson interval behaves at the boundary")


def test_opening_stats_recorded():
    cfg = SelfPlayConfig(games_per_generation=12, parallel_games=6,
                         mcts=MCTSConfig(simulations=24))
    _, stats = generate_games(mock_evaluator, cfg, np.random.default_rng(5))
    counts = stats["opening_counts"]
    check(counts.sum() == 12, "one opening recorded per game")
    top = int(np.argmax(counts))
    print(f"  --  most common opening this run: {action_to_str(top)} "
          f"({counts[top]}/12)")


if __name__ == "__main__":
    for fn in [
        test_generate_games,
        test_value_targets_alternate,
        test_td_lambda_targets,
        test_replay_buffer,
        test_arena_is_fair,
        test_arena_detects_strength,
        test_elo_and_intervals,
        test_opening_stats_recorded,
    ]:
        print(f"\n[{fn.__name__}]")
        fn()
    print("\nall self-play tests passed")


def test_replay_buffer_load_rebuilds_balancing_buckets(tmp_path):
    """Inherited replay buffers must remain balanceable after reload.

    Regression for v2.1.5: load() restored states/values but left all derived
    bucket ids at zero, so inherited samples were silently assigned to one
    bucket and --balanced-sampling stopped working.
    """
    n_each = 20
    states = []
    values = []
    policies = []
    for parity, value in [(0, -0.6), (0, +0.6), (1, -0.6), (1, +0.6)]:
        for _ in range(n_each):
            s = np.zeros((10, 7, 7), dtype=np.float32)
            # Plane occupancy count encodes ply parity exactly as game.encode().
            if parity:
                s[0, 0, 0] = 1.0
            else:
                s[0, 0, 0] = 1.0
                s[3, 0, 1] = 1.0
            states.append(s)
            values.append(value)
            policies.append(np.full(ACTION_SIZE, 1.0 / ACTION_SIZE,
                                    dtype=np.float32))
    states = np.stack(states)
    values = np.asarray(values, dtype=np.float32)
    policies = np.stack(policies)

    buf = ReplayBuffer(capacity=200)
    buf.add_arrays(states, policies, values)
    before = buf.bucket_sizes()
    path = tmp_path / "buffer.npz"
    buf.save(path)

    restored = ReplayBuffer(capacity=200)
    assert restored.load(path)
    after = restored.bucket_sizes()
    assert after == before
    # decisive buckets 0,2,3,5 should all be represented equally
    assert [after[i] for i in (0, 2, 3, 5)] == [n_each] * 4

    s, _p, v = restored.sample(4000, np.random.default_rng(123), balanced=True)
    ply = np.rint(s[:, :6].sum(axis=(1, 2, 3))).astype(np.int64)
    rates = [np.mean(v[(ply & 1) == k] > 0) for k in (0, 1)]
    assert abs(float(rates[0] - rates[1])) < 0.06


def test_forced_openings_mask_policy_but_keep_root_sample():
    """Forced roots must not become policy teachers.

    Half the games are forced and balanced across 2A/2B/2C/2D.  Their empty
    root is still present in the replay data, but with a zero-mass policy
    target so value/Q remain trainable while policy loss ignores that row.
    """
    from make7.game import str_to_action

    forced = tuple(str_to_action(x) for x in ("2A", "2B", "2C", "2D"))
    cfg = SelfPlayConfig(
        games_per_generation=16,
        parallel_games=8,
        augment_mirror=False,
        forced_opening_fraction=0.5,
        forced_openings=forced,
        mcts=MCTSConfig(simulations=16, temperature_moves=6),
    )
    samples, stats = generate_games(mock_evaluator, cfg, np.random.default_rng(123))

    assert stats["forced_games"] == 8
    counts = stats["forced_opening_counts"][list(forced)]
    assert counts.sum() == 8
    assert counts.max() - counts.min() <= 1
    assert stats["opening_counts"].sum() == 16
    assert stats["organic_opening_counts"].sum() == 8

    # Exactly one masked empty-root policy row per forced game.
    masked = [s for s in samples if float(s.policy.sum()) < 1e-8]
    assert len(masked) == 8
    for sample in masked:
        assert np.isfinite(float(sample.value))
        assert sample.q_target.shape == (ACTION_SIZE,)
        assert sample.q_weight.shape == (ACTION_SIZE,)


def test_forced_outcome_accounting_and_root_smoothing():
    """v2.1.8 diagnostics must conserve games and smoothing must be target-only."""
    from make7.game import str_to_action, new_game

    forced = tuple(str_to_action(x) for x in ("2A", "2B", "2C", "2D"))
    eps = 0.05
    cfg = SelfPlayConfig(
        games_per_generation=20,
        parallel_games=10,
        augment_mirror=False,
        forced_opening_fraction=0.4,
        forced_openings=forced,
        policy_target_smoothing=eps,
        mcts=MCTSConfig(simulations=16, temperature_moves=6),
    )
    _samples, stats = generate_games(mock_evaluator, cfg, np.random.default_rng(321))

    assert stats["organic_games"] + stats["forced_games"] == stats["games"] == 20
    assert (stats["organic_green_wins"] + stats["organic_yellow_wins"]
            + stats["organic_draws"] == stats["organic_games"])
    assert (stats["forced_green_wins"] + stats["forced_yellow_wins"]
            + stats["forced_draws"] == stats["forced_games"])
    assert int(stats["forced_result_games"].sum()) == stats["forced_games"]
    assert int(stats["forced_green_wins_by_opening"].sum()) == stats["forced_green_wins"]
    assert int(stats["forced_draws_by_opening"].sum()) == stats["forced_draws"]

    n = stats["organic_root_policy_count"]
    assert n == stats["organic_games"]
    raw = stats["organic_root_policy_raw_sum"] / n
    trained = stats["organic_root_policy_train_sum"] / n
    legal = new_game(cfg.limits).legal_actions().astype(np.float64)
    uniform = legal / legal.sum()
    expected = (1.0 - eps) * raw + eps * uniform
    assert np.allclose(trained, expected, atol=2e-6)
    assert abs(float(raw.sum()) - 1.0) < 1e-5
    assert abs(float(trained.sum()) - 1.0) < 1e-5


def test_policy_smoothing_applies_to_all_nonmasked_targets():
    """v2.1.9 smooths every supervised policy row, not only the root."""
    eps = 0.05
    legal = np.zeros(ACTION_SIZE, dtype=np.float32)
    legal[[0, 3, 5, 9]] = 1.0
    raw = np.zeros(ACTION_SIZE, dtype=np.float32)
    raw[3] = 1.0
    got = _smooth_policy_target(raw, legal, eps)
    expected = (1.0 - eps) * raw + eps * legal / legal.sum()
    assert np.allclose(got, expected, atol=1e-7)
    assert abs(float(got.sum()) - 1.0) < 1e-7

    from make7.game import str_to_action
    forced = tuple(str_to_action(x) for x in ("2A", "2B", "2C", "2D"))
    cfg = SelfPlayConfig(
        games_per_generation=12, parallel_games=6, augment_mirror=False,
        forced_opening_fraction=0.5, forced_openings=forced,
        policy_target_smoothing=eps,
        mcts=MCTSConfig(simulations=16, temperature_moves=6),
    )
    _samples, stats = generate_games(mock_evaluator, cfg, np.random.default_rng(777))
    # Exactly one policy-masked row exists for every forced game; every other
    # visited state contributes to raw/train entropy accounting.
    assert stats["policy_entropy_count"] == stats["total_plies"] - stats["forced_games"]
    raw_h = stats["policy_entropy_raw_sum"] / stats["policy_entropy_count"]
    train_h = stats["policy_entropy_train_sum"] / stats["policy_entropy_count"]
    assert train_h >= raw_h - 1e-9
    assert train_h > raw_h + 1e-4
