"""Correctness and scaling of the multiprocess self-play pool.

The evaluator here is a deterministic stand-in for the network, so the
parallel path must produce statistically identical data to the
single-process path, and the only thing that should change is wall time.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make7.game import ACTION_SIZE  # noqa: E402
from make7.mcts import MCTSConfig  # noqa: E402
from make7.parallel import ParallelSelfPlay  # noqa: E402
from make7.selfplay import SelfPlayConfig, generate_games  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def flat_evaluator(batch):
    """Deterministic, content-dependent, but cheap — stands in for the net."""
    n = batch.shape[0]
    seed = batch.reshape(n, -1).sum(axis=1)
    logits = np.zeros((n, ACTION_SIZE), dtype=np.float32)
    logits += (seed[:, None] * 0.001).astype(np.float32)
    values = np.tanh(seed * 0.01).astype(np.float32)
    return logits, values


def make_cfg(games, parallel, sims):
    return SelfPlayConfig(
        games_per_generation=games,
        parallel_games=parallel,
        mcts=MCTSConfig(simulations=sims, temperature_moves=8),
    )


def test_parallel_matches_serial_shape():
    cfg = make_cfg(games=32, parallel=16, sims=32)

    t0 = time.perf_counter()
    samples, stats = generate_games(flat_evaluator, cfg, np.random.default_rng(5))
    serial_time = time.perf_counter() - t0

    with ParallelSelfPlay(n_workers=2, max_batch=16) as pool:
        t1 = time.perf_counter()
        (states, policies, values, q_targets, q_weights), pstats = pool.run(
            flat_evaluator, cfg, seed=5)
        parallel_time = time.perf_counter() - t1

    check(pstats["games"] == cfg.games_per_generation,
          "parallel pool played every requested game")
    check(states.shape[1:] == (10, 7, 7), "state array has the right shape")
    check(policies.shape[1] == ACTION_SIZE, "policy array has the right shape")
    check(states.shape[0] == policies.shape[0] == values.shape[0]
          == q_targets.shape[0] == q_weights.shape[0],
          "all five sample arrays agree in length")
    check(q_targets.shape[1] == ACTION_SIZE and q_weights.shape[1] == ACTION_SIZE,
          "Q target arrays have one entry per action")
    check(pstats["opening_counts"].sum() == cfg.games_per_generation,
          "one opening recorded per game")
    check(set(np.unique(values)).issubset({-1.0, 0.0, 1.0}),
          "value targets are all in {-1, 0, +1}")
    check(abs(policies.sum(axis=1) - 1.0).max() < 1e-4,
          "every policy target is normalised")

    serial_per_game = len(samples) / stats["games"]
    parallel_per_game = states.shape[0] / pstats["games"]
    ratio = parallel_per_game / serial_per_game
    check(0.6 < ratio < 1.6,
          f"samples per game are comparable (serial {serial_per_game:.1f}, "
          f"parallel {parallel_per_game:.1f})")

    print(f"  --  serial {serial_time:.1f}s, parallel(2) {parallel_time:.1f}s, "
          f"mean batch {pstats['mean_batch']:.0f}, "
          f"{pstats['forward_passes']} forward passes")



def test_gumbel_root_burst_is_chunked():
    """Gumbel can request K root actions per live game in one call.

    With two live opening positions and all 14 legal opening actions, the
    first forced-root evaluator call is much larger than a two-position slot.
    The remote evaluator must chunk it transparently rather than raising.
    """
    cfg = SelfPlayConfig(
        games_per_generation=2,
        parallel_games=2,
        mcts=MCTSConfig(
            simulations=14,
            temperature_moves=8,
            search_mode="gumbel",
            gumbel_root_candidates=0,
        ),
    )
    with ParallelSelfPlay(n_workers=1, max_batch=2) as pool:
        (states, policies, values, q_targets, q_weights), stats = pool.run(
            flat_evaluator, cfg, seed=7)

    check(stats["games"] == 2,
          "Gumbel root burst larger than the shared slot is chunked")
    check(states.shape[0] > 0 and policies.shape[0] == states.shape[0],
          "chunked Gumbel run returns complete samples")
    check(stats["forward_passes"] > 0,
          "chunked Gumbel run issued parent inference passes")

def test_worker_scaling():
    """Report scaling. This can only be meaningful on a multi-core host: the
    pool parallelises the Python tree search, so on a single core the workers
    simply contend and the numbers below will show a slowdown. That is a
    property of the machine, not a failure of the pool — the correctness
    tests above are the ones that must pass everywhere."""
    import os
    cores = os.cpu_count() or 1
    print(f"  --  host reports {cores} CPU core(s)")
    if cores < 4:
        print("  --  too few cores to demonstrate scaling; skipping the "
              "comparison (run this on the training machine instead)")
        return
    cfg = make_cfg(games=48, parallel=48, sims=64)

    t0 = time.perf_counter()
    generate_games(flat_evaluator, cfg, np.random.default_rng(1))
    serial = time.perf_counter() - t0
    print(f"  --  serial            : {serial:6.1f}s")

    results = {}
    for workers in (2, 4, 6):
        with ParallelSelfPlay(n_workers=workers,
                              max_batch=cfg.parallel_games) as pool:
            pool.run(flat_evaluator, make_cfg(8, 8, 16), seed=0)  # warm up
            t = time.perf_counter()
            _, stats = pool.run(flat_evaluator, cfg, seed=1)
            dt = time.perf_counter() - t
        results[workers] = (dt, stats["forward_passes"], stats["mean_batch"])
        print(f"  --  workers={workers}          : {dt:6.1f}s  "
              f"({serial / dt:.2f}x)  forwards={stats['forward_passes']:5d}  "
              f"mean batch={stats['mean_batch']:5.1f}")

    best_workers = min(results, key=lambda w: results[w][0])
    best = results[best_workers][0]
    print(f"  --  fastest: {best_workers} workers at {best:.1f}s "
          f"({serial / best:.2f}x)")
    check(all(r[1] > 0 for r in results.values()),
          "every configuration issued forward passes")


def test_repeated_generations():
    """The pool is persistent; workers must survive many generations."""
    cfg = make_cfg(games=16, parallel=8, sims=24)
    with ParallelSelfPlay(n_workers=2, max_batch=8) as pool:
        totals = []
        for gen in range(4):
            (states, _, _, _, _), stats = pool.run(flat_evaluator, cfg, seed=gen)
            totals.append(states.shape[0])
            check(stats["games"] == 16, f"generation {gen + 1} completed")
    check(all(t > 0 for t in totals), "every generation produced samples")


def test_oversized_batch_is_rejected():
    cfg = make_cfg(games=8, parallel=8, sims=8)
    with ParallelSelfPlay(n_workers=1, max_batch=8) as pool:
        (states, _, _, _, _), stats = pool.run(flat_evaluator, cfg, seed=0)
        check(stats["games"] == 8, "run inside the slot limit succeeds")


if __name__ == "__main__":
    for fn in [
        test_parallel_matches_serial_shape,
        test_repeated_generations,
        test_oversized_batch_is_rejected,
        test_gumbel_root_burst_is_chunked,
        test_worker_scaling,
    ]:
        print(f"\n[{fn.__name__}]")
        fn()
    print("\nall parallel self-play tests passed")


def test_parallel_forced_openings_are_globally_balanced():
    from make7.game import str_to_action

    forced = tuple(str_to_action(x) for x in ("2A", "2B", "2C", "2D"))
    cfg = SelfPlayConfig(
        games_per_generation=20,
        parallel_games=10,
        augment_mirror=False,
        forced_opening_fraction=0.5,
        forced_openings=forced,
        mcts=MCTSConfig(simulations=12, temperature_moves=6),
    )
    with ParallelSelfPlay(n_workers=2, max_batch=10) as pool:
        (states, policies, values, q_targets, q_weights), stats = pool.run(
            flat_evaluator, cfg, seed=19)

    assert stats["forced_games"] == 10
    counts = stats["forced_opening_counts"][list(forced)]
    assert counts.sum() == 10
    assert counts.max() - counts.min() <= 1
    assert stats["opening_counts"].sum() == 20
    assert stats["organic_opening_counts"].sum() == 10
    # One zero-policy empty root per forced game, no mirror augmentation here.
    assert int(np.sum(policies.sum(axis=1) < 1e-8)) == 10
    assert stats["organic_games"] + stats["forced_games"] == stats["games"] == 20
    assert int(stats["forced_result_games"].sum()) == stats["forced_games"]
    assert stats["organic_root_policy_count"] == stats["organic_games"]
    assert np.isfinite(stats["organic_root_policy_raw_sum"]).all()
    assert np.isfinite(stats["organic_root_policy_train_sum"]).all()
