"""MCTS validation with a mock (network-free) evaluator.

If the search is wired up correctly it must find a mate-in-one and must
block a mate-in-one even when the evaluator carries no information at all,
because those signals come from terminal back-ups rather than the network.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make7.game import (  # noqa: E402
    ACTION_SIZE, action_to_str, new_game, str_to_action,
)
from make7.mcts import BatchedMCTS, MCTSConfig, Node, visits_to_policy  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def uniform_evaluator(batch):
    b = batch.shape[0]
    return np.zeros((b, ACTION_SIZE), dtype=np.float32), np.zeros(b, dtype=np.float32)


def noisy_evaluator(batch, rng=np.random.default_rng(0)):
    b = batch.shape[0]
    return (rng.normal(0, 1, (b, ACTION_SIZE)).astype(np.float32),
            rng.uniform(-0.5, 0.5, b).astype(np.float32))


def build(tokens):
    s = new_game()
    for tok in tokens:
        s = s.step(str_to_action(tok))
    return s


def test_visit_budget():
    cfg = MCTSConfig(simulations=128)
    mcts = BatchedMCTS(cfg, np.random.default_rng(1))
    states = [new_game() for _ in range(4)]
    visits = mcts.search(states, uniform_evaluator, add_noise=False)
    for v in visits:
        check(int(v.sum()) == 128, "root visit counts equal the simulation budget")
        break
    check(all(int(v.sum()) == 128 for v in visits), "budget holds for all games")


def test_only_legal_moves_visited():
    cfg = MCTSConfig(simulations=128)
    mcts = BatchedMCTS(cfg, np.random.default_rng(2))
    s = new_game()
    visits = mcts.search([s], uniform_evaluator, add_noise=False)[0]
    legal = s.legal_actions()
    check(not visits[~legal].any(), "no illegal action ever receives a visit")


def test_finds_mate_in_one():
    # GREEN has 2A,2B,2C on the bottom row; 1D completes 2+2+2+1 = 7.
    s = build(["2A", "1G", "2B", "1G", "2C", "1G"])
    winning = str_to_action("1D")
    cfg = MCTSConfig(simulations=200)
    mcts = BatchedMCTS(cfg, np.random.default_rng(3))
    visits = mcts.search([s], uniform_evaluator, add_noise=False)[0]
    best = int(np.argmax(visits))
    check(best == winning,
          f"search finds the mate in one ({action_to_str(best)} chosen)")
    share = visits[winning] / visits.sum()
    check(share > 0.5, f"winning move takes {share:.0%} of the visits")


def test_blocks_mate_in_one():
    # YELLOW has 2A,2B,2C on the bottom row and threatens 1D.
    # GREEN is to move and must occupy D0.
    s = build(["1G", "2A", "1G", "2B", "1G", "2C"])
    check(s.player == 1, "GREEN to move")
    cfg = MCTSConfig(simulations=400)
    mcts = BatchedMCTS(cfg, np.random.default_rng(4))
    visits = mcts.search([s], uniform_evaluator, add_noise=False)[0]
    best = int(np.argmax(visits))
    tile, col = best // 7 + 1, best % 7
    check(col == 3, f"search blocks in column D (played {action_to_str(best)})")


def test_terminal_root_is_safe():
    s = build(["2A", "1G", "2B", "1G", "2C", "1G", "1D"])
    check(s.is_terminal, "position is terminal")
    mcts = BatchedMCTS(MCTSConfig(simulations=32), np.random.default_rng(5))
    visits = mcts.search([s], uniform_evaluator, add_noise=False)[0]
    check(visits.sum() == 0, "a terminal root produces no visits and no crash")


def test_mixed_batch():
    """Terminal and live games in the same batch must not interfere."""
    live = new_game()
    dead = build(["2A", "1G", "2B", "1G", "2C", "1G", "1D"])
    mcts = BatchedMCTS(MCTSConfig(simulations=64), np.random.default_rng(6))
    visits = mcts.search([live, dead, live], noisy_evaluator, add_noise=True)
    check(visits[0].sum() == 64 and visits[2].sum() == 64, "live games searched")
    check(visits[1].sum() == 0, "terminal game skipped cleanly")


def test_temperature():
    v = np.array([0, 10, 30, 0] + [0] * 17, dtype=np.float32)
    greedy = visits_to_policy(v, 0.0)
    check(greedy[2] == 1.0, "temperature 0 gives the argmax")
    soft = visits_to_policy(v, 1.0)
    check(abs(soft.sum() - 1.0) < 1e-6 and soft[1] > 0, "temperature 1 normalises")


def test_throughput():
    """Rough cost of the tree code itself, excluding the network."""
    n_games, sims = 32, 100
    states = [new_game() for _ in range(n_games)]
    mcts = BatchedMCTS(MCTSConfig(simulations=sims), np.random.default_rng(8))
    t0 = time.perf_counter()
    mcts.search(states, uniform_evaluator, add_noise=True)
    dt = time.perf_counter() - t0
    total = n_games * sims
    print(f"  --  {total} simulations in {dt:.2f}s "
          f"({total / dt:,.0f} sims/s, batch size {n_games})")


def test_self_play_smoke():
    """A full game driven by search must terminate and stay legal."""
    mcts = BatchedMCTS(MCTSConfig(simulations=40), np.random.default_rng(9))
    s = new_game()
    plies = 0
    while not s.is_terminal:
        visits = mcts.search([s], noisy_evaluator, add_noise=True)[0]
        probs = visits_to_policy(visits, 1.0)
        a = int(np.random.default_rng(plies).choice(ACTION_SIZE, p=probs))
        check_legal = s.legal_actions()[a]
        if not check_legal:
            raise AssertionError("search proposed an illegal move")
        s = s.step(a)
        plies += 1
    print(f"  --  search-driven game ended after {plies} plies "
          f"(winner={s.winner})")


def test_gumbel_budget_and_coverage():
    """Gumbel root search must spend the full budget and inspect every legal
    opening action when the budget is large enough."""
    s = new_game()
    cfg = MCTSConfig(simulations=112, search_mode="gumbel",
                     gumbel_root_candidates=0)
    mcts = BatchedMCTS(cfg, np.random.default_rng(21))
    visits, policy = mcts.search([s], uniform_evaluator, add_noise=True,
                                 with_policy=True)
    v, p = visits[0], policy[0]
    legal = s.legal_actions()
    check(int(v.sum()) == 112, "Gumbel search spends the full root budget")
    check(np.all(v[legal] > 0), "Gumbel search visits every legal opening action")
    check(not v[~legal].any(), "Gumbel search never visits illegal actions")
    check(abs(float(p.sum()) - 1.0) < 1e-6,
          "Gumbel improved policy is normalised")
    check(not p[~legal].any(), "Gumbel improved policy masks illegal actions")


def test_gumbel_finds_mate_in_one():
    s = build(["2A", "1G", "2B", "1G", "2C", "1G"])
    winning = str_to_action("1D")
    cfg = MCTSConfig(simulations=240, search_mode="gumbel")
    mcts = BatchedMCTS(cfg, np.random.default_rng(22))
    visits, policy = mcts.search([s], uniform_evaluator, add_noise=False,
                                 with_policy=True)
    best = int(np.argmax(policy[0]))
    check(best == winning,
          f"Gumbel improved policy finds mate in one ({action_to_str(best)})")



def test_gumbel_action_q_target_sign():
    """A searched mate-in-one must supervise Q(s,a)=+1 for the mover."""
    s = build(["2A", "1G", "2B", "1G", "2C", "1G"])
    winning = str_to_action("1D")
    cfg = MCTSConfig(simulations=112, search_mode="gumbel")
    mcts = BatchedMCTS(cfg, np.random.default_rng(23))
    visits, q_values = mcts.search(
        [s], uniform_evaluator, add_noise=False, with_action_values=True)
    check(visits[0][winning] > 0, "winning action is actually searched")
    check(abs(float(q_values[0][winning]) - 1.0) < 1e-6,
          "mate-in-one root Q is +1 for the player to move")


def test_gumbel_rescaled_q_can_overturn_collapsed_prior():
    """A materially better searched action must be able to beat a huge prior.

    This is the failure seen in the v2.1.1 opening diagnostics: search Q
    preferred a low-prior move, but raw-Q scaling left the improved policy
    almost entirely on the old high-prior move.
    """
    node = Node(new_game())
    node.expanded = True
    node.legal[:2] = True
    node.illegal = ~node.legal
    node.P[0] = 0.9999
    node.P[1] = 0.0001
    node.N[0] = 52
    node.N[1] = 3
    node.W[0] = 52 * -0.24
    node.W[1] = 3 * -0.05
    node.total = 55
    cfg = MCTSConfig(search_mode="gumbel", gumbel_c_visit=50,
                     gumbel_c_scale=0.1, gumbel_rescale_q=True)
    mcts = BatchedMCTS(cfg, np.random.default_rng(24))
    scores = mcts._improved_policy_scores(node)
    check(scores[1] > scores[0],
          "rescaled completed Q can overturn a collapsed network prior")

if __name__ == "__main__":
    for fn in [
        test_visit_budget,
        test_only_legal_moves_visited,
        test_finds_mate_in_one,
        test_blocks_mate_in_one,
        test_terminal_root_is_safe,
        test_mixed_batch,
        test_temperature,
        test_throughput,
        test_self_play_smoke,
        test_gumbel_budget_and_coverage,
        test_gumbel_finds_mate_in_one,
        test_gumbel_action_q_target_sign,
        test_gumbel_rescaled_q_can_overturn_collapsed_prior,
    ]:
        print(f"\n[{fn.__name__}]")
        fn()
    print("\nall MCTS tests passed")
