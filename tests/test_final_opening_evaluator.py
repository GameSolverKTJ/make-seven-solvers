from types import SimpleNamespace

import numpy as np

from final_opening_evaluator import (
    _seed,
    cross_checkpoint_summaries,
    evaluate_post_opening_once,
    paired_repeat_differences,
)
from make7.game import ACTION_SIZE, THREE_ONLY, new_game, str_to_action


class ZeroEvaluator:
    def __call__(self, batch):
        n = batch.shape[0]
        return (np.zeros((n, ACTION_SIZE), dtype=np.float32),
                np.zeros(n, dtype=np.float32))

    def evaluate_with_q(self, batch):
        n = batch.shape[0]
        return (np.zeros((n, ACTION_SIZE), dtype=np.float32),
                np.zeros(n, dtype=np.float32),
                np.zeros((n, ACTION_SIZE), dtype=np.float32))


def args():
    return SimpleNamespace(
        search_mode="gumbel",
        c_puct=1.5,
        gumbel_root_candidates=0,
        gumbel_c_visit=50.0,
        gumbel_c_scale=0.10,
        gumbel_rescale_q=True,
    )


def test_seed_is_stable_and_changes_with_parts():
    assert _seed(1701, 1, 2, 3) == _seed(1701, 1, 2, 3)
    assert _seed(1701, 1, 2, 3) != _seed(1701, 1, 2, 4)


def test_post_opening_search_uses_requested_budget_and_green_view():
    root = new_game(THREE_ONLY)
    states = [root.step(str_to_action(x)) for x in ("2A", "2B", "2C", "2D")]
    rows = evaluate_post_opening_once(
        states, ZeroEvaluator(), args(), simulations=16, seed=7,
        stochastic_gumbel=False)
    assert len(rows) == 4
    for state, row in zip(states, rows):
        assert row["total_root_visits"] == 16
        assert row["selected_reply"]
        assert np.isfinite(row["opening_value_green"])
        assert np.isclose(
            row["opening_value_green"], -row["selected_reply_q_mover"])
        assert state.player == -1


def test_paired_repeat_differences_are_matched_by_repeat():
    rows = []
    vals = {
        0: {"2A": -0.5, "2B": -0.2},
        1: {"2A": -0.4, "2B": -0.1},
        2: {"2A": -0.6, "2B": -0.3},
    }
    for rep, by_opening in vals.items():
        for opening, value in by_opening.items():
            rows.append({
                "checkpoint": "c1", "generation": 7, "opening": opening,
                "repeat": rep, "simulations": 10000,
                "opening_value_green": value,
            })
    out = paired_repeat_differences(rows, ["2A", "2B"])
    assert len(out) == 1
    assert np.isclose(out[0]["mean_difference_a_minus_b"], -0.3)
    assert np.isclose(out[0]["fraction_a_greater_b"], 0.0)
    assert out[0]["paired_repeats"] == 3


def test_cross_checkpoint_summary_reports_rank_and_pairwise_sign():
    rows = [
        {"checkpoint": "c1", "opening": "2A", "deterministic_opening_value_green": -0.6},
        {"checkpoint": "c1", "opening": "2B", "deterministic_opening_value_green": -0.2},
        {"checkpoint": "c1", "opening": "2C", "deterministic_opening_value_green": 0.0},
        {"checkpoint": "c1", "opening": "2D", "deterministic_opening_value_green": -0.7},
        {"checkpoint": "c2", "opening": "2A", "deterministic_opening_value_green": -0.5},
        {"checkpoint": "c2", "opening": "2B", "deterministic_opening_value_green": -0.1},
        {"checkpoint": "c2", "opening": "2C", "deterministic_opening_value_green": 0.1},
        {"checkpoint": "c2", "opening": "2D", "deterministic_opening_value_green": -0.8},
    ]
    opening_rows, pair_rows = cross_checkpoint_summaries(
        rows, ["2A", "2B", "2C", "2D"])
    c = next(r for r in opening_rows if r["opening"] == "2C")
    assert np.isclose(c["mean_rank"], 1.0)
    assert c["best_rank_count"] == 2
    cd = next(r for r in pair_rows if r["opening_a"] == "2C" and r["opening_b"] == "2D")
    assert cd["same_nonzero_sign_across_checkpoints"] == 1
    assert np.isclose(cd["fraction_a_greater_b"], 1.0)
