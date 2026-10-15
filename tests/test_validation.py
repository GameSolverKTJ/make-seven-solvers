import csv

import numpy as np

from make7.game import new_game
from make7.validation import (
    ExactRecord,
    proposition_consistent,
    proposition_margin,
    write_proposition_predictions,
)


def rec(label=None, bound=None, proposition="GREEN_PROVED"):
    s = new_game()
    return ExactRecord(
        tokens=[], position="", proposition=proposition, group="root",
        label=label, bound=bound, state=s,
    )


def test_proposition_margin_decisive_and_bounds():
    assert proposition_margin(rec(label=1), 0.4) == 0.4
    assert proposition_margin(rec(label=-1, proposition="YELLOW_PROVED"), -0.4) == 0.4
    assert proposition_consistent(rec(label=1), 0.01)
    assert not proposition_consistent(rec(label=1), -0.01)
    assert proposition_consistent(rec(label=0, proposition="DRAW"), 0.1, 0.2)
    assert not proposition_consistent(rec(label=0, proposition="DRAW"), 0.3, 0.2)
    assert proposition_consistent(rec(bound="le0", proposition="GREEN_DISPROVED"), 0.19, 0.2)
    assert not proposition_consistent(rec(bound="le0", proposition="GREEN_DISPROVED"), 0.21, 0.2)
    assert proposition_consistent(rec(bound="ge0", proposition="YELLOW_DISPROVED"), -0.19, 0.2)
    assert not proposition_consistent(rec(bound="ge0", proposition="YELLOW_DISPROVED"), -0.21, 0.2)


def test_write_proposition_predictions(tmp_path):
    records = [
        rec(label=1, proposition="GREEN_PROVED"),
        rec(label=-1, proposition="YELLOW_PROVED"),
        rec(bound="le0", proposition="GREEN_DISPROVED"),
    ]
    values = np.array([0.5, -0.4, 0.1], dtype=np.float32)
    out = tmp_path / "exact_proposition_predictions.csv"
    metrics = write_proposition_predictions(out, 7, records, values, draw_band=0.2)

    assert metrics["proposition_supported"] == 3
    assert metrics["proposition_total"] == 3
    assert metrics["proposition_support_rate"] == 1.0

    with out.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert rows[0]["generation"] == "7"
    assert rows[0]["exact_proposition"] == "GREEN_PROVED"
    assert rows[0]["predicted_outcome"] == "GREEN"
    assert rows[0]["proposition_consistent"] == "1"


def test_stratified_split_keeps_decisive_classes_on_both_sides(tmp_path):
    from make7.validation import ExactRecord, stratified_split_by_group
    from make7.game import new_game
    s = new_game()
    records = []
    for label, prefix in [(1, 'G'), (-1, 'Y')]:
        for i in range(4):
            records.append(ExactRecord([], f'{prefix}{i}',
                                       'GREEN_WIN' if label == 1 else 'YELLOW_WIN',
                                       f'{prefix}{i}', label, None, s))
    cal, test = stratified_split_by_group(records, 0.5, seed=3)
    assert {r.label for r in cal} == {1, -1}
    assert {r.label for r in test} == {1, -1}
    assert {r.group for r in cal}.isdisjoint({r.group for r in test})


def test_optimal_threshold_repairs_global_shift():
    from make7.validation import ExactRecord, optimal_threshold, threshold_metrics
    from make7.game import new_game
    s = new_game()
    rec = []
    scores = []
    for i, x in enumerate([-0.4, -0.3, -0.2]):
        rec.append(ExactRecord([], f'g{i}', 'GREEN_WIN', f'g{i}', 1, None, s))
        scores.append(x)
    for i, x in enumerate([-0.9, -0.8, -0.7]):
        rec.append(ExactRecord([], f'y{i}', 'YELLOW_WIN', f'y{i}', -1, None, s))
        scores.append(x)
    scores = np.asarray(scores, dtype=np.float32)
    before = threshold_metrics(rec, scores, 0.0)['balanced_accuracy']
    t, m = optimal_threshold(rec, scores)
    assert before == 0.5
    assert -0.7 < t < -0.4
    assert m['balanced_accuracy'] == 1.0
