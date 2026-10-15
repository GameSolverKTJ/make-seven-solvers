import numpy as np

from train import _opening_entropy, _policy_entropy, _promotion_gate, _td_parity_diagnostics


def _states_for_parity(parities):
    s = np.zeros((len(parities), 10, 7, 7), dtype=np.float32)
    # The first six planes contain placed pieces. Put one tile in plane 0 for odd ply.
    for i, parity in enumerate(parities):
        if parity:
            s[i, 0, 0, 0] = 1.0
    return s


def test_td_parity_sign_gap_ignores_magnitude_only_difference():
    # Both parities have exactly 50/50 positive/negative signs, but odd targets
    # are much larger in magnitude. Old mean-value parity diagnostics could
    # misread this under TD(lambda); the sign gap must stay zero.
    states = _states_for_parity([0, 0, 1, 1])
    values = np.array([+0.1, -0.1, +0.9, -0.9], dtype=np.float32)
    sign_gap, abs_gap = _td_parity_diagnostics(states, values)
    assert sign_gap == 0.0
    assert abs_gap > 0.7


def test_td_parity_sign_gap_detects_shortcut():
    states = _states_for_parity([0, 0, 1, 1])
    values = np.array([+0.3, +0.8, -0.2, -0.7], dtype=np.float32)
    sign_gap, _ = _td_parity_diagnostics(states, values)
    assert sign_gap == 1.0


def test_policy_entropy_bounds():
    det = np.zeros((2, 21), dtype=np.float32)
    det[:, 3] = 1.0
    uni = np.full((2, 21), 1 / 21, dtype=np.float32)
    assert abs(_policy_entropy(det)) < 1e-12
    assert abs(_policy_entropy(uni) - 1.0) < 1e-6
    assert abs(_opening_entropy(np.ones(21)) - 1.0) < 1e-6


def test_dual_promotion_gate():
    promote, gate = _promotion_gate(0.56, 0.55, 0.82, 0.83, 0.01, True)
    assert promote and gate
    promote, gate = _promotion_gate(0.56, 0.55, 0.80, 0.83, 0.01, True)
    assert not promote and not gate
    promote, gate = _promotion_gate(0.54, 0.55, 0.90, 0.83, 0.01, True)
    assert not promote and not gate
    promote, gate = _promotion_gate(0.56, 0.55, np.nan, np.nan, 0.01, False)
    assert promote and gate
