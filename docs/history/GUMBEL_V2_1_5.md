# Gumbel Make7-AZ v2.1.5

v2.1.5 is a stability/diagnostics release. It does not change the game, model
architecture, Gumbel search, Q-head semantics, TD(lambda), or exact auxiliary
loss introduced in earlier v2.1.x releases.

## 1. Replay-buffer inheritance for fine-tuning

Starting a new run from a mature checkpoint with an empty replay buffer can
make the first generation overfit one unusually skewed self-play batch. A new
run can now inherit the checkpoint and replay buffer together:

```bash
python train.py \
  --run-dir runs/v215 \
  --init-run-dir runs/v213_old \
  ...
```

This resolves to `<old>/best.pt` and `<old>/buffer.npz`. Alternatively use
`--init-checkpoint` and `--init-buffer` explicitly.

## 2. TD(lambda)-safe parity diagnostics

The old `parity_gap` compared the mean continuous value target at even and odd
plies. Under TD(lambda), target magnitudes may legitimately differ by parity,
so that number could look large even when the win/loss sign was perfectly
balanced.

v2.1.5 defines:

- `parity_gap`: sign-gap in the *balanced training sample* (primary shortcut diagnostic)
- `parity_raw_sign_gap`: sign-gap in the raw replay distribution
- `parity_abs_gap`: raw difference in mean absolute TD target magnitude

A large `parity_abs_gap` by itself is not evidence of the old parity shortcut.

## 3. Policy-collapse diagnostics

`metrics.csv` now records:

- `policy_target_entropy`: mean policy-target entropy, normalised to [0,1]
- `opening_entropy`: entropy of the generation's empirical first-move distribution

Falling entropy together with a falling policy loss is direct evidence that the
training distribution is narrowing rather than simply becoming easier to fit.

## 4. Dual promotion gate

Arena-only promotion can cycle between policies. With an exact validation DB,
v2.1.5 therefore requires an arena-passing candidate also to preserve held-out
exact AUC:

```
arena_score >= promote_threshold
AND
candidate_holdout_auc >= incumbent_holdout_auc - tolerance
```

Defaults:

```
--promotion-holdout-gate
--promotion-holdout-tolerance 0.01
```

Only held-out proof groups (never used by exact auxiliary training) are used.
The extra exact evaluation is performed only after a candidate passes the arena
gate. Use `--no-promotion-holdout-gate` for the old arena-only behaviour.

New `metrics.csv` fields include `candidate_holdout_auc`,
`incumbent_holdout_auc`, and `holdout_gate_pass`.
