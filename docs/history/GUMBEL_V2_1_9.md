# Gumbel Make7-AZ v2.1.9

v2.1.9 changes only policy-target regularisation and diagnostics.

## Global target smoothing

`--policy-target-smoothing` (default `0.05`) is now applied to every
policy-supervised self-play state:

\[
\pi_{train}(s)=(1-\epsilon)\pi_{search}(s)+\epsilon U_{legal}(s).
\]

Search, Gumbel action selection, and self-play behaviour still use the raw
search distribution. Forced empty roots remain policy-masked; later states in
forced-opening games are smoothed normally.

This is intentionally different from v2.1.8, where smoothing affected only
organic empty roots and therefore touched only a small fraction of policy
training rows.

## Raw vs training entropy

Each generation now logs two policy entropy diagnostics from the positions
created in that generation:

- `policy_target_entropy_raw`: entropy of the raw MCTS/Gumbel policy teacher.
- `policy_target_entropy_train`: entropy after 5% smoothing, i.e. the teacher
  actually seen by the network.

The console renders these as `Hpi raw/train`, for example `Hpi 0.11/0.24`.
The legacy `policy_target_entropy` CSV column is retained and equals the
training entropy for compatibility with existing plotting scripts.

`Hopen` remains based on organic action choices only and is therefore not
artificially inflated by forced openings or target smoothing.
