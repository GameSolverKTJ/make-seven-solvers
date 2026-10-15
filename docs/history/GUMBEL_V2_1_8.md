# Gumbel Make7-AZ v2.1.8

v2.1.8 adds diagnostics and a deliberately small policy-target intervention for the opening-comparison experiment.

## Changes

1. **Organic vs forced outcome logging**
   - `metrics.csv`: `organic_green_win_rate`, `forced_green_win_rate`
   - `opening_outcomes.csv`: game/win/draw/loss counts, win rates and Green scores for organic games, all forced games, and each configured forced opening separately.

2. **Organic root policy logging**
   - `organic_root_policy.csv` stores the mean raw search policy at the empty root and the mean policy target actually used for training.
   - This prevents smoothing from being mistaken for a genuine recovery of the search distribution.

3. **Organic-root policy-target smoothing**
   - New option: `--policy-target-smoothing`, default `0.05`.
   - Only organic empty-root **training targets** are changed:
     `pi_train = 0.95*pi_search + 0.05*Uniform(legal)`.
   - MCTS search, move sampling, forced-opening actions, downstream policy targets, value targets and Q targets are unchanged.
   - Forced empty roots remain policy-masked exactly as in v2.1.7.

The intervention therefore does not force 2A/2B/2C/2D to be equal and does not alter self-play action selection directly. It only prevents the learned root prior from driving legal alternatives all the way to zero.
