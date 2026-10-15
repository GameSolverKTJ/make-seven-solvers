# Final opening evaluation (v2.1.11)

`final_opening_evaluator.py` is the frozen-checkpoint evaluator for the EE.
Training is unchanged from v2.1.9.

## Final analysis design

The evaluator deliberately excludes stochastic full-game robustness matches
from the final analysis. Those matches can amplify small search perturbations
over many plies and therefore measure error accumulation / policy dynamics
rather than the opening's value itself.

The final evaluator uses three measurements.

### 1. Primary: post-opening high-budget search

For each forced opening (`2A 2B 2C 2D` by default), the opening is applied to
the board first. Search then begins from the resulting YELLOW-to-move state.
The primary value is

```
GREEN opening value = - Q_YELLOW(selected best reply)
```

The evaluator performs a budget sweep (default 500/2000/5000/10000) to test
whether relative opening values converge as search effort increases. Empty-board
policy probabilities are not used for the ranking.

### 2. Root-value repeat / uncertainty analysis

At the primary budget, the same post-opening states are searched repeatedly
with independent Gumbel root perturbations. The script reports, per opening:

- mean opening value
- SD
- 95% CI
- modal best reply and its frequency

It also computes paired opening-value differences using matched repeat indices,
for example `V(2C) - V(2D)`, with a 95% CI and the fraction of repeats for which
the difference has the expected sign. These are written to
`pairwise_root_value.csv`.

### 3. Deterministic PV

After forcing each opening, both sides use temperature 0, no Gumbel/Dirichlet
perturbation, and fixed high-budget search until terminal. This provides a
representative principal variation. PV length is descriptive only; a slower
loss is not treated as a higher game-theoretic value.

## Multiple-checkpoint reproducibility

When two or more checkpoints are supplied, the evaluator additionally writes:

- `checkpoint_opening_summary.csv`: mean primary value, SD, mean rank, and how
  often each opening ranks first across checkpoints.
- `checkpoint_pairwise_consistency.csv`: pairwise primary-value differences
  across checkpoints, including whether the sign is identical for every
  checkpoint.

This is the preferred way to show that a result such as `2C > 2D` is not a
single-checkpoint accident.

## Recommended final command

```bash
python final_opening_evaluator.py \
  --checkpoint gen52=/path/to/gen52_best.pt \
  --checkpoint gen73=/path/to/gen73_best.pt \
  --openings 2A 2B 2C 2D \
  --tile-limits three_only \
  --budgets 500 2000 5000 10000 \
  --primary-budget 10000 \
  --value-repeats 24 \
  --pv-simulations 2000 \
  --search-mode gumbel \
  --gumbel-root-candidates 0 \
  --gumbel-c-visit 50 \
  --gumbel-c-scale 0.10 \
  --symmetrise \
  --device cuda \
  --seed 1701 \
  --out-dir final_opening_eval
```

## Output files

- `budget_sweep.csv` — deterministic value at each search budget.
- `search_repeat_details.csv` — every root-value repeat.
- `search_repeat_summary.csv` — mean/SD/95% CI and reply stability.
- `pairwise_root_value.csv` — paired root-value differences within checkpoint.
- `pv_summary.csv` / `pv_moves.csv` — deterministic principal variations.
- `final_summary.csv` — compact per-checkpoint final table.
- `checkpoint_opening_summary.csv` — opening stability across checkpoints.
- `checkpoint_pairwise_consistency.csv` — pairwise difference consistency across checkpoints.
- `run_metadata.json` — exact evaluation settings.

For the EE, use the highest-budget deterministic post-opening searched Q as the
primary estimate, the budget sweep as a convergence check, root-value repeats
as uncertainty analysis, deterministic PV as a representative line of play,
and checkpoint consistency as a reproducibility check.
