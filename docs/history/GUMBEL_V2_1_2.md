# Gumbel Make7-AZ v2.1.2

v2.1.2 fixes the policy-improvement bottleneck seen in the v2.1.1 opening
logs and adds per-proposition DF-PN validation.

## 1. Gumbel completed-Q transform

The v2.1.1 run could search every opening action yet still return an improved
policy dominated by a collapsed prior. v2.1.2 changes the root transform to:

1. complete root Q values;
2. min-max rescale legal completed Q to `[0, 1]`;
3. multiply by `(c_visit + max_root_visits) * c_scale`;
4. add the transformed Q to the network log-prior.

Defaults are now:

```
--gumbel-c-visit 50
--gumbel-c-scale 0.10
--gumbel-rescale-q
```

Use `--no-gumbel-rescale-q` only for an ablation that reproduces the v2.1.1
raw-Q behaviour.

## 2. Q-head supervision

`--q-min-visits` now defaults to 3. In the current sequential-halving schedule,
three visits are enough for an opening action to survive the first round, so
low-prior actions discovered by search can now supervise the auxiliary Q head.
The existing `sqrt(N)` weighting still downweights noisy low-visit targets.

## 3. Exact-proposition validation

Whenever `--validation-database` is supplied, validation now writes two files.

### `validation.csv`

The existing group-balanced metrics remain and these columns are added:

- `proposition_support_rate`
- `decisive_support_rate`
- `bound_support_rate`
- `proposition_supported`
- `proposition_total`

These are diagnostic record-level rates. Continue to use group-balanced AUC as
the main checkpoint-selection metric unless there is a reason to change it.

### `exact_proposition_predictions.csv`

One row is appended for every exact proposition at every validation generation.
Columns include:

- generation
- exact position string and ply
- shallow validation group
- side to move
- exact proposition (`GREEN_PROVED`, `YELLOW_PROVED`, `GREEN_DISPROVED`, ...)
- network value from GREEN's point of view
- network-estimated outcome
- signed proposition margin
- whether the estimate is consistent with the exact proposition

A positive `proposition_margin` means that the network estimate supports the
exact proposition under the configured criterion. The margin is not a
probability.

The draw / uncertainty band is controlled by:

```
--validation-draw-band 0.2
```

Decisive GREEN/YELLOW propositions still use the sign of the value, matching
the existing validation metrics. The draw band is used for draws and one-sided
`DISPROVED` bounds.

## Recommended run

```bash
export EXACT_DB=/path/to/exact_propositions_campaign.txt
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6

python train.py \
  --run-dir runs/gumbel_v2_1_2 \
  --generations 100 \
  --preset coord_base \
  --tile-limits three_only \
  --search-mode gumbel \
  --simulations 200 \
  --gumbel-root-candidates 0 \
  --gumbel-c-visit 50 \
  --gumbel-c-scale 0.10 \
  --gumbel-rescale-q \
  --symmetrise \
  --balanced-sampling \
  --td-lambda 0.7 \
  --q-loss-weight 0.25 \
  --q-min-visits 3 \
  --games-per-generation 1024 \
  --parallel-games 256 \
  --workers 6 \
  --gather-timeout 0.5 \
  --train-steps 640 \
  --batch-size 1024 \
  --lr 2e-3 \
  --arena-games 200 \
  --validation-database "$EXACT_DB" \
  --validation-every 1 \
  --validation-select auc
```
