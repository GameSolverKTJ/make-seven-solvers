# Gumbel Make7-AZ v2

This branch adds a Gumbel AlphaZero-style root search to the existing Make 7
AlphaZero implementation while preserving the original PUCT path for direct
A/B comparison.

## What changed

### 1. Gumbel root search + sequential halving

`MCTSConfig(search_mode="gumbel")`:

- evaluates every legal root action (by default),
- perturbs root priors with independent Gumbel noise during self-play,
- allocates simulations by sequential halving,
- forces each surviving root action to receive search effort,
- uses ordinary PUCT below the root.

This specifically prevents an opening move from receiving zero search merely
because its current policy prior is small.

### 2. Search visits are separated from the policy target

Sequential-halving visit counts are allocation counts, not a calibrated policy.
In Gumbel mode `BatchedMCTS.search(..., with_policy=True)` therefore returns an
**improved policy** separately from the raw visits:

    pi'(a) proportional to exp(log pi(a) + sigma(Q_completed(a)))

Unvisited Q values are completed by the root network value. The Gumbel sample
is used to decide what to investigate, but is not written into the training
target.

### 3. Mirror-symmetrised inference is on by default in train.py

Make 7 is exactly left-right symmetric. v2 therefore uses `NetEvaluator(...,
symmetrise=True)` by default in training. Disable only for speed experiments:

    --no-symmetrise

### 4. Original AlphaZero remains available

Use:

    --search-mode puct

for the original root PUCT + Dirichlet implementation. This makes controlled
experiments straightforward.

## Recommended first run

```bash
python train.py \
  --run-dir runs/gumbel_v2 \
  --generations 100 \
  --preset coord_base \
  --tile-limits three_only \
  --search-mode gumbel \
  --simulations 200 \
  --gumbel-root-candidates 0 \
  --gumbel-c-visit 8 \
  --gumbel-c-scale 0.10 \
  --balanced-sampling \
  --td-lambda 0.7 \
  --games-per-generation 1024 \
  --parallel-games 256 \
  --workers 6 \
  --gather-timeout 0.5 \
  --train-steps 640 \
  --batch-size 1024 \
  --lr 2e-3 \
  --arena-games 200
```

For a clean comparison against the previous run, keep every option identical
and change only:

```bash
--search-mode puct
```

versus

```bash
--search-mode gumbel
```

## Opening analysis

`analyse_openings.py` now defaults to Gumbel search and reports the Gumbel
improved policy in the root share column.

```bash
python analyse_openings.py \
  --checkpoint runs/gumbel_v2/best_validation.pt \
  --search-mode gumbel \
  --games 400 \
  --simulations 400
```

## Parameters worth sweeping

Keep the first experiment small. Suggested grid:

- `gumbel_c_scale`: 0.05, 0.10, 0.20
- `simulations`: 100, 200, 400
- `gumbel_root_candidates`: 0 (all legal) first; then 8 only if speed matters

Do **not** interpret lower training loss alone as evidence that the search is
better. Compare exact-DB AUC, mirror consistency, colour-fixed strength, and
opening-rank stability.

## Tests

The new MCTS tests verify that Gumbel search:

- spends exactly the requested simulation budget,
- visits every legal opening action when the budget permits,
- never visits illegal actions,
- returns a normalized improved policy,
- still finds a mate in one.

A one-generation CPU smoke train also completes successfully in Gumbel mode.

The pre-existing CoordConv test was also corrected to inspect the policy head
rather than the deliberately zero-initialised value head. This makes the test
consistent with the model's neutral-value initialisation invariant.
