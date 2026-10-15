# Gumbel Make7-AZ v2.1.3

v2.1.3 adds leakage-safe value calibration diagnostics and optional exact-guided
value fine-tuning on top of v2.1.2.

## What changed

- Exact propositions are split by the shallow proof group, never by record.
- The split is stratified by whether a group contains GREEN wins, YELLOW wins,
  both, or neither. This prevents a plain random group split from placing nearly
  all YELLOW-win proof groups on one side.
- `validation.csv` now logs both raw and mirror-symmetrised validation.
- The primary validation view follows `--symmetrise` (on by default), while
  `raw_*` and `sym_*` columns are always written.
- A classification threshold is fitted only on the calibration groups and then
  evaluated on untouched test groups. This is diagnostic only; MCTS continues
  to use continuous values.
- Optional exact auxiliary value supervision uses only fully decisive positions
  from the calibration/train groups. Sampling is balanced by class and proof
  group, so a huge proof cluster cannot dominate.
- `--init-checkpoint` can start a new fine-tuning run from an existing
  `best_validation.pt`.

## New validation.csv columns

- `raw_auc`, `raw_balanced_accuracy`
- `sym_auc`, `sym_balanced_accuracy`
- `optimal_threshold`
- `calibration_balanced_accuracy`
- `test_balanced_accuracy_at_zero`
- `calibrated_balanced_accuracy`
- `test_mcc_at_zero`, `calibrated_mcc`
- `calibrated_green_win_accuracy`, `calibrated_yellow_win_accuracy`
- calibration/test record and group counts

`auc` remains the recommended checkpoint-selection metric because it is
threshold-free. `optimal_threshold` is for diagnosis, not for changing MCTS
win/loss logic.

## Recommended short exact-guided fine-tune from gen38

```bash
export EXACT_DB=/path/to/exact_propositions_campaign.txt
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6

python train.py \
  --run-dir runs/gumbel_v2_1_3_ft \
  --init-checkpoint /path/to/best_validation.pt \
  --generations 50 \
  --tile-limits three_only \
  --search-mode gumbel \
  --simulations 200 \
  --gumbel-root-candidates 0 \
  --gumbel-c-visit 50 \
  --gumbel-c-scale 0.10 \
  --symmetrise \
  --balanced-sampling \
  --td-lambda 0.7 \
  --q-loss-weight 0.25 \
  --q-min-visits 3 \
  --exact-loss-weight 0.05 \
  --exact-batch-size 128 \
  --validation-database "$EXACT_DB" \
  --validation-every 1 \
  --validation-select auc \
  --validation-calibration-fraction 0.5 \
  --validation-calibration-seed 1701 \
  --games-per-generation 1024 \
  --parallel-games 256 \
  --workers 6 \
  --gather-timeout 0.5 \
  --train-steps 320 \
  --batch-size 1024 \
  --lr 5e-4 \
  --lr-final 1e-4 \
  --lr-decay-generations 12 \
  --arena-games 200
```

For a clean control, run the same command with `--exact-loss-weight 0`.
