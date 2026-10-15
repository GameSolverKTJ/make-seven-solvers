# Gumbel Make7-AZ v2.1.4

Validation leakage and opening-symmetry fix.

## Changes from v2.1.3

- `holdout_auc`, `holdout_balanced_accuracy`, and `holdout_mcc` are computed only on exact-proof groups that are excluded from exact auxiliary training.
- Raw and mirror-symmetrised holdout metrics are both logged separately.
- `--validation-select` now defaults to `holdout_auc`. Legacy all-database metrics remain available only for diagnostics/backward compatibility.
- Resuming does not reuse a best-validation score produced with a different selection metric.
- Console output labels held-out and all-database exact metrics separately.
- Opening values now use the same `NetEvaluator` as the production solver, including mirror symmetrisation. Mirror-equivalent openings should therefore agree up to floating-point noise.

The exact calibration/train split remains group-stratified. Threshold calibration is still diagnostic only and is never inserted into MCTS.
