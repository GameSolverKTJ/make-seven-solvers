# Gumbel Make7-AZ v2.1.11

v2.1.11 simplifies the final EE opening evaluator. Training is unchanged.

Changes from v2.1.10:

- Removed stochastic full-game robustness matches from the final evaluator and
  all default outputs.
- Retained the post-opening high-budget budget sweep as the primary analysis.
- Retained repeated primary-budget root searches, now with direct paired
  opening-difference summaries (`pairwise_root_value.csv`).
- Retained deterministic temperature-0 principal variations.
- Added cross-checkpoint opening value/rank summaries.
- Added cross-checkpoint pairwise-difference sign-consistency summaries.
- `final_summary.csv` no longer contains robustness-match fields.

The purpose is to measure opening value as directly as possible without
allowing small stochastic search errors to accumulate over an entire game.
