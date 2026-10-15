# Gumbel Make7-AZ v2.1.6

## Replay-buffer balanced-sampling fix

When a saved replay buffer was loaded in v2.1.5, `states`, `policies`, `values`, Q targets and Q weights were restored, but the derived `buckets` array used by `--balanced-sampling` was not rebuilt. The inherited samples therefore defaulted to bucket 0 until overwritten by new self-play.

v2.1.6 rebuilds bucket ids from the loaded states and TD(lambda) value targets immediately after loading and invalidates the cached bucket index.

This is especially important for `--init-run-dir`, where most of the replay buffer may be inherited.
