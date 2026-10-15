# Gumbel Make7-AZ v2.1.7 — Balanced forced-opening self-play

v2.1.7 adds branch-coverage self-play for the EE opening comparison without
turning the forced move into a policy teacher.

## Why

A mature policy can concentrate almost all organic self-play on one opening
family (for example 2C/2E).  That makes value estimates for 2A/2B/2D depend on
far less downstream experience even when Gumbel searches every root action.
The research question, however, is explicitly the relative value of
2A/2B/2C/2D.

## Design

By default `train.py` forces 25% of self-play games, balanced across:

    2A 2B 2C 2D

Mirror augmentation supplies the exactly symmetric 2G/2F/2E trajectories.
Use `--forced-opening-fraction 0` to disable the intervention.

For a forced game the empty root is still searched normally.  The sample
keeps:

- the root value target;
- searched Q(s,a) targets and weights;

but its policy target is set to zero mass.  The training loss recognises such
a row as masked and excludes it from policy cross-entropy.  Therefore the
network is not taught that 2A/2B/2C/2D was preferred merely because the
experimenter forced it.

The actual move is then forced and the remainder of the game is ordinary
self-play.

## Global balance with multiprocessing

The parent process computes the exact number of forced games and distributes
both counts and a rotating opening offset across workers.  This avoids each
worker independently over-allocating the first item in the list when the
forced count is not divisible by four.

## Diagnostics

`opening_sources.csv` records organic and forced opening counts separately.

`metrics.csv` adds:

- `forced_opening_entropy`
- `forced_games`
- `forced_opening_fraction_actual`

`opening_entropy` / console `Hopen` now uses only **organic** games.  Forced
openings therefore cannot make a collapsed policy look diverse.

## Recommended use

For the current 2A/2B/2C/2D EE question:

    --forced-opening-fraction 0.25 \
    --forced-openings 2A 2B 2C 2D

Keep v2.1.6 replay-buffer inheritance, balanced sampling, exact holdout gating,
and mirror symmetrisation enabled.
