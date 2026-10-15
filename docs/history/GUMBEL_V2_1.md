# Gumbel Make7-AZ v2.1.1

## v2.1.1 parallel Gumbel fix

Gumbel sequential halving can request several root-action leaf evaluations per
live game in one MCTS call.  With multiprocessing this can be much larger than
the per-worker shared-memory slot (for example 171 live games x 14 legal
opening actions = 2394 positions).  The remote evaluator now transparently
splits such bursts into slot-sized chunks.  No training command change is
required.

v2.1 adds an auxiliary action-value head to Gumbel Make7-AZ v2 without letting
that head drive search yet. The goal is to learn, and separately monitor, the
value of each legal root action while preserving the v2 Gumbel/PUCT search
behaviour.

## Architecture

The shared trunk now feeds three heads:

- policy logits `P(s,a)` — 21 actions
- state value `V(s)` — scalar in `[-1, 1]`
- auxiliary action values `Q_head(s,a)` — 21 values in `[-1, 1]`

`Make7Net.forward()` still returns `(policy, value)` for backward compatibility.
Training uses `Make7Net.forward_with_q()` which returns `(policy, value, q)`.

The Q head is `1x1 conv -> FC hidden -> 21 outputs -> tanh`. Its final layer is
zero-initialised, so a fresh model (and a v2 checkpoint upgraded to v2.1) starts
with exactly neutral `Q_head=0`.

## Q targets

Q targets come from the actual searched root action values `W(a)/N(a)`.
Crucially:

- only actions with `N(a) >= --q-min-visits` are supervised;
- unvisited/completed Q values are NEVER used as labels;
- action weights are `sqrt(N(a))`, so better-read moves count more without the
  final sequential-halving survivor dominating the entire loss.

The auxiliary loss is a weighted Huber loss:

    L = L_policy + value_weight * L_value + q_weight * L_Q

Defaults:

    --q-loss-weight 0.25
    --q-min-visits 4

The Q head is deliberately not consulted by MCTS in v2.1. It can affect the
shared representation through training, but there is no direct `Q_head -> move
selection` path. That makes v2.1 a clean intermediate experiment before any
v2.2 search integration.

## Opening diagnostics

Each generation now writes `opening_q_diagnostics.csv`. For all 21 actions it
records:

- `policy_*`: network root prior after the legal-action mask
- `qhead_*`: direct Q-head prediction
- `mctsq_*`: searched root `W/N`
- `visits_*`: deterministic diagnostic search visits (no root noise)
- `improved_*`: Gumbel improved policy returned by search

`metrics.csv` also contains `q_loss` and `q_actions_per_position` (the mean
number of actions per self-play position that passed the Q supervision visit
threshold). When resuming a v2 run, the old metrics file is migrated by column
name so the new fields are added without corrupting earlier rows.

This makes it possible to distinguish cases such as:

- Q-head thinks 2D is good but policy suppresses it;
- policy likes 2D but searched Q rejects it;
- Q-head and MCTS-Q agree while the final improved policy remains unstable.

## Compatibility

- v2 checkpoints load in v2.1. Missing `q_*` parameters are initialised to the
  neutral zero-output Q head.
- old replay buffers load in v2.1. Their missing Q targets/weights are filled
  with zeros, so they contribute no Q loss until new self-play data arrives.
- arena, validation, opening analysis and other code using `net(x) -> (p,v)`
  continue to work unchanged.

## Recommended run

```bash
python train.py \
  --run-dir runs/gumbel_v21 \
  --generations 100 \
  --preset coord_base \
  --tile-limits three_only \
  --search-mode gumbel \
  --simulations 200 \
  --gumbel-root-candidates 0 \
  --gumbel-c-visit 8 \
  --gumbel-c-scale 0.10 \
  --q-loss-weight 0.25 \
  --q-min-visits 4 \
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

For the cleanest ablation, compare this against the same v2 search/training run
with:

```bash
--q-loss-weight 0
```

Everything else should remain identical.