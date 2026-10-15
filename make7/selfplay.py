"""Self-play data generation.

Many games advance in lockstep so that every MCTS simulation step can send
one batch to the GPU.  This is the single most important performance choice
in the whole pipeline: a 7x7 position evaluated on its own leaves an 11 GB
card almost completely idle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .game import ACTION_SIZE, Make7State, mirror_action, new_game
from .mcts import BatchedMCTS, MCTSConfig, visits_to_policy


@dataclass
class SelfPlayConfig:
    games_per_generation: int = 256
    parallel_games: int = 128
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    augment_mirror: bool = True
    # Weight on the final game result when forming the value target:
    #     target = value_mix * z + (1 - value_mix) * root_Q
    # 1.0 reproduces plain AlphaZero. Lower values blend in the search's own
    # estimate, which is far less noisy than a single playthrough: z is the
    # outcome of one game in which a large fraction of the moves were sampled
    # for exploration rather than chosen, whereas root_Q summarises hundreds
    # of simulations at that exact position.
    #
    # This matters here beyond ordinary variance reduction. Because the input
    # planes are written from the mover's point of view, the network can read
    # the side to move off the parity of the tile count. If self-play results
    # are lopsided, every first-player position carries a similar z and the
    # network can score well by keying on parity instead of on the position,
    # which then degrades its play for that side and makes the imbalance
    # worse. Mixing in root_Q breaks that loop, since root_Q varies with the
    # position even when z does not.
    # TD(lambda) for the value target. 1.0 is plain AlphaZero: every position
    # in a game is labelled with the final result. Lower values bootstrap
    # from the search values of later positions instead.
    #
    # The problem it addresses: a single game supplies one sample of the
    # outcome, and here roughly half of every game's moves are sampled for
    # exploration rather than chosen, so an opening position's label is close
    # to a coin flip no matter how good the position is. The search has
    # already spent hundreds of simulations evaluating each of those
    # positions and that work is otherwise thrown away.
    #
    # Because the players alternate, a value seen from ply t+1 is negated
    # when read from ply t, which is where the minus signs in the recursion
    # come from:
    #
    #     G_t = -[(1 - lambda) * q_{t+1} + lambda * G_{t+1}]
    #     G_last = z
    #
    # lambda = 1 collapses to the alternating final result, lambda = 0 to a
    # one-step bootstrap off the next position's search value. Around 0.7
    # keeps most of the outcome signal while cutting the variance that comes
    # from the exploratory opening.
    td_lambda: float = 1.0
    value_mix: float = 1.0
    # Tile supply for self-play. None = unlimited; THREE_ONLY caps 3s at four
    # per player, matching the ruleset the exact database was proved under.
    limits: tuple | None = None
    # Opening randomisation: play this many uniformly random plies before
    # handing over to the search.  Keeps early-generation buffers diverse.
    random_opening_plies: int = 0
    # Auxiliary Q-head supervision. Only root actions visited at least this
    # many times receive a target. Unvisited/completed Q is never used as a
    # label, preventing the self-referential failure seen with value_mix.
    q_min_visits: int = 4
    # v2.1.7: force a fraction of games to begin with selected openings so
    # low-prior branches receive enough downstream training data.  The forced
    # root still runs MCTS and contributes value/Q targets, but its policy
    # target is masked out so the network is never taught that the forced
    # action itself was preferred by search.
    forced_opening_fraction: float = 0.0
    forced_openings: tuple[int, ...] = ()
    # v2.1.9: smooth every non-masked policy training target, not just the
    # organic empty root. Search/action selection still uses the unsmoothed
    # distribution. Forced roots remain policy-masked, while later positions
    # in forced games are smoothed normally.
    policy_target_smoothing: float = 0.05
    # Internal fields used by the parallel dispatcher to make the global
    # forced-game count and action allocation exact across workers.
    forced_opening_count: int | None = None
    forced_opening_offset: int = 0


@dataclass
class Sample:
    state: np.ndarray      # (10, 7, 7) float32
    policy: np.ndarray     # (21,) float32, sums to 1
    value: np.float32      # outcome for the player to move, in [-1, 1]
    q_target: np.ndarray   # (21,) searched root Q(s,a), mover-relative
    q_weight: np.ndarray   # (21,) sqrt(visits) for sufficiently visited actions


_MIRROR_INDEX = np.array([mirror_action(a) for a in range(ACTION_SIZE)])


def _normalised_policy_entropy(policy: np.ndarray) -> float:
    """Entropy of one policy row, normalised by log(ACTION_SIZE)."""
    p = np.asarray(policy, dtype=np.float64)
    mass = float(p.sum())
    if mass <= 1e-12:
        return float("nan")
    p = p / mass
    ent = -float(np.where(p > 0, p * np.log(np.clip(p, 1e-12, 1.0)), 0.0).sum())
    return ent / float(np.log(ACTION_SIZE))


def _smooth_policy_target(policy: np.ndarray, legal_actions: np.ndarray,
                          epsilon: float) -> np.ndarray:
    """Mix a policy teacher with a uniform distribution over legal moves.

    This function changes only the learning target. Search and action sampling
    must continue to use the raw policy supplied separately by the caller.
    """
    raw = np.asarray(policy, dtype=np.float32)
    eps = float(np.clip(epsilon, 0.0, 1.0))
    if eps <= 0.0:
        return raw.astype(np.float32, copy=True)
    legal = np.asarray(legal_actions, dtype=np.float32)
    legal_mass = float(legal.sum())
    if legal_mass <= 0.0:
        return raw.astype(np.float32, copy=True)
    uniform_legal = legal / legal_mass
    out = ((1.0 - eps) * raw + eps * uniform_legal).astype(np.float32)
    out /= max(float(out.sum()), 1e-12)
    return out


def _mirror_samples(samples: list[Sample], states: list[Make7State]
                    ) -> list[Sample]:
    out = []
    for sample, state in zip(samples, states):
        mirrored_policy = np.zeros_like(sample.policy)
        mirrored_policy[_MIRROR_INDEX] = sample.policy
        mirrored_q = np.zeros_like(sample.q_target)
        mirrored_w = np.zeros_like(sample.q_weight)
        mirrored_q[_MIRROR_INDEX] = sample.q_target
        mirrored_w[_MIRROR_INDEX] = sample.q_weight
        out.append(Sample(state.mirror().encode(), mirrored_policy, sample.value,
                          mirrored_q, mirrored_w))
    return out


def generate_games(evaluator, cfg: SelfPlayConfig,
                   rng: np.random.Generator | None = None,
                   progress=None) -> tuple[list[Sample], dict]:
    """Play `cfg.games_per_generation` games and return training samples.

    Games run in a fixed set of slots that are refilled as soon as a game
    ends, rather than starting 256 games together and waiting for the last
    one.  That matters more than it sounds: mean game length here is about
    22 plies but the longest run to nearly 49, so a start-together batch
    spends its whole second half issuing forward passes for a handful of
    stragglers.  Refilling keeps every batch full, which roughly halves the
    number of forward passes for the same amount of training data.

    Also returns a statistics dict used for the convergence plots: game
    lengths, result split, and how often each opening move was chosen.
    """
    rng = rng or np.random.default_rng()
    mcts = BatchedMCTS(cfg.mcts, rng)

    samples: list[Sample] = []
    stats = {
        "games": 0, "green_wins": 0, "yellow_wins": 0, "draws": 0,
        "total_plies": 0, "q_supervised_actions": 0, "q_positions": 0,
        "opening_counts": np.zeros(ACTION_SIZE, dtype=np.int64),
        "organic_opening_counts": np.zeros(ACTION_SIZE, dtype=np.int64),
        "forced_opening_counts": np.zeros(ACTION_SIZE, dtype=np.int64),
        "forced_games": 0,
        "organic_games": 0, "organic_green_wins": 0,
        "organic_yellow_wins": 0, "organic_draws": 0,
        "forced_green_wins": 0, "forced_yellow_wins": 0, "forced_draws": 0,
        "forced_result_games": np.zeros(ACTION_SIZE, dtype=np.int64),
        "forced_green_wins_by_opening": np.zeros(ACTION_SIZE, dtype=np.int64),
        "forced_draws_by_opening": np.zeros(ACTION_SIZE, dtype=np.int64),
        "organic_root_policy_raw_sum": np.zeros(ACTION_SIZE, dtype=np.float64),
        "organic_root_policy_train_sum": np.zeros(ACTION_SIZE, dtype=np.float64),
        "organic_root_policy_count": 0,
        "policy_entropy_raw_sum": 0.0,
        "policy_entropy_train_sum": 0.0,
        "policy_entropy_count": 0,
    }

    target_games = cfg.games_per_generation
    n_slots = min(cfg.parallel_games, target_games)

    # Build the whole opening plan up front.  This makes the number of forced
    # games deterministic and balances the selected openings to within one
    # game.  In parallel mode the parent assigns each worker a count and
    # offset so the *global* allocation is balanced, not merely each worker.
    forced_actions_cfg = tuple(int(a) for a in cfg.forced_openings)
    frac = float(np.clip(cfg.forced_opening_fraction, 0.0, 1.0))
    if cfg.forced_opening_count is None:
        n_forced = int(round(target_games * frac)) if forced_actions_cfg else 0
    else:
        n_forced = int(np.clip(cfg.forced_opening_count, 0, target_games))
        if not forced_actions_cfg:
            n_forced = 0
    forced_seq = [
        forced_actions_cfg[(cfg.forced_opening_offset + j) % len(forced_actions_cfg)]
        for j in range(n_forced)
    ] if forced_actions_cfg else []
    opening_plan: list[int | None] = forced_seq + [None] * (target_games - n_forced)
    rng.shuffle(opening_plan)
    plan_cursor = 0

    def fresh() -> tuple[Make7State, int | None]:
        nonlocal plan_cursor
        forced_action = opening_plan[plan_cursor]
        plan_cursor += 1
        s = new_game(cfg.limits)
        # Random pre-play and a forced first move are intentionally not mixed:
        # a forced-opening game must expose the true empty root to MCTS so its
        # value/Q targets remain usable.
        if forced_action is None:
            for _ in range(cfg.random_opening_plies):
                if s.is_terminal:
                    break
                s = s.step(int(rng.choice(s.legal_action_list())))
        return s, forced_action

    initial = [fresh() for _ in range(n_slots)]
    slots = [x[0] for x in initial]
    forced_actions = [x[1] for x in initial]
    histories: list[list[tuple[Make7State, np.ndarray, float, np.ndarray, np.ndarray]]] = [
        [] for _ in range(n_slots)]
    move_numbers = [int(s.ply) for s in slots]
    live = [True] * n_slots
    started = n_slots
    finished = 0

    while finished < target_games:
        idx = [i for i in range(n_slots) if live[i]]
        if not idx:
            break
        need_q = cfg.value_mix < 1.0 or cfg.td_lambda < 1.0
        use_improved_policy = cfg.mcts.search_mode == "gumbel"
        if use_improved_policy and need_q:
            visits, improved_policies, root_values, root_action_q = mcts.search(
                [slots[i] for i in idx], evaluator, add_noise=True,
                with_policy=True, with_values=True, with_action_values=True)
        elif use_improved_policy:
            visits, improved_policies, root_action_q = mcts.search(
                [slots[i] for i in idx], evaluator, add_noise=True,
                with_policy=True, with_action_values=True)
            root_values = None
        elif need_q:
            visits, root_values, root_action_q = mcts.search(
                [slots[i] for i in idx], evaluator, add_noise=True,
                with_values=True, with_action_values=True)
            improved_policies = None
        else:
            visits, root_action_q = mcts.search(
                [slots[i] for i in idx], evaluator, add_noise=True,
                with_action_values=True)
            improved_policies = None
            root_values = None

        for k, i in enumerate(idx):
            state = slots[i]
            counts = visits[k]
            total = counts.sum()
            if total <= 0:
                live[i] = False
                continue

            move_number = move_numbers[i]
            temperature = (cfg.mcts.temperature
                           if move_number < cfg.mcts.temperature_moves else 0.0)
            if improved_policies is not None:
                target_policy = improved_policies[k].astype(np.float32, copy=False)
                probs = visits_to_policy(target_policy, temperature)
            else:
                target_policy = (counts / total).astype(np.float32)
                probs = visits_to_policy(counts, temperature)
            if probs.sum() <= 0:
                live[i] = False
                continue

            # In PUCT mode the target remains the raw visit distribution. In
            # Gumbel mode sequential-halving visits reflect the allocation
            # schedule, so the search returns a separate improved policy based
            # on the prior plus completed root Q-values.
            # Q-head targets are only the Q values of actions that were
            # actually searched enough to be informative. In particular, we
            # do NOT use completed Q for unvisited actions. Weighting by
            # sqrt(N) gives better-read actions more influence without letting
            # one dominant move overwhelm the whole auxiliary loss.
            q_target = root_action_q[k].astype(np.float32, copy=True)
            q_weight = np.where(
                counts >= cfg.q_min_visits, np.sqrt(counts), 0.0
            ).astype(np.float32)
            stats["q_supervised_actions"] += int(np.count_nonzero(q_weight))
            stats["q_positions"] += 1

            forced_root = (move_number == 0 and forced_actions[i] is not None)
            raw_policy = target_policy.astype(np.float32, copy=False)
            if forced_root:
                # Keep root value/Q supervision but remove the policy teacher.
                # train.py treats a zero-mass policy row as masked.
                policy_for_training = np.zeros_like(target_policy, dtype=np.float32)
            else:
                # v2.1.9: smoothing applies to every policy-supervised state.
                # ``probs`` above was already formed from raw search output, so
                # neither MCTS nor action sampling is changed by this operation.
                policy_for_training = _smooth_policy_target(
                    raw_policy, state.legal_actions(), cfg.policy_target_smoothing)
                raw_h = _normalised_policy_entropy(raw_policy)
                train_h = _normalised_policy_entropy(policy_for_training)
                if np.isfinite(raw_h) and np.isfinite(train_h):
                    stats["policy_entropy_raw_sum"] += raw_h
                    stats["policy_entropy_train_sum"] += train_h
                    stats["policy_entropy_count"] += 1

                if move_number == 0 and forced_actions[i] is None:
                    # Keep root-specific raw/train distributions as a separate
                    # diagnostic so target smoothing cannot masquerade as a
                    # recovered organic search policy.
                    stats["organic_root_policy_raw_sum"] += raw_policy
                    stats["organic_root_policy_train_sum"] += policy_for_training
                    stats["organic_root_policy_count"] += 1

            histories[i].append((
                state,
                policy_for_training,
                root_values[k] if root_values is not None else 0.0,
                q_target,
                q_weight,
            ))

            if forced_root:
                action = int(forced_actions[i])
                if action < 0 or action >= ACTION_SIZE or not state.legal_actions()[action]:
                    raise ValueError(f"forced opening action {action} is illegal at root")
                stats["forced_games"] += 1
                stats["forced_opening_counts"][action] += 1
            else:
                action = int(rng.choice(ACTION_SIZE, p=probs))
                if move_number == 0:
                    stats["organic_opening_counts"][action] += 1

            if move_number == 0:
                stats["opening_counts"][action] += 1

            state = state.step(action)
            slots[i] = state
            move_numbers[i] += 1

            if not state.is_terminal:
                continue

            stats["games"] += 1
            stats["total_plies"] += state.ply
            if state.winner == 1:
                stats["green_wins"] += 1
            elif state.winner == -1:
                stats["yellow_wins"] += 1
            else:
                stats["draws"] += 1

            # Outcome diagnostics by opening source.  These are deliberately
            # observational only: no promotion or loss term depends on them.
            forced_action = forced_actions[i]
            if forced_action is None:
                stats["organic_games"] += 1
                if state.winner == 1:
                    stats["organic_green_wins"] += 1
                elif state.winner == -1:
                    stats["organic_yellow_wins"] += 1
                else:
                    stats["organic_draws"] += 1
            else:
                a0 = int(forced_action)
                stats["forced_result_games"][a0] += 1
                if state.winner == 1:
                    stats["forced_green_wins"] += 1
                    stats["forced_green_wins_by_opening"][a0] += 1
                elif state.winner == -1:
                    stats["forced_yellow_wins"] += 1
                else:
                    stats["forced_draws"] += 1
                    stats["forced_draws_by_opening"][a0] += 1

            game_states = [st for st, _, _, _, _ in histories[i]]
            mix, lam = cfg.value_mix, cfg.td_lambda
            history = histories[i]
            n_positions = len(history)

            # TD(lambda) returns, computed backwards from the end of the game.
            returns = [0.0] * n_positions
            for k in range(n_positions - 1, -1, -1):
                z_k = state.result_for(history[k][0].player)
                if lam >= 1.0 or k == n_positions - 1:
                    returns[k] = z_k
                else:
                    q_next = history[k + 1][2]
                    returns[k] = -((1.0 - lam) * q_next + lam * returns[k + 1])

            batch = []
            for k, (st, pol, q, q_target, q_weight) in enumerate(history):
                target = mix * returns[k] + (1.0 - mix) * q
                batch.append(Sample(st.encode(), pol, np.float32(target),
                                    q_target, q_weight))
            samples.extend(batch)
            if cfg.augment_mirror:
                samples.extend(_mirror_samples(batch, game_states))

            finished += 1
            if progress is not None:
                progress(finished, target_games)

            if started < target_games:
                slots[i], forced_actions[i] = fresh()
                histories[i] = []
                move_numbers[i] = int(slots[i].ply)
                started += 1
            else:
                live[i] = False

    return samples, stats


def _bucket_ids(states: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Group samples by (side to move, sign of the value target).

    The side to move is recoverable from the board: planes 0..5 hold every
    placed tile, so their sum is the ply count and its parity says whose turn
    it is. That is precisely what makes the shortcut below possible, and
    knowing the bucket is what lets us take it away.
    """
    ply = np.rint(states[:, :6].sum(axis=(1, 2, 3))).astype(np.int64)
    parity = ply % 2
    sign = np.sign(values).astype(np.int64) + 1      # 0 loss, 1 draw, 2 win
    return parity * 3 + sign


class ReplayBuffer:
    """Fixed-capacity ring buffer of training samples.

    Optionally samples training batches balanced across
    (side to move) x (outcome), which removes a shortcut that otherwise
    dominates value learning here.

    Because the board is encoded from the mover's point of view, the value
    target for a decided game is -1 at every position where the losing side
    is to move and +1 at every position where the winner is. Games alternate,
    so within one game the target is a perfect function of the ply parity —
    no board content required. When self-play is lopsided, and here the
    first-player rate has swung between 0.11 and 0.63, the same rule
    generalises across games and the value head can score very well by
    reading the tile count instead of the position. The measured symptom is a
    model that names YELLOW wins with 99.8% accuracy while getting GREEN wins
    right only 37% of the time.

    Drawing each (parity, outcome) bucket equally makes parity uninformative
    about the target, so the only way left to reduce the loss is to look at
    the board.
    """

    def __init__(self, capacity: int = 400_000):
        self.capacity = capacity
        self.states = np.zeros((capacity, 10, 7, 7), dtype=np.float32)
        self.policies = np.zeros((capacity, ACTION_SIZE), dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.q_targets = np.zeros((capacity, ACTION_SIZE), dtype=np.float32)
        self.q_weights = np.zeros((capacity, ACTION_SIZE), dtype=np.float32)
        self.buckets = np.zeros(capacity, dtype=np.int8)
        self._bucket_index: list[np.ndarray] | None = None
        self.size = 0
        self.cursor = 0

    def add(self, samples: list[Sample]) -> None:
        if not samples:
            return
        states = np.stack([s.state for s in samples])
        policies = np.stack([s.policy for s in samples])
        values = np.asarray([s.value for s in samples], dtype=np.float32)
        q_targets = np.stack([s.q_target for s in samples])
        q_weights = np.stack([s.q_weight for s in samples])
        self.add_arrays(states, policies, values, q_targets, q_weights)

    def add_arrays(self, states: np.ndarray, policies: np.ndarray,
                   values: np.ndarray, q_targets: np.ndarray | None = None,
                   q_weights: np.ndarray | None = None) -> None:
        """Bulk insert, used by the multiprocess pool which already returns
        stacked arrays. Writes in at most two slices instead of per-sample,
        and reproduces the per-sample ring semantics exactly — including the
        case where a single call is larger than the whole buffer."""
        total = len(states)
        if total == 0:
            return
        if q_targets is None:
            q_targets = np.zeros((total, ACTION_SIZE), dtype=np.float32)
        if q_weights is None:
            q_weights = np.zeros((total, ACTION_SIZE), dtype=np.float32)
        cap = self.capacity
        start = self.cursor

        if total >= cap:
            # Only the final `cap` samples survive. Writing them at
            # (cursor + total - cap) % cap leaves the ring in the same state
            # as feeding every sample through `add` one at a time.
            states = states[-cap:]
            policies = policies[-cap:]
            values = values[-cap:]
            q_targets = q_targets[-cap:]
            q_weights = q_weights[-cap:]
            start = (self.cursor + total) % cap
            n = cap
        else:
            n = total

        first = min(n, cap - start)
        self.states[start:start + first] = states[:first]
        self.policies[start:start + first] = policies[:first]
        self.values[start:start + first] = values[:first]
        self.q_targets[start:start + first] = q_targets[:first]
        self.q_weights[start:start + first] = q_weights[:first]
        rest = n - first
        if rest:
            self.states[:rest] = states[first:]
            self.policies[:rest] = policies[first:]
            self.values[:rest] = values[first:]
            self.q_targets[:rest] = q_targets[first:]
            self.q_weights[:rest] = q_weights[first:]

        ids = _bucket_ids(states, values)
        self.buckets[start:start + first] = ids[:first]
        if rest:
            self.buckets[:rest] = ids[first:]

        self.cursor = (self.cursor + total) % cap
        self.size = min(self.size + total, cap)
        self._bucket_index = None      # contents changed; rebuild on demand

    def _rebuild_bucket_index(self) -> None:
        live = self.buckets[:self.size]
        self._bucket_index = [np.flatnonzero(live == b) for b in range(6)]

    def bucket_sizes(self) -> list[int]:
        if self._bucket_index is None:
            self._rebuild_bucket_index()
        return [len(ix) for ix in self._bucket_index]

    def sample(self, batch_size: int, rng: np.random.Generator,
               balanced: bool = False, include_q: bool = False):
        def take(idx):
            base = (self.states[idx], self.policies[idx], self.values[idx])
            if include_q:
                return base + (self.q_targets[idx], self.q_weights[idx])
            return base

        if not balanced:
            idx = rng.integers(0, self.size, size=batch_size)
            return take(idx)

        if self._bucket_index is None:
            self._rebuild_bucket_index()

        # Buckets 0..2 are (loss, draw, win) for one parity, 3..5 for the
        # other. Only the decisive ones are equalised. Draws are left at
        # their natural frequency: they are rare here — often under 1% of the
        # buffer — and forcing them to a quarter of every batch would teach
        # the network that games are drawn far more often than they are.
        decisive = [0, 2, 3, 5]
        draws = [1, 4]
        live_decisive = [b for b in decisive if len(self._bucket_index[b]) > 0]
        if len(live_decisive) < 2:
            idx = rng.integers(0, self.size, size=batch_size)
            return take(idx)

        draw_total = sum(len(self._bucket_index[b]) for b in draws)
        draw_share = draw_total / max(self.size, 1)
        n_draw = int(round(batch_size * draw_share))
        n_decisive = batch_size - n_draw

        picks = []
        base, extra = divmod(n_decisive, len(live_decisive))
        for k, b in enumerate(live_decisive):
            ix = self._bucket_index[b]
            n = base + (1 if k < extra else 0)
            if n:
                picks.append(ix[rng.integers(0, len(ix), size=n)])

        if n_draw and draw_total:
            live_draws = [b for b in draws if len(self._bucket_index[b]) > 0]
            base_d, extra_d = divmod(n_draw, len(live_draws))
            for k, b in enumerate(live_draws):
                ix = self._bucket_index[b]
                n = base_d + (1 if k < extra_d else 0)
                if n:
                    picks.append(ix[rng.integers(0, len(ix), size=n)])

        idx = np.concatenate(picks)
        rng.shuffle(idx)
        return take(idx)

    def __len__(self) -> int:
        return self.size

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #

    def save(self, path) -> None:
        """Write the buffer to disk so a restart does not throw away several
        generations of self-play. Only the filled region is stored."""
        path = Path(path)
        # np.savez appends ".npz" unless the name already ends with it, so
        # the temporary name has to end that way too or the rename misses.
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(
            tmp,
            states=self.states[:self.size],
            policies=self.policies[:self.size],
            values=self.values[:self.size],
            q_targets=self.q_targets[:self.size],
            q_weights=self.q_weights[:self.size],
            cursor=np.int64(self.cursor),
            capacity=np.int64(self.capacity),
        )
        tmp.replace(path)          # atomic, so an interrupted save cannot
                                   # leave a half-written buffer behind

    def load(self, path) -> bool:
        """Restore a saved buffer. Returns False if the file is missing or
        unreadable, in which case the buffer is left empty."""
        path = Path(path)
        if not path.exists():
            return False
        try:
            with np.load(path) as data:
                states = data["states"]
                policies = data["policies"]
                values = data["values"]
                q_targets = (data["q_targets"] if "q_targets" in data.files
                             else np.zeros((len(states), ACTION_SIZE), dtype=np.float32))
                q_weights = (data["q_weights"] if "q_weights" in data.files
                             else np.zeros((len(states), ACTION_SIZE), dtype=np.float32))
                cursor = int(data["cursor"])
        except Exception:
            return False

        n = min(len(states), self.capacity)
        self.states[:n] = states[-n:]
        self.policies[:n] = policies[-n:]
        self.values[:n] = values[-n:]
        self.q_targets[:n] = q_targets[-n:]
        self.q_weights[:n] = q_weights[-n:]
        self.size = n
        self.cursor = cursor % self.capacity if n >= self.capacity else n

        # v2.1.6: rebuild balanced-sampling bucket metadata after loading.
        # The bucket ids are derived from the *actual stored TD(lambda) value
        # targets* and ply parity.  Earlier versions restored states/values but
        # left ``self.buckets`` at its zero-initialised default, so every
        # inherited sample was silently treated as bucket 0 until overwritten
        # by new self-play.  That broke --balanced-sampling exactly when a
        # replay buffer was inherited into a new run.
        if n:
            self.buckets[:n] = _bucket_ids(self.states[:n], self.values[:n])
        if n < self.capacity:
            self.buckets[n:] = 0
        self._bucket_index = None
        return True
