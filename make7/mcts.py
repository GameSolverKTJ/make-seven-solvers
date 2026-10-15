"""Batched Monte-Carlo tree search for Make 7.

Two root-search modes are supported:

``puct``
    The original AlphaZero search. Every simulation chooses the root move by
    PUCT and the visit distribution is used as the policy target.

``gumbel``
    Gumbel AlphaZero-style root search. Legal root actions are ranked by a
    sampled Gumbel perturbation plus the network prior, then compared with
    sequential halving. Each surviving root action is forced to receive
    search effort before the candidate set is reduced. Below the root the
    tree still uses ordinary PUCT.

The important implementation detail is that sequential-halving visit counts
are *not* a good training target: they are partly determined by the allocation
schedule. ``search(..., with_policy=True)`` therefore returns a separate
improved root policy based on prior logits and completed root Q-values.

Value convention: a node's value is always the expected result for the
player to move AT that node, in [-1, +1]. Backing up therefore flips the sign
at every level.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .game import ACTION_SIZE, Make7State


@dataclass
class MCTSConfig:
    simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.8
    dirichlet_epsilon: float = 0.25
    # Moves after which self-play switches from sampling to argmax.
    temperature_moves: int = 12
    temperature: float = 1.0

    # Root search. ``puct`` exactly preserves the original implementation.
    # ``gumbel`` uses sequential halving at the root and PUCT below it.
    search_mode: str = "puct"
    # 0 means all legal root actions. At the Make 7 opening this is only 14,
    # so using all actions is recommended.
    gumbel_root_candidates: int = 0
    # Scale used by the monotone Q transform in the Gumbel score:
    #   log prior + gumbel + (c_visit + maxN) * c_scale * completed_Q
    # Defaults follow the released MCTX-style completed-Q transform: a
    # substantial visit offset and a small value scale after Q normalisation.
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 0.10
    gumbel_rescale_q: bool = True

    def __post_init__(self):
        if self.search_mode not in ("puct", "gumbel"):
            raise ValueError("search_mode must be 'puct' or 'gumbel'")
        if self.simulations < 1:
            raise ValueError("simulations must be >= 1")
        if self.gumbel_root_candidates < 0:
            raise ValueError("gumbel_root_candidates must be >= 0")


class Node:
    """A search node with cached state and legal-action mask."""

    __slots__ = ("P", "N", "W", "children", "legal", "illegal", "expanded",
                 "state", "total", "prior_value")

    def __init__(self, state=None):
        self.P = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.N = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.W = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.children: dict[int, "Node"] = {}
        self.legal = np.zeros(ACTION_SIZE, dtype=bool)
        self.illegal = np.ones(ACTION_SIZE, dtype=bool)
        self.expanded = False
        self.state = state
        self.total = 0.0
        # Network V(s), used to complete Q for root actions that have not yet
        # been visited. This is the same mover-relative value convention as W/N.
        self.prior_value = 0.0

    @property
    def visit_count(self) -> int:
        return int(self.total)

    def q(self) -> np.ndarray:
        return self.W / np.maximum(self.N, 1.0)

    def completed_q(self) -> np.ndarray:
        q = self.q().astype(np.float64)
        return np.where(self.N > 0, q, float(self.prior_value))

    def select(self, c_puct: float) -> int:
        scores = (self.W / np.maximum(self.N, 1.0)
                  + (c_puct * np.sqrt(max(self.total, 1.0))) * self.P
                  / (1.0 + self.N))
        scores[self.illegal] = -np.inf
        return int(scores.argmax())


def _expand(node: Node, state: Make7State, policy_logits: np.ndarray,
            value: float | None = None) -> None:
    legal = state.legal_actions()
    node.legal = legal
    node.illegal = ~legal
    logits = policy_logits.astype(np.float64).copy()
    logits[~legal] = -np.inf
    finite = logits[legal]
    if finite.size == 0:
        node.P.fill(0.0)
    else:
        logits -= finite.max()
        p = np.exp(logits)
        p[~legal] = 0.0
        s = p.sum()
        if s <= 0 or not np.isfinite(s):
            p = legal.astype(np.float64)
            s = p.sum()
        node.P = (p / s).astype(np.float32)
    if value is not None:
        node.prior_value = float(value)
    node.expanded = True


def _add_dirichlet_noise(node: Node, alpha: float, eps: float,
                         rng: np.random.Generator) -> None:
    idx = np.flatnonzero(node.legal)
    if idx.size == 0:
        return
    noise = rng.dirichlet([alpha] * idx.size)
    node.P[idx] = (1.0 - eps) * node.P[idx] + eps * noise.astype(np.float32)


def _backup(path: list[tuple[Node, int]], leaf_value: float) -> None:
    value = leaf_value
    for node, action in reversed(path):
        value = -value
        node.N[action] += 1.0
        node.W[action] += value
        node.total += 1.0


def _softmax_masked(scores: np.ndarray, legal: np.ndarray) -> np.ndarray:
    out = np.zeros(ACTION_SIZE, dtype=np.float32)
    idx = np.flatnonzero(legal)
    if idx.size == 0:
        return out
    x = scores[idx].astype(np.float64)
    x -= np.max(x)
    ex = np.exp(x)
    denom = ex.sum()
    if denom <= 0 or not np.isfinite(denom):
        out[idx] = 1.0 / idx.size
    else:
        out[idx] = (ex / denom).astype(np.float32)
    return out


class BatchedMCTS:
    """Batched PUCT or Gumbel-root MCTS for a list of states."""

    def __init__(self, config: MCTSConfig, rng: np.random.Generator | None = None):
        self.cfg = config
        self.rng = rng if rng is not None else np.random.default_rng()

    def search(self, states: list[Make7State], evaluator,
               add_noise: bool = True, with_values: bool = False,
               with_policy: bool = False, with_action_values: bool = False):
        """Search a batch of positions.

        By default returns a list of root visit-count vectors.

        ``with_policy=True`` additionally returns a policy target. For PUCT it
        is simply normalised visits. For Gumbel search it is the improved policy
        obtained from prior logits plus completed root Q, not the sequential-
        halving allocation counts.

        ``with_values=True`` additionally returns each root's backed-up mean
        value from the mover's point of view.

        ``with_action_values=True`` returns the raw searched Q(s,a) at the
        root. Unvisited actions are zero here; callers must use the returned
        visit counts as a mask and must NOT substitute completed Q as a
        supervised target.
        """
        if self.cfg.search_mode == "gumbel":
            visits, policies, values, roots = self._search_gumbel(
                states, evaluator, add_noise=add_noise)
        else:
            visits, policies, values, roots = self._search_puct(
                states, evaluator, add_noise=add_noise)

        action_values = [r.q().astype(np.float32, copy=True) for r in roots]

        outputs = [visits]
        if with_policy:
            outputs.append(policies)
        if with_values:
            outputs.append(values)
        if with_action_values:
            outputs.append(action_values)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    # ------------------------------------------------------------------ #
    # Original AlphaZero PUCT path

    def _initial_roots(self, states, evaluator, add_noise, dirichlet):
        roots = [Node(states[i]) for i in range(len(states))]
        active = [i for i, s in enumerate(states) if not s.is_terminal]
        if active:
            batch = np.stack([states[i].encode() for i in active])
            logits, values = evaluator(batch)
            for k, i in enumerate(active):
                _expand(roots[i], states[i], logits[k], float(values[k]))
                if add_noise and dirichlet:
                    _add_dirichlet_noise(
                        roots[i], self.cfg.dirichlet_alpha,
                        self.cfg.dirichlet_epsilon, self.rng)
        return roots, active

    def _search_puct(self, states, evaluator, add_noise):
        roots, active = self._initial_roots(
            states, evaluator, add_noise, dirichlet=True)

        for _ in range(self.cfg.simulations):
            pending_states: list[Make7State] = []
            pending_meta: list[tuple[Node, list[tuple[Node, int]]]] = []

            for i in active:
                node = roots[i]
                state = node.state
                path: list[tuple[Node, int]] = []

                while True:
                    if state.is_terminal:
                        _backup(path, state.result_for(state.player))
                        break
                    if not node.expanded:
                        pending_states.append(state)
                        pending_meta.append((node, path))
                        break
                    action = node.select(self.cfg.c_puct)
                    path.append((node, action))
                    child = node.children.get(action)
                    if child is None:
                        child = Node(state.step(action))
                        node.children[action] = child
                    state = child.state
                    node = child

            self._evaluate_pending(pending_states, pending_meta, evaluator)

        visits = [r.N.copy() for r in roots]
        policies = [visits_to_policy(v, 1.0) for v in visits]
        values = [self._root_value(r) for r in roots]
        return visits, policies, values, roots

    # ------------------------------------------------------------------ #
    # Gumbel AlphaZero-style root search

    def _search_gumbel(self, states, evaluator, add_noise):
        # Dirichlet is intentionally disabled. Gumbel perturbations are the
        # root exploration mechanism in this mode.
        roots, active = self._initial_roots(
            states, evaluator, add_noise=False, dirichlet=False)

        candidates: dict[int, np.ndarray] = {}
        gumbels: dict[int, np.ndarray] = {}
        budgets_used = {i: 0 for i in active}

        for i in active:
            root = roots[i]
            legal = np.flatnonzero(root.legal)
            # log(P) is stable here because legal priors are strictly positive
            # after the softmax in _expand.
            logp = np.full(ACTION_SIZE, -np.inf, dtype=np.float64)
            logp[legal] = np.log(np.maximum(root.P[legal], 1e-30))
            g = np.zeros(ACTION_SIZE, dtype=np.float64)
            if add_noise:
                # numpy's standard Gumbel has the required Gumbel(0,1) law.
                g[legal] = self.rng.gumbel(size=legal.size)
            gumbels[i] = g

            k = self.cfg.gumbel_root_candidates or legal.size
            k = max(1, min(int(k), legal.size, self.cfg.simulations))
            rank = logp + g
            chosen = legal[np.argsort(rank[legal])[::-1][:k]]
            candidates[i] = chosen.astype(np.int64)

        # Number of sequential-halving rounds is determined by the largest
        # candidate set. Rounds with a singleton are skipped for that root.
        max_k = max((len(candidates[i]) for i in active), default=0)
        rounds = max(1, int(math.ceil(math.log2(max_k)))) if max_k > 1 else 1

        for round_index in range(rounds):
            # Allocate roughly an equal share of the total budget to every
            # round, then evenly among surviving candidates. At least one
            # sweep is made whenever budget permits.
            remaining_rounds = max(1, rounds - round_index)
            sweeps = {}
            for i in active:
                cand = candidates[i]
                if len(cand) <= 1:
                    continue
                remaining = self.cfg.simulations - budgets_used[i]
                if remaining <= 0:
                    continue
                per_round = max(1, remaining // remaining_rounds)
                sweeps[i] = max(1, per_round // len(cand))

            max_sweeps = max(sweeps.values(), default=0)
            for sweep in range(max_sweeps):
                requests: list[tuple[int, int]] = []
                for i in active:
                    n_sweeps = sweeps.get(i, 0)
                    if sweep >= n_sweeps:
                        continue
                    for action in candidates[i]:
                        if budgets_used[i] >= self.cfg.simulations:
                            break
                        requests.append((i, int(action)))
                        budgets_used[i] += 1
                self._forced_root_batch(roots, requests, evaluator)

            # Reduce each candidate set by approximately half using the
            # Gumbel + prior + monotone completed-Q score.
            for i in active:
                cand = candidates[i]
                if len(cand) <= 1:
                    continue
                scores = self._gumbel_scores(roots[i], gumbels[i])
                keep = max(1, (len(cand) + 1) // 2)
                order = cand[np.argsort(scores[cand])[::-1]]
                candidates[i] = order[:keep]

        # Spend every leftover simulation on the final survivor. This makes
        # root visit sums exactly equal to the requested budget, preserving a
        # useful invariant for diagnostics and tests.
        while True:
            requests = []
            any_left = False
            for i in active:
                if budgets_used[i] >= self.cfg.simulations:
                    continue
                any_left = True
                cand = candidates[i]
                if len(cand) == 0:
                    continue
                scores = self._gumbel_scores(roots[i], gumbels[i])
                action = int(cand[np.argmax(scores[cand])])
                requests.append((i, action))
                budgets_used[i] += 1
            if requests:
                self._forced_root_batch(roots, requests, evaluator)
            if not any_left:
                break

        visits = [r.N.copy() for r in roots]
        policies = []
        for i, root in enumerate(roots):
            if i not in gumbels:
                policies.append(np.zeros(ACTION_SIZE, dtype=np.float32))
                continue
            # Training target is deterministic given the completed Q and the
            # network prior: Gumbel noise chooses what to investigate, but is
            # not baked into the target itself.
            scores = self._improved_policy_scores(root)
            policies.append(_softmax_masked(scores, root.legal))
        values = [self._root_value(r) for r in roots]
        return visits, policies, values, roots

    def _forced_root_batch(self, roots: list[Node], requests, evaluator):
        """Run one simulation for each explicitly chosen root action."""
        pending_states: list[Make7State] = []
        pending_meta: list[tuple[Node, list[tuple[Node, int]]]] = []

        for root_index, root_action in requests:
            root = roots[root_index]
            state = root.state
            path: list[tuple[Node, int]] = [(root, root_action)]
            child = root.children.get(root_action)
            if child is None:
                child = Node(state.step(root_action))
                root.children[root_action] = child
            state = child.state
            node = child

            while True:
                if state.is_terminal:
                    _backup(path, state.result_for(state.player))
                    break
                if not node.expanded:
                    pending_states.append(state)
                    pending_meta.append((node, path))
                    break
                action = node.select(self.cfg.c_puct)
                path.append((node, action))
                nxt = node.children.get(action)
                if nxt is None:
                    nxt = Node(state.step(action))
                    node.children[action] = nxt
                state = nxt.state
                node = nxt

        self._evaluate_pending(pending_states, pending_meta, evaluator)

    def _evaluate_pending(self, pending_states, pending_meta, evaluator):
        if not pending_states:
            return
        batch = np.stack([s.encode() for s in pending_states])
        logits, values = evaluator(batch)
        for k, (node, path) in enumerate(pending_meta):
            _expand(node, pending_states[k], logits[k], float(values[k]))
            _backup(path, float(values[k]))

    def _q_scale(self, root: Node) -> float:
        max_n = float(root.N[root.legal].max()) if root.legal.any() else 0.0
        return (self.cfg.gumbel_c_visit + max_n) * self.cfg.gumbel_c_scale

    def _transformed_completed_q(self, root: Node) -> np.ndarray:
        """Completed root Q after the Gumbel policy-improvement transform.

        Raw Q magnitudes drift during training.  If they are inserted into
        ``log prior + scale * Q`` directly, a collapsed prior can become
        effectively impossible to overturn even when search ranks another
        action higher.  The released MCTX implementation rescales completed
        Q values to [0, 1] before applying the visit-dependent multiplier.
        Make 7 uses the same transform by default in v2.1.2.
        """
        q = root.completed_q().astype(np.float64, copy=True)
        idx = np.flatnonzero(root.legal)
        if idx.size == 0:
            return q
        if self.cfg.gumbel_rescale_q:
            lo = float(np.min(q[idx]))
            hi = float(np.max(q[idx]))
            span = hi - lo
            if span > 1e-8:
                q[idx] = (q[idx] - lo) / span
            else:
                q[idx] = 0.0
        q[idx] *= self._q_scale(root)
        return q

    def _improved_policy_scores(self, root: Node) -> np.ndarray:
        scores = np.full(ACTION_SIZE, -np.inf, dtype=np.float64)
        idx = np.flatnonzero(root.legal)
        if idx.size == 0:
            return scores
        logp = np.log(np.maximum(root.P[idx].astype(np.float64), 1e-30))
        q = self._transformed_completed_q(root)[idx]
        scores[idx] = logp + q
        return scores

    def _gumbel_scores(self, root: Node, gumbel: np.ndarray) -> np.ndarray:
        return self._improved_policy_scores(root) + gumbel

    @staticmethod
    def _root_value(root: Node) -> float:
        total = float(root.N.sum())
        return float(root.W.sum() / total) if total > 0 else 0.0


def visits_to_policy(visits: np.ndarray, temperature: float) -> np.ndarray:
    """Normalised move distribution from non-negative visit-like weights."""
    total = visits.sum()
    if total <= 0:
        return np.zeros_like(visits)
    if temperature <= 1e-3:
        out = np.zeros_like(visits)
        out[int(np.argmax(visits))] = 1.0
        return out
    scaled = visits ** (1.0 / temperature)
    return scaled / scaled.sum()
