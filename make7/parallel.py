"""Multiprocess self-play with a central batching inference server.

Why this shape
--------------
On a 1080 Ti a forward pass over this network costs about the same for a
batch of 64 as for a batch of 256 — it is latency bound, not throughput
bound.  That single fact rules out the obvious parallelisation.  Giving each
worker its own copy of the network splits one batch of 256 into four of 64,
so the number of forward passes quadruples while each still costs 11 ms, and
the GPU serialises them.  Self-play gets *slower*.

So the workers here hold no network at all.  They run the tree search, which
is the part that is actually CPU bound, and post leaf positions to the parent
through shared memory.  The parent concatenates whatever has arrived from
every worker into one batch, runs a single forward pass, and writes the
results back.  Batch size stays large, forward-pass count stays the same as
the single-process path, and only the Python tree work is parallelised.

Consequences worth knowing:
  * VRAM use does not grow with worker count — there is only ever one model
    and one CUDA context.
  * Wall time per generation approaches max(GPU time, Python time / workers)
    rather than their sum.
  * Messages on the queues are three-element tuples; the position data never
    gets pickled.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import replace
from multiprocessing import shared_memory

import numpy as np

from .game import ACTION_SIZE
from .selfplay import SelfPlayConfig, generate_games

PLANES = 10
BOARD = 7
STATE_FLOATS = PLANES * BOARD * BOARD


class _SharedSlot:
    """One worker's request and response buffers."""

    def __init__(self, max_batch: int, create: bool = True, names=None):
        self.max_batch = max_batch
        req_bytes = max_batch * STATE_FLOATS * 4
        pol_bytes = max_batch * ACTION_SIZE * 4
        val_bytes = max_batch * 4

        if create:
            self.req_shm = shared_memory.SharedMemory(create=True, size=req_bytes)
            self.pol_shm = shared_memory.SharedMemory(create=True, size=pol_bytes)
            self.val_shm = shared_memory.SharedMemory(create=True, size=val_bytes)
        else:
            self.req_shm = shared_memory.SharedMemory(name=names[0])
            self.pol_shm = shared_memory.SharedMemory(name=names[1])
            self.val_shm = shared_memory.SharedMemory(name=names[2])

        self.req = np.ndarray((max_batch, PLANES, BOARD, BOARD),
                              dtype=np.float32, buffer=self.req_shm.buf)
        self.pol = np.ndarray((max_batch, ACTION_SIZE),
                              dtype=np.float32, buffer=self.pol_shm.buf)
        self.val = np.ndarray((max_batch,), dtype=np.float32,
                              buffer=self.val_shm.buf)

    @property
    def names(self):
        return (self.req_shm.name, self.pol_shm.name, self.val_shm.name)

    def close(self):
        for shm in (self.req_shm, self.pol_shm, self.val_shm):
            shm.close()

    def unlink(self):
        for shm in (self.req_shm, self.pol_shm, self.val_shm):
            try:
                shm.unlink()
            except FileNotFoundError:
                pass


class _RemoteEvaluator:
    """Drop-in replacement for `NetEvaluator` that defers to the parent."""

    def __init__(self, wid, slot, request_q, response_q):
        self.wid = wid
        self.slot = slot
        self.request_q = request_q
        self.response_q = response_q

    def __call__(self, batch: np.ndarray):
        """Evaluate an arbitrary-size batch through this worker's slot.

        The shared-memory slot is sized for the worker's number of concurrent
        games.  Ordinary PUCT never asks for more leaves than that in one
        evaluator call, but Gumbel sequential halving can deliberately expand
        several root actions per live game at once (e.g. 14 x 171 positions at
        the opening).  Split such bursts into slot-sized requests instead of
        requiring an enormous shared slot or failing outright.

        A worker waits for each chunk before submitting the next one.  This
        preserves the parent's one-outstanding-request-per-worker protocol and
        still lets the parent concatenate one chunk from every active worker
        into a large GPU batch.
        """
        n = int(batch.shape[0])
        if n == 0:
            return (np.zeros((0, ACTION_SIZE), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32))

        logits = np.empty((n, ACTION_SIZE), dtype=np.float32)
        values = np.empty((n,), dtype=np.float32)
        cap = self.slot.max_batch

        for start in range(0, n, cap):
            stop = min(start + cap, n)
            m = stop - start
            self.slot.req[:m] = batch[start:stop]
            self.request_q.put((self.wid, m))
            self.response_q.get()
            logits[start:stop] = self.slot.pol[:m]
            values[start:stop] = self.slot.val[:m]

        return logits, values


def _worker_main(wid, names, max_batch, task_q, request_q, response_q, result_q):
    slot = _SharedSlot(max_batch, create=False, names=names)
    evaluator = _RemoteEvaluator(wid, slot, request_q, response_q)
    try:
        while True:
            task = task_q.get()
            if task is None:
                break
            cfg, seed = task
            samples, stats = generate_games(
                evaluator, cfg, np.random.default_rng(seed))
            request_q.put((wid, -1))          # tell the server this worker idled

            if samples:
                states = np.stack([s.state for s in samples])
                policies = np.stack([s.policy for s in samples])
                values = np.asarray([s.value for s in samples], dtype=np.float32)
                q_targets = np.stack([s.q_target for s in samples])
                q_weights = np.stack([s.q_weight for s in samples])
            else:
                states = np.zeros((0, PLANES, BOARD, BOARD), dtype=np.float32)
                policies = np.zeros((0, ACTION_SIZE), dtype=np.float32)
                values = np.zeros((0,), dtype=np.float32)
                q_targets = np.zeros((0, ACTION_SIZE), dtype=np.float32)
                q_weights = np.zeros((0, ACTION_SIZE), dtype=np.float32)
            result_q.put((wid, states, policies, values, q_targets, q_weights, stats))
    finally:
        slot.close()


class ParallelSelfPlay:
    """Persistent worker pool. Create once, call `run` per generation."""

    def __init__(self, n_workers: int, max_batch: int, start_method: str = "spawn"):
        self.n_workers = n_workers
        self.max_batch = max_batch
        self.ctx = mp.get_context(start_method)

        self.slots = [_SharedSlot(max_batch) for _ in range(n_workers)]
        self.task_qs = [self.ctx.Queue() for _ in range(n_workers)]
        self.response_qs = [self.ctx.Queue() for _ in range(n_workers)]
        self.request_q = self.ctx.Queue()
        self.result_q = self.ctx.Queue()

        self.procs = []
        for wid in range(n_workers):
            p = self.ctx.Process(
                target=_worker_main,
                args=(wid, self.slots[wid].names, max_batch, self.task_qs[wid],
                      self.request_q, self.response_qs[wid], self.result_q),
                daemon=True,
            )
            p.start()
            self.procs.append(p)

    # ------------------------------------------------------------------ #

    def run(self, evaluator, cfg: SelfPlayConfig, seed: int,
            gather_timeout: float = 0.0):
        """Play `cfg.games_per_generation` games spread over the pool.

        `evaluator` runs in this process and is the only thing that touches
        the GPU.

        `gather_timeout` is how long the server blocks, after draining every
        request already queued, hoping a straggler will post too.  This is a
        genuine trade-off rather than a free win:

          * 0.0 serves whatever has arrived. Batches are smaller, but no time
            is ever spent waiting. Usually the right default.
          * A large value (say 0.5) turns the server into a barrier: it waits
            for every active worker, which maximises batch size but makes
            every worker run at the pace of the slowest one at each step.

        Workers drift apart because they sit at different move numbers with
        different numbers of live games, so a small non-zero timeout tends to
        be the worst of both: it rarely completes the batch and pays the full
        wait almost every time.
        """
        per_worker = _split(cfg.games_per_generation, self.n_workers)
        batch_per_worker = min(cfg.parallel_games, self.max_batch)

        # Allocate forced games globally, then split them across workers.
        # The cumulative offset means extras in a non-divisible 2A/2B/2C/2D
        # schedule rotate across workers instead of always favouring 2A.
        forced_total = 0
        if cfg.forced_openings and cfg.forced_opening_fraction > 0:
            forced_total = int(round(
                cfg.games_per_generation *
                float(np.clip(cfg.forced_opening_fraction, 0.0, 1.0))))
        forced_per_worker = _split(forced_total, self.n_workers)
        forced_offset = int(cfg.forced_opening_offset)

        idle = 0
        self._evaluated = 0
        for wid, n_games in enumerate(per_worker):
            if n_games == 0:
                idle += 1
                continue
            n_forced = min(forced_per_worker[wid], n_games)
            worker_cfg = replace(
                cfg,
                games_per_generation=n_games,
                parallel_games=min(batch_per_worker, n_games),
                forced_opening_count=n_forced,
                forced_opening_offset=forced_offset,
            )
            forced_offset += n_forced
            self.task_qs[wid].put((worker_cfg, seed * 1000 + wid))

        active = self.n_workers - idle
        pending: dict[int, int] = {}
        forwards = 0
        gpu_seconds = 0.0

        while active > 0:
            wid, n = self.request_q.get()
            if n == -1:
                active -= 1
                continue
            pending[wid] = n

            # Always take everything already sitting in the queue; this costs
            # nothing and is what makes a zero timeout worthwhile.
            while len(pending) < active:
                try:
                    wid2, n2 = self.request_q.get_nowait()
                except queue.Empty:
                    break
                if n2 == -1:
                    active -= 1
                else:
                    pending[wid2] = n2

            if gather_timeout > 0:
                deadline = time.perf_counter() + gather_timeout
                while len(pending) < active:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        wid2, n2 = self.request_q.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if n2 == -1:
                        active -= 1
                    else:
                        pending[wid2] = n2

            if pending:
                t0 = time.perf_counter()
                self._serve(evaluator, pending)
                gpu_seconds += time.perf_counter() - t0
                forwards += 1
                pending = {}

        samples_states, samples_pol, samples_val = [], [], []
        samples_q, samples_qw = [], []
        merged = {"games": 0, "green_wins": 0, "yellow_wins": 0, "draws": 0,
                  "total_plies": 0, "q_supervised_actions": 0, "q_positions": 0,
                  "forced_games": 0,
                  "organic_games": 0, "organic_green_wins": 0,
                  "organic_yellow_wins": 0, "organic_draws": 0,
                  "forced_green_wins": 0, "forced_yellow_wins": 0,
                  "forced_draws": 0, "organic_root_policy_count": 0,
                  "policy_entropy_raw_sum": 0.0,
                  "policy_entropy_train_sum": 0.0,
                  "policy_entropy_count": 0,
                  "opening_counts": np.zeros(ACTION_SIZE, dtype=np.int64),
                  "organic_opening_counts": np.zeros(ACTION_SIZE, dtype=np.int64),
                  "forced_opening_counts": np.zeros(ACTION_SIZE, dtype=np.int64),
                  "forced_result_games": np.zeros(ACTION_SIZE, dtype=np.int64),
                  "forced_green_wins_by_opening": np.zeros(ACTION_SIZE, dtype=np.int64),
                  "forced_draws_by_opening": np.zeros(ACTION_SIZE, dtype=np.int64),
                  "organic_root_policy_raw_sum": np.zeros(ACTION_SIZE, dtype=np.float64),
                  "organic_root_policy_train_sum": np.zeros(ACTION_SIZE, dtype=np.float64)}
        for _ in range(self.n_workers - idle):
            _wid, states, policies, values, q_targets, q_weights, stats = self.result_q.get()
            samples_states.append(states)
            samples_pol.append(policies)
            samples_val.append(values)
            samples_q.append(q_targets)
            samples_qw.append(q_weights)
            for key in ("games", "green_wins", "yellow_wins", "draws",
                        "total_plies", "q_supervised_actions", "q_positions",
                        "forced_games", "organic_games", "organic_green_wins",
                        "organic_yellow_wins", "organic_draws",
                        "forced_green_wins", "forced_yellow_wins",
                        "forced_draws", "organic_root_policy_count",
                        "policy_entropy_raw_sum", "policy_entropy_train_sum",
                        "policy_entropy_count"):
                merged[key] += stats[key]
            merged["opening_counts"] += stats["opening_counts"]
            merged["organic_opening_counts"] += stats["organic_opening_counts"]
            merged["forced_opening_counts"] += stats["forced_opening_counts"]
            merged["forced_result_games"] += stats["forced_result_games"]
            merged["forced_green_wins_by_opening"] += stats["forced_green_wins_by_opening"]
            merged["forced_draws_by_opening"] += stats["forced_draws_by_opening"]
            merged["organic_root_policy_raw_sum"] += stats["organic_root_policy_raw_sum"]
            merged["organic_root_policy_train_sum"] += stats["organic_root_policy_train_sum"]

        arrays = (np.concatenate(samples_states) if samples_states else None,
                  np.concatenate(samples_pol) if samples_pol else None,
                  np.concatenate(samples_val) if samples_val else None,
                  np.concatenate(samples_q) if samples_q else None,
                  np.concatenate(samples_qw) if samples_qw else None)
        merged["forward_passes"] = forwards
        merged["gpu_seconds"] = gpu_seconds
        merged["mean_batch"] = self._evaluated / max(forwards, 1)
        return arrays, merged

    def _serve(self, evaluator, pending: dict[int, int]) -> None:
        order = list(pending.items())
        batch = np.concatenate(
            [self.slots[wid].req[:n] for wid, n in order], axis=0)
        logits, values = evaluator(batch)
        self._evaluated += batch.shape[0]
        offset = 0
        for wid, n in order:
            slot = self.slots[wid]
            slot.pol[:n] = logits[offset:offset + n]
            slot.val[:n] = values[offset:offset + n]
            offset += n
            self.response_qs[wid].put(1)

    def close(self):
        for q in self.task_qs:
            q.put(None)
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        for slot in self.slots:
            slot.close()
            slot.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _split(total: int, parts: int) -> list[int]:
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]
