"""Policy-value residual network for Make 7 and its inference wrapper.

Sizing for an 11 GB card
------------------------
A 7x7 board is tiny, so depth is cheap and the real constraint is how fast
self-play can fill the replay buffer, not memory.  The presets below are
chosen so that even `large` leaves several GB free for the self-play
evaluator running in the same process.

    preset   blocks  channels  params   train batch   approx. VRAM
    small       6       96      1.1 M      1024          ~1.5 GB
    base       10      128      3.4 M      1024          ~2.5 GB
    large      15      192     11.5 M       768          ~5.5 GB

`base` is the recommended default: on a 7x7 board with 21 actions it has
ample capacity, and the extra self-play throughput is worth more than extra
depth at this scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .game import ACTION_SIZE, BOARD_H, BOARD_W

INPUT_PLANES = 10


@dataclass
class ModelConfig:
    blocks: int = 10
    channels: int = 128
    value_hidden: int = 256
    q_hidden: int = 256
    head_channels: int = 32
    coord_conv: bool = False
    residual: bool = True

    @staticmethod
    def preset(name: str) -> "ModelConfig":
        if name == "small":
            return ModelConfig(blocks=6, channels=96)
        if name == "base":
            return ModelConfig(blocks=10, channels=128)
        if name == "large":
            return ModelConfig(blocks=15, channels=192)
        if name == "shallow":
            # Plain stacked convolutions with coordinates, no residual path.
            # Four 3x3 layers already give every cell a view of the whole
            # 7x7 board, so depth beyond that buys compositional capacity
            # rather than reach.
            return ModelConfig(blocks=4, channels=128, coord_conv=True,
                               residual=False)
        if name == "shallow_wide":
            return ModelConfig(blocks=4, channels=192, coord_conv=True,
                               residual=False)
        if name == "coord_base":
            # The standard residual tower, but with coordinates added. Use
            # this to separate the effect of CoordConv from the effect of
            # removing depth.
            return ModelConfig(blocks=10, channels=128, coord_conv=True)
        raise ValueError(f"unknown preset {name!r}")


def _coordinate_planes() -> torch.Tensor:
    """Two constant planes holding the normalised row and column index.

    Convolution is translation equivariant, but almost nothing about this
    board is: the red squares sit at fixed cells, gravity makes rows mean
    different things, and a run summing to seven is cut short at the edges.
    Handing the absolute position to every layer removes the need to infer
    it from zero padding.
    """
    rows = torch.linspace(-1.0, 1.0, BOARD_H).view(1, 1, BOARD_H, 1)
    cols = torch.linspace(-1.0, 1.0, BOARD_W).view(1, 1, 1, BOARD_W)
    return torch.cat([rows.expand(1, 1, BOARD_H, BOARD_W),
                      cols.expand(1, 1, BOARD_H, BOARD_W)], dim=1)


class ConvBlock(nn.Module):
    """Plain 3x3 convolution, batch norm, ReLU."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class Make7Net(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        c, hc = cfg.channels, cfg.head_channels

        in_planes = INPUT_PLANES + (2 if cfg.coord_conv else 0)
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        block = ResidualBlock if cfg.residual else ConvBlock
        self.blocks = nn.Sequential(*[block(c) for _ in range(cfg.blocks)])
        if cfg.coord_conv:
            # Registered as a buffer so it moves with .to(device) and is
            # saved with the checkpoint.
            self.register_buffer("coords", _coordinate_planes(),
                                 persistent=False)

        self.policy_conv = nn.Sequential(
            nn.Conv2d(c, hc, 1, bias=False),
            nn.BatchNorm2d(hc),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(hc * BOARD_H * BOARD_W, ACTION_SIZE)

        self.value_conv = nn.Sequential(
            nn.Conv2d(c, hc, 1, bias=False),
            nn.BatchNorm2d(hc),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(hc * BOARD_H * BOARD_W, cfg.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.value_hidden, 1),
        )
        # Start the value head at exactly tanh(0) = 0.
        #
        # With default initialisation the pre-tanh logit can land far enough
        # from zero that tanh saturates, and a saturated tanh has almost no
        # gradient, so the head never recovers. That is not hypothetical: the
        # plain (non-residual) variant hit it on the first generation, with
        # every position scored at +1.000 and a value loss of 1.96 — exactly
        # what a constant +1 gives against zero-mean targets. Zeroing the last
        # layer costs nothing and removes the failure mode for every variant,
        # so the architectures stay comparable.
        nn.init.zeros_(self.value_fc[-1].weight)
        nn.init.zeros_(self.value_fc[-1].bias)

        # Auxiliary action-value head. This predicts Q(s,a) for all 21 actions
        # but is deliberately NOT used by MCTS selection in v2.1. Its targets
        # come only from actually visited root actions, avoiding self-training
        # on completed/unvisited Q estimates.
        self.q_conv = nn.Sequential(
            nn.Conv2d(c, hc, 1, bias=False),
            nn.BatchNorm2d(hc),
            nn.ReLU(inplace=True),
        )
        self.q_fc = nn.Sequential(
            nn.Linear(hc * BOARD_H * BOARD_W, cfg.q_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.q_hidden, ACTION_SIZE),
        )
        # Start neutral, exactly like the state-value head. A new v2.1 run
        # therefore begins with Q(s,a)=0 for every action, and a v2 checkpoint
        # can be upgraded without injecting a random action preference.
        nn.init.zeros_(self.q_fc[-1].weight)
        nn.init.zeros_(self.q_fc[-1].bias)

        # The policy head starts near-uniform for the same reason: a confident
        # random prior at initialisation sends the first generation of search
        # down arbitrary lines.
        nn.init.normal_(self.policy_fc.weight, std=0.01)
        nn.init.zeros_(self.policy_fc.bias)

    def _trunk(self, x):
        if self.cfg.coord_conv:
            x = torch.cat([x, self.coords.expand(x.shape[0], -1, -1, -1)], dim=1)
        return self.blocks(self.stem(x))

    def forward(self, x):
        """Compatibility path: policy logits and state value only.

        Keeping the original two-output signature means all v2 arena,
        validation and analysis code continues to work unchanged.
        """
        h = self._trunk(x)
        p = self.policy_fc(self.policy_conv(h).flatten(1))
        v = torch.tanh(self.value_fc(self.value_conv(h).flatten(1))).squeeze(-1)
        return p, v

    def forward_with_q(self, x):
        """Policy logits, V(s), and auxiliary Q(s,a) in [-1, 1]."""
        h = self._trunk(x)
        p = self.policy_fc(self.policy_conv(h).flatten(1))
        v = torch.tanh(self.value_fc(self.value_conv(h).flatten(1))).squeeze(-1)
        q = torch.tanh(self.q_fc(self.q_conv(h).flatten(1)))
        return p, v, q

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class NetEvaluator:
    """Adapts `Make7Net` to the callable interface `BatchedMCTS` expects.

    Optionally averages the prediction with its left-right mirror, which is
    an exact symmetry of this board and costs one extra forward pass.
    """

    def __init__(self, net: Make7Net, device: torch.device | str = "cuda",
                 use_amp: bool = True, symmetrise: bool = False):
        self.net = net
        self.device = torch.device(device)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.symmetrise = symmetrise
        # Advanced indexing requires the index tensor to live on the same
        # device as the tensor being indexed.
        self._mirror_idx = _mirror_action_index().to(self.device)

    @torch.inference_mode()
    def __call__(self, batch: np.ndarray):
        self.net.eval()
        x = torch.from_numpy(np.ascontiguousarray(batch)).to(
            self.device, non_blocking=True)
        if self.symmetrise:
            x = torch.cat([x, torch.flip(x, dims=[3])], dim=0)

        amp = torch.autocast(device_type=self.device.type,
                             dtype=torch.float16, enabled=self.use_amp)
        with amp:
            logits, values = self.net(x)
        logits = logits.float()
        values = values.float()

        if self.symmetrise:
            n = batch.shape[0]
            direct_l, mirror_l = logits[:n], logits[n:]
            mirror_l = mirror_l[:, self._mirror_idx]
            logits = 0.5 * (direct_l + mirror_l)
            values = 0.5 * (values[:n] + values[n:])

        return logits.cpu().numpy(), values.cpu().numpy()

    @torch.inference_mode()
    def evaluate_with_q(self, batch: np.ndarray):
        """Like ``__call__`` but also return the auxiliary Q(s,a) head.

        MCTS deliberately calls ``__call__`` and therefore cannot see this
        head in v2.1. The Q head is diagnostic/auxiliary only.
        """
        self.net.eval()
        x = torch.from_numpy(np.ascontiguousarray(batch)).to(
            self.device, non_blocking=True)
        if self.symmetrise:
            x = torch.cat([x, torch.flip(x, dims=[3])], dim=0)

        amp = torch.autocast(device_type=self.device.type,
                             dtype=torch.float16, enabled=self.use_amp)
        with amp:
            logits, values, q_values = self.net.forward_with_q(x)
        logits = logits.float()
        values = values.float()
        q_values = q_values.float()

        if self.symmetrise:
            n = batch.shape[0]
            direct_l, mirror_l = logits[:n], logits[n:]
            direct_q, mirror_q = q_values[:n], q_values[n:]
            mirror_l = mirror_l[:, self._mirror_idx]
            mirror_q = mirror_q[:, self._mirror_idx]
            logits = 0.5 * (direct_l + mirror_l)
            q_values = 0.5 * (direct_q + mirror_q)
            values = 0.5 * (values[:n] + values[n:])

        return (logits.cpu().numpy(), values.cpu().numpy(),
                q_values.cpu().numpy())


def _mirror_action_index() -> torch.Tensor:
    """Permutation that maps a mirrored policy back to the original frame."""
    from .game import mirror_action
    return torch.tensor([mirror_action(a) for a in range(ACTION_SIZE)],
                        dtype=torch.long)


def save_checkpoint(path, net: Make7Net, optimizer=None, generation: int = 0,
                    extra: dict | None = None) -> None:
    payload = {
        "model_config": vars(net.cfg),
        "state_dict": net.state_dict(),
        "generation": generation,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path, device="cuda", with_optimizer=False):
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**payload["model_config"])
    net = Make7Net(cfg).to(device)
    missing, unexpected = net.load_state_dict(payload["state_dict"], strict=False)
    # v2 checkpoints predate the auxiliary Q head. Missing q_* parameters are
    # expected and remain at their neutral zero-output initialisation. Any
    # other mismatch is almost certainly a real checkpoint/code incompatibility.
    bad_missing = [k for k in missing if not k.startswith("q_")]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={bad_missing}, unexpected={unexpected}")
    if with_optimizer:
        return net, payload.get("optimizer"), payload.get("generation", 0)
    return net, payload.get("generation", 0)
