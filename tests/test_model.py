"""Architecture checks. Requires torch, so run this on the training machine.

    python tests/test_model.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make7.game import ACTION_SIZE, THREE_ONLY, new_game  # noqa: E402
from make7.model import (  # noqa: E402
    INPUT_PLANES, Make7Net, ModelConfig, NetEvaluator, _coordinate_planes,
    load_checkpoint, save_checkpoint,
)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def test_coordinate_planes():
    coords = _coordinate_planes()
    check(coords.shape == (1, 2, 7, 7), "coordinate tensor is (1, 2, 7, 7)")
    check(torch.allclose(coords[0, 0, 0, :], torch.full((7,), -1.0)),
          "row plane is -1 along the top row")
    check(torch.allclose(coords[0, 0, 6, :], torch.full((7,), 1.0)),
          "row plane is +1 along the bottom row")
    check(torch.allclose(coords[0, 1, :, 0], torch.full((7,), -1.0)),
          "column plane is -1 down the left edge")
    check(torch.allclose(coords[0, 1, :, 6], torch.full((7,), 1.0)),
          "column plane is +1 down the right edge")
    # The planes must break translation symmetry, which is the whole point.
    check(not torch.allclose(coords[0, 0], torch.roll(coords[0, 0], 1, dims=0)),
          "row plane changes under a vertical shift")


def test_forward_shapes():
    for preset in ("small", "base", "shallow", "shallow_wide", "coord_base"):
        cfg = ModelConfig.preset(preset)
        net = Make7Net(cfg).eval()
        x = torch.randn(5, INPUT_PLANES, 7, 7)
        with torch.inference_mode():
            p, v = net(x)
        check(p.shape == (5, ACTION_SIZE) and v.shape == (5,),
              f"{preset}: forward gives (5, 21) and (5,)")
        check(bool((v.abs() <= 1.0).all()), f"{preset}: values lie in [-1, 1]")


def test_parameter_counts():
    print(f"  {'preset':14} {'blocks':>6} {'ch':>5} {'coord':>6} "
          f"{'resid':>6} {'params':>12}")
    counts = {}
    for preset in ("small", "base", "large", "shallow", "shallow_wide",
                   "coord_base"):
        cfg = ModelConfig.preset(preset)
        net = Make7Net(cfg)
        counts[preset] = net.num_parameters()
        print(f"  {preset:14} {cfg.blocks:6d} {cfg.channels:5d} "
              f"{str(cfg.coord_conv):>6} {str(cfg.residual):>6} "
              f"{net.num_parameters():12,}")
    check(counts["shallow"] < counts["base"],
          "shallow has fewer parameters than base")


def test_coordconv_changes_output():
    """A coordinate-aware net must distinguish a position from its vertical
    shift; without coordinates the stem alone cannot."""
    torch.manual_seed(0)
    cfg = ModelConfig.preset("shallow")
    net = Make7Net(cfg).eval()
    x = torch.zeros(1, INPUT_PLANES, 7, 7)
    x[0, 0, 2, 3] = 1.0
    y = torch.zeros(1, INPUT_PLANES, 7, 7)
    y[0, 0, 4, 3] = 1.0
    with torch.inference_mode():
        pa, _ = net(x)
        pb, _ = net(y)
    # The value head is deliberately zero-initialised, so initial values must
    # be identical. The policy head remains a valid probe of whether absolute
    # coordinates reach the trunk.
    check(not torch.allclose(pa, pb),
          "coordconv net separates two vertically shifted inputs in policy")


def test_value_head_starts_neutral():
    """Every preset must begin at value 0 with a near-uniform policy."""
    for preset in ("base", "shallow", "shallow_wide", "coord_base"):
        net = Make7Net(ModelConfig.preset(preset)).eval()
        x = torch.randn(64, INPUT_PLANES, 7, 7)
        with torch.inference_mode():
            logits, v = net(x)
        check(bool((v.abs() < 1e-5).all()),
              f"{preset}: value head outputs exactly 0 at initialisation")
        spread = logits.std(dim=1).mean().item()
        check(spread < 0.2,
              f"{preset}: policy logits start near-uniform (std {spread:.3f})")


def test_value_head_can_learn():
    """A saturated tanh has no gradient; confirm the head actually moves.

    Fits a constant target for a few steps and checks the output follows.
    """
    torch.manual_seed(0)
    for preset in ("base", "shallow"):
        net = Make7Net(ModelConfig.preset(preset)).train()
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
        x = torch.randn(128, INPUT_PLANES, 7, 7)
        target = torch.full((128,), 0.6)
        for _ in range(60):
            opt.zero_grad(set_to_none=True)
            _, v = net(x)
            loss = torch.nn.functional.mse_loss(v, target)
            loss.backward()
            opt.step()
        net.eval()
        with torch.inference_mode():
            _, v = net(x)
        err = (v.mean() - 0.6).abs().item()
        check(err < 0.15,
              f"{preset}: value head reaches the target "
              f"(mean {v.mean():.3f}, loss {loss.item():.4f})")


def test_value_head_starts_at_zero():
    """Every variant must report exactly 0 before training.

    A tanh value head that initialises near a saturation point has almost no
    gradient and can stay stuck there. The plain tower did exactly that on
    its first generation: every opening scored +1.000 and the value loss sat
    at 1.96, which is what a constant +1 gives against zero-mean targets.
    """
    for preset in ("small", "base", "shallow", "shallow_wide", "coord_base"):
        torch.manual_seed(0)
        net = Make7Net(ModelConfig.preset(preset)).eval()
        x = torch.randn(32, INPUT_PLANES, 7, 7)
        with torch.inference_mode():
            _, v = net(x)
        check(bool(v.abs().max() < 1e-6),
              f"{preset}: value head starts at 0 "
              f"(max |v| = {float(v.abs().max()):.2e})")


def test_policy_head_starts_near_uniform():
    for preset in ("base", "shallow"):
        torch.manual_seed(0)
        net = Make7Net(ModelConfig.preset(preset)).eval()
        x = torch.randn(32, INPUT_PLANES, 7, 7)
        with torch.inference_mode():
            logits, _ = net(x)
        spread = float(logits.std())
        check(spread < 0.2,
              f"{preset}: initial policy logits are nearly flat "
              f"(std = {spread:.3f})")


def test_checkpoint_round_trip(tmp=Path("/tmp/make7_arch_test.pt")):
    cfg = ModelConfig.preset("shallow")
    net = Make7Net(cfg)
    save_checkpoint(tmp, net, generation=7)
    loaded, gen = load_checkpoint(tmp, device="cpu")
    check(gen == 7, "generation survives the round trip")
    check(loaded.cfg.coord_conv and not loaded.cfg.residual,
          "coord_conv and residual flags survive the round trip")
    x = torch.randn(3, INPUT_PLANES, 7, 7)
    net.eval()
    loaded.eval()
    with torch.inference_mode():
        check(torch.allclose(net(x)[1], loaded(x)[1], atol=1e-6),
              "restored network reproduces the original outputs")
    tmp.unlink(missing_ok=True)


def test_evaluator_on_real_positions():
    cfg = ModelConfig.preset("shallow")
    net = Make7Net(cfg)
    ev = NetEvaluator(net, device="cpu", use_amp=False)
    state = new_game(THREE_ONLY)
    batch = np.stack([state.encode() for _ in range(4)])
    logits, values = ev(batch)
    check(logits.shape == (4, ACTION_SIZE), "evaluator returns (4, 21) logits")
    check(values.shape == (4,), "evaluator returns 4 values")
    check(np.allclose(values, values[0]),
          "identical positions receive identical values")


def test_speed():
    import time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    for preset in ("base", "coord_base", "shallow", "shallow_wide"):
        net = Make7Net(ModelConfig.preset(preset)).to(device).eval()
        x = torch.randn(256, INPUT_PLANES, 7, 7, device=device)
        with torch.inference_mode():
            for _ in range(5):
                net(x)
            if device == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(30):
                p, v = net(x)
                p.cpu()
            if device == "cuda":
                torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / 30
        print(f"  {preset:14} batch 256: {dt * 1000:7.2f} ms  "
              f"({256 / dt:9,.0f} pos/s)")



def test_q_head_shape_and_neutral_start():
    """Auxiliary Q head starts at zero and preserves the old forward API."""
    net = Make7Net(ModelConfig.preset("small"))
    x = torch.randn(3, INPUT_PLANES, 7, 7)
    with torch.no_grad():
        p, v = net(x)
        p2, v2, q = net.forward_with_q(x)
    check(tuple(q.shape) == (3, 21), "Q head has one value per action")
    check(torch.allclose(q, torch.zeros_like(q)), "Q head starts neutral at zero")
    check(torch.allclose(p, p2), "forward_with_q preserves policy output")
    check(torch.allclose(v, v2), "forward_with_q preserves value output")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    for fn in [
        test_coordinate_planes,
        test_forward_shapes,
        test_parameter_counts,
        test_coordconv_changes_output,
        test_value_head_starts_at_zero,
        test_policy_head_starts_near_uniform,
        test_value_head_starts_neutral,
        test_value_head_can_learn,
        test_checkpoint_round_trip,
        test_evaluator_on_real_positions,
        test_q_head_shape_and_neutral_start,
        test_speed,
    ]:
        print(f"\n[{fn.__name__}]")
        fn()
    print("\nall model tests passed")
