"""Exercise the original GIFT generator API on a tiny simulated CPU fixture.

This bounded interface smoke demonstration is not the manuscript's training
controller, Navier--Stokes benchmark, or equation-parameter recovery experiment.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

START = time.perf_counter()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import hashlib
import json
import platform

import numpy as np
import torch

from gift import GIFTGenerator, load_generator, rollout_rk4


def digest(path: Path) -> str:
    """Identify exact saved bytes without recording machine-specific paths."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; an existing path is refused.")
    parser.add_argument("--data", type=Path, default=HERE / "demo_data.npz",
                        help="NPZ with float32 state and time_derivative [batch,8,8].")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    torch.set_num_threads(1)
    torch.manual_seed(20261023)
    torch.use_deterministic_algorithms(True)
    with np.load(args.data, allow_pickle=False) as dataset:
        state = torch.from_numpy(dataset["state"].copy())
        target = torch.from_numpy(dataset["time_derivative"].copy())
    if state.ndim != 3 or state.shape[0] < 1 or state.shape[1:] != (8, 8) or target.shape != state.shape:
        raise ValueError("Expected matching [batch,8,8] state and time_derivative arrays.")
    if state.dtype != torch.float32 or target.dtype != torch.float32:
        raise ValueError("The input arrays must use float32.")
    if not torch.isfinite(state).all() or not torch.isfinite(target).all():
        raise ValueError("The supplied fields must be finite.")

    model = GIFTGenerator(cutoff=1, rank=1, state_scale=1.0).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    losses = []
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(state) - target).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite smoke loss.")
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = model(state)
        final_loss = float((prediction - target).square().mean())
        components = model.components_hat(state)
        scaled = model.components_hat(2.0 * state)
        homogeneity_errors = {
            name: float((scaled[name] - factor * components[name]).abs().max())
            for name, factor in (("constant", 1.0), ("linear", 2.0), ("quadratic", 4.0))
        }
    if not final_loss < losses[0]:
        raise RuntimeError("The bounded toy objective did not decrease.")
    if max(homogeneity_errors.values()) > 1e-5:
        raise RuntimeError("The component amplitude-response check failed.")

    checkpoint = args.output / "smoke_model.pt"
    torch.save({"format_version": 3,
                "model_configuration": model.configuration(),
                "model_state_dict": model.state_dict()}, checkpoint)
    restored = load_generator(checkpoint, device="cpu")
    with torch.no_grad():
        restored_prediction = restored(state)
    if not torch.equal(prediction, restored_prediction):
        raise RuntimeError("The saved/reloaded generator changed its output.")

    trajectory = rollout_rk4(restored, state[0], dt=0.01, steps=4).numpy()
    if trajectory.shape != (5, 8, 8) or not np.isfinite(trajectory).all():
        raise RuntimeError("The short RK4 trajectory has invalid shape/values.")
    np.savez(args.output / "smoke_arrays.npz", prediction=prediction.numpy(),
             trajectory=trajectory, losses=np.array(losses + [final_loss]))
    summary = {
        "schema": "gift.cpu-api-smoke.v1",
        "status": "passed",
        "scope": "30 Adam updates on four supplied toy derivative fields; component amplitude responses; checkpoint reload; four-step RK4. No manuscript metric or full training protocol is reproduced.",
        "environment": {
            "python": platform.python_version(), "os": platform.platform(),
            "machine_architecture": platform.machine(),
            "torch": torch.__version__, "numpy": np.__version__,
            "device": "cpu", "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "configuration": {"seed": 20261023, "field_shape": list(state.shape),
                          "updates": 30, "adam_learning_rate": 0.02,
                          "generator": model.configuration(),
                          "rk4_dt": 0.01, "rk4_steps": 4},
        "initial_mse": losses[0], "final_mse": final_loss,
        "component_homogeneity_max_abs_errors": homogeneity_errors,
        "checkpoint_reload_prediction_bitwise_equal": True,
        "trajectory_shape": list(trajectory.shape), "all_outputs_finite": True,
        "seconds_including_imports_to_summary": time.perf_counter() - START,
        "input_sha256": digest(args.data),
        "output_files": {name: {"bytes": (args.output / name).stat().st_size,
                                "sha256": digest(args.output / name)}
                         for name in ("smoke_model.pt", "smoke_arrays.npz")},
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in
                      ("status", "initial_mse", "final_mse", "trajectory_shape",
                       "checkpoint_reload_prediction_bitwise_equal",
                       "seconds_including_imports_to_summary")}))


if __name__ == "__main__":
    main()
