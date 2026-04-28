from pathlib import Path
import logging
from typing import Any, Dict, Tuple

import numpy as np

try:
    from .bootstrap_train import BootstrapConfig, BootstrapTrainer
    from .nn_step_counter_backend import NNStepCounter
except ImportError:
    from bootstrap_train import BootstrapConfig, BootstrapTrainer
    from nn_step_counter_backend import NNStepCounter


LOGGER = logging.getLogger(__name__)


class StepCounter:
    """
    Step counter that trains model from packed phyphox data if checkpoint is missing,
    then uses the trained NN model for inference.
    """

    def __init__(self):
        checkpoint = self._bootstrap_train()
        self._backend = NNStepCounter(checkpoint=checkpoint, device="auto")

    def _bootstrap_train(self) -> Dict[str, Any]:
        this_dir = Path(__file__).resolve().parent
        repo_root = this_dir.parent

        config = BootstrapConfig(
            packed_cache=repo_root / "newsubmission" / "models" / "phyphox_data_packed.pt",
        )
        LOGGER.info("Starting fresh bootstrap training in memory")
        trainer = BootstrapTrainer(config)
        trained_checkpoint = trainer.train_fresh()
        LOGGER.info("Bootstrap training finished")
        return trained_checkpoint

    def reset(self) -> None:
        self._backend.reset()

    @property
    def total_steps(self) -> int:
        return int(self._backend.total_steps)

    @property
    def step_timestamps(self):
        return self._backend.step_timestamps

    def _validate_input(self, data: dict) -> Tuple[np.ndarray, np.ndarray]:
        if "time" not in data or "acc" not in data:
            raise ValueError("Input must contain keys 'time' and 'acc'.")

        t = np.asarray(data["time"], dtype=float)
        acc = np.asarray(data["acc"], dtype=float)

        if t.ndim != 1:
            raise ValueError("data['time'] must be a 1D array.")
        if acc.ndim != 2 or acc.shape[1] != 3:
            raise ValueError("data['acc'] must have shape (N, 3).")
        if acc.shape[0] != t.shape[0]:
            raise ValueError("data['time'] and data['acc'] must have matching length.")

        return t, acc

    def _prepare_offline_stream(self, t: np.ndarray, acc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if t.size == 0:
            return t, acc

        order = np.argsort(t)
        t = t[order]
        acc = acc[order]

        unique_mask = np.r_[True, np.diff(t) > 0]
        t = t[unique_mask]
        acc = acc[unique_mask]

        if t.size > 0:
            t = t - t[0]

        return t, acc

    def _resample_to_rate(self, t: np.ndarray, acc: np.ndarray, target_hz: float) -> Tuple[np.ndarray, np.ndarray]:
        if t.size < 2:
            return t, acc

        t0 = float(t[0])
        t1 = float(t[-1])
        if t1 <= t0:
            return t, acc

        dt = 1.0 / float(target_hz)
        t_new = np.arange(t0, t1 + 1e-12, dt, dtype=float)
        acc_new = np.column_stack(
            [
                np.interp(t_new, t, acc[:, 0]),
                np.interp(t_new, t, acc[:, 1]),
                np.interp(t_new, t, acc[:, 2]),
            ]
        )
        return t_new, acc_new

    def update(self, data_chunk: dict) -> dict:
        out = self._backend.update(data_chunk)

        new_step_timestamps = np.asarray(out.get("new_step_timestamps", np.asarray([], dtype=float)), dtype=float)
        if new_step_timestamps.ndim != 1:
            new_step_timestamps = new_step_timestamps.reshape(-1)

        diagnostics = out.get("diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = {}

        return {
            "new_steps": int(max(0, int(out.get("new_steps", 0)))),
            "total_steps": int(max(0, int(out.get("total_steps", 0)))),
            "new_step_timestamps": new_step_timestamps,
            "diagnostics": diagnostics,
        }

    def run_offline(self, data: dict) -> dict:
        t, acc = self._validate_input(data)
        t, acc = self._prepare_offline_stream(t, acc)
        t, acc = self._resample_to_rate(t, acc, self._backend.sample_rate_hz)
        out = self._backend.run_offline({"time": t, "acc": acc})

        step_timestamps = np.asarray(out.get("step_timestamps", np.asarray([], dtype=float)), dtype=float)
        if step_timestamps.ndim != 1:
            step_timestamps = step_timestamps.reshape(-1)

        diagnostics = out.get("diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = {}

        return {
            "step_count": int(max(0, int(out.get("step_count", 0)))),
            "step_timestamps": step_timestamps,
            "diagnostics": diagnostics,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    counter = StepCounter()
    LOGGER.info("StepCounter ready. Current total_steps=%d", counter.total_steps)
