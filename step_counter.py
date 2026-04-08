from pathlib import Path
from typing import Tuple

import numpy as np
from infer_nn_step_counter import NNStepCounter


class StepCounter:
    """
    One step counter class for both offline and real-time usage.
    """

    def __init__(
        self,
        model_path: str = "artifacts_all_retrain/stepnet_tcn_best.pt",
        device: str = "auto",
        prob_threshold: float = 0.68,
        min_step_interval: float = 0.33,
        context_seconds: float = 4.0,
    ):
        """
        Initialize the NN-backed step counter.
        """
        resolved_model = Path(model_path)
        if not resolved_model.exists():
            raise FileNotFoundError(
                f"Trained model checkpoint not found: {resolved_model}. "
                "Train a model first or pass a valid model_path to StepCounter(...)."
            )

        self._backend = NNStepCounter(
            model_path=str(resolved_model),
            device=device,
            prob_threshold=prob_threshold,
            min_step_interval=min_step_interval,
            context_seconds=context_seconds,
        )

    def reset(self) -> None:
        """
        Reset internal state such as buffers and cumulative count.
        After reset(), total_steps should be 0.
        """
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
        """
        Real-time update: process a chunk of new samples.
        """
        return self._backend.update(data_chunk)

    def run_offline(self, data: dict) -> dict:
        """
        Offline processing: process a full recording.
        """
        t, acc = self._validate_input(data)
        t, acc = self._prepare_offline_stream(t, acc)
        t, acc = self._resample_to_rate(t, acc, self._backend.sample_rate_hz)
        return self._backend.run_offline({"time": t, "acc": acc})
