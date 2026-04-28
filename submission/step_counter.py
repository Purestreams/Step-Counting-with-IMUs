from pathlib import Path
import logging
import subprocess
import sys
from typing import Tuple

import numpy as np
from infer_nn_step_counter import NNStepCounter


LOGGER = logging.getLogger(__name__)


class StepCounter:
    """
    One step counter class for both offline and real-time usage.
    """

    def __init__(self):
        """
        Initialize the step counter.
        """
        submission_dir = Path(__file__).resolve().parent
        resolved_model = self._resolve_model_path(submission_dir)

        if not resolved_model.exists():
            self._bootstrap_train(resolved_model)

        if not resolved_model.exists():
            raise FileNotFoundError(f"Trained model checkpoint not found after bootstrap: {resolved_model}")

        self._backend = NNStepCounter(model_path=str(resolved_model), device="auto")

    def _resolve_model_path(self, submission_dir: Path) -> Path:
        candidates = [
            submission_dir / "artifacts_all_retrain" / "stepnet_tcn_best.pt",
            submission_dir.parent / "artifacts_all_retrain" / "stepnet_tcn_best.pt",
        ]
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    def _bootstrap_train(self, resolved_model: Path) -> None:
        submission_dir = Path(__file__).resolve().parent
        bootstrap_script = submission_dir / "bootstrap_train.py"
        if not bootstrap_script.exists():
            raise FileNotFoundError(f"Missing bootstrap script: {bootstrap_script}")

        LOGGER.info("Model checkpoint missing, starting bootstrap training: %s", resolved_model)

        cmd = [
            sys.executable,
            str(bootstrap_script),
            "--model-path",
            str(resolved_model),
        ]

        proc = subprocess.run(cmd, cwd=str(submission_dir), check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Bootstrap training failed with exit code {proc.returncode}")
        LOGGER.info("Bootstrap training finished with exit code %d", proc.returncode)

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
        """
        Offline processing: process a full recording.
        """
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


def _run_default_tests() -> None:
    submission_dir = Path(__file__).resolve().parent
    run_tests_script = submission_dir / "test_step_counter.py"
    if not run_tests_script.exists():
        raise FileNotFoundError(f"Missing test runner: {run_tests_script}")

    LOGGER.info("Running default step counter tests via %s", run_tests_script)
    cmd = [sys.executable, str(run_tests_script)]
    proc = subprocess.run(cmd, cwd=str(submission_dir), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"test_step_counter.py failed with exit code {proc.returncode}")
    LOGGER.info("Step counter tests completed with exit code %d", proc.returncode)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    _run_default_tests()
