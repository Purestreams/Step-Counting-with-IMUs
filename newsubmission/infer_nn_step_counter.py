from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from models.tcn_stepnet import build_model


class NNStepCounter:
    def __init__(
        self,
        model_path: str | None = None,
        checkpoint: Dict[str, Any] | None = None,
        device: str = "auto",
        prob_threshold: float = 0.68,
        min_step_interval: float = 0.33,
        context_seconds: float = 4.0,
    ):
        self._device = self._pick_device(device)

        if checkpoint is None:
            if model_path is None:
                raise ValueError("Either checkpoint or model_path must be provided")
            ckpt_path = Path(model_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            checkpoint = torch.load(str(ckpt_path), map_location=self._device)

        model_name = str(checkpoint.get("model_name", "tcn"))
        self.model = build_model(model_name)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self._device)
        self.model.eval()

        self.feature_mean = np.asarray(checkpoint.get("feature_mean", [0.0, 0.0, 0.0, 0.0]), dtype=float)
        self.feature_std = np.asarray(checkpoint.get("feature_std", [1.0, 1.0, 1.0, 1.0]), dtype=float)
        self.feature_std = np.where(self.feature_std < 1e-6, 1.0, self.feature_std)

        self.sample_rate_hz = float(checkpoint.get("sample_rate_hz", 50.0))
        self.window_size = int(checkpoint.get("window_size", 200))
        self.context_samples = int(max(self.window_size, round(context_seconds * self.sample_rate_hz)))

        self.prob_threshold = float(prob_threshold)
        self.min_step_interval = float(min_step_interval)

        self.reset()

    def _pick_device(self, flag: str) -> torch.device:
        f = flag.lower().strip()
        if f == "cpu":
            return torch.device("cpu")
        if f == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if f == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def reset(self) -> None:
        self.total_steps = 0
        self.step_timestamps = []
        self._last_step_time = None

        self._context_x = np.empty((0, 4), dtype=float)
        self._context_t = np.empty((0,), dtype=float)

        self._prob_prev2 = None
        self._prob_prev1 = None
        self._time_prev2 = None
        self._time_prev1 = None

    def _validate_input(self, data: dict) -> Tuple[np.ndarray, np.ndarray]:
        if "time" not in data or "acc" not in data:
            raise ValueError("Input must contain keys 'time' and 'acc'.")

        t = np.asarray(data["time"], dtype=float)
        acc = np.asarray(data["acc"], dtype=float)

        if t.ndim != 1:
            raise ValueError("data['time'] must be 1D.")
        if acc.ndim != 2 or acc.shape[1] != 3:
            raise ValueError("data['acc'] must have shape (N,3).")
        if t.shape[0] != acc.shape[0]:
            raise ValueError("time and acc lengths must match.")

        return t, acc

    def _can_accept_step(self, t: float) -> bool:
        if self._last_step_time is None:
            return True
        return (t - self._last_step_time) >= self.min_step_interval

    def update(self, data_chunk: dict) -> dict:
        t, acc = self._validate_input(data_chunk)

        if t.size == 0:
            return {
                "new_steps": 0,
                "total_steps": int(self.total_steps),
                "new_step_timestamps": np.asarray([], dtype=float),
                "diagnostics": {
                    "backend": "nn",
                    "threshold": float(self.prob_threshold),
                    "samples_seen": 0,
                },
            }

        amag = np.sqrt(np.sum(acc * acc, axis=1, keepdims=True))
        x_chunk = np.concatenate([acc, amag], axis=1)

        if self._context_x.size > 0:
            x_input = np.concatenate([self._context_x, x_chunk], axis=0)
            t_input = np.concatenate([self._context_t, t], axis=0)
        else:
            x_input = x_chunk
            t_input = t

        x_norm = (x_input - self.feature_mean) / self.feature_std
        x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=self._device).unsqueeze(0)

        with torch.no_grad():
            event_logits, _ = self.model(x_tensor)
            probs = torch.sigmoid(event_logits).squeeze(0).detach().cpu().numpy()

        n_new = t.shape[0]
        probs_new = probs[-n_new:]
        t_new = t_input[-n_new:]

        new_steps = []
        for i in range(n_new):
            pi = float(probs_new[i])
            ti = float(t_new[i])

            if self._prob_prev1 is not None and self._prob_prev2 is not None:
                left = self._prob_prev2
                mid = self._prob_prev1
                right = pi
                if mid > left and mid >= right and mid >= self.prob_threshold and self._time_prev1 is not None:
                    if self._can_accept_step(self._time_prev1):
                        self.total_steps += 1
                        self._last_step_time = self._time_prev1
                        self.step_timestamps.append(self._time_prev1)
                        new_steps.append(self._time_prev1)

            self._prob_prev2 = self._prob_prev1
            self._prob_prev1 = pi
            self._time_prev2 = self._time_prev1
            self._time_prev1 = ti

        self._context_x = x_input[-self.context_samples:]
        self._context_t = t_input[-self.context_samples:]

        return {
            "new_steps": int(len(new_steps)),
            "total_steps": int(self.total_steps),
            "new_step_timestamps": np.asarray(new_steps, dtype=float),
            "diagnostics": {
                "backend": "nn",
                "threshold": float(self.prob_threshold),
                "samples_seen": int(n_new),
                "mean_prob": float(np.mean(probs_new)),
            },
        }

    def run_offline(self, data: dict) -> dict:
        self.reset()
        out = self.update(data)

        step_timestamps = np.asarray(self.step_timestamps, dtype=float)
        if step_timestamps.ndim != 1:
            step_timestamps = step_timestamps.reshape(-1)

        diagnostics = dict(out.get("diagnostics", {}))
        diagnostics["mode"] = "offline"

        return {
            "step_count": int(max(0, self.total_steps)),
            "step_timestamps": step_timestamps,
            "diagnostics": diagnostics,
        }
