from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class CausalResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.drop(h)
        return F.gelu(h + x)


class CausalTCNStepNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        channels: int = 64,
        kernel_size: int = 5,
        num_blocks: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, channels, kernel_size=1)
        blocks = []
        for i in range(num_blocks):
            blocks.append(CausalResBlock(channels, kernel_size=kernel_size, dilation=2 ** i, dropout=dropout))
        self.tcn = nn.Sequential(*blocks)
        self.event_head = nn.Conv1d(channels, 1, kernel_size=1)
        self.count_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x.transpose(1, 2)
        h = self.input_proj(h)
        h = self.tcn(h)
        event_logits = self.event_head(h).squeeze(1)
        count_pred = self.count_head(h).squeeze(-1)
        return event_logits, count_pred


class TcnStepCounter:
    """
    Standalone TCN step counter using only a pretrained checkpoint.

    Expected input format for update/run_offline:
      data = {
        "time": 1D array-like of timestamps (seconds),
        "acc":  2D array-like with shape (N, 3) for [ax, ay, az]
      }
    """

    def __init__(
        self,
        model_path: str = "artifacts_all_retrain/stepnet_tcn_best.pt",
        device: str = "auto",
        prob_threshold: float = 0.68,
        min_step_interval: float = 0.33,
        context_seconds: float = 4.0,
    ):
        checkpoint_path = Path(model_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        self._device = self._pick_device(device)
        checkpoint = torch.load(str(checkpoint_path), map_location=self._device)

        self.model = CausalTCNStepNet()
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
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

    @staticmethod
    def _pick_device(flag: str) -> torch.device:
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

    @staticmethod
    def _validate_input(data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        if "time" not in data or "acc" not in data:
            raise ValueError("Input must contain keys 'time' and 'acc'.")

        t = np.asarray(data["time"], dtype=float)
        acc = np.asarray(data["acc"], dtype=float)

        if t.ndim != 1:
            raise ValueError("data['time'] must be 1D.")
        if acc.ndim != 2 or acc.shape[1] != 3:
            raise ValueError("data['acc'] must have shape (N, 3).")
        if t.shape[0] != acc.shape[0]:
            raise ValueError("time and acc lengths must match.")

        return t, acc

    @staticmethod
    def _prepare_offline_stream(t: np.ndarray, acc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    @staticmethod
    def _resample_to_rate(t: np.ndarray, acc: np.ndarray, target_hz: float) -> Tuple[np.ndarray, np.ndarray]:
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

    def _can_accept_step(self, t: float) -> bool:
        if self._last_step_time is None:
            return True
        return (t - self._last_step_time) >= self.min_step_interval

    def update(self, data_chunk: Dict) -> Dict:
        t, acc = self._validate_input(data_chunk)

        if t.size == 0:
            return {
                "new_steps": 0,
                "total_steps": int(self.total_steps),
                "new_step_timestamps": np.asarray([], dtype=float),
                "diagnostics": {
                    "backend": "tcn",
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
                "backend": "tcn",
                "threshold": float(self.prob_threshold),
                "samples_seen": int(n_new),
                "mean_prob": float(np.mean(probs_new)),
            },
        }

    def run_offline(self, data: Dict) -> Dict:
        t, acc = self._validate_input(data)
        t, acc = self._prepare_offline_stream(t, acc)
        t, acc = self._resample_to_rate(t, acc, self.sample_rate_hz)

        self.reset()
        out = self.update({"time": t, "acc": acc})

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


StepCounter = TcnStepCounter
