from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class ACFParams:
    sample_rate_hz: float = 50.0
    step_hz_min: float = 1.2
    step_hz_max: float = 2.8
    detrend_seconds: float = 1.0
    smooth_seconds: float = 0.14
    prominence_std: float = 0.7
    min_distance_ratio: float = 0.65
    min_acf_peak: float = 0.11


class ACFStepCounter:
    """Orientation-robust, ACF-guided offline step counting from accelerometer data."""

    def __init__(self, params: ACFParams | None = None):
        self.params = params or ACFParams()

    @staticmethod
    def _moving_average(x: np.ndarray, window_size: int) -> np.ndarray:
        if window_size <= 1:
            return x.copy()
        kernel = np.ones(window_size, dtype=float) / float(window_size)
        return np.convolve(x, kernel, mode="same")

    @staticmethod
    def _autocorr_fft(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return np.asarray([], dtype=float)
        x0 = x - np.mean(x)
        if np.allclose(x0, 0.0):
            return np.zeros_like(x0)

        n = int(2 ** np.ceil(np.log2(max(1, 2 * x0.size - 1))))
        spectrum = np.fft.rfft(x0, n=n)
        acf = np.fft.irfft(spectrum * np.conj(spectrum), n=n)[: x0.size]
        if acf[0] <= 1e-12:
            return np.zeros_like(acf)
        return acf / acf[0]

    @staticmethod
    def _select_with_min_distance(candidates: np.ndarray, values: np.ndarray, min_distance: int) -> np.ndarray:
        if candidates.size == 0:
            return candidates
        if min_distance <= 1:
            return np.sort(candidates)

        order = np.argsort(values[candidates])[::-1]
        selected_sorted: list[int] = []
        for idx in candidates[order]:
            idx_int = int(idx)
            pos = bisect.bisect_left(selected_sorted, idx_int)

            too_close_left = pos > 0 and (idx_int - selected_sorted[pos - 1] < min_distance)
            too_close_right = pos < len(selected_sorted) and (selected_sorted[pos] - idx_int < min_distance)

            if not too_close_left and not too_close_right:
                selected_sorted.insert(pos, idx_int)

        if not selected_sorted:
            return np.asarray([], dtype=int)
        return np.asarray(selected_sorted, dtype=int)

    def _estimate_period_samples(self, signal: np.ndarray, sample_rate_hz: float) -> tuple[int, float]:
        p = self.params
        acf = self._autocorr_fft(signal)
        if acf.size < 3:
            return 0, 0.0

        lag_min = max(1, int(np.floor(sample_rate_hz / p.step_hz_max)))
        lag_max = min(acf.size - 1, int(np.ceil(sample_rate_hz / p.step_hz_min)))
        if lag_max <= lag_min:
            return 0, 0.0

        lag_slice = acf[lag_min : lag_max + 1]
        if lag_slice.size == 0:
            return 0, 0.0

        local_idx = int(np.argmax(lag_slice))
        best_lag = lag_min + local_idx
        best_peak = float(lag_slice[local_idx])
        return int(best_lag), best_peak

    @staticmethod
    def _estimate_sample_rate_hz(t: np.ndarray, fallback_hz: float) -> float:
        if t.size < 3:
            return float(fallback_hz)
        dt = np.diff(t)
        dt = dt[dt > 1e-9]
        if dt.size == 0:
            return float(fallback_hz)
        fs = 1.0 / float(np.median(dt))
        if not np.isfinite(fs) or fs < 5.0:
            return float(fallback_hz)
        return float(fs)

    def run_offline(self, data: Dict[str, np.ndarray]) -> Dict[str, object]:
        if "time" not in data or "acc" not in data:
            raise ValueError("data must contain 'time' and 'acc'")

        t = np.asarray(data["time"], dtype=float)
        acc = np.asarray(data["acc"], dtype=float)

        if t.ndim != 1:
            raise ValueError("data['time'] must be 1D")
        if acc.ndim != 2 or acc.shape[1] != 3:
            raise ValueError("data['acc'] must be shape (N, 3)")
        if t.size != acc.shape[0]:
            raise ValueError("time and acc length mismatch")

        if t.size < 5:
            return {
                "step_count": 0,
                "step_timestamps": np.asarray([], dtype=float),
                "diagnostics": {"reason": "too_few_samples"},
            }

        order = np.argsort(t)
        t = t[order]
        acc = acc[order]
        unique_mask = np.r_[True, np.diff(t) > 0]
        t = t[unique_mask]
        acc = acc[unique_mask]

        amag = np.sqrt(np.sum(acc * acc, axis=1))
        p = self.params
        fs_hz = self._estimate_sample_rate_hz(t, p.sample_rate_hz)

        detrend_win = max(1, int(round(p.detrend_seconds * fs_hz)))
        smooth_win = max(1, int(round(p.smooth_seconds * fs_hz)))

        baseline = self._moving_average(amag, detrend_win)
        signal = amag - baseline
        signal = self._moving_average(signal, smooth_win)

        period_samples, acf_peak = self._estimate_period_samples(signal, sample_rate_hz=fs_hz)
        if period_samples <= 0:
            return {
                "step_count": 0,
                "step_timestamps": np.asarray([], dtype=float),
                "diagnostics": {
                    "reason": "no_periodicity",
                    "acf_peak": float(acf_peak),
                    "period_samples": int(max(period_samples, 0)),
                    "estimated_sample_rate_hz": float(fs_hz),
                },
            }

        mean_signal = float(np.mean(signal))
        std_signal = float(np.std(signal))
        q65 = float(np.quantile(signal, 0.65))
        threshold = max(mean_signal + p.prominence_std * std_signal, q65)

        if acf_peak < p.min_acf_peak:
            threshold = max(mean_signal + 0.25 * std_signal, float(np.quantile(signal, 0.55)))

        is_peak = (signal[1:-1] > signal[:-2]) & (signal[1:-1] >= signal[2:]) & (signal[1:-1] >= threshold)
        peak_idx = np.where(is_peak)[0] + 1

        min_distance = max(1, int(round(period_samples * p.min_distance_ratio)))
        peak_idx = self._select_with_min_distance(peak_idx, signal, min_distance)

        step_timestamps = t[peak_idx].astype(float) if peak_idx.size else np.asarray([], dtype=float)
        if step_timestamps.ndim != 1:
            step_timestamps = step_timestamps.reshape(-1)

        diagnostics = {
                "acf_peak": float(acf_peak),
                "period_samples": int(period_samples),
                "threshold": threshold,
                "min_distance_samples": int(min_distance),
                "estimated_sample_rate_hz": float(fs_hz),
            }

        return {
            "step_count": int(max(0, int(step_timestamps.size))),
            "step_timestamps": np.asarray(step_timestamps, dtype=float),
            "diagnostics": diagnostics if isinstance(diagnostics, dict) else {},
        }


class StepCounter:
    """
    One step counter class for both offline and real-time usage.
    Uses ACFStepCounter as backend and conforms to the required I/O template.
    """

    def __init__(self):
        self._backend = ACFStepCounter()
        self.reset()

    def reset(self) -> None:
        self.total_steps = 0
        self._all_times = np.asarray([], dtype=float)
        self._all_acc = np.zeros((0, 3), dtype=float)
        self._step_timestamps = np.asarray([], dtype=float)

    @staticmethod
    def _validate_input(data: dict) -> tuple[np.ndarray, np.ndarray]:
        if "time" not in data or "acc" not in data:
            raise ValueError("Input must contain keys 'time' and 'acc'.")

        t = np.asarray(data["time"], dtype=float)
        acc = np.asarray(data["acc"], dtype=float)

        if t.ndim != 1:
            raise ValueError("data['time'] must be a 1D array.")
        if acc.ndim != 2 or acc.shape[1] != 3:
            raise ValueError("data['acc'] must have shape (N, 3).")
        if t.shape[0] != acc.shape[0]:
            raise ValueError("data['time'] and data['acc'] must have matching length.")

        return t, acc

    def update(self, data_chunk: dict) -> dict:
        t_chunk, acc_chunk = self._validate_input(data_chunk)

        if t_chunk.size:
            self._all_times = np.concatenate([self._all_times, t_chunk.astype(float)], axis=0)
            self._all_acc = np.concatenate([self._all_acc, acc_chunk.astype(float)], axis=0)

        out = self.run_offline({"time": self._all_times, "acc": self._all_acc})
        all_step_ts = np.asarray(out.get("step_timestamps", np.asarray([], dtype=float)), dtype=float).reshape(-1)

        prev_count = int(self.total_steps)
        new_count = int(all_step_ts.size)
        delta = max(0, new_count - prev_count)
        new_step_timestamps = all_step_ts[prev_count:new_count] if delta > 0 else np.asarray([], dtype=float)

        self.total_steps = new_count
        self._step_timestamps = all_step_ts

        diagnostics = out.get("diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = {}

        return {
            "new_steps": int(delta),
            "total_steps": int(self.total_steps),
            "new_step_timestamps": np.asarray(new_step_timestamps, dtype=float),
            "diagnostics": diagnostics,
        }

    def run_offline(self, data: dict) -> dict:
        t, acc = self._validate_input(data)
        out = self._backend.run_offline({"time": t, "acc": acc})

        step_timestamps = np.asarray(out.get("step_timestamps", np.asarray([], dtype=float)), dtype=float)
        if step_timestamps.ndim != 1:
            step_timestamps = step_timestamps.reshape(-1)

        diagnostics = out.get("diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = {}

        count_raw = out.get("step_count", step_timestamps.size)
        if isinstance(count_raw, (int, np.integer, float, np.floating)):
            step_count = int(max(0, int(count_raw)))
        else:
            step_count = int(step_timestamps.size)

        return {
            "step_count": step_count,
            "step_timestamps": step_timestamps,
            "diagnostics": diagnostics,
        }
