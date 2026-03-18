import numpy as np
from typing import Tuple


class StepCounter:
    """
    One step counter class for both offline and real-time usage.
    """

    def __init__(self):
        """
        Initialize the step counter.
        """
        self._alpha = 0.25
        self._stats_alpha = 0.01
        self._threshold_std_gain = 0.35
        self._min_threshold = 0.03
        self._min_step_interval = 0.25
        self._max_step_interval = 2.0
        self._prominence_gain = 0.05
        self._min_prominence = 0.01
        self._init_warmup_samples = 200
        self.reset()

    def reset(self) -> None:
        """
        Reset internal state such as buffers and cumulative count.
        After reset(), total_steps should be 0.
        """
        self.total_steps = 0
        self.step_timestamps = []

        self._ema_prev = None
        self._det_prev2 = None
        self._det_prev1 = None
        self._time_prev2 = None
        self._time_prev1 = None

        self._running_mean = 0.0
        self._running_var = 0.0
        self._stats_initialized = False
        self._sample_count = 0

        self._last_step_time = None

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

    def _update_running_stats(self, x: float) -> float:
        if not self._stats_initialized:
            self._running_mean = x
            self._running_var = 1e-6
            self._stats_initialized = True
            return 0.0

        delta = x - self._running_mean
        self._running_mean += self._stats_alpha * delta
        centered = x - self._running_mean
        self._running_var = (1.0 - self._stats_alpha) * self._running_var + self._stats_alpha * (centered ** 2)
        return centered

    def _current_threshold(self) -> float:
        std = float(np.sqrt(max(self._running_var, 1e-8)))
        adaptive = self._threshold_std_gain * std
        return float(max(self._min_threshold, adaptive))

    def _can_accept_step(self, candidate_t: float) -> bool:
        if self._last_step_time is None:
            return True

        dt = candidate_t - self._last_step_time
        if dt < self._min_step_interval:
            return False

        if dt > self._max_step_interval:
            return True

        return True

    def _process_sample(self, t: float, acc_xyz: np.ndarray) -> float:
        amag = float(np.sqrt(np.dot(acc_xyz, acc_xyz)))

        if self._ema_prev is None:
            smoothed = amag
        else:
            smoothed = self._alpha * amag + (1.0 - self._alpha) * self._ema_prev
        self._ema_prev = smoothed

        centered = self._update_running_stats(smoothed)
        self._sample_count += 1

        return centered

    def update(self, data_chunk: dict) -> dict:
        """
        Real-time update: process a chunk of new samples.
        """
        t, acc = self._validate_input(data_chunk)

        new_steps_ts = []

        if t.size == 0:
            return {
                "new_steps": 0,
                "total_steps": int(self.total_steps),
                "new_step_timestamps": np.asarray([], dtype=float),
                "diagnostics": {
                    "threshold": float(self._current_threshold()) if self._stats_initialized else 0.0,
                    "running_mean": float(self._running_mean),
                    "running_std": float(np.sqrt(max(self._running_var, 0.0))),
                    "samples_seen": int(self._sample_count),
                },
            }

        for idx in range(t.shape[0]):
            ti = float(t[idx])
            si = self._process_sample(ti, acc[idx])

            if self._det_prev1 is not None and self._det_prev2 is not None:
                mid_val = self._det_prev1
                left_val = self._det_prev2
                right_val = si

                is_peak = mid_val > left_val and mid_val >= right_val

                if is_peak and self._sample_count >= self._init_warmup_samples:
                    thr = self._current_threshold()
                    std = float(np.sqrt(max(self._running_var, 1e-8)))
                    prominence = mid_val - max(left_val, right_val)
                    prominence_thr = max(self._min_prominence, self._prominence_gain * std)

                    if (
                        mid_val >= thr
                        and prominence >= prominence_thr
                        and self._time_prev1 is not None
                        and self._can_accept_step(self._time_prev1)
                    ):
                        self.total_steps += 1
                        self._last_step_time = self._time_prev1
                        self.step_timestamps.append(self._time_prev1)
                        new_steps_ts.append(self._time_prev1)

            self._det_prev2 = self._det_prev1
            self._det_prev1 = si
            self._time_prev2 = self._time_prev1
            self._time_prev1 = ti

        return {
            "new_steps": int(len(new_steps_ts)),
            "total_steps": int(self.total_steps),
            "new_step_timestamps": np.asarray(new_steps_ts, dtype=float),
            "diagnostics": {
                "threshold": float(self._current_threshold()),
                "running_mean": float(self._running_mean),
                "running_std": float(np.sqrt(max(self._running_var, 0.0))),
                "samples_seen": int(self._sample_count),
            },
        }

    def run_offline(self, data: dict) -> dict:
        """
        Offline processing: process a full recording.
        """
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
