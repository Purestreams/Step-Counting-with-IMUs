import argparse
import time
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests

from step_counter import StepCounter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time phyphox step counting demo")
    parser.add_argument("--base-url", type=str, required=True, help="Example: http://192.168.0.42:8080")
    parser.add_argument("--time-buffer", type=str, default="time", help="phyphox time buffer name")
    parser.add_argument("--acc-x", type=str, default="ax", help="phyphox x-acc buffer name")
    parser.add_argument("--acc-y", type=str, default="ay", help="phyphox y-acc buffer name")
    parser.add_argument("--acc-z", type=str, default="az", help="phyphox z-acc buffer name")
    parser.add_argument("--poll-interval", type=float, default=0.05, help="Polling interval in seconds")
    parser.add_argument("--window-seconds", type=float, default=12.0, help="Visible curve window length")
    return parser.parse_args()


def fetch_new(base_url: str, time_buf: str, acc_bufs: List[str], last_t: Optional[float]) -> dict:
    if last_t is None:
        params = {time_buf: "", **{b: "" for b in acc_bufs}}
    else:
        params = {time_buf: str(last_t)}
        for b in acc_bufs:
            params[b] = f"{last_t}|{time_buf}"

    response = requests.get(base_url.rstrip("/") + "/get", params=params, timeout=2.0)
    response.raise_for_status()
    return response.json()


def extract_chunk(payload: dict, time_buf: str, acc_bufs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(payload["buffer"][time_buf]["buffer"], dtype=float)
    if t.size == 0:
        return t, np.empty((0, 3), dtype=float)

    ax = np.asarray(payload["buffer"][acc_bufs[0]]["buffer"], dtype=float)
    ay = np.asarray(payload["buffer"][acc_bufs[1]]["buffer"], dtype=float)
    az = np.asarray(payload["buffer"][acc_bufs[2]]["buffer"], dtype=float)

    n = min(t.size, ax.size, ay.size, az.size)
    t = t[:n]
    acc = np.stack([ax[:n], ay[:n], az[:n]], axis=1)
    return t, acc


def main() -> None:
    args = parse_args()

    counter = StepCounter()
    time_buf = args.time_buffer
    acc_bufs = [args.acc_x, args.acc_y, args.acc_z]

    last_t = None
    session_id = None

    t_hist = np.asarray([], dtype=float)
    amag_hist = np.asarray([], dtype=float)
    step_hist = np.asarray([], dtype=float)

    plt.ion()
    fig, ax = plt.subplots(figsize=(11, 5))
    (curve_line,) = ax.plot([], [], lw=1.8, label="|a| (m/s^2)")
    (step_points,) = ax.plot([], [], "ro", ms=5, label="Detected steps")

    text_handle = ax.text(
        0.02,
        0.95,
        "Total steps: 0",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    ax.set_title("Real-time Step Counting from phyphox")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration magnitude (m/s^2)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    while plt.fignum_exists(fig.number):
        try:
            payload = fetch_new(args.base_url, time_buf, acc_bufs, last_t)
        except requests.RequestException:
            time.sleep(args.poll_interval)
            continue

        status = payload.get("status", {})
        new_session = status.get("session")
        if session_id is None:
            session_id = new_session
        elif new_session is not None and new_session != session_id:
            counter.reset()
            t_hist = np.asarray([], dtype=float)
            amag_hist = np.asarray([], dtype=float)
            step_hist = np.asarray([], dtype=float)
            last_t = None
            session_id = new_session
            time.sleep(args.poll_interval)
            continue

        t, acc = extract_chunk(payload, time_buf, acc_bufs)
        if t.size == 0:
            time.sleep(args.poll_interval)
            plt.pause(0.001)
            continue

        out = counter.update({"time": t, "acc": acc})
        new_step_ts = out["new_step_timestamps"]

        if new_step_ts.size > 0:
            for step_t in new_step_ts:
                print(f"[STEP] t={float(step_t):.3f}s total={out['total_steps']}")

        amag = np.sqrt(np.sum(acc * acc, axis=1))
        t_hist = np.concatenate([t_hist, t])
        amag_hist = np.concatenate([amag_hist, amag])
        if new_step_ts.size > 0:
            step_hist = np.concatenate([step_hist, new_step_ts])

        last_t = float(t[-1])

        window_start = max(float(t_hist[-1]) - float(args.window_seconds), float(t_hist[0]))
        keep = t_hist >= window_start
        t_view = t_hist[keep]
        a_view = amag_hist[keep]

        step_keep = step_hist >= window_start
        step_view = step_hist[step_keep]

        if t_view.size > 0:
            curve_line.set_data(t_view, a_view)
            ax.set_xlim(t_view[0], t_view[-1] + 1e-6)

            if a_view.size > 0:
                pad = max(0.2, 0.15 * (a_view.max() - a_view.min() + 1e-6))
                ax.set_ylim(a_view.min() - pad, a_view.max() + pad)

            if step_view.size > 0:
                step_vals = np.interp(step_view, t_view, a_view)
                step_points.set_data(step_view, step_vals)
            else:
                step_points.set_data([], [])

        text_handle.set_text(
            f"Total steps: {out['total_steps']}\n"
            f"New steps in chunk: {out['new_steps']}\n"
            f"Last chunk samples: {t.size}"
        )

        fig.canvas.draw_idle()
        plt.pause(0.001)
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
