import argparse
import threading
import time
from typing import Dict, List, Optional

import numpy as np
import requests
from flask import Flask, jsonify, render_template_string

from infer_nn_step_counter import NNStepCounter
from step_counter import StepCounter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web realtime phyphox step counting demo")
    parser.add_argument("--base-url", type=str, required=True, help="Example: http://192.168.0.42:8080")
    parser.add_argument("--time-buffer", type=str, default="time", help="phyphox time buffer name")
    parser.add_argument("--acc-x", type=str, default="ax", help="phyphox x-acc buffer name")
    parser.add_argument("--acc-y", type=str, default="ay", help="phyphox y-acc buffer name")
    parser.add_argument("--acc-z", type=str, default="az", help="phyphox z-acc buffer name")
    parser.add_argument("--poll-interval", type=float, default=0.05, help="Polling interval in seconds")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Web server host")
    parser.add_argument("--port", type=int, default=8000, help="Web server port")
    parser.add_argument("--max-points", type=int, default=6000, help="Max points kept in memory")
    parser.add_argument("--backend", type=str, default="heuristic", choices=["heuristic", "nn"], help="Step counter backend")
    parser.add_argument("--model-path", type=str, default=None, help="Path to NN checkpoint (.pt) when backend=nn")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="NN inference device")
    parser.add_argument("--nn-prob-threshold", type=float, default=0.68, help="Event probability threshold for NN backend")
    parser.add_argument("--nn-context-seconds", type=float, default=4.0, help="Context length for NN streaming inference")
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

def _extract_series(payload: dict, key: str) -> np.ndarray:
    return np.asarray(payload["buffer"][key]["buffer"], dtype=float)


def _find_first_key(keys: List[str], candidates: List[str]) -> Optional[str]:
    keyset = {k.lower(): k for k in keys}
    for candidate in candidates:
        found = keyset.get(candidate.lower())
        if found is not None:
            return found
    for key in keys:
        low = key.lower()
        if any(candidate in low for candidate in candidates):
            return key
    return None


class RealtimeWebDemo:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        if args.backend == "nn":
            if args.model_path is None:
                raise ValueError("--model-path is required when --backend nn")
            self.counter = NNStepCounter(
                model_path=args.model_path,
                device=args.device,
                prob_threshold=args.nn_prob_threshold,
                min_step_interval=0.30,
                context_seconds=args.nn_context_seconds,
            )
        else:
            self.counter = StepCounter()
        self.time_buf = args.time_buffer
        self.acc_bufs = [args.acc_x, args.acc_y, args.acc_z]

        self.last_t: Optional[float] = None
        self.session_id: Optional[str] = None
        self.last_error: str = ""
        self.last_update_wall_time = 0.0

        self.t_hist = np.asarray([], dtype=float)
        self.ax_hist = np.asarray([], dtype=float)
        self.ay_hist = np.asarray([], dtype=float)
        self.az_hist = np.asarray([], dtype=float)
        self.amag_hist = np.asarray([], dtype=float)

        self.step_ts_hist = np.asarray([], dtype=float)
        self.step_total_hist = np.asarray([], dtype=int)

        self.total_samples = 0
        self.total_chunks = 0
        self.last_chunk_samples = 0
        self.last_chunk_steps = 0
        self.last_diagnostics: Dict[str, object] = {}

        self.status: Dict[str, object] = {}

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._poll_loop, daemon=True)

    def _probe_and_autodetect_buffers(self) -> bool:
        try:
            response = requests.get(self.args.base_url.rstrip("/") + "/get", timeout=2.0)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException:
            return False

        return self._autodetect_buffers_from_payload(payload)

    def _autodetect_buffers_from_payload(self, payload: dict) -> bool:
        buffer = payload.get("buffer", {})
        if not isinstance(buffer, dict) or len(buffer) == 0:
            return False

        keys = list(buffer.keys())
        time_key = _find_first_key(keys, ["time", "timestamp", "t"])
        x_key = _find_first_key(keys, ["ax", "accx", "acc_x", "x"])
        y_key = _find_first_key(keys, ["ay", "accy", "acc_y", "y"])
        z_key = _find_first_key(keys, ["az", "accz", "acc_z", "z"])

        if time_key and x_key and y_key and z_key:
            self.time_buf = time_key
            self.acc_bufs = [x_key, y_key, z_key]
            return True

        return False

    def start(self) -> None:
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.worker.join(timeout=2.0)

    def _reset_for_new_session(self, new_session: Optional[str]) -> None:
        self.counter.reset()
        self.last_t = None
        self.session_id = new_session

        self.t_hist = np.asarray([], dtype=float)
        self.ax_hist = np.asarray([], dtype=float)
        self.ay_hist = np.asarray([], dtype=float)
        self.az_hist = np.asarray([], dtype=float)
        self.amag_hist = np.asarray([], dtype=float)

        self.step_ts_hist = np.asarray([], dtype=float)
        self.step_total_hist = np.asarray([], dtype=int)

        self.total_samples = 0
        self.total_chunks = 0
        self.last_chunk_samples = 0
        self.last_chunk_steps = 0
        self.last_diagnostics = {}

    def _append_histories(
        self,
        t: np.ndarray,
        ax: np.ndarray,
        ay: np.ndarray,
        az: np.ndarray,
        new_step_ts: np.ndarray,
    ) -> None:
        amag = np.sqrt(ax * ax + ay * ay + az * az)

        self.t_hist = np.concatenate([self.t_hist, t])
        self.ax_hist = np.concatenate([self.ax_hist, ax])
        self.ay_hist = np.concatenate([self.ay_hist, ay])
        self.az_hist = np.concatenate([self.az_hist, az])
        self.amag_hist = np.concatenate([self.amag_hist, amag])

        if new_step_ts.size > 0:
            self.step_ts_hist = np.concatenate([self.step_ts_hist, new_step_ts])
            new_totals = np.arange(
                self.counter.total_steps - new_step_ts.size + 1,
                self.counter.total_steps + 1,
                dtype=int,
            )
            self.step_total_hist = np.concatenate([self.step_total_hist, new_totals])

        if self.t_hist.size > self.args.max_points:
            keep = self.args.max_points
            self.t_hist = self.t_hist[-keep:]
            self.ax_hist = self.ax_hist[-keep:]
            self.ay_hist = self.ay_hist[-keep:]
            self.az_hist = self.az_hist[-keep:]
            self.amag_hist = self.amag_hist[-keep:]

        if self.step_ts_hist.size > self.args.max_points:
            keep = self.args.max_points
            self.step_ts_hist = self.step_ts_hist[-keep:]
            self.step_total_hist = self.step_total_hist[-keep:]

    def _poll_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                payload = fetch_new(self.args.base_url, self.time_buf, self.acc_bufs, self.last_t)
            except requests.RequestException as exc:
                with self.lock:
                    self.last_error = f"{type(exc).__name__}: {str(exc)}"
                time.sleep(self.args.poll_interval)
                continue

            status = payload.get("status", {})
            new_session = status.get("session")

            with self.lock:
                self.status = status
                if self.session_id is None:
                    self.session_id = new_session
                elif new_session is not None and new_session != self.session_id:
                    self._reset_for_new_session(new_session)

            try:
                t = _extract_series(payload, self.time_buf)
                ax = _extract_series(payload, self.acc_bufs[0])
                ay = _extract_series(payload, self.acc_bufs[1])
                az = _extract_series(payload, self.acc_bufs[2])
            except (KeyError, TypeError, ValueError) as exc:
                recovered = self._autodetect_buffers_from_payload(payload)
                if not recovered:
                    recovered = self._probe_and_autodetect_buffers()

                if recovered:
                    with self.lock:
                        self.last_error = (
                            f"Buffer names auto-detected: time={self.time_buf}, acc={self.acc_bufs}"
                        )
                    time.sleep(self.args.poll_interval)
                    continue

                with self.lock:
                    self.last_error = (
                        f"BufferParseError: {str(exc)}. "
                        f"Configured time='{self.time_buf}', acc={self.acc_bufs}"
                    )
                time.sleep(self.args.poll_interval)
                continue

            n = int(min(t.size, ax.size, ay.size, az.size))
            if n <= 0:
                time.sleep(self.args.poll_interval)
                continue

            t = t[:n]
            ax = ax[:n]
            ay = ay[:n]
            az = az[:n]
            acc = np.stack([ax, ay, az], axis=1)

            out = self.counter.update({"time": t, "acc": acc})
            new_step_ts = out["new_step_timestamps"]

            with self.lock:
                self.last_t = float(t[-1])
                self.total_chunks += 1
                self.total_samples += n
                self.last_chunk_samples = n
                self.last_chunk_steps = int(out["new_steps"])
                self.last_diagnostics = dict(out.get("diagnostics", {}))
                self.last_update_wall_time = time.time()
                self.last_error = ""
                self._append_histories(t, ax, ay, az, new_step_ts)

            time.sleep(self.args.poll_interval)

    def snapshot(self) -> dict:
      with self.lock:
        thr_obj = self.last_diagnostics.get("threshold", 0.0)
        threshold = float(thr_obj) if isinstance(thr_obj, (int, float, np.floating)) else 0.0
        return {
          "time": self.t_hist.tolist(),
          "ax": self.ax_hist.tolist(),
          "ay": self.ay_hist.tolist(),
          "az": self.az_hist.tolist(),
          "amag": self.amag_hist.tolist(),
          "step_timestamps": self.step_ts_hist.tolist(),
          "step_totals": self.step_total_hist.tolist(),
          "stats": {
            "total_steps": int(self.counter.total_steps),
            "total_samples": int(self.total_samples),
            "total_chunks": int(self.total_chunks),
            "last_chunk_samples": int(self.last_chunk_samples),
            "last_chunk_steps": int(self.last_chunk_steps),
            "last_sensor_time": float(self.last_t) if self.last_t is not None else None,
            "last_update_unix": float(self.last_update_wall_time),
            "threshold": threshold,
            "status": self.status,
            "last_error": self.last_error,
            "buffer_names": {
              "time": self.time_buf,
              "acc": self.acc_bufs,
            },
          },
        }


PAGE_HTML = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Realtime Step Counting Dashboard</title>
  <script src=\"https://cdn.tailwindcss.com\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js\"></script>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      margin: 20px;
      color: #111827;
      background: #ffffff;
    }
    .page { max-width: 1200px; margin: 0 auto; }
    .header-row { display: flex; justify-content: space-between; gap: 12px; align-items: flex-end; flex-wrap: wrap; }
    .button-row { display: flex; gap: 8px; }
    .btn {
      border: 1px solid #d0d7de;
      background: #f6f8fa;
      color: #111827;
      border-radius: 8px;
      padding: 8px 12px;
      font-weight: 600;
      cursor: pointer;
    }
    .btn:hover { background: #eaeef2; }
    .dashboard-card {
      background: #ffffff;
      border: 1px solid #d0d7de;
      border-radius: 8px;
      padding: 12px;
    }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .stats-grid { display: grid; grid-template-columns: repeat(2, minmax(180px, 1fr)); gap: 8px 14px; }
    .charts-grid { display: grid; grid-template-columns: 1fr; gap: 14px; }
    @media (min-width: 1200px) {
      .charts-grid { grid-template-columns: 1fr 1fr; }
    }
    .chart-wrap { position: relative; width: 100%; height: clamp(280px, 40vh, 430px); }
    .chart-wrap canvas { width: 100% !important; height: 100% !important; display: block; }
    .subtitle { color: #4b5563; margin-top: 4px; }
  </style>
</head>
<body>
  <main class=\"page\">
    <section class=\"dashboard-card\" style=\"margin-bottom:14px;\">
      <div class=\"header-row\">
        <div>
          <h1 style=\"font-size:28px; font-weight:700; margin:0;\">Realtime Step Counting Dashboard</h1>
          <p class=\"subtitle\">Streaming from phyphox with live acceleration and detected step points.</p>
        </div>
        <div class=\"button-row\">
          <button id=\"resetTimeBtn\" class=\"btn\">Reset Time</button>
          <button id=\"resetZoomBtn\" class=\"btn\">Reset Zoom</button>
        </div>
      </div>
    </section>

    <section class=\"charts-grid\">
      <article class=\"dashboard-card\">
        <h2 style=\"font-size:20px; font-weight:700; margin:0 0 10px 0;\">Acceleration Data (All buffered points)</h2>
        <div class=\"chart-wrap\">
          <canvas id=\"accChart\"></canvas>
        </div>
      </article>
      <article class=\"dashboard-card\">
        <h2 style=\"font-size:20px; font-weight:700; margin:0 0 10px 0;\">Magnitude + Step Markers</h2>
        <div class=\"chart-wrap\">
          <canvas id=\"magChart\"></canvas>
        </div>
      </article>
    </section>

    <section class=\"dashboard-card\" style=\"margin-top:14px;\">
      <h2 style=\"font-size:20px; font-weight:700; margin:0 0 10px 0;\">Stats</h2>
      <div class=\"stats-grid\" id=\"stats\"></div>
      <p style=\"margin-top:10px;\" id=\"error\"></p>
    </section>
  </main>

<script>
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
Chart.defaults.font.size = 13;
Chart.defaults.color = '#111827';

const accCtx = document.getElementById('accChart').getContext('2d');
const magCtx = document.getElementById('magChart').getContext('2d');
const resetTimeBtn = document.getElementById('resetTimeBtn');
const resetZoomBtn = document.getElementById('resetZoomBtn');

const zoomOptions = {
  pan: {
    enabled: true,
    mode: 'xy',
    modifierKey: 'shift'
  },
  zoom: {
    wheel: { enabled: true },
    pinch: { enabled: true },
    drag: {
      enabled: true,
      borderColor: 'rgba(17, 24, 39, 0.8)',
      borderWidth: 1,
      backgroundColor: 'rgba(91, 206, 250, 0.15)'
    },
    mode: 'xy'
  }
};

let timeOffset = 0;
let latestRawTime = 0;

const accChart = new Chart(accCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: 'ax', data: [], borderColor: '#1f77b4', pointRadius: 0, borderWidth: 1.6, parsing: false },
      { label: 'ay', data: [], borderColor: '#6b7280', pointRadius: 0, borderWidth: 1.3, parsing: false },
      { label: 'az', data: [], borderColor: '#9ca3af', pointRadius: 0, borderWidth: 1.3, parsing: false },
    ]
  },
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      zoom: zoomOptions
    },
    scales: {
      x: {
        type: 'linear',
        title: { display: true, text: 'Time (s)' },
        ticks: { callback: (value) => Number(value).toFixed(0) }
      },
      y: { title: { display: true, text: 'm/s²' } }
    }
  }
});

const magChart = new Chart(magCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: '|a|', data: [], borderColor: '#1f77b4', pointRadius: 0, borderWidth: 1.8, parsing: false },
      { label: 'Steps', data: [], borderColor: '#d62728', backgroundColor: '#d62728', showLine: false, pointRadius: 6, pointHoverRadius: 8, pointStyle: 'crossRot', parsing: false },
    ]
  },
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      zoom: zoomOptions
    },
    scales: {
      x: {
        type: 'linear',
        title: { display: true, text: 'Time (s)' },
        ticks: { callback: (value) => Number(value).toFixed(0) }
      },
      y: { title: { display: true, text: 'm/s²' } }
    }
  }
});

function interpY(x, xs, ys) {
  if (xs.length === 0) return null;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1) {
    const mid = Math.floor((lo + hi) / 2);
    if (xs[mid] <= x) lo = mid; else hi = mid;
  }
  const x0 = xs[lo], x1 = xs[hi], y0 = ys[lo], y1 = ys[hi];
  const r = (x - x0) / Math.max(1e-9, (x1 - x0));
  return y0 + r * (y1 - y0);
}

function renderStats(stats) {
  const el = document.getElementById('stats');
  const rows = [
    ['total_steps', stats.total_steps],
    ['total_samples', stats.total_samples],
    ['total_chunks', stats.total_chunks],
    ['last_chunk_samples', stats.last_chunk_samples],
    ['last_chunk_steps', stats.last_chunk_steps],
    ['threshold', Number(stats.threshold).toFixed(4)],
    ['last_sensor_time', stats.last_sensor_time],
    ['time_offset', Number(timeOffset).toFixed(2)],
    ['session', stats.status && stats.status.session],
    ['measuring', stats.status && stats.status.measuring],
    ['buffers', JSON.stringify(stats.buffer_names)],
  ];
  el.innerHTML = rows.map(([k, v]) => `<div><b>${k}</b>: ${v}</div>`).join('');
  const err = document.getElementById('error');
  err.textContent = stats.last_error ? `last_error: ${stats.last_error}` : '';
}

function toDisplaySeries(t, values) {
  const out = [];
  for (let i = 0; i < t.length && i < values.length; i++) {
    const tx = t[i] - timeOffset;
    if (Number.isFinite(tx) && tx >= 0) {
      out.push({ x: tx, y: values[i] });
    }
  }
  return out;
}

resetTimeBtn.addEventListener('click', () => {
  timeOffset = latestRawTime;
});

resetZoomBtn.addEventListener('click', () => {
  if (typeof accChart.resetZoom === 'function') {
    accChart.resetZoom();
  }
  if (typeof magChart.resetZoom === 'function') {
    magChart.resetZoom();
  }
});

async function tick() {
  try {
    const res = await fetch('/api/state');
    const j = await res.json();

    const t = j.time;
    const ax = j.ax;
    const ay = j.ay;
    const az = j.az;
    const amag = j.amag;
    if (t.length > 0) {
      latestRawTime = t[t.length - 1];
    }

    accChart.data.labels = [];
    accChart.data.datasets[0].data = toDisplaySeries(t, ax);
    accChart.data.datasets[1].data = toDisplaySeries(t, ay);
    accChart.data.datasets[2].data = toDisplaySeries(t, az);
    accChart.update('none');

    magChart.data.labels = [];
    magChart.data.datasets[0].data = toDisplaySeries(t, amag);
    const stepPoints = j.step_timestamps
      .map(st => ({ x: st - timeOffset, y: interpY(st, t, amag) }))
      .filter(p => Number.isFinite(p.x) && p.x >= 0 && Number.isFinite(p.y));
    magChart.data.datasets[1].data = stepPoints;
    magChart.update('none');

    renderStats(j.stats);
  } catch (e) {
    document.getElementById('error').textContent = 'dashboard fetch error: ' + e;
  }
}

setInterval(tick, 250);
tick();
</script>
</body>
</html>
"""


def build_app(demo: RealtimeWebDemo) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(PAGE_HTML)

    @app.route("/api/state")
    def api_state():
        return jsonify(demo.snapshot())

    return app


def main() -> None:
    args = parse_args()
    demo = RealtimeWebDemo(args)
    app = build_app(demo)

    demo.start()
    try:
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    finally:
        demo.stop()


if __name__ == "__main__":
    main()
