import argparse
import threading
import time
from typing import Dict, List, Optional

import numpy as np
import requests
from flask import Flask, jsonify, render_template_string

from step_counter import StepCounter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web realtime phyphox step counting demo")
    parser.add_argument("--base-url", type=str, required=True, help="Example: http://192.168.0.42:8080")
    parser.add_argument("--time-buffer", type=str, default="time", help="phyphox time buffer name")
    parser.add_argument("--acc-x", type=str, default="ax", help="phyphox x-acc buffer name")
    parser.add_argument("--acc-y", type=str, default="ay", help="phyphox y-acc buffer name")
    parser.add_argument("--acc-z", type=str, default="az", help="phyphox z-acc buffer name")
    parser.add_argument("--poll-interval", type=float, default=0.05, help="Polling interval in seconds")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web server host")
    parser.add_argument("--port", type=int, default=8000, help="Web server port")
    parser.add_argument("--max-points", type=int, default=6000, help="Max points kept in memory")
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


class RealtimeWebDemo:
    def __init__(self, args: argparse.Namespace):
        self.args = args
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

        self.status: Dict[str, object] = {}

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._poll_loop, daemon=True)

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
                t = np.asarray(payload["buffer"][self.time_buf]["buffer"], dtype=float)
                ax = np.asarray(payload["buffer"][self.acc_bufs[0]]["buffer"], dtype=float)
                ay = np.asarray(payload["buffer"][self.acc_bufs[1]]["buffer"], dtype=float)
                az = np.asarray(payload["buffer"][self.acc_bufs[2]]["buffer"], dtype=float)
            except (KeyError, TypeError, ValueError) as exc:
                with self.lock:
                    self.last_error = f"BufferParseError: {str(exc)}"
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
                self.last_update_wall_time = time.time()
                self.last_error = ""
                self._append_histories(t, ax, ay, az, new_step_ts)

            time.sleep(self.args.poll_interval)

    def snapshot(self) -> dict:
        with self.lock:
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
                    "threshold": float(self.counter._current_threshold()),
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
  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
  <link href=\"https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;700;800&display=swap\" rel=\"stylesheet\">
  <script src=\"https://cdn.tailwindcss.com\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js\"></script>
  <style>
    :root {
      --theme-blue: #5BCEFA;
      --theme-pink: #F5A9B8;
      --theme-white: #FFFFFF;
    }
    body {
      font-family: 'Plus Jakarta Sans', ui-sans-serif, system-ui, -apple-system, sans-serif;
      background:
        radial-gradient(1200px 600px at 0% 0%, rgba(91, 206, 250, 0.20), transparent 60%),
        radial-gradient(1200px 600px at 100% 0%, rgba(245, 169, 184, 0.20), transparent 60%),
        var(--theme-white);
    }
    .dashboard-card {
      background: rgba(255, 255, 255, 0.86);
      border: 1px solid rgba(17, 24, 39, 0.08);
      backdrop-filter: blur(8px);
      box-shadow: 0 10px 20px rgba(17, 24, 39, 0.07);
    }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .stats-grid { display: grid; grid-template-columns: repeat(2, minmax(180px, 1fr)); gap: 8px 14px; }
    .chart-wrap { position: relative; width: 100%; height: clamp(280px, 40vh, 430px); }
    .chart-wrap canvas { width: 100% !important; height: 100% !important; display: block; }
  </style>
</head>
<body class=\"min-h-screen text-black antialiased\">
  <main class=\"max-w-7xl mx-auto px-4 py-6 md:px-6 md:py-8\">
    <section class=\"dashboard-card rounded-2xl p-5 md:p-6 mb-5\">
      <div class=\"flex flex-col gap-3 md:flex-row md:items-end md:justify-between\">
        <div>
          <h1 class=\"text-2xl md:text-3xl font-extrabold tracking-tight\">Realtime Step Counting Dashboard</h1>
          <p class=\"text-sm md:text-base text-black/70 mt-1\">Streaming from phyphox with live acceleration and step analytics.</p>
        </div>
        <div class=\"flex items-center gap-3\">
          <button id=\"resetTimeBtn\" class=\"rounded-xl px-3 py-2 text-sm font-semibold border border-black/10 bg-white hover:bg-black hover:text-white transition-colors\">Reset Time</button>
          <button id=\"resetZoomBtn\" class=\"rounded-xl px-3 py-2 text-sm font-semibold border border-black/10 bg-white hover:bg-black hover:text-white transition-colors\">Reset Zoom</button>
        </div>
      </div>
    </section>

    <section class=\"grid grid-cols-1 xl:grid-cols-2 gap-4\">
      <article class=\"dashboard-card rounded-2xl p-4 md:p-5\">
        <h2 class=\"text-lg font-bold mb-3\">Acceleration Data (All buffered points)</h2>
        <div class=\"chart-wrap\">
          <canvas id=\"accChart\"></canvas>
        </div>
      </article>
      <article class=\"dashboard-card rounded-2xl p-4 md:p-5\">
        <h2 class=\"text-lg font-bold mb-3\">Magnitude + Step Markers</h2>
        <div class=\"chart-wrap\">
          <canvas id=\"magChart\"></canvas>
        </div>
      </article>
    </section>

    <section class=\"dashboard-card rounded-2xl p-4 md:p-5 mt-4\">
      <h2 class=\"text-lg font-bold mb-3\">Stats</h2>
      <div class=\"stats-grid text-sm md:text-base\" id=\"stats\"></div>
      <p class=\"text-sm md:text-base mt-3\" id=\"error\"></p>
    </section>
  </main>

<script>
Chart.defaults.font.family = "'Plus Jakarta Sans', ui-sans-serif, system-ui, -apple-system, sans-serif";
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
      { label: 'ax', data: [], borderColor: '#5BCEFA', pointRadius: 0, borderWidth: 2, parsing: false },
      { label: 'ay', data: [], borderColor: '#F5A9B8', pointRadius: 0, borderWidth: 2, parsing: false },
      { label: 'az', data: [], borderColor: '#6b7280', pointRadius: 0, borderWidth: 1.8, parsing: false },
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
      { label: '|a|', data: [], borderColor: '#5BCEFA', pointRadius: 0, borderWidth: 2, parsing: false },
      { label: 'Steps', data: [], borderColor: '#F5A9B8', backgroundColor: '#F5A9B8', showLine: false, pointRadius: 8, pointHoverRadius: 10, pointStyle: 'triangle', parsing: false },
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
