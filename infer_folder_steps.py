from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from infer_nn_step_counter import NNStepCounter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NN step inference on folders in testdata and generate interactive HTML reports"
    )
    parser.add_argument("--test-root", type=Path, default=Path("testdata"))
    parser.add_argument("--folder-glob", type=str, default="test*", help="Folder pattern under test-root")
    parser.add_argument("--raw-file", type=str, default="Raw Data.csv", help="CSV file name inside each folder")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts_all/stepnet_tcn_best.pt"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--prob-threshold", type=float, default=0.68)
    parser.add_argument("--context-seconds", type=float, default=4.0)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts_all/folder_inference_html"))
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    return parser.parse_args()


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip()).strip("-").lower()
    return slug or "recording"


def find_recording_dirs(test_root: Path, folder_glob: str, raw_file: str) -> List[Path]:
    if not test_root.exists():
        raise FileNotFoundError(f"Test root does not exist: {test_root}")

    selected: List[Path] = []
    for folder in sorted(test_root.glob(folder_glob)):
        if folder.is_dir() and (folder / raw_file).exists():
            selected.append(folder)

    if selected:
        return selected

    fallback: List[Path] = []
    for folder in sorted(test_root.iterdir()):
        if folder.is_dir() and (folder / raw_file).exists():
            fallback.append(folder)

    return fallback


def load_raw_data(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3), dtype=float)
    t = arr[:, 0].astype(float)
    acc = arr[:, 1:4].astype(float)

    order = np.argsort(t)
    t = t[order]
    acc = acc[order]

    unique_mask = np.r_[True, np.diff(t) > 0]
    t = t[unique_mask]
    acc = acc[unique_mask]

    if t.size == 0:
        raise ValueError(f"No valid samples in {csv_path}")

    t = t - t[0]
    return t, acc


def estimate_fs(t: np.ndarray) -> float:
    if t.size < 3:
        return float("nan")
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        return float("nan")
    return 1.0 / float(np.median(dt))


def resample_to_rate(t: np.ndarray, acc: np.ndarray, target_hz: float) -> Tuple[np.ndarray, np.ndarray]:
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


def render_recording_html(
    folder_name: str,
    result: Dict[str, Any],
    time_raw: np.ndarray,
    acc_raw: np.ndarray,
    step_times: np.ndarray,
) -> str:
    amag = np.sqrt(np.sum(acc_raw * acc_raw, axis=1))
    step_amag = np.interp(step_times, time_raw, amag) if step_times.size > 0 else np.asarray([], dtype=float)

    t_json = json.dumps(time_raw.tolist())
    x_json = json.dumps(acc_raw[:, 0].tolist())
    y_json = json.dumps(acc_raw[:, 1].tolist())
    z_json = json.dumps(acc_raw[:, 2].tolist())
    amag_json = json.dumps(amag.tolist())
    st_json = json.dumps(step_times.tolist())
    sa_json = json.dumps(step_amag.tolist())

    summary_json = json.dumps(result, indent=2)

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{folder_name} - Step Inference</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }}
    .card {{ max-width: 1200px; margin: 0 auto; }}
    .meta {{ background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; margin-bottom: 14px; }}
    .plot {{ width: 100%; height: 640px; }}
    pre {{ overflow-x: auto; background: #0d1117; color: #c9d1d9; border-radius: 8px; padding: 12px; }}
  </style>
</head>
<body>
  <div class=\"card\">
    <h1>{folder_name}</h1>
    <div class=\"meta\">
      <strong>Estimated steps:</strong> {result['step_count']}<br/>
      <strong>Duration:</strong> {result['duration_seconds']:.2f} s<br/>
      <strong>Raw samples:</strong> {result['samples_raw']}<br/>
      <strong>Raw rate (estimated):</strong> {result['sample_rate_raw_hz']:.2f} Hz<br/>
      <strong>Model rate:</strong> {result['sample_rate_model_hz']:.1f} Hz
    </div>

    <div id=\"plot\" class=\"plot\"></div>

    <h2>Inference Result</h2>
    <pre>{summary_json}</pre>
  </div>

  <script>
    const t = {t_json};
    const ax = {x_json};
    const ay = {y_json};
    const az = {z_json};
    const amag = {amag_json};
    const st = {st_json};
    const sa = {sa_json};

    const traces = [
      {{ x: t, y: ax, mode: 'lines', name: 'acc_x', line: {{ width: 1.3 }} }},
      {{ x: t, y: ay, mode: 'lines', name: 'acc_y', line: {{ width: 1.3 }} }},
      {{ x: t, y: az, mode: 'lines', name: 'acc_z', line: {{ width: 1.3 }} }},
      {{ x: t, y: amag, mode: 'lines', name: 'acc_mag', line: {{ width: 1.8 }} }},
      {{
        x: st,
        y: sa,
        mode: 'markers',
        name: 'detected_steps',
        marker: {{ color: '#6a4c93', size: 8, symbol: 'x' }}
      }}
    ];

    const layout = {{
      title: 'Acceleration and Detected Step Points',
      xaxis: {{ title: 'Time (s)' }},
      yaxis: {{ title: 'Acceleration (m/s²)' }},
      legend: {{ orientation: 'h' }},
      hovermode: 'x unified',
      margin: {{ l: 55, r: 20, t: 45, b: 45 }}
    }};

    Plotly.newPlot('plot', traces, layout, {{ responsive: true, displaylogo: false }});
  </script>
</body>
</html>
"""


def render_index_html(rows: List[Dict[str, Any]]) -> str:
    row_html = "\n".join(
        f"<tr><td><a href=\"{r['report_file']}\">{r['folder_name']}</a></td>"
        f"<td>{r['step_count']}</td><td>{r['duration_seconds']:.2f}</td>"
        f"<td>{r['samples_raw']}</td><td>{r['sample_rate_raw_hz']:.2f}</td></tr>"
        for r in rows
    )

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Folder Step Inference Reports</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 1200px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 8px 10px; text-align: left; }}
    th {{ background: #f6f8fa; }}
  </style>
</head>
<body>
  <h1>Folder Step Inference Reports</h1>
  <table>
    <thead>
      <tr>
        <th>Folder</th>
        <th>Estimated Steps</th>
        <th>Duration (s)</th>
        <th>Raw Samples</th>
        <th>Raw Rate (Hz)</th>
      </tr>
    </thead>
    <tbody>
      {row_html}
    </tbody>
  </table>
</body>
</html>
"""


def main() -> None:
    args = parse_args()

    recording_dirs = find_recording_dirs(args.test_root, args.folder_glob, args.raw_file)
    if not recording_dirs:
        raise RuntimeError(
            f"No recording folder found under {args.test_root} with pattern '{args.folder_glob}' containing {args.raw_file}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    counter = NNStepCounter(
        model_path=str(args.model_path),
        device=args.device,
        prob_threshold=args.prob_threshold,
        context_seconds=args.context_seconds,
    )

    iterator = recording_dirs
    if not args.no_progress:
        iterator = tqdm(recording_dirs, desc="Infer folders")

    rows: List[Dict[str, Any]] = []

    for folder in iterator:
        csv_path = folder / args.raw_file

        t_raw, acc_raw = load_raw_data(csv_path)
        fs_raw = estimate_fs(t_raw)

        t_use, acc_use = resample_to_rate(t_raw, acc_raw, counter.sample_rate_hz)
        out = counter.run_offline({"time": t_use, "acc": acc_use})

        step_times = np.asarray(out.get("step_timestamps", np.asarray([], dtype=float)), dtype=float)
        duration = float(t_raw[-1] - t_raw[0]) if t_raw.size > 1 else 0.0

        row: Dict[str, Any] = {
            "folder_name": folder.name,
            "step_count": int(out["step_count"]),
            "duration_seconds": duration,
            "samples_raw": int(t_raw.size),
            "sample_rate_raw_hz": float(fs_raw),
            "sample_rate_model_hz": float(counter.sample_rate_hz),
            "steps_per_min": (float(out["step_count"]) / duration * 60.0) if duration > 1e-9 else 0.0,
            "source_csv": str(csv_path),
        }

        html = render_recording_html(folder.name, row, t_raw, acc_raw, step_times)
        report_file = f"{slugify(folder.name)}.html"
        (args.out_dir / report_file).write_text(html, encoding="utf-8")

        row["report_file"] = report_file
        rows.append(row)

    index_html = render_index_html(rows)
    (args.out_dir / "index.html").write_text(index_html, encoding="utf-8")
    (args.out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"Processed folders: {len(rows)}")
    print(f"Summary JSON: {args.out_dir / 'summary.json'}")
    print(f"Index HTML : {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
