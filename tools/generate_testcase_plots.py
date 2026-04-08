from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from step_counter import StepCounter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate plots for test1/test2 IMU recordings with detected steps"
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=Path("testdata"),
        help="Root folder that contains test case folders",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["test1-84step", "test2-100steps"],
        help="Test case folder names under --test-root",
    )
    parser.add_argument(
        "--raw-file",
        type=str,
        default="Raw Data.csv",
        help="CSV file name inside each test case folder",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("report/fig"),
        help="Output directory for generated plots",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts_all_retrain/stepnet_tcn_best.pt"),
        help="Path to trained model checkpoint used by StepCounter",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Inference device",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG output DPI",
    )
    return parser.parse_args()


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


def estimate_sample_rate_hz(t: np.ndarray) -> float:
    if t.size < 3:
        return float("nan")
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        return float("nan")
    return float(1.0 / np.median(dt))


def parse_ground_truth(folder_name: str) -> Optional[int]:
    match = re.search(r"(\d+)step", folder_name)
    if match is None:
        return None
    return int(match.group(1))


def make_plot(
    case_name: str,
    t: np.ndarray,
    acc: np.ndarray,
    step_timestamps: np.ndarray,
    predicted_steps: int,
    output_path: Path,
    dpi: int,
) -> Dict[str, Any]:
    ax = acc[:, 0]
    ay = acc[:, 1]
    az = acc[:, 2]
    amag = np.sqrt(np.sum(acc * acc, axis=1))

    gt = parse_ground_truth(case_name)
    abs_error = None if gt is None else abs(predicted_steps - gt)
    rel_error = None if (gt is None or gt == 0) else 100.0 * abs_error / gt

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(t, ax, lw=1.0, label="acc_x")
    axes[0].plot(t, ay, lw=1.0, label="acc_y")
    axes[0].plot(t, az, lw=1.0, label="acc_z")
    axes[0].set_ylabel("Acceleration (m/s²)")
    axes[0].set_title(f"{case_name} - 3-axis acceleration")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right", ncol=3)

    axes[1].plot(t, amag, lw=1.4, color="black", label="|a|")
    if step_timestamps.size > 0:
        step_amag = np.interp(step_timestamps, t, amag)
        axes[1].scatter(step_timestamps, step_amag, s=18, color="#7b2cbf", label="detected steps", zorder=5)

    subtitle = f"Predicted={predicted_steps}"
    if gt is not None:
        subtitle += f", GT={gt}, AbsErr={abs_error}"
        if rel_error is not None:
            subtitle += f", RelErr={rel_error:.2f}%"

    axes[1].set_title(subtitle)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("|a| (m/s²)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    duration_seconds = float(t[-1] - t[0]) if t.size > 1 else 0.0
    sample_rate_hz = estimate_sample_rate_hz(t)

    row: Dict[str, Any] = {
        "case": case_name,
        "figure": str(output_path),
        "predicted_steps": int(predicted_steps),
        "ground_truth_steps": gt,
        "abs_error": abs_error,
        "rel_error_percent": rel_error,
        "duration_seconds": duration_seconds,
        "samples": int(t.size),
        "sample_rate_hz": sample_rate_hz,
        "detected_step_events": int(step_timestamps.size),
    }
    return row


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    counter = StepCounter(model_path=str(args.model_path), device=args.device)

    summary: List[Dict[str, Any]] = []
    for case_name in args.cases:
        csv_path = args.test_root / case_name / args.raw_file
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for case '{case_name}': {csv_path}")

        t, acc = load_raw_data(csv_path)
        out = counter.run_offline({"time": t, "acc": acc})
        step_timestamps = np.asarray(out["step_timestamps"], dtype=float)
        predicted_steps = int(out["step_count"])

        fig_path = args.out_dir / f"{case_name}_plot.png"
        row = make_plot(
            case_name=case_name,
            t=t,
            acc=acc,
            step_timestamps=step_timestamps,
            predicted_steps=predicted_steps,
            output_path=fig_path,
            dpi=args.dpi,
        )
        summary.append(row)
        print(f"Saved plot: {fig_path}")

    summary_path = args.out_dir / "testcase_plot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
