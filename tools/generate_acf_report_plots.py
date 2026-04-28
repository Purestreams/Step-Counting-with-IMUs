#!/usr/bin/env python3
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

from acf_oxwalk.acf_step_counter import StepCounter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ACF plots for the report")
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


def parse_ground_truth(folder_name: str) -> Optional[int]:
    match = re.search(r"(\d+)step", folder_name)
    if match is None:
        return None
    return int(match.group(1))


def run_acf_cases(
    counter: StepCounter,
    test_root: Path,
    cases: List[str],
    raw_file: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case_name in cases:
        csv_path = test_root / case_name / raw_file
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for case '{case_name}': {csv_path}")

        t, acc = load_raw_data(csv_path)
        out = counter.run_offline({"time": t, "acc": acc})

        step_timestamps = np.asarray(out.get("step_timestamps", np.asarray([], dtype=float)), dtype=float)
        if step_timestamps.ndim != 1:
            step_timestamps = step_timestamps.reshape(-1)

        step_count = int(out.get("step_count", int(step_timestamps.size)))
        gt = parse_ground_truth(case_name)
        abs_error = None if gt is None else abs(step_count - gt)
        rel_error = None if (gt is None or gt == 0) else 100.0 * abs_error / gt

        diagnostics = out.get("diagnostics", {})
        if not isinstance(diagnostics, dict):
            diagnostics = {}

        rows.append(
            {
                "case": case_name,
                "ground_truth": gt,
                "estimated": step_count,
                "abs_error": abs_error,
                "rel_error_percent": rel_error,
                "samples": int(t.size),
                "duration_sec": float(t[-1] - t[0]) if t.size > 1 else 0.0,
                "time": t,
                "acc": acc,
                "step_timestamps": step_timestamps,
                "diagnostics": diagnostics,
            }
        )

    return rows


def plot_gt_vs_est(rows: List[Dict[str, Any]], output_path: Path, dpi: int) -> None:
    labels = [r["case"] for r in rows]
    gt_values = [r["ground_truth"] if r["ground_truth"] is not None else 0 for r in rows]
    est_values = [r["estimated"] for r in rows]

    x = np.arange(len(rows))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=150)
    ax.bar(x - width / 2, gt_values, width=width, label="Ground truth", color="#4C78A8", edgecolor="black", linewidth=0.4)
    ax.bar(x + width / 2, est_values, width=width, label="ACF estimate", color="#59A14F", edgecolor="black", linewidth=0.4)

    ax.set_xlabel("Session")
    ax.set_ylabel("Step count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_signal_with_detected_steps(rows: List[Dict[str, Any]], output_path: Path, dpi: int) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(11.2, 3.0 * n), sharex=False, squeeze=False)
    axes = axes.reshape(-1)

    for ax, row in zip(axes, rows):
        t = row["time"]
        acc = row["acc"]
        step_timestamps = row["step_timestamps"]
        amag = np.sqrt(np.sum(acc * acc, axis=1))

        ax.plot(t, amag, lw=1.2, color="black", label="|a|")
        if step_timestamps.size > 0:
            y_steps = np.interp(step_timestamps, t, amag)
            ax.scatter(step_timestamps, y_steps, s=14, color="#E15759", label="detected steps", zorder=4)

        title = f"{row['case']}: GT={row['ground_truth']}, Est={row['estimated']}, AbsErr={row['abs_error']}"
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("|a| (m/s²)")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    eval_rows = [r for r in rows if r["ground_truth"] is not None]
    abs_errors = [float(r["abs_error"]) for r in eval_rows if r["abs_error"] is not None]
    rel_errors = [float(r["rel_error_percent"]) for r in eval_rows if r["rel_error_percent"] is not None]

    clean_rows: List[Dict[str, Any]] = []
    for r in rows:
        clean_rows.append(
            {
                "case": r["case"],
                "ground_truth": r["ground_truth"],
                "estimated": r["estimated"],
                "abs_error": r["abs_error"],
                "rel_error_percent": r["rel_error_percent"],
                "samples": r["samples"],
                "duration_sec": r["duration_sec"],
                "diagnostics": r["diagnostics"],
            }
        )

    return {
        "results": clean_rows,
        "mae": float(np.mean(abs_errors)) if abs_errors else 0.0,
        "mape_percent": float(np.mean(rel_errors)) if rel_errors else 0.0,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    counter = StepCounter()
    rows = run_acf_cases(
        counter=counter,
        test_root=args.test_root,
        cases=list(args.cases),
        raw_file=args.raw_file,
    )

    gt_vs_est_png = args.out_dir / "acf_self_sessions_gt_vs_est.png"
    signal_png = args.out_dir / "acf_testcases_signal.png"
    summary_json = args.out_dir / "acf_analysis_summary.json"

    plot_gt_vs_est(rows, gt_vs_est_png, args.dpi)
    plot_signal_with_detected_steps(rows, signal_png, args.dpi)

    summary = build_summary(rows)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved plot: {gt_vs_est_png}")
    print(f"Saved plot: {signal_png}")
    print(f"Saved summary: {summary_json}")
    print(f"ACF MAE={summary['mae']:.3f}, MAPE={summary['mape_percent']:.3f}%")


if __name__ == "__main__":
    main()