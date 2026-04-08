#!/usr/bin/env python3
import json
import re
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_JSON = PROJECT_ROOT / "calibration_subset10_results.json"
SELF_SUMMARY_JSON = PROJECT_ROOT / "artifacts_all_retrain" / "folder_inference_html" / "summary.json"
FIG_DIR = PROJECT_ROOT / "report" / "fig"

CALIBRATION_PNG = FIG_DIR / "calibration_subset_abs_error.png"
SELF_SESSIONS_PNG = FIG_DIR / "self_sessions_gt_vs_est.png"
ANALYSIS_SUMMARY_JSON = FIG_DIR / "analysis_summary.json"

MODALITY_ORDER = ["Hip_100Hz", "Hip_25Hz", "Wrist_100Hz", "Wrist_25Hz", "Unknown"]
MODALITY_COLORS = {
    "Hip_100Hz": "#4C78A8",
    "Hip_25Hz": "#59A14F",
    "Wrist_100Hz": "#E15759",
    "Wrist_25Hz": "#F28E2B",
    "Unknown": "#9D9DA1",
}


def infer_modality(file_path: str) -> str:
    for key in MODALITY_ORDER:
        if key != "Unknown" and key in file_path:
            return key
    return "Unknown"


def short_sample_label(file_path: str) -> str:
    p = Path(file_path)
    stem = p.stem
    parts = stem.split("_")
    return parts[0] if parts else stem


def extract_gt_from_folder_name(folder_name: str) -> int:
    match = re.search(r"-(\d+)steps?$", folder_name)
    if not match:
        raise ValueError(f"Could not infer ground-truth steps from folder name: {folder_name}")
    return int(match.group(1))


def load_calibration_data(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    per_sample = data.get("per_sample", [])
    rows = []
    for item in per_sample:
        file_path = item.get("file", "")
        abs_error = float(item.get("abs_error", 0.0))
        rows.append(
            {
                "file": file_path,
                "label": short_sample_label(file_path),
                "modality": infer_modality(file_path),
                "abs_error": abs_error,
            }
        )
    return rows


def load_self_session_data(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    rows = []
    for item in data:
        session = item.get("folder_name", "")
        gt = extract_gt_from_folder_name(session)
        est = int(item.get("step_count", 0))
        abs_error = abs(est - gt)
        rel_error_pct = (abs_error / gt * 100.0) if gt > 0 else 0.0
        rows.append(
            {
                "session": session,
                "gt": gt,
                "est": est,
                "abs_error": abs_error,
                "rel_error_pct": rel_error_pct,
            }
        )
    return rows


def plot_calibration_abs_error(rows, output_path: Path):
    labels = [r["label"] for r in rows]
    values = [r["abs_error"] for r in rows]
    modalities = [r["modality"] for r in rows]
    colors = [MODALITY_COLORS.get(m, MODALITY_COLORS["Unknown"]) for m in modalities]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    x = list(range(len(rows)))
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.4)

    ax.set_xlabel("Sample")
    ax.set_ylabel("Absolute error (steps)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    legend_handles = [
        Patch(facecolor=MODALITY_COLORS[m], edgecolor="black", linewidth=0.4, label=m)
        for m in MODALITY_ORDER
        if any(r["modality"] == m for r in rows)
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, ncol=min(4, len(legend_handles)), frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_self_sessions_gt_vs_est(rows, output_path: Path):
    sessions = [r["session"] for r in rows]
    gt_values = [r["gt"] for r in rows]
    est_values = [r["est"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    x = list(range(len(rows)))
    width = 0.36

    ax.bar([i - width / 2 for i in x], gt_values, width=width, label="Ground truth", color="#4C78A8", edgecolor="black", linewidth=0.4)
    ax.bar([i + width / 2 for i in x], est_values, width=width, label="Estimated", color="#F28E2B", edgecolor="black", linewidth=0.4)

    ax.set_xlabel("Session")
    ax.set_ylabel("Step count")
    ax.set_xticks(x)
    ax.set_xticklabels(sessions)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_summary(calibration_rows, self_rows):
    cal_errors = [r["abs_error"] for r in calibration_rows]
    self_abs_errors = [r["abs_error"] for r in self_rows]
    self_rel_errors = [r["rel_error_pct"] for r in self_rows]

    summary = {
        "calibration": {
            "mean_abs_error": mean(cal_errors) if cal_errors else 0.0,
            "median_abs_error": median(cal_errors) if cal_errors else 0.0,
            "max_abs_error": max(cal_errors) if cal_errors else 0.0,
            "min_abs_error": min(cal_errors) if cal_errors else 0.0,
            "n": len(cal_errors),
        },
        "self_sessions": {
            "mae": mean(self_abs_errors) if self_abs_errors else 0.0,
            "mape": mean(self_rel_errors) if self_rel_errors else 0.0,
            "entries": self_rows,
        },
    }
    return summary


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    calibration_rows = load_calibration_data(CALIBRATION_JSON)
    self_rows = load_self_session_data(SELF_SUMMARY_JSON)

    plot_calibration_abs_error(calibration_rows, CALIBRATION_PNG)
    plot_self_sessions_gt_vs_est(self_rows, SELF_SESSIONS_PNG)

    summary = build_summary(calibration_rows, self_rows)
    with ANALYSIS_SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote: {CALIBRATION_PNG}")
    print(f"Wrote: {SELF_SESSIONS_PNG}")
    print(f"Wrote: {ANALYSIS_SUMMARY_JSON}")
    print(
        "Calibration MAE={:.3f}, median={:.3f}, max={:.3f}, min={:.3f}, n={}".format(
            summary["calibration"]["mean_abs_error"],
            summary["calibration"]["median_abs_error"],
            summary["calibration"]["max_abs_error"],
            summary["calibration"]["min_abs_error"],
            summary["calibration"]["n"],
        )
    )
    print(
        "Self sessions MAE={:.3f}, MAPE={:.3f}%".format(
            summary["self_sessions"]["mae"],
            summary["self_sessions"]["mape"],
        )
    )


if __name__ == "__main__":
    main()
