from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from step_counter import StepCounter


def load_raw_data(csv_path: Path):
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3), dtype=float)
    t = arr[:, 0].astype(float)
    acc = arr[:, 1:4].astype(float)

    order = np.argsort(t)
    t = t[order]
    acc = acc[order]

    unique_mask = np.r_[True, np.diff(t) > 0]
    t = t[unique_mask]
    acc = acc[unique_mask]

    if t.size:
        t = t - t[0]

    return t, acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run submission StepCounter on test1/test2 datasets")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root folder containing testdata/",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts_all_retrain/stepnet_tcn_best.pt"),
        help="Model checkpoint path relative to submission folder",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("test_results.json"),
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cases = [
        ("test1-84step", 84),
        ("test2-100steps", 100),
    ]

    counter = StepCounter(model_path=str(args.model_path), device="auto")

    results = []
    for case_name, gt_steps in cases:
        csv_path = args.project_root / "testdata" / case_name / "Raw Data.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing test CSV: {csv_path}")

        t, acc = load_raw_data(csv_path)
        out = counter.run_offline({"time": t, "acc": acc})

        est_steps = int(out["step_count"])
        abs_err = abs(est_steps - gt_steps)
        rel_err = (abs_err / gt_steps * 100.0) if gt_steps else 0.0

        results.append(
            {
                "case": case_name,
                "ground_truth": int(gt_steps),
                "estimated": int(est_steps),
                "abs_error": int(abs_err),
                "rel_error_percent": float(rel_err),
                "samples": int(t.size),
                "duration_sec": float(t[-1] - t[0]) if t.size > 1 else 0.0,
            }
        )

    summary = {
        "results": results,
        "mae": float(np.mean([item["abs_error"] for item in results])),
        "mape_percent": float(np.mean([item["rel_error_percent"] for item in results])),
    }

    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
