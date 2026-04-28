from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from step_counter import StepCounter


LOGGER = logging.getLogger(__name__)


def load_raw_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
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
    parser = argparse.ArgumentParser(description="Evaluate StepCounter on test1/test2 cases")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing testdata/",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("test_results.json"),
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()

    cases = [
        ("test1-84step", 84),
        ("test2-100steps", 100),
    ]

    LOGGER.info("Initializing StepCounter")
    counter = StepCounter()

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

        case_result = {
            "case": case_name,
            "ground_truth": int(gt_steps),
            "estimated": int(est_steps),
            "abs_error": int(abs_err),
            "rel_error_percent": float(rel_err),
            "samples": int(t.size),
            "duration_sec": float(t[-1] - t[0]) if t.size > 1 else 0.0,
        }
        results.append(case_result)
        LOGGER.info(
            "Case=%s GT=%d EST=%d abs_error=%d rel_error=%.2f%%",
            case_name,
            gt_steps,
            est_steps,
            abs_err,
            rel_err,
        )

    summary = {
        "results": results,
        "mae": float(np.mean([item["abs_error"] for item in results])),
        "mape_percent": float(np.mean([item["rel_error_percent"] for item in results])),
    }

    out_json = args.out_json if args.out_json.is_absolute() else Path(__file__).resolve().parent / args.out_json
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Evaluation summary saved to: %s", out_json)
    LOGGER.info("MAE=%.3f, MAPE=%.3f%%", summary["mae"], summary["mape_percent"])


if __name__ == "__main__":
    main()
