from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acf_oxwalk.acf_step_counter import ACFParams, ACFStepCounter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test ACFStepCounter on test1/test2 datasets")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root containing testdata/",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=Path("acf_oxwalk/results/acf_best_params.json"),
        help="JSON file containing best params from tuner",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("acf_oxwalk/results/acf_test1_test2_results.json"),
        help="Output summary JSON path",
    )
    return parser.parse_args()


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


def resolve_params(config_json_path: Path) -> ACFParams:
    payload = json.loads(config_json_path.read_text(encoding="utf-8"))
    if "best" in payload and isinstance(payload["best"], dict) and "params" in payload["best"]:
        params_dict = payload["best"]["params"]
    elif "params" in payload:
        params_dict = payload["params"]
    else:
        raise ValueError(f"Cannot find params in config: {config_json_path}")

    return ACFParams(**params_dict)


def main() -> None:
    args = parse_args()

    config_json_path = args.config_json if args.config_json.is_absolute() else args.project_root / args.config_json
    out_json_path = args.out_json if args.out_json.is_absolute() else args.project_root / args.out_json

    if not config_json_path.exists():
        raise FileNotFoundError(f"Missing config json: {config_json_path}")

    params = resolve_params(config_json_path)
    counter = ACFStepCounter(params=params)

    cases = [
        ("test1-84step", 84),
        ("test2-100steps", 100),
    ]

    results: list[dict[str, object]] = []
    for case_name, gt_steps in cases:
        csv_path = args.project_root / "testdata" / case_name / "Raw Data.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing test CSV: {csv_path}")

        t, acc = load_raw_data(csv_path)
        out = counter.run_offline({"time": t, "acc": acc})

        est_steps = int(out["step_count"])
        abs_err = abs(est_steps - gt_steps)
        rel_err = (abs_err / gt_steps * 100.0) if gt_steps else 0.0

        result = {
            "case": case_name,
            "ground_truth": int(gt_steps),
            "estimated": int(est_steps),
            "abs_error": int(abs_err),
            "rel_error_percent": float(rel_err),
            "samples": int(t.size),
            "duration_sec": float(t[-1] - t[0]) if t.size > 1 else 0.0,
            "diagnostics": out.get("diagnostics", {}),
        }
        results.append(result)

    summary = {
        "config_json": str(config_json_path),
        "params": asdict(params),
        "results": results,
        "mae": float(np.mean([item["abs_error"] for item in results])),
        "mape_percent": float(np.mean([item["rel_error_percent"] for item in results])),
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_json_path}")


if __name__ == "__main__":
    main()
