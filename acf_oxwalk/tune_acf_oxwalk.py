from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.oxwalk_dataset import list_oxwalk_records, load_oxwalk_file
from acf_oxwalk.acf_step_counter import ACFParams, ACFStepCounter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search ACF parameters on OxWalk.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("testdata/OxWalk_Dec2022"),
        help="OxWalk root containing Hip_100Hz/Hip_25Hz/Wrist_100Hz/Wrist_25Hz",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=50.0,
        help="Target sample rate for unified evaluation.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("acf_oxwalk/results"),
        help="Output folder for search summaries.",
    )
    parser.add_argument(
        "--limit-records",
        type=int,
        default=0,
        help="Optional cap on number of records for quick experiments (0 = all).",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=100.0,
        help="Use only first N seconds of each record for faster tuning (0 = full length).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker threads for parallel parameter evaluation.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    return parser.parse_args()


def build_grid(sample_rate_hz: float) -> List[ACFParams]:
    step_hz_min = [1.2]
    step_hz_max = [2.8]
    detrend_seconds = [0.6, 1.0]
    smooth_seconds = [0.06, 0.10, 0.14]
    prominence_std = [0.3, 0.5, 0.7]
    min_distance_ratio = [0.45, 0.55, 0.65]
    min_acf_peak = [0.12, 0.18, 0.24]

    grid: List[ACFParams] = []
    for vals in itertools.product(
        step_hz_min,
        step_hz_max,
        detrend_seconds,
        smooth_seconds,
        prominence_std,
        min_distance_ratio,
        min_acf_peak,
    ):
        p = ACFParams(
            sample_rate_hz=sample_rate_hz,
            step_hz_min=vals[0],
            step_hz_max=vals[1],
            detrend_seconds=vals[2],
            smooth_seconds=vals[3],
            prominence_std=vals[4],
            min_distance_ratio=vals[5],
            min_acf_peak=vals[6],
        )
        grid.append(p)
    return grid


def load_records(
    dataset_root: Path,
    sample_rate_hz: float,
    limit_records: int,
    max_seconds: float,
) -> List[Dict[str, object]]:
    records = list_oxwalk_records(dataset_root)
    payloads: List[Dict[str, object]] = []
    for rec in records:
        t, acc, ann = load_oxwalk_file(rec.file_path, target_hz=sample_rate_hz)
        if max_seconds > 0:
            mask = t <= (float(t[0]) + float(max_seconds))
            t = t[mask]
            acc = acc[mask]
            ann = ann[mask]
            if t.size > 0:
                t = t - float(t[0])

        if t.size == 0:
            continue

        payloads.append(
            {
                "file_path": str(rec.file_path),
                "participant": rec.participant,
                "modality": rec.modality,
                "time": t,
                "acc": acc,
                "gt_steps": int(np.sum(ann > 0.5)),
            }
        )

    if limit_records > 0:
        payloads = payloads[:limit_records]
    return payloads


def evaluate_params(params: ACFParams, records: List[Dict[str, object]]) -> Dict[str, object]:
    counter = ACFStepCounter(params=params)

    abs_errors: List[int] = []
    rel_errors: List[float] = []
    per_record: List[Dict[str, object]] = []

    for rec in records:
        out = counter.run_offline({"time": rec["time"], "acc": rec["acc"]})
        pred = int(out["step_count"])
        gt = int(rec["gt_steps"])
        abs_err = abs(pred - gt)
        rel_err = float(abs_err / gt * 100.0) if gt > 0 else 0.0

        abs_errors.append(abs_err)
        rel_errors.append(rel_err)

        per_record.append(
            {
                "file_path": rec["file_path"],
                "participant": rec["participant"],
                "modality": rec["modality"],
                "ground_truth": gt,
                "predicted": pred,
                "abs_error": abs_err,
                "rel_error_percent": rel_err,
            }
        )

    mae = float(np.mean(abs_errors)) if abs_errors else 0.0
    mape = float(np.mean(rel_errors)) if rel_errors else 0.0
    return {
        "params": asdict(params),
        "mae": mae,
        "mape_percent": mape,
        "n_records": len(records),
        "per_record": per_record,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.dataset_root, args.sample_rate_hz, args.limit_records, args.max_seconds)
    if not records:
        raise RuntimeError(f"No OxWalk records loaded from {args.dataset_root}")

    grid = build_grid(args.sample_rate_hz)
    all_results: List[Dict[str, object]] = []

    max_workers = max(1, int(args.num_workers))
    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_params, params, records) for params in grid]

        if args.no_progress:
            for future in cf.as_completed(futures):
                all_results.append(future.result())
        else:
            with tqdm(total=len(futures), desc="Searching ACF params") as pbar:
                for future in cf.as_completed(futures):
                    all_results.append(future.result())
                    pbar.update(1)

    all_results.sort(key=lambda x: (x["mae"], x["mape_percent"]))
    best = all_results[0]

    summary = {
        "dataset_root": str(args.dataset_root),
        "sample_rate_hz": float(args.sample_rate_hz),
        "max_seconds": float(args.max_seconds),
        "n_records": len(records),
        "grid_size": len(grid),
        "num_workers": int(max_workers),
        "best": {
            "params": best["params"],
            "mae": best["mae"],
            "mape_percent": best["mape_percent"],
        },
        "top10": [
            {
                "rank": idx + 1,
                "params": item["params"],
                "mae": item["mae"],
                "mape_percent": item["mape_percent"],
            }
            for idx, item in enumerate(all_results[:10])
        ],
    }

    (args.out_dir / "acf_best_params.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "acf_grid_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print(json.dumps(summary["best"], indent=2))
    print(f"Saved: {args.out_dir / 'acf_best_params.json'}")
    print(f"Saved: {args.out_dir / 'acf_grid_results.json'}")


if __name__ == "__main__":
    main()
