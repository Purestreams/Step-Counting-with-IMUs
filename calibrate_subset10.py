import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from step_counter import StepCounter


@dataclass(frozen=True)
class Params:
    alpha: float
    stats_alpha: float
    threshold_std_gain: float
    min_threshold: float
    min_step_interval: float
    prominence_gain: float
    min_prominence: float


@dataclass
class SampleResult:
    file: str
    duration_s: float
    ground_truth: int
    estimated: int
    abs_error: int
    rel_error_pct: float


def _apply_params(counter: StepCounter, params: Params) -> None:
    counter._alpha = float(params.alpha)
    counter._stats_alpha = float(params.stats_alpha)
    counter._threshold_std_gain = float(params.threshold_std_gain)
    counter._min_threshold = float(params.min_threshold)
    counter._min_step_interval = float(params.min_step_interval)
    counter._prominence_gain = float(params.prominence_gain)
    counter._min_prominence = float(params.min_prominence)


def _load_window(csv_path: Path, window_seconds: float) -> Tuple[np.ndarray, np.ndarray, int]:
    numeric = np.loadtxt(
        csv_path,
        delimiter=",",
        skiprows=1,
        usecols=(1, 2, 3, 4),
        dtype=float,
    )
    timestamps = np.genfromtxt(
        csv_path,
        delimiter=",",
        skip_header=1,
        usecols=(0,),
        dtype="U32",
    )

    ts = np.asarray(timestamps, dtype="datetime64[ms]")
    t = (ts - ts[0]) / np.timedelta64(1, "s")
    t = t.astype(float)

    if t.size == 0:
        raise ValueError(f"No samples in file: {csv_path}")

    win = float(window_seconds)
    ann = numeric[:, 3]
    step_idx = np.flatnonzero(ann > 0.5)

    if step_idx.size > 0 and t[-1] > win:
        center_idx = int(step_idx[step_idx.size // 2])
        center_t = float(t[center_idx])
        start_t = max(0.0, center_t - 0.5 * win)
        end_t = start_t + win
        window_mask = (t >= start_t) & (t <= end_t)
    else:
        window_mask = t <= win

    if not np.any(window_mask):
        window_mask = np.ones_like(t, dtype=bool)

    acc = numeric[window_mask, :3]
    gt = int(np.rint(np.sum(numeric[window_mask, 3])))
    t_win = t[window_mask]

    if t_win.size >= 1:
        t_win = t_win - t_win[0]

    return t_win, acc, gt


def _gather_files(dataset_root: Path) -> List[Path]:
    modality_dirs = [
        dataset_root / "Hip_100Hz",
        dataset_root / "Hip_25Hz",
        dataset_root / "Wrist_100Hz",
        dataset_root / "Wrist_25Hz",
    ]

    files: List[Path] = []
    for modality_dir in modality_dirs:
        if modality_dir.exists():
            files.extend(sorted(modality_dir.glob("*.csv")))

    if not files:
        raise FileNotFoundError(f"No CSV files found under {dataset_root}")

    return files


def _select_subset(files: List[Path], n_samples: int, seed: int) -> List[Path]:
    if n_samples > len(files):
        raise ValueError(f"Requested {n_samples} samples but only {len(files)} files found.")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(files), size=n_samples, replace=False)
    idx = np.sort(idx)
    return [files[i] for i in idx]


def _build_candidates(rng: np.random.Generator, n_trials: int) -> List[Params]:
    baseline = Params(
        alpha=0.25,
        stats_alpha=0.01,
        threshold_std_gain=0.35,
        min_threshold=0.03,
        min_step_interval=0.25,
        prominence_gain=0.05,
        min_prominence=0.01,
    )

    candidates = {baseline}
    while len(candidates) < n_trials:
        p = Params(
            alpha=float(rng.choice([0.20, 0.23, 0.25, 0.28, 0.30])),
            stats_alpha=float(rng.choice([0.006, 0.008, 0.01, 0.012, 0.015])),
            threshold_std_gain=float(rng.choice([0.28, 0.32, 0.35, 0.38, 0.42])),
            min_threshold=float(rng.choice([0.02, 0.025, 0.03, 0.035])),
            min_step_interval=float(rng.choice([0.22, 0.25, 0.28, 0.30])),
            prominence_gain=float(rng.choice([0.03, 0.05, 0.07, 0.09])),
            min_prominence=float(rng.choice([0.005, 0.01, 0.015, 0.02])),
        )
        candidates.add(p)

    return list(candidates)


def _evaluate_subset(samples: List[Tuple[Path, np.ndarray, np.ndarray, int]], params: Params) -> Tuple[float, float, List[SampleResult]]:
    counter = StepCounter()
    _apply_params(counter, params)

    results: List[SampleResult] = []
    abs_errors: List[float] = []
    rel_errors: List[float] = []

    for file_path, t, acc, gt in samples:
        out = counter.run_offline({"time": t, "acc": acc})
        pred = int(out["step_count"])
        abs_error = abs(pred - gt)
        rel_error_pct = 0.0 if gt == 0 else 100.0 * abs_error / gt

        abs_errors.append(float(abs_error))
        rel_errors.append(float(rel_error_pct))

        results.append(
            SampleResult(
                file=str(file_path),
                duration_s=float(t[-1] - t[0]) if t.size > 1 else 0.0,
                ground_truth=int(gt),
                estimated=pred,
                abs_error=int(abs_error),
                rel_error_pct=float(rel_error_pct),
            )
        )

    mae = float(np.mean(abs_errors)) if abs_errors else float("inf")
    mape = float(np.mean(rel_errors)) if rel_errors else float("inf")
    return mae, mape, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate StepCounter on a deterministic 10-sample OxWalk subset.")
    parser.add_argument("--dataset-root", type=Path, default=Path("testdata/OxWalk_Dec2022"))
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--window-seconds", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--out-json", type=Path, default=Path("calibration_subset10_results.json"))
    args = parser.parse_args()

    all_files = _gather_files(args.dataset_root)
    subset_files = _select_subset(all_files, args.n_samples, args.seed)

    loaded_samples: List[Tuple[Path, np.ndarray, np.ndarray, int]] = []
    for file_path in subset_files:
        t, acc, gt = _load_window(file_path, args.window_seconds)
        loaded_samples.append((file_path, t, acc, gt))

    rng = np.random.default_rng(args.seed)
    candidates = _build_candidates(rng, args.trials)

    best_params = None
    best_mae = float("inf")
    best_mape = float("inf")
    best_results: List[SampleResult] = []

    for params in candidates:
        mae, mape, per_sample = _evaluate_subset(loaded_samples, params)
        if (mae < best_mae) or (np.isclose(mae, best_mae) and mape < best_mape):
            best_mae = mae
            best_mape = mape
            best_params = params
            best_results = per_sample

    if best_params is None:
        raise RuntimeError("No candidate parameters were evaluated.")

    payload: Dict[str, object] = {
        "seed": int(args.seed),
        "dataset_root": str(args.dataset_root),
        "n_samples": int(args.n_samples),
        "window_seconds": float(args.window_seconds),
        "trials": int(args.trials),
        "selected_files": [str(p) for p in subset_files],
        "best_params": asdict(best_params),
        "best_mae": float(best_mae),
        "best_mape": float(best_mape),
        "per_sample": [asdict(x) for x in best_results],
    }

    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Calibration complete")
    print(f"Saved: {args.out_json}")
    print("Best parameters:")
    for k, v in asdict(best_params).items():
        print(f"  {k} = {v}")
    print(f"Subset MAE  : {best_mae:.3f} steps")
    print(f"Subset MAPE : {best_mape:.3f}%")


if __name__ == "__main__":
    main()
