from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


MODALITY_DIRS = ["Hip_100Hz", "Hip_25Hz", "Wrist_100Hz", "Wrist_25Hz"]


@dataclass(frozen=True)
class WindowSpec:
    sample_rate_hz: float = 50.0
    window_seconds: float = 4.0
    stride_seconds: float = 0.5
    label_sigma_seconds: float = 0.06

    @property
    def window_size(self) -> int:
        return int(round(self.window_seconds * self.sample_rate_hz))

    @property
    def stride_size(self) -> int:
        return int(round(self.stride_seconds * self.sample_rate_hz))


@dataclass(frozen=True)
class Record:
    file_path: Path
    participant: str
    modality: str


class OxWalkWindowDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[np.ndarray, np.ndarray, float]],
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
    ):
        self.samples = samples
        self.feature_mean = feature_mean
        self.feature_std = feature_std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y_event, y_count = self.samples[idx]
        if self.feature_mean is not None and self.feature_std is not None:
            x = (x - self.feature_mean) / self.feature_std

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y_event": torch.tensor(y_event, dtype=torch.float32),
            "y_count": torch.tensor([y_count], dtype=torch.float32),
        }


def parse_participant_id(file_path: Path) -> str:
    token = file_path.stem.split("_")[0]
    return token


def list_oxwalk_records(dataset_root: Path) -> List[Record]:
    records: List[Record] = []
    for modality in MODALITY_DIRS:
        folder = dataset_root / modality
        if not folder.exists():
            continue
        for file_path in sorted(folder.glob("*.csv")):
            records.append(
                Record(
                    file_path=file_path,
                    participant=parse_participant_id(file_path),
                    modality=modality,
                )
            )
    if not records:
        raise FileNotFoundError(f"No OxWalk csv files found under {dataset_root}")
    return records


def select_participants(records: Sequence[Record], n_participants: int, seed: int) -> List[str]:
    participants = sorted({r.participant for r in records})
    if n_participants > len(participants):
        raise ValueError(f"Requested {n_participants} participants but only {len(participants)} are available")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(participants), size=n_participants, replace=False))
    return [participants[i] for i in idx]


def split_participants(
    participants: Sequence[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[str]]:
    arr = np.asarray(list(participants))
    rng = np.random.default_rng(seed)
    rng.shuffle(arr)

    n_total = len(arr)
    n_test = max(1, int(round(n_total * test_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = n_total - n_val - n_test
    if n_train < 1:
        n_train = 1
        n_val = max(1, n_total - n_test - n_train)

    train = arr[:n_train].tolist()
    val = arr[n_train:n_train + n_val].tolist()
    test = arr[n_train + n_val:].tolist()

    if not test:
        test = val[-1:]
        val = val[:-1] if len(val) > 1 else train[-1:]

    return {"train": train, "val": val, "test": test}


def load_oxwalk_file(file_path: Path, target_hz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    numeric = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=(1, 2, 3, 4), dtype=float)
    timestamp = np.genfromtxt(file_path, delimiter=",", skip_header=1, usecols=(0,), dtype="U32")

    ts = np.asarray(timestamp, dtype="datetime64[ms]")
    t = ((ts - ts[0]) / np.timedelta64(1, "s")).astype(float)
    acc = numeric[:, :3]
    annotation = numeric[:, 3]

    if t.size < 2:
        return t, acc, annotation

    dt = 1.0 / float(target_hz)
    t_new = np.arange(float(t[0]), float(t[-1]) + 1e-9, dt)

    x_new = np.interp(t_new, t, acc[:, 0])
    y_new = np.interp(t_new, t, acc[:, 1])
    z_new = np.interp(t_new, t, acc[:, 2])
    acc_new = np.stack([x_new, y_new, z_new], axis=1)

    step_times = t[annotation > 0.5]
    ann_new = np.zeros_like(t_new)
    if step_times.size > 0:
        step_idx = np.clip(np.rint(step_times * target_hz).astype(int), 0, t_new.size - 1)
        ann_new[step_idx] = 1.0

    return t_new, acc_new, ann_new


def smooth_event_labels(labels: np.ndarray, sample_rate_hz: float, sigma_seconds: float) -> np.ndarray:
    sigma = max(1e-6, sigma_seconds * sample_rate_hz)
    radius = int(max(1, round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= np.max(kernel)
    y = np.convolve(labels, kernel, mode="same")
    return np.clip(y, 0.0, 1.0)


def build_windows(
    acc: np.ndarray,
    labels: np.ndarray,
    spec: WindowSpec,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    window = spec.window_size
    stride = max(1, spec.stride_size)

    if acc.shape[0] < window:
        return []

    labels_smoothed = smooth_event_labels(labels, spec.sample_rate_hz, spec.label_sigma_seconds)
    amag = np.sqrt(np.sum(acc * acc, axis=1, keepdims=True))
    feats = np.concatenate([acc, amag], axis=1)

    samples: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for start in range(0, feats.shape[0] - window + 1, stride):
        end = start + window
        x = feats[start:end]
        y_event = labels_smoothed[start:end]
        y_count = float(np.sum(labels[start:end] > 0.5))
        samples.append((x, y_event, y_count))

    return samples


def build_split_datasets(
    dataset_root: Path,
    n_participants: Optional[int],
    seed: int,
    val_ratio: float,
    test_ratio: float,
    spec: WindowSpec,
    show_progress: bool = True,
    packed_cache_path: Optional[Path] = None,
    rebuild_packed_cache: bool = False,
) -> Tuple[Dict[str, OxWalkWindowDataset], Dict[str, object]]:
    def _build_packed_records(records_to_pack: Sequence[Record]) -> List[Dict[str, Any]]:
        iterator = records_to_pack
        if show_progress:
            iterator = tqdm(records_to_pack, desc="Packing records", leave=False)

        packed: List[Dict[str, Any]] = []
        for rec in iterator:
            _, acc, ann = load_oxwalk_file(rec.file_path, target_hz=spec.sample_rate_hz)
            packed.append(
                {
                    "participant": rec.participant,
                    "modality": rec.modality,
                    "file_path": str(rec.file_path),
                    "acc": acc.astype(np.float32),
                    "ann": ann.astype(np.float32),
                }
            )
        return packed

    records = list_oxwalk_records(dataset_root)
    all_participants = sorted({r.participant for r in records})

    packed_records: Optional[List[Dict[str, Any]]] = None
    if packed_cache_path is not None:
        packed_cache_path = Path(packed_cache_path)
        can_load_cache = packed_cache_path.exists() and (not rebuild_packed_cache)
        if can_load_cache:
            payload = torch.load(str(packed_cache_path), map_location="cpu", weights_only=False)
            cached_sr = float(payload.get("sample_rate_hz", -1.0))
            if abs(cached_sr - spec.sample_rate_hz) < 1e-6:
                packed_records = payload.get("records", [])
                print(f"Loaded packed cache: {packed_cache_path}")
            else:
                print(
                    f"Packed cache sample-rate mismatch ({cached_sr} vs {spec.sample_rate_hz}); rebuilding {packed_cache_path}"
                )

        if packed_records is None:
            packed_records = _build_packed_records(records)
            packed_cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "version": 1,
                    "sample_rate_hz": float(spec.sample_rate_hz),
                    "records": packed_records,
                },
                str(packed_cache_path),
            )
            print(f"Saved packed cache: {packed_cache_path}")

    if n_participants is None or n_participants <= 0 or n_participants >= len(all_participants):
        selected_participants = list(all_participants)
    else:
        selected_participants = select_participants(records, n_participants=n_participants, seed=seed)

    split = split_participants(selected_participants, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    by_participant: Dict[str, List[Record]] = {}
    if packed_records is None:
        for rec in records:
            if rec.participant in selected_participants:
                by_participant.setdefault(rec.participant, []).append(rec)

    split_samples: Dict[str, List[Tuple[np.ndarray, np.ndarray, float]]] = {"train": [], "val": [], "test": []}
    split_files: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    for split_name, participants in split.items():
        if packed_records is None:
            records_for_split: List[Record] = []
            for pid in participants:
                records_for_split.extend(by_participant.get(pid, []))

            iterator = records_for_split
            if show_progress:
                iterator = tqdm(records_for_split, desc=f"Loading {split_name}", leave=False)

            for rec in iterator:
                _, acc, ann = load_oxwalk_file(rec.file_path, target_hz=spec.sample_rate_hz)
                windows = build_windows(acc, ann, spec)
                if windows:
                    split_samples[split_name].extend(windows)
                    split_files[split_name].append(str(rec.file_path))
        else:
            recs_for_split = [
                rec
                for rec in packed_records
                if rec.get("participant") in participants
            ]

            iterator = recs_for_split
            if show_progress:
                iterator = tqdm(recs_for_split, desc=f"Loading {split_name}", leave=False)

            for rec in iterator:
                acc = np.asarray(rec["acc"], dtype=float)
                ann = np.asarray(rec["ann"], dtype=float)
                windows = build_windows(acc, ann, spec)
                if windows:
                    split_samples[split_name].extend(windows)
                    split_files[split_name].append(str(rec.get("file_path", "")))

    if len(split_samples["train"]) == 0:
        raise RuntimeError("No training windows produced. Increase n_participants or adjust window settings.")

    x_train = np.concatenate([s[0] for s in split_samples["train"]], axis=0)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    datasets = {
        key: OxWalkWindowDataset(samples=value, feature_mean=mean, feature_std=std)
        for key, value in split_samples.items()
    }

    meta: Dict[str, object] = {
        "selected_participants": selected_participants,
        "split_participants": split,
        "split_files": split_files,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "spec": {
            "sample_rate_hz": spec.sample_rate_hz,
            "window_seconds": spec.window_seconds,
            "stride_seconds": spec.stride_seconds,
            "label_sigma_seconds": spec.label_sigma_seconds,
            "window_size": spec.window_size,
        },
        "n_windows": {k: len(v) for k, v in split_samples.items()},
    }

    if packed_cache_path is not None:
        meta["packed_cache_path"] = str(packed_cache_path)
        meta["used_packed_cache"] = True
        meta["rebuild_packed_cache"] = bool(rebuild_packed_cache)

    return datasets, meta
