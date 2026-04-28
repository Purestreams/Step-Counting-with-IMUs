from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
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


class PackedWindowDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[np.ndarray, np.ndarray, float]],
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
    ):
        self.samples = list(samples)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)
        self.feature_std = np.where(self.feature_std < 1e-6, 1.0, self.feature_std).astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x, y_event, y_count = self.samples[idx]
        x = (x - self.feature_mean) / self.feature_std
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y_event": torch.tensor(y_event, dtype=torch.float32),
            "y_count": torch.tensor([y_count], dtype=torch.float32),
        }


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class CausalResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.drop(h)
        return F.gelu(h + x)


class CausalTCNStepNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        channels: int = 64,
        kernel_size: int = 5,
        num_blocks: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, channels, kernel_size=1)
        blocks = []
        for i in range(num_blocks):
            blocks.append(CausalResBlock(channels, kernel_size=kernel_size, dilation=2 ** i, dropout=dropout))
        self.tcn = nn.Sequential(*blocks)
        self.event_head = nn.Conv1d(channels, 1, kernel_size=1)
        self.count_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x.transpose(1, 2)
        h = self.input_proj(h)
        h = self.tcn(h)
        event_logits = self.event_head(h).squeeze(1)
        count_pred = self.count_head(h).squeeze(-1)
        return event_logits, count_pred


class LSTMStepNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.event_head = nn.Linear(hidden_size, 1)
        self.count_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.lstm(x)
        event_logits = self.event_head(h).squeeze(-1)
        pooled = h.mean(dim=1)
        count_pred = self.count_head(pooled).squeeze(-1)
        return event_logits, count_pred


def build_model(model_name: str) -> nn.Module:
    name = model_name.lower()
    if name == "tcn":
        return CausalTCNStepNet()
    if name == "lstm":
        return LSTMStepNet()
    raise ValueError(f"Unsupported model: {model_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural step counter on phyphox packed data")
    parser.add_argument(
        "--packed-data",
        type=Path,
        default=Path("submission/models/phyphox_data_packed_50hz.pt"),
        help="Path to packed phyphox training data (.pt)",
    )
    parser.add_argument(
        "--n-participants",
        type=int,
        default=0,
        help="Number of participants to use (<=0 means use all available participants)",
    )
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--label-sigma-seconds", type=float, default=0.06)
    parser.add_argument("--model", type=str, default="tcn", choices=["tcn", "lstm"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--count-loss-weight", type=float, default=0.15)
    parser.add_argument("--event-pos-weight", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    return parser.parse_args()


def pick_device(device_flag: str) -> torch.device:
    if device_flag == "cpu":
        return torch.device("cpu")
    if device_flag == "mps":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if device_flag == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def smooth_event_labels(labels: np.ndarray, sample_rate_hz: float, sigma_seconds: float) -> np.ndarray:
    sigma = max(1e-6, sigma_seconds * sample_rate_hz)
    radius = int(max(1, round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= np.max(kernel)
    y = np.convolve(labels, kernel, mode="same")
    return np.clip(y, 0.0, 1.0)


def build_windows(acc: np.ndarray, labels: np.ndarray, spec: WindowSpec) -> List[Tuple[np.ndarray, np.ndarray, float]]:
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
        x = feats[start:end].astype(np.float32)
        y_event = labels_smoothed[start:end].astype(np.float32)
        y_count = float(np.sum(labels[start:end] > 0.5))
        samples.append((x, y_event, y_count))
    return samples


def split_participants(participants: Sequence[str], val_ratio: float, test_ratio: float, seed: int) -> Dict[str, List[str]]:
    pids = list(participants)
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)

    n_total = len(pids)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    n_test = max(1 if n_total >= 3 else 0, n_test)
    n_val = max(1 if n_total >= 3 else 0, n_val)

    if n_test + n_val >= n_total:
        n_test = max(0, min(n_test, n_total - 2))
        n_val = max(0, min(n_val, n_total - 1 - n_test))

    test = pids[:n_test]
    val = pids[n_test : n_test + n_val]
    train = pids[n_test + n_val :]

    if len(train) == 0:
        if len(val) > 0:
            train = [val.pop()]
        elif len(test) > 0:
            train = [test.pop()]

    return {"train": train, "val": val, "test": test}


def load_records_from_packed(packed_data_path: Path, expected_sample_rate: float) -> Tuple[List[Dict[str, Any]], float]:
    if not packed_data_path.exists():
        raise FileNotFoundError(f"Packed data not found: {packed_data_path}")

    payload = torch.load(str(packed_data_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("Packed payload must be a dict with keys including 'records'.")

    sample_rate_hz = float(payload.get("sample_rate_hz", -1.0))
    if sample_rate_hz <= 0:
        raise RuntimeError("Invalid sample_rate_hz in packed payload.")
    if abs(sample_rate_hz - expected_sample_rate) > 1e-6:
        raise RuntimeError(
            f"Packed sample_rate_hz mismatch: payload={sample_rate_hz} vs --sample-rate={expected_sample_rate}"
        )

    records = payload.get("records", [])
    if not isinstance(records, list) or len(records) == 0:
        raise RuntimeError("Packed payload has no records.")

    return records, sample_rate_hz


def build_split_datasets_from_packed(
    records: List[Dict[str, Any]],
    n_participants: int,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    spec: WindowSpec,
    show_progress: bool,
) -> Tuple[Dict[str, PackedWindowDataset], Dict[str, Any]]:
    all_participants = sorted({str(r.get("participant")) for r in records if r.get("participant") is not None})
    if len(all_participants) == 0:
        raise RuntimeError("No participant IDs found in packed records.")

    if n_participants <= 0 or n_participants >= len(all_participants):
        selected_participants = list(all_participants)
    else:
        rng = np.random.default_rng(seed)
        selected_participants = sorted(rng.choice(all_participants, size=n_participants, replace=False).tolist())

    split = split_participants(selected_participants, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    split_samples: Dict[str, List[Tuple[np.ndarray, np.ndarray, float]]] = {"train": [], "val": [], "test": []}
    split_files: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    for split_name, participants in split.items():
        recs = [rec for rec in records if str(rec.get("participant")) in participants]
        iterator = recs
        if show_progress:
            iterator = tqdm(recs, desc=f"Loading {split_name}", leave=False)

        for rec in iterator:
            acc = np.asarray(rec["acc"], dtype=np.float32)
            ann = np.asarray(rec["ann"], dtype=np.float32)
            windows = build_windows(acc, ann, spec)
            if windows:
                split_samples[split_name].extend(windows)
                split_files[split_name].append(str(rec.get("file_path", "")))

    if len(split_samples["train"]) == 0:
        raise RuntimeError("No training windows produced. Adjust participant count or window settings.")

    x_train = np.concatenate([s[0] for s in split_samples["train"]], axis=0)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    datasets = {
        key: PackedWindowDataset(samples=value, feature_mean=mean, feature_std=std)
        for key, value in split_samples.items()
    }

    meta: Dict[str, Any] = {
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
    return datasets, meta


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    count_loss_weight: float,
    event_pos_weight: float,
    show_progress: bool,
    progress_desc: str,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(event_pos_weight, device=device))
    mse = nn.MSELoss()

    total_loss = 0.0
    total_event_loss = 0.0
    total_count_loss = 0.0
    total_count_mae = 0.0
    n_batches = 0

    iterator = loader
    if show_progress:
        iterator = tqdm(loader, desc=progress_desc, leave=False)

    for batch in iterator:
        x = batch["x"].to(device)
        y_event = batch["y_event"].to(device)
        y_count = batch["y_count"].to(device).squeeze(-1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        event_logits, count_pred = model(x)
        event_loss = bce(event_logits, y_event)
        count_loss = mse(count_pred, y_count)
        loss = event_loss + count_loss_weight * count_loss

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        count_mae = torch.mean(torch.abs(count_pred.detach() - y_count)).item()

        total_loss += float(loss.item())
        total_event_loss += float(event_loss.item())
        total_count_loss += float(count_loss.item())
        total_count_mae += float(count_mae)
        n_batches += 1

    if n_batches == 0:
        return {
            "loss": float("inf"),
            "event_loss": float("inf"),
            "count_loss": float("inf"),
            "count_mae": float("inf"),
        }

    return {
        "loss": total_loss / n_batches,
        "event_loss": total_event_loss / n_batches,
        "count_loss": total_count_loss / n_batches,
        "count_mae": total_count_mae / n_batches,
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    spec = WindowSpec(
        sample_rate_hz=args.sample_rate,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        label_sigma_seconds=args.label_sigma_seconds,
    )

    print(f"Packed data path: {args.packed_data}")
    records, loaded_sr = load_records_from_packed(args.packed_data, expected_sample_rate=spec.sample_rate_hz)
    print(f"Loaded packed records: {len(records)} @ {loaded_sr:.1f} Hz")

    datasets, meta = build_split_datasets_from_packed(
        records=records,
        n_participants=args.n_participants,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        spec=spec,
        show_progress=not args.no_progress,
    )

    device = pick_device(args.device)
    print(f"Using device: {device}")
    model = build_model(args.model).to(device)

    train_loader = make_loader(datasets["train"], args.batch_size, shuffle=True)
    val_loader = make_loader(datasets["val"], args.batch_size, shuffle=False)
    test_loader = make_loader(datasets["test"], args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / f"stepnet_{args.model}_best.pt"
    metrics_path = args.out_dir / f"stepnet_{args.model}_metrics.json"

    history: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0

    spec_meta = cast(Dict[str, Any], meta.get("spec", {}))
    saved_sample_rate = float(spec_meta.get("sample_rate_hz") or args.sample_rate)
    saved_window_seconds = float(spec_meta.get("window_seconds") or args.window_seconds)
    default_window_size = int(round(args.window_seconds * args.sample_rate))
    saved_window_size = int(spec_meta.get("window_size") or default_window_size)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            count_loss_weight=args.count_loss_weight,
            event_pos_weight=args.event_pos_weight,
            show_progress=not args.no_progress,
            progress_desc=f"Epoch {epoch:02d} train",
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                count_loss_weight=args.count_loss_weight,
                event_pos_weight=args.event_pos_weight,
                show_progress=not args.no_progress,
                progress_desc=f"Epoch {epoch:02d} val",
            )

        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_count_mae={val_metrics['count_mae']:.3f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "feature_mean": meta["feature_mean"],
                    "feature_std": meta["feature_std"],
                    "sample_rate_hz": saved_sample_rate,
                    "window_size": saved_window_size,
                    "window_seconds": saved_window_seconds,
                    "split_participants": meta["split_participants"],
                    "selected_participants": meta["selected_participants"],
                    "seed": args.seed,
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    if not ckpt_path.exists():
        raise RuntimeError("No checkpoint was saved. Increase --epochs or check dataset settings.")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            device=device,
            optimizer=None,
            count_loss_weight=args.count_loss_weight,
            event_pos_weight=args.event_pos_weight,
            show_progress=not args.no_progress,
            progress_desc="Test",
        )

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test": test_metrics,
        "meta": meta,
        "history": history,
        "checkpoint": str(ckpt_path),
        "packed_data": str(args.packed_data),
    }

    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved metrics   : {metrics_path}")


if __name__ == "__main__":
    main()
