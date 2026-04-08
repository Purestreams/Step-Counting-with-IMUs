from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, cast

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.oxwalk_dataset import WindowSpec, build_split_datasets
from models.tcn_stepnet import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural step counter on OxWalk")
    parser.add_argument("--dataset-root", type=Path, default=Path("testdata/OxWalk_Dec2022"))
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


def make_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


def run_epoch(
    model: torch.nn.Module,
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

    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(event_pos_weight, device=device))
    mse = torch.nn.MSELoss()

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
        return {"loss": float("inf"), "event_loss": float("inf"), "count_loss": float("inf"), "count_mae": float("inf")}

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

    datasets, meta = build_split_datasets(
        dataset_root=args.dataset_root,
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

    history = []
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

    checkpoint = torch.load(ckpt_path, map_location=device)
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
    }

    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved metrics   : {metrics_path}")


if __name__ == "__main__":
    main()
