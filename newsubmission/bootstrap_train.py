from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.oxwalk_dataset import OxWalkWindowDataset, WindowSpec, build_windows, split_participants
from models.tcn_stepnet import build_model


LOGGER = logging.getLogger(__name__)


@dataclass
class BootstrapConfig:
    packed_cache: Path
    seed: int = 20260408
    n_participants: int = 10
    sample_rate: float = 50.0
    window_seconds: float = 4.0
    stride_seconds: float = 0.5
    label_sigma_seconds: float = 0.06
    epochs: int = 2
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    count_loss_weight: float = 0.15
    event_pos_weight: float = 3.0
    patience: int = 3
    model_name: str = "tcn"
    device: str = "auto"
    val_ratio: float = 0.2
    test_ratio: float = 0.2


class BootstrapTrainer:
    def __init__(self, config: BootstrapConfig):
        self.config = config

    @staticmethod
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

    @staticmethod
    def make_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)

    @staticmethod
    def run_epoch(
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        optimizer: torch.optim.Optimizer | None,
        count_loss_weight: float,
        event_pos_weight: float,
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

        for batch in loader:
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

    def train_fresh(self) -> Dict[str, Any]:
        cfg = self.config

        if not cfg.packed_cache.exists():
            raise FileNotFoundError(f"Missing packed cache: {cfg.packed_cache}")

        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        spec = WindowSpec(
            sample_rate_hz=cfg.sample_rate,
            window_seconds=cfg.window_seconds,
            stride_seconds=cfg.stride_seconds,
            label_sigma_seconds=cfg.label_sigma_seconds,
        )

        payload = torch.load(str(cfg.packed_cache), map_location="cpu", weights_only=False)
        cached_sr = float(payload.get("sample_rate_hz", -1.0))
        if abs(cached_sr - cfg.sample_rate) > 1e-6:
            raise RuntimeError(
                f"Packed cache sample rate mismatch: {cached_sr} (cache) vs {cfg.sample_rate} (config)"
            )

        packed_records = payload.get("records", [])
        if not isinstance(packed_records, list) or len(packed_records) == 0:
            raise RuntimeError(f"Packed cache has no records: {cfg.packed_cache}")

        all_participants = sorted({str(r.get("participant", "")) for r in packed_records if r.get("participant")})
        if not all_participants:
            raise RuntimeError("Packed cache does not contain participant IDs")

        if cfg.n_participants <= 0 or cfg.n_participants >= len(all_participants):
            selected_participants = list(all_participants)
        else:
            rng = np.random.default_rng(cfg.seed)
            idx = np.sort(rng.choice(len(all_participants), size=cfg.n_participants, replace=False))
            selected_participants = [all_participants[i] for i in idx]

        split = split_participants(selected_participants, val_ratio=cfg.val_ratio, test_ratio=cfg.test_ratio, seed=cfg.seed)

        split_samples: Dict[str, list[tuple[np.ndarray, np.ndarray, float]]] = {"train": [], "val": [], "test": []}
        split_files: Dict[str, list[str]] = {"train": [], "val": [], "test": []}

        for split_name, participants in split.items():
            recs_for_split = [rec for rec in packed_records if rec.get("participant") in participants]
            for rec in recs_for_split:
                acc = np.asarray(rec["acc"], dtype=float)
                ann = np.asarray(rec["ann"], dtype=float)
                windows = build_windows(acc, ann, spec)
                if windows:
                    split_samples[split_name].extend(windows)
                    split_files[split_name].append(str(rec.get("file_path", "")))

        if len(split_samples["train"]) == 0:
            raise RuntimeError("No training windows produced from packed cache")

        x_train = np.concatenate([s[0] for s in split_samples["train"]], axis=0)
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)

        datasets = {
            key: OxWalkWindowDataset(samples=value, feature_mean=mean, feature_std=std)
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
            "packed_cache_path": str(cfg.packed_cache),
            "used_packed_cache": True,
            "rebuild_packed_cache": False,
        }

        device = self.pick_device(cfg.device)
        model = build_model(cfg.model_name).to(device)

        train_loader = self.make_loader(datasets["train"], cfg.batch_size, shuffle=True)
        val_loader = self.make_loader(datasets["val"], cfg.batch_size, shuffle=False)
        test_loader = self.make_loader(datasets["test"], cfg.batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        history = []
        best_val = float("inf")
        best_epoch = -1
        bad_epochs = 0
        best_checkpoint: Dict[str, Any] | None = None
        spec_meta = cast(Dict[str, Any], meta.get("spec", {}))
        saved_sample_rate = float(spec_meta.get("sample_rate_hz") or cfg.sample_rate)
        saved_window_seconds = float(spec_meta.get("window_seconds") or cfg.window_seconds)
        default_window_size = int(round(cfg.window_seconds * cfg.sample_rate))
        saved_window_size = int(spec_meta.get("window_size") or default_window_size)

        for epoch in range(1, cfg.epochs + 1):
            train_metrics = self.run_epoch(
                model=model,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                count_loss_weight=cfg.count_loss_weight,
                event_pos_weight=cfg.event_pos_weight,
            )

            with torch.no_grad():
                val_metrics = self.run_epoch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    optimizer=None,
                    count_loss_weight=cfg.count_loss_weight,
                    event_pos_weight=cfg.event_pos_weight,
                )

            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

            LOGGER.info(
                "[Epoch %02d] train_loss=%.4f val_loss=%.4f val_count_mae=%.3f",
                epoch,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["count_mae"],
            )

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_epoch = epoch
                bad_epochs = 0
                best_checkpoint = {
                    "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "model_name": cfg.model_name,
                    "feature_mean": meta["feature_mean"],
                    "feature_std": meta["feature_std"],
                    "sample_rate_hz": saved_sample_rate,
                    "window_size": saved_window_size,
                    "window_seconds": saved_window_seconds,
                    "split_participants": meta["split_participants"],
                    "selected_participants": meta["selected_participants"],
                    "seed": cfg.seed,
                }
            else:
                bad_epochs += 1

            if bad_epochs >= cfg.patience:
                LOGGER.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                break

        if best_checkpoint is None:
            raise RuntimeError("Training did not produce a valid checkpoint")

        checkpoint = best_checkpoint
        model.load_state_dict(checkpoint["model_state_dict"])

        with torch.no_grad():
            test_metrics = self.run_epoch(
                model=model,
                loader=test_loader,
                device=device,
                optimizer=None,
                count_loss_weight=cfg.count_loss_weight,
                event_pos_weight=cfg.event_pos_weight,
            )

        summary = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "test": test_metrics,
            "meta": meta,
            "history": history,
            "packed_cache": str(cfg.packed_cache),
        }

        checkpoint["training_summary"] = summary

        LOGGER.info("Bootstrap training complete (in-memory checkpoint)")
        return checkpoint
