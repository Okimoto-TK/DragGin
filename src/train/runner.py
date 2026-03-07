from __future__ import annotations

import argparse
import bisect
import json
import logging
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from src.model.fusion import MultiScaleFusion
from src.model.head import RegressionHead, masked_huber_loss
from src.model.wno import WNO1DEncoder

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


class NpyShardDataset(Dataset):
    def __init__(self, shard_paths: list[str], y_key: str = "y") -> None:
        self.shard_paths = [str(path) for path in shard_paths]
        self.y_key = y_key
        self._shards: list[dict[str, Any]] = []
        self._offsets: list[int] = [0]

        for path in self.shard_paths:
            shard = np.load(path, allow_pickle=True).item()
            if self.y_key not in shard:
                raise KeyError(f"y_key '{self.y_key}' not found in shard: {path}")
            n_rows = int(len(shard["codes"]))
            self._shards.append(shard)
            self._offsets.append(self._offsets[-1] + n_rows)

    def __len__(self) -> int:
        return self._offsets[-1]

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        shard_idx = bisect.bisect_right(self._offsets, idx) - 1
        row_idx = idx - self._offsets[shard_idx]
        return shard_idx, row_idx

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shard_idx, row_idx = self._locate(idx)
        shard = self._shards[shard_idx]

        return {
            "code": str(shard["codes"][row_idx]),
            "asof_date": str(shard["asof_dates"][row_idx]),
            "x_micro": torch.as_tensor(shard["X_micro"][row_idx], dtype=torch.float32),
            "x_mezzo": torch.as_tensor(shard["X_mezzo"][row_idx], dtype=torch.float32),
            "x_macro": torch.as_tensor(shard["X_macro"][row_idx], dtype=torch.float32),
            "mask_micro": torch.as_tensor(shard["mask_micro"][row_idx], dtype=torch.bool),
            "mask_mezzo": torch.as_tensor(shard["mask_mezzo"][row_idx], dtype=torch.bool),
            "mask_macro": torch.as_tensor(shard["mask_macro"][row_idx], dtype=torch.bool),
            "y": torch.as_tensor(shard[self.y_key][row_idx], dtype=torch.float32),
            "dp_ok": torch.as_tensor(shard["dp_ok"][row_idx], dtype=torch.bool),
            "label_ok": torch.as_tensor(shard["label_ok"][row_idx], dtype=torch.bool),
            "loss_mask": torch.as_tensor(shard["loss_mask"][row_idx], dtype=torch.bool),
        }


class ShardBatchIterator:
    def __init__(
        self,
        shard_paths: list[str],
        batch_size: int,
        y_key: str = "y",
        shuffle: bool = True,
        buffer: bool = False,
        row_indices_by_shard: dict[int, np.ndarray] | None = None,
    ) -> None:
        self.shard_paths = [str(path) for path in shard_paths]
        self.batch_size = int(batch_size)
        self.y_key = y_key
        self.shuffle = bool(shuffle)
        self.buffer = bool(buffer)
        self.row_indices_by_shard = row_indices_by_shard or {}
        self._rng = np.random.default_rng()
        self._sizes = [self._get_shard_size(path) for path in self.shard_paths]
        self._total_rows = int(sum(self._selected_rows_count(i) for i in range(len(self.shard_paths))))
        self._buffered_shards: list[dict[str, Any]] | None = None
        self.last_shard_order: list[str] = []
        self.last_row_orders: dict[str, list[int]] = {}

        if self.buffer:
            self._buffered_shards = [self._load_shard(path) for path in self.shard_paths]

    def _get_shard_size(self, path: str) -> int:
        shard = self._load_shard(path)
        n_rows = int(len(shard["codes"]))
        del shard
        return n_rows

    def _selected_rows_count(self, shard_idx: int) -> int:
        rows = self.row_indices_by_shard.get(shard_idx)
        if rows is None:
            return int(self._sizes[shard_idx])
        return int(len(rows))

    def _load_shard(self, path: str) -> dict[str, Any]:
        shard = np.load(path, allow_pickle=True).item()
        if self.y_key not in shard:
            raise KeyError(f"y_key '{self.y_key}' not found in shard: {path}")
        return shard

    def __len__(self) -> int:
        if self._total_rows == 0:
            return 0
        return int(math.ceil(self._total_rows / max(1, self.batch_size)))

    def __iter__(self):
        if not self.shard_paths:
            return
        shard_indices = np.arange(len(self.shard_paths), dtype=np.int64)
        if self.shuffle:
            shard_indices = self._rng.permutation(shard_indices)
        self.last_shard_order = [self.shard_paths[int(i)] for i in shard_indices.tolist()]
        self.last_row_orders = {}

        batch_samples: list[dict[str, Any]] = []
        for shard_idx in shard_indices.tolist():
            selected_rows = self.row_indices_by_shard.get(shard_idx)
            if selected_rows is None:
                row_indices = np.arange(self._sizes[shard_idx], dtype=np.int64)
            else:
                row_indices = np.asarray(selected_rows, dtype=np.int64).copy()

            if row_indices.size == 0:
                continue
            if self.shuffle:
                row_indices = self._rng.permutation(row_indices)

            path = self.shard_paths[shard_idx]
            self.last_row_orders[path] = row_indices.tolist()
            shard = self._buffered_shards[shard_idx] if self._buffered_shards is not None else self._load_shard(path)

            for row_idx in row_indices.tolist():
                batch_samples.append(
                    {
                        "code": str(shard["codes"][row_idx]),
                        "asof_date": str(shard["asof_dates"][row_idx]),
                        "x_micro": torch.as_tensor(shard["X_micro"][row_idx], dtype=torch.float32),
                        "x_mezzo": torch.as_tensor(shard["X_mezzo"][row_idx], dtype=torch.float32),
                        "x_macro": torch.as_tensor(shard["X_macro"][row_idx], dtype=torch.float32),
                        "mask_micro": torch.as_tensor(shard["mask_micro"][row_idx], dtype=torch.bool),
                        "mask_mezzo": torch.as_tensor(shard["mask_mezzo"][row_idx], dtype=torch.bool),
                        "mask_macro": torch.as_tensor(shard["mask_macro"][row_idx], dtype=torch.bool),
                        "y": torch.as_tensor(shard[self.y_key][row_idx], dtype=torch.float32),
                        "dp_ok": torch.as_tensor(shard["dp_ok"][row_idx], dtype=torch.bool),
                        "label_ok": torch.as_tensor(shard["label_ok"][row_idx], dtype=torch.bool),
                        "loss_mask": torch.as_tensor(shard["loss_mask"][row_idx], dtype=torch.bool),
                    }
                )
                if len(batch_samples) == self.batch_size:
                    yield collate_batch(batch_samples)
                    batch_samples = []

            if self._buffered_shards is None:
                del shard

        if batch_samples:
            yield collate_batch(batch_samples)


def _build_row_splits(
    shard_paths: list[str], val_ratio: float, split_seed: int, y_key: str
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], int, int]:
    shard_sizes: list[int] = []
    for path in shard_paths:
        shard = np.load(path, allow_pickle=True).item()
        if y_key not in shard:
            raise KeyError(f"y_key '{y_key}' not found in shard: {path}")
        shard_sizes.append(int(len(shard["codes"])))
        del shard

    offsets = [0]
    for size in shard_sizes:
        offsets.append(offsets[-1] + size)
    total = offsets[-1]
    if total < 2:
        raise ValueError("At least 2 samples are required when val_shards is not provided.")

    ratio = float(min(max(val_ratio, 0.0), 0.99))
    val_size = int(round(total * ratio))
    val_size = max(1, min(total - 1, val_size))

    g = torch.Generator()
    g.manual_seed(int(split_seed))
    perm = torch.randperm(total, generator=g).tolist()
    val_indices = perm[:val_size]
    train_indices = perm[val_size:]

    def _to_rows(global_indices: list[int]) -> dict[int, np.ndarray]:
        grouped: dict[int, list[int]] = {}
        for idx in global_indices:
            shard_idx = bisect.bisect_right(offsets, idx) - 1
            row_idx = idx - offsets[shard_idx]
            grouped.setdefault(shard_idx, []).append(int(row_idx))
        return {k: np.asarray(v, dtype=np.int64) for k, v in grouped.items()}

    return _to_rows(train_indices), _to_rows(val_indices), len(train_indices), len(val_indices)


def collate_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "code": [sample["code"] for sample in samples],
        "asof_date": [sample["asof_date"] for sample in samples],
        "x_micro": torch.stack([sample["x_micro"] for sample in samples], dim=0),
        "x_mezzo": torch.stack([sample["x_mezzo"] for sample in samples], dim=0),
        "x_macro": torch.stack([sample["x_macro"] for sample in samples], dim=0),
        "mask_micro": torch.stack([sample["mask_micro"] for sample in samples], dim=0),
        "mask_mezzo": torch.stack([sample["mask_mezzo"] for sample in samples], dim=0),
        "mask_macro": torch.stack([sample["mask_macro"] for sample in samples], dim=0),
        "y": torch.stack([sample["y"] for sample in samples], dim=0),
        "dp_ok": torch.stack([sample["dp_ok"] for sample in samples], dim=0),
        "label_ok": torch.stack([sample["label_ok"] for sample in samples], dim=0),
        "loss_mask": torch.stack([sample["loss_mask"] for sample in samples], dim=0),
    }


class MultiScaleRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int = 6,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.0,
        enable_dynamic_threshold: bool = True,
        enable_free_branch: bool = True,
        init_lambda_micro: float = 1.5,
        init_lambda_mezzo: float = 0.8,
        init_lambda_macro: float = 0.3,
        use_seq_context: bool = False,
    ) -> None:
        super().__init__()
        self.micro_encoder = WNO1DEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            enable_dynamic_threshold=enable_dynamic_threshold,
            init_lambda=init_lambda_micro,
        )
        self.mezzo_encoder = WNO1DEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            enable_dynamic_threshold=enable_dynamic_threshold,
            init_lambda=init_lambda_mezzo,
        )
        self.macro_encoder = WNO1DEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            enable_dynamic_threshold=enable_dynamic_threshold,
            init_lambda=init_lambda_macro,
        )

        self.fusion = MultiScaleFusion(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            enable_free_branch=enable_free_branch,
        )
        self.head = RegressionHead(hidden_dim=hidden_dim, use_seq_context=use_seq_context, dropout=dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        micro_seq, micro_pool, aux_micro = self.micro_encoder(batch["x_micro"], batch["mask_micro"])
        mezzo_seq, mezzo_pool, aux_mezzo = self.mezzo_encoder(batch["x_mezzo"], batch["mask_mezzo"])
        macro_seq, macro_pool, aux_macro = self.macro_encoder(batch["x_macro"], batch["mask_macro"])

        fused_seq, fused_pool, aux_fusion = self.fusion(
            micro_seq=micro_seq,
            mezzo_seq=mezzo_seq,
            macro_seq=macro_seq,
            micro_pool=micro_pool,
            mezzo_pool=mezzo_pool,
            macro_pool=macro_pool,
            mask_micro=batch["mask_micro"],
            mask_mezzo=batch["mask_mezzo"],
            mask_macro=batch["mask_macro"],
        )
        y_hat = self.head(fused_seq=fused_seq, fused_pool=fused_pool)
        return y_hat, {"micro": aux_micro, "mezzo": aux_mezzo, "macro": aux_macro, "fusion": aux_fusion}


@dataclass
class TrainConfig:
    train_shards: list[str]
    val_shards: list[str]
    batch_size: int
    grad_accum_steps: int
    num_epochs: int
    lr: float
    weight_decay: float
    hidden_dim: int
    num_heads: int
    dropout: float
    exp_name: str
    out_dir: str
    y_key: str = "y"
    in_dim: int = 6
    enable_dynamic_threshold: bool = True
    enable_free_branch: bool = True
    init_lambda_micro: float = 1.5
    init_lambda_mezzo: float = 0.8
    init_lambda_macro: float = 0.3
    use_seq_context: bool = False
    clip_grad_norm: float | None = None
    val_ratio: float = 0.15
    split_seed: int = 42
    buffer: bool = False


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"train_runner_{log_path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def run_training(config: TrainConfig, raise_on_error: bool = True) -> dict[str, Any]:
    out_dir = Path(config.out_dir)
    logs_dir = out_dir / "logs"
    runs_dir = out_dir / "runs" / config.exp_name
    metrics_dir = out_dir / "metrics"
    feedback_path = out_dir / "reports" / "feedback" / f"{config.exp_name}.yaml"

    logs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(logs_dir / "train.log")

    writer = SummaryWriter(log_dir=str(runs_dir))
    curve_path = metrics_dir / "curve.json"
    history: dict[str, list[dict[str, float | int]]] = {"train": [], "val": []}

    train_sample_count = 0
    val_sample_count = 0
    if config.val_shards:
        train_loader = ShardBatchIterator(
            config.train_shards,
            batch_size=config.batch_size,
            y_key=config.y_key,
            shuffle=True,
            buffer=config.buffer,
        )
        val_loader = ShardBatchIterator(
            config.val_shards,
            batch_size=config.batch_size,
            y_key=config.y_key,
            shuffle=False,
            buffer=config.buffer,
        )
        train_sample_count = train_loader._total_rows
        val_sample_count = val_loader._total_rows
    else:
        train_rows, val_rows, train_sample_count, val_sample_count = _build_row_splits(
            config.train_shards,
            val_ratio=config.val_ratio,
            split_seed=config.split_seed,
            y_key=config.y_key,
        )
        train_loader = ShardBatchIterator(
            config.train_shards,
            batch_size=config.batch_size,
            y_key=config.y_key,
            shuffle=True,
            buffer=config.buffer,
            row_indices_by_shard=train_rows,
        )
        val_loader = ShardBatchIterator(
            config.train_shards,
            batch_size=config.batch_size,
            y_key=config.y_key,
            shuffle=False,
            buffer=config.buffer,
            row_indices_by_shard=val_rows,
        )

    model = MultiScaleRegressor(
        in_dim=config.in_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        dropout=config.dropout,
        enable_dynamic_threshold=config.enable_dynamic_threshold,
        enable_free_branch=config.enable_free_branch,
        init_lambda_micro=config.init_lambda_micro,
        init_lambda_mezzo=config.init_lambda_mezzo,
        init_lambda_macro=config.init_lambda_macro,
        use_seq_context=config.use_seq_context,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = None
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    global_step = 0
    epochs_completed = 0
    best_val = {"loss": float("inf"), "huber": float("inf"), "mae": float("inf"), "mse": float("inf")}
    last_train_row: dict[str, Any] = {}
    last_val_row: dict[str, Any] = {"loss": 0.0}

    try:
        for epoch in range(config.num_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_train_iter = iter(train_loader)
            logger.info(
                "epoch=%d train_shard_iter_start buffer=%s train_shards=%d sample_shards=%s",
                epoch + 1,
                str(config.buffer).lower(),
                len(config.train_shards),
                [Path(p).name for p in train_loader.last_shard_order[:3]],
            )

            for batch_idx, raw_batch in enumerate(epoch_train_iter):
                batch = _to_device(raw_batch, device)

                with autocast(enabled=use_amp):
                    y_hat, aux = model(batch)
                    loss, metrics = masked_huber_loss(y_hat=y_hat, y_true=batch["y"], loss_mask=batch["loss_mask"])
                    scaled_loss = loss / max(1, config.grad_accum_steps)

                if scaled_loss.requires_grad:
                    scaler.scale(scaled_loss).backward()

                should_step = (batch_idx + 1) % max(1, config.grad_accum_steps) == 0 or (batch_idx + 1) == len(train_loader)
                if not should_step:
                    continue

                if config.clip_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)

                if scaled_loss.requires_grad:
                    scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1

                batch_size_current = int(batch["y"].shape[0])
                num_valid = int(metrics["num_valid"])
                valid_ratio = float(num_valid / max(1, batch_size_current))
                micro_lambda_mean = float(aux["micro"]["lambda_mean"].mean().detach().item())
                mezzo_lambda_mean = float(aux["mezzo"]["lambda_mean"].mean().detach().item())
                macro_lambda_mean = float(aux["macro"]["lambda_mean"].mean().detach().item())
                gate = aux["fusion"]["gate"]
                gate_mean = float(gate.mean().detach().item())
                gate_std = float(gate.std(unbiased=False).detach().item())

                train_row = {
                    "step": int(global_step),
                    "loss": float(loss.detach().item()),
                    "huber": float(metrics["huber"].detach().item()),
                    "mae": float(metrics["mae"].detach().item()),
                    "mse": float(metrics["mse"].detach().item()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "num_valid": int(num_valid),
                    "valid_ratio": float(valid_ratio),
                    "micro_lambda_mean": micro_lambda_mean,
                    "mezzo_lambda_mean": mezzo_lambda_mean,
                    "macro_lambda_mean": macro_lambda_mean,
                    "gate_mean": gate_mean,
                    "gate_std": gate_std,
                }
                history["train"].append(train_row)
                last_train_row = train_row

                writer.add_scalar("train/loss", train_row["loss"], global_step)
                writer.add_scalar("train/huber", train_row["huber"], global_step)
                writer.add_scalar("train/mae", train_row["mae"], global_step)
                writer.add_scalar("train/mse", train_row["mse"], global_step)
                writer.add_scalar("train/lr", train_row["lr"], global_step)
                writer.add_scalar("data/num_valid", train_row["num_valid"], global_step)
                writer.add_scalar("data/valid_ratio", train_row["valid_ratio"], global_step)
                writer.add_scalar("model/micro_lambda_mean", train_row["micro_lambda_mean"], global_step)
                writer.add_scalar("model/mezzo_lambda_mean", train_row["mezzo_lambda_mean"], global_step)
                writer.add_scalar("model/macro_lambda_mean", train_row["macro_lambda_mean"], global_step)
                writer.add_scalar("model/gate_mean", train_row["gate_mean"], global_step)
                writer.add_scalar("model/gate_std", train_row["gate_std"], global_step)

                logger.info(
                    "epoch=%d step=%d train_loss=%.6f valid_ratio=%.4f",
                    epoch + 1,
                    global_step,
                    train_row["loss"],
                    train_row["valid_ratio"],
                )
                curve_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

            model.eval()
            val_loss_sum = 0.0
            val_huber_sum = 0.0
            val_mae_sum = 0.0
            val_mse_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for raw_batch in val_loader:
                    batch = _to_device(raw_batch, device)
                    with autocast(enabled=use_amp):
                        y_hat, _ = model(batch)
                        val_loss, val_metrics = masked_huber_loss(y_hat=y_hat, y_true=batch["y"], loss_mask=batch["loss_mask"])
                    val_loss_sum += float(val_loss.detach().item())
                    val_huber_sum += float(val_metrics["huber"].detach().item())
                    val_mae_sum += float(val_metrics["mae"].detach().item())
                    val_mse_sum += float(val_metrics["mse"].detach().item())
                    val_batches += 1

            denom = max(1, val_batches)
            val_row = {
                "epoch": int(epoch + 1),
                "loss": float(val_loss_sum / denom),
                "huber": float(val_huber_sum / denom),
                "mae": float(val_mae_sum / denom),
                "mse": float(val_mse_sum / denom),
            }
            history["val"].append(val_row)
            last_val_row = val_row
            epochs_completed = epoch + 1

            writer.add_scalar("val/loss", val_row["loss"], epochs_completed)
            writer.add_scalar("val/huber", val_row["huber"], epochs_completed)
            writer.add_scalar("val/mae", val_row["mae"], epochs_completed)
            writer.add_scalar("val/mse", val_row["mse"], epochs_completed)
            logger.info(
                "epoch=%d val_loss=%.6f val_huber=%.6f val_mae=%.6f val_mse=%.6f",
                epochs_completed,
                val_row["loss"],
                val_row["huber"],
                val_row["mae"],
                val_row["mse"],
            )
            curve_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

            for key in ("loss", "huber", "mae", "mse"):
                best_val[key] = min(best_val[key], val_row[key])

        feedback = {
            "meta": {
                "exp_name": config.exp_name,
                "status": "ok",
                "epochs_completed": int(epochs_completed),
            },
            "data": {
                "train_samples": int(train_sample_count),
                "val_samples": int(val_sample_count),
                "final_train_valid_ratio": float(last_train_row.get("valid_ratio", 0.0)),
                "final_val_loss": float(last_val_row.get("loss", 0.0)),
            },
            "model": {
                "micro_lambda_mean": float(last_train_row.get("micro_lambda_mean", 0.0)),
                "mezzo_lambda_mean": float(last_train_row.get("mezzo_lambda_mean", 0.0)),
                "macro_lambda_mean": float(last_train_row.get("macro_lambda_mean", 0.0)),
                "gate_mean": float(last_train_row.get("gate_mean", 0.0)),
                "gate_std": float(last_train_row.get("gate_std", 0.0)),
            },
            "metrics": {
                "best_val_loss": float(best_val["loss"] if best_val["loss"] != float("inf") else 0.0),
                "best_val_huber": float(best_val["huber"] if best_val["huber"] != float("inf") else 0.0),
                "best_val_mae": float(best_val["mae"] if best_val["mae"] != float("inf") else 0.0),
                "best_val_mse": float(best_val["mse"] if best_val["mse"] != float("inf") else 0.0),
            },
        }
        _write_yaml(feedback_path, feedback)
        writer.flush()
        writer.close()
        return {"history": history, "feedback": feedback}
    except Exception:
        error_tail = traceback.format_exc()[-4000:]
        crash_feedback = {
            "meta": {
                "exp_name": config.exp_name,
                "status": "crash",
                "epochs_completed": int(epochs_completed),
            },
            "data": {
                "train_samples": int(train_sample_count),
                "val_samples": int(val_sample_count),
                "final_train_valid_ratio": float(last_train_row.get("valid_ratio", 0.0)),
                "final_val_loss": float(last_val_row.get("loss", 0.0)),
            },
            "model": {
                "micro_lambda_mean": float(last_train_row.get("micro_lambda_mean", 0.0)),
                "mezzo_lambda_mean": float(last_train_row.get("mezzo_lambda_mean", 0.0)),
                "macro_lambda_mean": float(last_train_row.get("macro_lambda_mean", 0.0)),
                "gate_mean": float(last_train_row.get("gate_mean", 0.0)),
                "gate_std": float(last_train_row.get("gate_std", 0.0)),
            },
            "metrics": {
                "best_val_loss": float(best_val["loss"] if best_val["loss"] != float("inf") else 0.0),
                "best_val_huber": float(best_val["huber"] if best_val["huber"] != float("inf") else 0.0),
                "best_val_mae": float(best_val["mae"] if best_val["mae"] != float("inf") else 0.0),
                "best_val_mse": float(best_val["mse"] if best_val["mse"] != float("inf") else 0.0),
            },
            "error_traceback": error_tail,
        }
        _write_yaml(feedback_path, crash_feedback)
        writer.flush()
        writer.close()
        if raise_on_error:
            raise
        return {"history": history, "feedback": crash_feedback}


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train multi-scale regressor")
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--val-shards", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--y-key", type=str, default="y")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--buffer", action="store_true")

    args = parser.parse_args()
    return TrainConfig(
        train_shards=args.train_shards,
        val_shards=args.val_shards or [],
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        exp_name=args.exp_name,
        out_dir=args.out_dir,
        y_key=args.y_key,
        val_ratio=args.val_ratio,
        buffer=args.buffer,
    )


def main() -> None:
    config = _parse_args()
    run_training(config)


if __name__ == "__main__":
    main()
