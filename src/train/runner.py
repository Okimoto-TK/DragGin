from __future__ import annotations

import bisect
import json
import logging
import math
import traceback
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


def _load_shard_payload(path: str, y_key: str) -> dict[str, Any]:
    shard = np.load(path, allow_pickle=True).item()
    if y_key not in shard:
        raise KeyError(f"y_key '{y_key}' not found in shard: {path}")
    return shard




class ShardBatchIterator:
    def __init__(
        self,
        shard_paths: list[str],
        batch_size: int,
        y_key: str = "y",
        shuffle: bool = True,
        buffer: bool = False,
        row_indices_by_shard: dict[int, np.ndarray] | None = None,
        num_workers: int = 1,
    ) -> None:
        self.shard_paths = [str(path) for path in shard_paths]
        self.batch_size = int(batch_size)
        self.y_key = y_key
        self.shuffle = bool(shuffle)
        self.buffer = bool(buffer)
        self.row_indices_by_shard = row_indices_by_shard or {}
        self.num_workers = max(1, int(num_workers))
        self._rng = np.random.default_rng()
        self._sizes = [self._get_shard_size(path) for path in self.shard_paths]
        self._total_rows = int(sum(self._selected_rows_count(i) for i in range(len(self.shard_paths))))
        self._buffered_shards: list[dict[str, Any]] | None = None
        self.last_shard_order: list[str] = []
        self.last_row_orders: dict[str, list[int]] = {}

        if self.buffer:
            if self.num_workers > 1 and len(self.shard_paths) > 0:
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    self._buffered_shards = list(executor.map(_load_shard_payload, self.shard_paths, [self.y_key] * len(self.shard_paths)))
            else:
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
        return _load_shard_payload(path, self.y_key)

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
        ordered_indices = shard_indices.tolist()
        self.last_shard_order = [self.shard_paths[int(i)] for i in ordered_indices]
        self.last_row_orders = {}

        row_orders: dict[int, np.ndarray] = {}
        for shard_idx in ordered_indices:
            selected_rows = self.row_indices_by_shard.get(shard_idx)
            if selected_rows is None:
                row_indices = np.arange(self._sizes[shard_idx], dtype=np.int64)
            else:
                row_indices = np.asarray(selected_rows, dtype=np.int64).copy()
            if row_indices.size == 0:
                continue
            if self.shuffle:
                row_indices = self._rng.permutation(row_indices)
            row_orders[shard_idx] = row_indices
            path = self.shard_paths[shard_idx]
            self.last_row_orders[path] = row_indices.tolist()

        batch_chunks: list[tuple[dict[str, Any], np.ndarray]] = []
        buffered_rows = 0

        def _materialize_batch(chunks: list[tuple[dict[str, Any], np.ndarray]]) -> dict[str, Any]:
            if len(chunks) == 1:
                shard, rows = chunks[0]
                return _batch_from_shard_rows(shard=shard, rows=rows, y_key=self.y_key)
            return _batch_from_shard_chunks(chunks=chunks, y_key=self.y_key)

        def _consume_shard(shard_idx: int, shard: dict[str, Any]) -> list[dict[str, Any]]:
            nonlocal batch_chunks, buffered_rows
            emitted: list[dict[str, Any]] = []
            row_indices = row_orders[shard_idx]
            start = 0
            while start < row_indices.size:
                take = min(self.batch_size - buffered_rows, int(row_indices.size - start))
                rows = np.ascontiguousarray(row_indices[start : start + take], dtype=np.int64)
                batch_chunks.append((shard, rows))
                buffered_rows += int(rows.size)
                start += take
                if buffered_rows == self.batch_size:
                    emitted.append(_materialize_batch(batch_chunks))
                    batch_chunks = []
                    buffered_rows = 0
            return emitted

        if self._buffered_shards is not None:
            for shard_idx in ordered_indices:
                if shard_idx not in row_orders:
                    continue
                for out_batch in _consume_shard(shard_idx, self._buffered_shards[shard_idx]):
                    yield out_batch
        elif self.num_workers > 1:
            pending: deque[tuple[int, Future[dict[str, Any]]]] = deque()
            loadable_indices = [idx for idx in ordered_indices if idx in row_orders]
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                ptr = 0
                while ptr < len(loadable_indices) and len(pending) < self.num_workers:
                    shard_idx = loadable_indices[ptr]
                    pending.append((shard_idx, executor.submit(_load_shard_payload, self.shard_paths[shard_idx], self.y_key)))
                    ptr += 1

                while pending:
                    shard_idx, fut = pending.popleft()
                    shard = fut.result()
                    for out_batch in _consume_shard(shard_idx, shard):
                        yield out_batch
                    del shard

                    if ptr < len(loadable_indices):
                        next_idx = loadable_indices[ptr]
                        pending.append((next_idx, executor.submit(_load_shard_payload, self.shard_paths[next_idx], self.y_key)))
                        ptr += 1
        else:
            for shard_idx in ordered_indices:
                if shard_idx not in row_orders:
                    continue
                shard = self._load_shard(self.shard_paths[shard_idx])
                for out_batch in _consume_shard(shard_idx, shard):
                    yield out_batch
                del shard

        if buffered_rows > 0:
            yield _materialize_batch(batch_chunks)



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


def _batch_from_shard_rows(shard: dict[str, Any], rows: np.ndarray, y_key: str) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=np.int64)
    rows_contig = np.ascontiguousarray(rows)
    return {
        "code": shard["codes"][rows_contig].astype(str).tolist(),
        "asof_date": shard["asof_dates"][rows_contig].astype(str).tolist(),
        "x_micro": torch.from_numpy(np.ascontiguousarray(shard["X_micro"][rows_contig], dtype=np.float32)),
        "x_mezzo": torch.from_numpy(np.ascontiguousarray(shard["X_mezzo"][rows_contig], dtype=np.float32)),
        "x_macro": torch.from_numpy(np.ascontiguousarray(shard["X_macro"][rows_contig], dtype=np.float32)),
        "mask_micro": torch.from_numpy(np.ascontiguousarray(shard["mask_micro"][rows_contig])).to(torch.bool),
        "mask_mezzo": torch.from_numpy(np.ascontiguousarray(shard["mask_mezzo"][rows_contig])).to(torch.bool),
        "mask_macro": torch.from_numpy(np.ascontiguousarray(shard["mask_macro"][rows_contig])).to(torch.bool),
        "y": torch.from_numpy(np.ascontiguousarray(shard[y_key][rows_contig], dtype=np.float32)),
        "dp_ok": torch.from_numpy(np.ascontiguousarray(shard["dp_ok"][rows_contig])).to(torch.bool),
        "label_ok": torch.from_numpy(np.ascontiguousarray(shard["label_ok"][rows_contig])).to(torch.bool),
        "loss_mask": torch.from_numpy(np.ascontiguousarray(shard["loss_mask"][rows_contig])).to(torch.bool),
    }


def _batch_from_shard_chunks(chunks: list[tuple[dict[str, Any], np.ndarray]], y_key: str) -> dict[str, Any]:
    codes: list[str] = []
    asof_dates: list[str] = []

    def _cat_contiguous(key: str, dtype: np.dtype[Any] | None = None) -> np.ndarray:
        arrays = []
        for shard, rows in chunks:
            rows_contig = np.ascontiguousarray(rows)
            arr = shard[key][rows_contig]
            if dtype is not None:
                arr = arr.astype(dtype, copy=False)
            arrays.append(np.ascontiguousarray(arr))
        return np.ascontiguousarray(np.concatenate(arrays, axis=0))

    for shard, rows in chunks:
        rows_contig = np.ascontiguousarray(rows)
        codes.extend(shard["codes"][rows_contig].astype(str).tolist())
        asof_dates.extend(shard["asof_dates"][rows_contig].astype(str).tolist())

    return {
        "code": codes,
        "asof_date": asof_dates,
        "x_micro": torch.from_numpy(_cat_contiguous("X_micro", np.float32)),
        "x_mezzo": torch.from_numpy(_cat_contiguous("X_mezzo", np.float32)),
        "x_macro": torch.from_numpy(_cat_contiguous("X_macro", np.float32)),
        "mask_micro": torch.from_numpy(_cat_contiguous("mask_micro")).to(torch.bool),
        "mask_mezzo": torch.from_numpy(_cat_contiguous("mask_mezzo")).to(torch.bool),
        "mask_macro": torch.from_numpy(_cat_contiguous("mask_macro")).to(torch.bool),
        "y": torch.from_numpy(_cat_contiguous(y_key, np.float32)),
        "dp_ok": torch.from_numpy(_cat_contiguous("dp_ok")).to(torch.bool),
        "label_ok": torch.from_numpy(_cat_contiguous("label_ok")).to(torch.bool),
        "loss_mask": torch.from_numpy(_cat_contiguous("loss_mask")).to(torch.bool),
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
        gate_temperature: float = 1.0,
        use_seq_context: bool = True,
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
            gate_temperature=gate_temperature,
        )
        self.head = RegressionHead(hidden_dim=hidden_dim, use_seq_context=use_seq_context, dropout=dropout)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        force_gate_value: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
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
            force_gate_value=force_gate_value,
        )
        y_hat = self.head(fused_seq=fused_seq, fused_pool=fused_pool, mask_macro=batch["mask_macro"])
        aux_fusion["fused_pool"] = fused_pool
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
    gate_temperature: float = 1.0
    gate_std_target: float = 0.10
    gate_std_reg: float = 1e-2
    gate_mean_target: float = 0.50
    gate_mean_reg: float = 1e-2
    gate_entropy_reg: float = 5e-3
    gate_warmup_steps: int = 500
    y_key: str = "y"
    in_dim: int = 6
    enable_dynamic_threshold: bool = True
    enable_free_branch: bool = True
    init_lambda_micro: float = 1.5
    init_lambda_mezzo: float = 0.8
    init_lambda_macro: float = 0.3
    use_seq_context: bool = True
    clip_grad_norm: float | None = None
    gate_lr: float | None = None
    gate_clip_grad_norm: float | None = 0.2
    scheduler_name: str = "plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 6
    scheduler_min_lr: float = 1e-6
    finite_skip_max_warn: int = 20
    val_ratio: float = 0.15
    split_seed: int = 42
    buffer: bool = False
    num_workers: int = 1
    pin_memory: bool = False
    prefetch_cuda: bool = False
    enable_compile: bool = False
    compile_mode: str = "reduce-overhead"
    log_every: int = 10
    curve_save_every: int = 100
    hist_every: int = 100
    checkpoint: str | None = None
    save_every: int = 1


def _build_checkpoint_payload(
    *,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: AdamW,
    scaler: GradScaler,
    history: dict[str, list[dict[str, float | int]]],
    best_val: dict[str, float],
    scheduler: ReduceLROnPlateau | None = None,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "history": history,
        "best_val": best_val,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


class CUDAPrefetcher:
    def __init__(self, loader: Any, device: torch.device, pin_memory: bool) -> None:
        self.loader_iter = iter(loader)
        self.device = device
        self.pin_memory = pin_memory
        self.stream = torch.cuda.Stream(device=device)
        self.next_batch: dict[str, Any] | None = None
        self._preload()

    def _preload(self) -> None:
        try:
            raw_batch = next(self.loader_iter)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            self.next_batch = _to_device(raw_batch, self.device, pin_memory=self.pin_memory)

    def next(self) -> dict[str, Any] | None:
        torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            return None
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                value.record_stream(torch.cuda.current_stream(device=self.device))
        self._preload()
        return batch


def _to_device(batch: dict[str, Any], device: torch.device, pin_memory: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensor = value
            if pin_memory and tensor.device.type == "cpu":
                tensor = tensor.pin_memory()
            out[key] = tensor.to(device, non_blocking=True)
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


def _weighted_metric_value(metric: torch.Tensor, num_valid: int) -> float:
    return float(metric.detach().item()) * float(max(0, int(num_valid)))




def _is_finite_tensor(x: torch.Tensor | None) -> bool:
    if x is None:
        return True
    return bool(torch.isfinite(x).all().item())


def _safe_scalar(value: float | int) -> float | None:
    v = float(value)
    return v if math.isfinite(v) else None


def _safe_add_scalar(writer: SummaryWriter, tag: str, value: float | int, step: int) -> None:
    safe = _safe_scalar(value)
    if safe is not None:
        writer.add_scalar(tag, safe, step)


def _all_grads_finite(model: nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is not None and not _is_finite_tensor(p.grad):
            return False
    return True


def _resolve_gate_lr(config: TrainConfig) -> float:
    if config.gate_lr is not None:
        return float(config.gate_lr)
    return float(config.lr) * 0.5


def _gate_reg_decay_scale(*, progress: float) -> tuple[float, float, float]:
    """Late-phase decay for gate regularizers to reduce over-constraint near convergence."""
    if progress <= 0.2:
        return 1.0, 1.0, 1.0
    return 0.7, 0.3, 0.3




def _compute_gate_stats(gate: torch.Tensor, gate_logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute scalar statistics on vector gates (global over B,H unless stated otherwise)."""
    gate_mean = gate.mean()
    gate_global_std = gate.std(unbiased=False)
    gate_logits_mean = gate_logits.mean()
    gate_logits_std = gate_logits.std(unbiased=False)

    # mean_b(std_h(g)): channel variation within each sample.
    gate_channel_std = gate.std(dim=-1, unbiased=False).mean()
    # std_b(mean_h(g)): sample-level preference variation.
    gate_sample_mean_std = gate.mean(dim=-1).std(unbiased=False)
    # std_h(mean_b(g)): long-term channel preference divergence.
    gate_channel_mean_std = gate.mean(dim=0).std(unbiased=False)

    gate_saturation_frac_low = (gate < 0.1).to(dtype=gate.dtype).mean()
    gate_saturation_frac_high = (gate > 0.9).to(dtype=gate.dtype).mean()
    gate_mid_frac = ((gate >= 0.25) & (gate <= 0.75)).to(dtype=gate.dtype).mean()

    return {
        "gate_mean": gate_mean,
        "gate_global_std": gate_global_std,
        "gate_logits_mean": gate_logits_mean,
        "gate_logits_std": gate_logits_std,
        "gate_channel_std": gate_channel_std,
        "gate_sample_mean_std": gate_sample_mean_std,
        "gate_channel_mean_std": gate_channel_mean_std,
        "gate_saturation_frac_low": gate_saturation_frac_low,
        "gate_saturation_frac_high": gate_saturation_frac_high,
        "gate_mid_frac": gate_mid_frac,
    }


def _compute_global_pearson(y_hat: torch.Tensor, y_true: torch.Tensor) -> float:
    if y_hat.numel() < 2 or y_true.numel() < 2:
        return float("nan")
    y_hat_f = y_hat.to(dtype=torch.float32)
    y_true_f = y_true.to(dtype=torch.float32)
    y_hat_c = y_hat_f - y_hat_f.mean()
    y_true_c = y_true_f - y_true_f.mean()
    denom = torch.sqrt((y_hat_c.pow(2).sum()) * (y_true_c.pow(2).sum()))
    if not torch.isfinite(denom) or float(denom.item()) <= 0.0:
        return float("nan")
    corr = (y_hat_c * y_true_c).sum() / denom
    return float(corr.detach().item()) if torch.isfinite(corr) else float("nan")


def _compute_mean_y_true_when_yhat_gt_threshold(
    y_hat: torch.Tensor, y_true: torch.Tensor, threshold: float = 1.0
) -> tuple[float, int]:
    mask = y_hat > float(threshold)
    count = int(mask.sum().item())
    if count <= 0:
        return float("nan"), 0
    value = y_true[mask].to(dtype=torch.float32).mean()
    return (float(value.detach().item()) if torch.isfinite(value) else float("nan")), count


def _compute_sign_acc_when_abs_yhat_gt_threshold(
    y_hat: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5
) -> tuple[float, int]:
    mask = y_hat.abs() > float(threshold)
    count = int(mask.sum().item())
    if count <= 0:
        return float("nan"), 0
    correct = (y_hat[mask] > 0) == (y_true[mask] > 0)
    acc = correct.to(dtype=torch.float32).mean()
    return (float(acc.detach().item()) if torch.isfinite(acc) else float("nan")), count


def _safe_add_histogram(writer: SummaryWriter, tag: str, values: torch.Tensor, step: int) -> None:
    if values is None:
        return
    det = values.detach()
    if det.numel() == 0 or not torch.isfinite(det).all():
        return
    writer.add_histogram(tag, det.to(dtype=torch.float32).flatten().cpu(), step)


def run_training(config: TrainConfig, raise_on_error: bool = True) -> dict[str, Any]:
    out_dir = Path(config.out_dir)
    logs_dir = out_dir / "logs"
    runs_dir = out_dir / "runs" / config.exp_name
    metrics_dir = out_dir / "metrics"
    checkpoints_dir = out_dir / "checkpoints" / config.exp_name
    feedback_path = out_dir / "reports" / "feedback" / f"{config.exp_name}.yaml"

    logs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(logs_dir / "train.log")

    writer = SummaryWriter(log_dir=str(runs_dir))
    curve_path = metrics_dir / "curve.json"
    history: dict[str, list[dict[str, float | int]]] = {"train": [], "val": []}
    start_epoch = 0

    train_sample_count = 0
    val_sample_count = 0
    if config.val_shards:
        train_loader = ShardBatchIterator(
            config.train_shards,
            batch_size=config.batch_size,
            y_key=config.y_key,
            shuffle=True,
            buffer=config.buffer,
            num_workers=config.num_workers,
        )
        val_loader = ShardBatchIterator(
            config.val_shards,
            batch_size=config.batch_size,
            y_key=config.y_key,
            shuffle=False,
            buffer=config.buffer,
            num_workers=config.num_workers,
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
            num_workers=config.num_workers,
        )
        val_loader = ShardBatchIterator(
            config.train_shards,
            batch_size=config.batch_size,
            y_key=config.y_key,
            shuffle=False,
            buffer=config.buffer,
            row_indices_by_shard=val_rows,
            num_workers=config.num_workers,
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
        gate_temperature=config.gate_temperature,
        use_seq_context=config.use_seq_context,
    )
    gate_last_layer = model.fusion.gated.gate_mlp[-1]

    gate_reg_enabled = bool(model.fusion.gated.enable_free_branch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    model.to(device)
    if device.type == "cuda" and config.enable_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode=config.compile_mode)

    gate_lr = _resolve_gate_lr(config)
    gate_last_lr = float(config.lr) * 0.5
    gate_module = model.fusion.gated.gate_mlp
    gate_all_named_params = [(name, p) for name, p in gate_module.named_parameters() if p.requires_grad]
    gate_last_named_params = [(name, p) for name, p in gate_all_named_params if name.startswith("2.")]
    gate_other_named_params = [(name, p) for name, p in gate_all_named_params if not name.startswith("2.")]
    gate_param_ids = {id(p) for _, p in gate_all_named_params}
    main_params = [p for p in model.parameters() if id(p) not in gate_param_ids]

    optimizer = AdamW(
        [
            {"params": main_params, "lr": float(config.lr), "weight_decay": float(config.weight_decay), "name": "main"},
            {
                "params": [p for _, p in gate_other_named_params],
                "lr": float(gate_lr),
                "weight_decay": float(config.weight_decay),
                "name": "gate_other",
            },
            {
                "params": [p for _, p in gate_last_named_params],
                "lr": float(gate_last_lr),
                # Keep logits/bias calibration stable: no decay on gate output layer.
                "weight_decay": 0.0,
                "name": "gate_last_no_decay",
            },
        ]
    )
    scheduler: ReduceLROnPlateau | None = None
    if str(config.scheduler_name).lower() == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config.scheduler_factor),
            patience=max(1, int(config.scheduler_patience)),
            min_lr=float(config.scheduler_min_lr),
        )
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    global_step = 0
    epochs_completed = 0
    train_steps_per_epoch = max(1, len(train_loader))
    total_train_steps = max(1, train_steps_per_epoch * max(1, int(config.num_epochs)))
    best_val = {"loss": float("inf"), "huber": float("inf"), "mae": float("inf"), "mse": float("inf")}
    latest_ckpt_path = checkpoints_dir / "latest.ckpt"
    best_ckpt_path = checkpoints_dir / "best.ckpt"

    if config.checkpoint is not None:
        checkpoint_path = Path(config.checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler is not None and isinstance(scheduler_state, dict):
            scheduler.load_state_dict(scheduler_state)

        resume_main_lr = float(config.lr)
        resume_gate_lr = _resolve_gate_lr(config)
        resume_gate_last_lr = float(config.lr) * 0.5
        for idx, pg in enumerate(optimizer.param_groups):
            group_name = str(pg.get("name", ""))
            is_gate_group = group_name.startswith("gate_") or idx > 0
            if group_name == "gate_last_no_decay":
                pg["lr"] = float(resume_gate_last_lr)
                pg["weight_decay"] = 0.0
            else:
                pg["lr"] = float(resume_gate_lr if is_gate_group else resume_main_lr)
                pg["weight_decay"] = float(config.weight_decay)
            if "initial_lr" in pg:
                pg["initial_lr"] = float(pg["lr"])
        if scheduler is not None:
            scheduler.min_lrs = [float(config.scheduler_min_lr)] * len(optimizer.param_groups)
            scheduler._last_lr = [float(pg["lr"]) for pg in optimizer.param_groups]
        logger.info(
            "resume_override_optim_hparams lrs=%s weight_decay=%s",
            [float(pg["lr"]) for pg in optimizer.param_groups],
            [float(pg["weight_decay"]) for pg in optimizer.param_groups],
        )
        scaler_state = checkpoint.get("scaler_state_dict")
        if isinstance(scaler_state, dict):
            scaler.load_state_dict(scaler_state)
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        epochs_completed = start_epoch
        history_ckpt = checkpoint.get("history")
        if isinstance(history_ckpt, dict) and "train" in history_ckpt and "val" in history_ckpt:
            history = history_ckpt
        best_val_ckpt = checkpoint.get("best_val")
        if isinstance(best_val_ckpt, dict):
            for key in ("loss", "huber", "mae", "mse"):
                if key in best_val_ckpt:
                    best_val[key] = float(best_val_ckpt[key])
        logger.info("resumed_from=%s start_epoch=%d global_step=%d", str(checkpoint_path), start_epoch, global_step)
    logger.info(
        "optimizer_param_groups=%s",
        [
            {
                "idx": idx,
                "name": str(pg.get("name", f"group_{idx}")),
                "lr": float(pg["lr"]),
                "weight_decay": float(pg.get("weight_decay", 0.0)),
                "num_params": int(len(pg.get("params", []))),
            }
            for idx, pg in enumerate(optimizer.param_groups)
        ],
    )
    last_train_row: dict[str, Any] = {}
    last_val_row: dict[str, Any] = {"loss": 0.0}

    try:
        for epoch in range(start_epoch, start_epoch + config.num_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_train_iter = iter(train_loader)
            logger.info(
                "epoch=%d train_shard_iter_start buffer=%s workers=%d train_shards=%d sample_shards=%s",
                epoch + 1,
                str(config.buffer).lower(),
                int(config.num_workers),
                len(config.train_shards),
                [Path(p).name for p in train_loader.last_shard_order[:3]],
            )

            finite_skip_warn_count = 0

            def _warn_finite_skip(reason: str, batch: dict[str, Any], batch_idx: int) -> None:
                nonlocal finite_skip_warn_count
                if finite_skip_warn_count >= int(config.finite_skip_max_warn):
                    return
                finite_skip_warn_count += 1
                code = batch.get("code", [])
                asof = batch.get("asof_date", [])
                code0 = code[0] if isinstance(code, list | tuple) and len(code) > 0 else "n/a"
                asof0 = asof[0] if isinstance(asof, list | tuple) and len(asof) > 0 else "n/a"
                logger.warning(
                    "skip_non_finite epoch=%d step=%d batch_idx=%d reason=%s code=%s asof_date=%s",
                    epoch + 1,
                    global_step,
                    batch_idx,
                    reason,
                    str(code0),
                    str(asof0),
                )

            def _train_on_batch(batch: dict[str, Any], batch_idx: int) -> None:
                nonlocal global_step, last_train_row
                force_gate_value = 0.5 if global_step < int(config.gate_warmup_steps) else None
                try:
                    with autocast(enabled=use_amp):
                        y_hat, aux = model(batch, force_gate_value=force_gate_value)
                        loss, metrics = masked_huber_loss(y_hat=y_hat, y_true=batch["y"], loss_mask=batch["loss_mask"])
                        gate = aux["fusion"]["gate"]
                        gate_logits = aux["fusion"]["gate_logits"]
                        guided_pool = aux["fusion"]["guided_pool"]
                        free_pool = aux["fusion"]["free_pool"]
                        fused_pool = aux["fusion"].get("fused_pool", None)
                        gate_stats = _compute_gate_stats(gate=gate, gate_logits=gate_logits)
                        gate_mean = gate_stats["gate_mean"]
                        gate_std = gate_stats["gate_global_std"]
                        if gate_reg_enabled:
                            progress = min(1.0, float(global_step) / float(total_train_steps))
                            gate_std_scale, gate_mean_scale, gate_entropy_scale = _gate_reg_decay_scale(progress=progress)
                            effective_gate_std_reg = float(config.gate_std_reg) * gate_std_scale
                            effective_gate_mean_reg = float(config.gate_mean_reg) * gate_mean_scale
                            effective_gate_entropy_reg = float(config.gate_entropy_reg) * gate_entropy_scale
                            gate_target = torch.as_tensor(config.gate_std_target, dtype=gate_std.dtype, device=gate_std.device)
                            gate_std_penalty = torch.relu(gate_target - gate_std) ** 2
                            gate_mean_target = torch.as_tensor(config.gate_mean_target, dtype=gate_mean.dtype, device=gate_mean.device)
                            gate_mean_penalty = (gate_mean - gate_mean_target) ** 2
                            gate_eps = torch.as_tensor(1e-6, dtype=gate.dtype, device=gate.device)
                            gate_clamped = gate.clamp(gate_eps, 1.0 - gate_eps)
                            gate_entropy = -(gate_clamped * torch.log(gate_clamped) + (1.0 - gate_clamped) * torch.log(1.0 - gate_clamped)).mean()
                            gate_entropy_penalty = -gate_entropy
                            total_loss = (
                                loss
                                + effective_gate_std_reg * gate_std_penalty
                                + effective_gate_mean_reg * gate_mean_penalty
                                + effective_gate_entropy_reg * gate_entropy_penalty
                            )
                        else:
                            gate_std_penalty = torch.zeros((), dtype=loss.dtype, device=loss.device)
                            gate_mean_penalty = torch.zeros((), dtype=loss.dtype, device=loss.device)
                            gate_entropy_penalty = torch.zeros((), dtype=loss.dtype, device=loss.device)
                            effective_gate_std_reg = 0.0
                            effective_gate_mean_reg = 0.0
                            effective_gate_entropy_reg = 0.0
                            total_loss = loss
                        scaled_loss = total_loss / max(1, config.grad_accum_steps)
                except RuntimeError as exc:
                    _warn_finite_skip(reason=f"forward_runtime_error:{exc}", batch=batch, batch_idx=batch_idx)
                    optimizer.zero_grad(set_to_none=True)
                    return

                if not _is_finite_tensor(y_hat):
                    _warn_finite_skip(reason="y_hat_non_finite", batch=batch, batch_idx=batch_idx)
                    optimizer.zero_grad(set_to_none=True)
                    return
                for name, tensor in {
                    "gate_logits": gate_logits,
                    "gate": gate,
                    "guided_pool": guided_pool,
                    "free_pool": free_pool,
                    "fused_pool": fused_pool,
                    "total_loss": total_loss,
                }.items():
                    if not _is_finite_tensor(tensor):
                        _warn_finite_skip(reason=f"{name}_non_finite", batch=batch, batch_idx=batch_idx)
                        optimizer.zero_grad(set_to_none=True)
                        return

                if scaled_loss.requires_grad:
                    scaler.scale(scaled_loss).backward()

                should_step = (batch_idx + 1) % max(1, config.grad_accum_steps) == 0 or (batch_idx + 1) == len(train_loader)
                if not should_step:
                    return

                has_non_finite = False
                did_unscale = False
                if scaled_loss.requires_grad:
                    scaler.unscale_(optimizer)
                    did_unscale = True
                    if not _all_grads_finite(model):
                        has_non_finite = True
                    gate_weight_grad = gate_last_layer.weight.grad
                    gate_bias_grad = gate_last_layer.bias.grad
                    if not _is_finite_tensor(gate_weight_grad) or not _is_finite_tensor(gate_bias_grad):
                        has_non_finite = True
                    if has_non_finite:
                        _warn_finite_skip(reason="non_finite_grad", batch=batch, batch_idx=batch_idx)
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        return
                    if config.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
                    if config.gate_clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(gate_last_layer.parameters(), config.gate_clip_grad_norm)

                gate_last_grad_norm = 0.0
                if gate_last_layer.weight.grad is not None:
                    gate_last_grad_norm = float(gate_last_layer.weight.grad.detach().norm().item())
                elif gate_last_layer.bias.grad is not None:
                    gate_last_grad_norm = float(gate_last_layer.bias.grad.detach().norm().item())
                if not math.isfinite(gate_last_grad_norm):
                    _warn_finite_skip(reason="gate_last_grad_norm_non_finite", batch=batch, batch_idx=batch_idx)
                    optimizer.zero_grad(set_to_none=True)
                    if did_unscale:
                        scaler.update()
                    return

                if scaled_loss.requires_grad:
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                batch_size_current = int(batch["y"].shape[0])
                num_valid = int(metrics["num_valid"])
                valid_ratio = float(num_valid / max(1, batch_size_current))
                gate_last_weight_norm = float(gate_last_layer.weight.detach().norm().item())
                gate_last_bias = float(gate_last_layer.bias.detach().mean().item())
                guided_pool_norm = float(guided_pool.detach().norm(dim=-1).mean().item())
                free_pool_norm = float(free_pool.detach().norm(dim=-1).mean().item())
                guided_free_gap = float(torch.abs(guided_pool - free_pool).mean().detach().item())
                train_row = {
                    "step": int(global_step),
                    "loss": float(loss.detach().item()),
                    "total_loss": float(total_loss.detach().item()),
                    "huber": float(metrics["huber"].detach().item()),
                    "mae": float(metrics["mae"].detach().item()),
                    "mse": float(metrics["mse"].detach().item()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "num_valid": int(num_valid),
                    "valid_ratio": float(valid_ratio),
                    "micro_lambda_mean": float(aux["micro"]["lambda_mean"].mean().detach().item()),
                    "mezzo_lambda_mean": float(aux["mezzo"]["lambda_mean"].mean().detach().item()),
                    "macro_lambda_mean": float(aux["macro"]["lambda_mean"].mean().detach().item()),
                    "gate_mean": float(gate_mean.detach().item()),
                    "gate_std": float(gate_std.detach().item()),
                    "gate_global_std": float(gate_stats["gate_global_std"].detach().item()),
                    "gate_logits_mean": float(gate_stats["gate_logits_mean"].detach().item()),
                    "gate_logits_std": float(gate_stats["gate_logits_std"].detach().item()),
                    "gate_channel_std": float(gate_stats["gate_channel_std"].detach().item()),
                    "gate_sample_mean_std": float(gate_stats["gate_sample_mean_std"].detach().item()),
                    "gate_channel_mean_std": float(gate_stats["gate_channel_mean_std"].detach().item()),
                    "gate_saturation_frac_low": float(gate_stats["gate_saturation_frac_low"].detach().item()),
                    "gate_saturation_frac_high": float(gate_stats["gate_saturation_frac_high"].detach().item()),
                    "gate_mid_frac": float(gate_stats["gate_mid_frac"].detach().item()),
                    "gate_std_penalty": float(gate_std_penalty.detach().item()),
                    "gate_mean_penalty": float(gate_mean_penalty.detach().item()),
                    "gate_entropy_penalty": float(gate_entropy_penalty.detach().item()),
                    "effective_gate_mean_reg": float(effective_gate_mean_reg),
                    "effective_gate_std_reg": float(effective_gate_std_reg),
                    "effective_gate_entropy_reg": float(effective_gate_entropy_reg),
                    "gate_last_bias": gate_last_bias,
                    "gate_last_weight_norm": gate_last_weight_norm,
                    "gate_last_grad_norm": gate_last_grad_norm,
                    "guided_pool_norm": guided_pool_norm,
                    "free_pool_norm": free_pool_norm,
                    "guided_free_gap": guided_free_gap,
                    "force_gate_value": float(force_gate_value) if force_gate_value is not None else -1.0,
                }
                finite_row = all(_safe_scalar(v) is not None for v in train_row.values() if isinstance(v, float | int))
                if not finite_row:
                    _warn_finite_skip(reason="train_row_non_finite", batch=batch, batch_idx=batch_idx)
                    return
                history["train"].append(train_row)
                last_train_row = train_row

                _safe_add_scalar(writer, "train/loss", train_row["loss"], global_step)
                _safe_add_scalar(writer, "train/total_loss", train_row["total_loss"], global_step)
                _safe_add_scalar(writer, "train/gate_std_penalty", train_row["gate_std_penalty"], global_step)
                _safe_add_scalar(writer, "train/gate_mean_penalty", train_row["gate_mean_penalty"], global_step)
                _safe_add_scalar(writer, "train/gate_entropy_penalty", train_row["gate_entropy_penalty"], global_step)
                _safe_add_scalar(writer, "train/effective_gate_mean_reg", train_row["effective_gate_mean_reg"], global_step)
                _safe_add_scalar(writer, "train/effective_gate_std_reg", train_row["effective_gate_std_reg"], global_step)
                _safe_add_scalar(writer, "train/effective_gate_entropy_reg", train_row["effective_gate_entropy_reg"], global_step)
                _safe_add_scalar(writer, "train/huber", train_row["huber"], global_step)
                _safe_add_scalar(writer, "train/mae", train_row["mae"], global_step)
                _safe_add_scalar(writer, "train/mse", train_row["mse"], global_step)
                _safe_add_scalar(writer, "train/lr", train_row["lr"], global_step)
                _safe_add_scalar(writer, "train/force_gate_value", train_row["force_gate_value"], global_step)
                _safe_add_scalar(writer, "train/gate_last_grad_norm", train_row["gate_last_grad_norm"], global_step)
                _safe_add_scalar(writer, "data/num_valid", train_row["num_valid"], global_step)
                _safe_add_scalar(writer, "data/valid_ratio", train_row["valid_ratio"], global_step)
                _safe_add_scalar(writer, "model/micro_lambda_mean", train_row["micro_lambda_mean"], global_step)
                _safe_add_scalar(writer, "model/mezzo_lambda_mean", train_row["mezzo_lambda_mean"], global_step)
                _safe_add_scalar(writer, "model/macro_lambda_mean", train_row["macro_lambda_mean"], global_step)
                _safe_add_scalar(writer, "train_model/gate_mean", train_row["gate_mean"], global_step)
                _safe_add_scalar(writer, "train_model/gate_std", train_row["gate_std"], global_step)
                _safe_add_scalar(writer, "train_model/gate_global_std", train_row["gate_global_std"], global_step)
                _safe_add_scalar(writer, "train_model/gate_channel_std", train_row["gate_channel_std"], global_step)
                _safe_add_scalar(writer, "train_model/gate_sample_mean_std", train_row["gate_sample_mean_std"], global_step)
                _safe_add_scalar(writer, "train_model/gate_channel_mean_std", train_row["gate_channel_mean_std"], global_step)
                _safe_add_scalar(writer, "train_model/gate_saturation_frac_low", train_row["gate_saturation_frac_low"], global_step)
                _safe_add_scalar(writer, "train_model/gate_saturation_frac_high", train_row["gate_saturation_frac_high"], global_step)
                _safe_add_scalar(writer, "train_model/gate_mid_frac", train_row["gate_mid_frac"], global_step)
                _safe_add_scalar(writer, "train_model/gate_logits_mean", train_row["gate_logits_mean"], global_step)
                _safe_add_scalar(writer, "train_model/gate_logits_std", train_row["gate_logits_std"], global_step)
                _safe_add_scalar(writer, "train_model/gate_last_bias", train_row["gate_last_bias"], global_step)
                _safe_add_scalar(writer, "train_model/gate_last_weight_norm", train_row["gate_last_weight_norm"], global_step)
                _safe_add_scalar(writer, "train_model/guided_pool_norm", train_row["guided_pool_norm"], global_step)
                _safe_add_scalar(writer, "train_model/free_pool_norm", train_row["free_pool_norm"], global_step)
                _safe_add_scalar(writer, "train_model/guided_free_gap", train_row["guided_free_gap"], global_step)

                if global_step % max(1, int(config.hist_every)) == 0:
                    # Histograms are interval-logged to avoid oversized TensorBoard event files.
                    _safe_add_histogram(writer, "model/gate_values_hist", gate, global_step)
                    _safe_add_histogram(writer, "model/gate_logits_hist", gate_logits, global_step)
                    _safe_add_histogram(writer, "model/gate_channel_mean_hist", gate.mean(dim=0), global_step)
                    _safe_add_histogram(writer, "model/gate_channel_std_hist", gate.std(dim=0, unbiased=False), global_step)

                if global_step % max(1, config.log_every) == 0:
                    logger.info("epoch=%d step=%d train_loss=%.6f valid_ratio=%.4f", epoch + 1, global_step, train_row["loss"], train_row["valid_ratio"])
                if global_step % max(1, config.curve_save_every) == 0:
                    curve_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

            if device.type == "cuda" and config.prefetch_cuda:
                prefetcher = CUDAPrefetcher(train_loader, device=device, pin_memory=config.pin_memory)
                batch_idx = 0
                while True:
                    batch = prefetcher.next()
                    if batch is None:
                        break
                    _train_on_batch(batch, batch_idx)
                    batch_idx += 1
            else:
                for batch_idx, raw_batch in enumerate(epoch_train_iter):
                    batch = _to_device(raw_batch, device, pin_memory=config.pin_memory)
                    _train_on_batch(batch, batch_idx)

            model.eval()
            val_loss_weighted_sum = 0.0
            val_huber_weighted_sum = 0.0
            val_mae_weighted_sum = 0.0
            val_mse_weighted_sum = 0.0
            val_num_valid_total = 0
            val_preds_chunks: list[torch.Tensor] = []
            val_targets_chunks: list[torch.Tensor] = []
            val_gate_stats_acc = {
                "gate_mean": 0.0,
                "gate_global_std": 0.0,
                "gate_logits_mean": 0.0,
                "gate_logits_std": 0.0,
                "gate_channel_std": 0.0,
                "gate_sample_mean_std": 0.0,
                "gate_channel_mean_std": 0.0,
                "gate_saturation_frac_low": 0.0,
                "gate_saturation_frac_high": 0.0,
                "gate_mid_frac": 0.0,
                "guided_pool_norm": 0.0,
                "free_pool_norm": 0.0,
                "guided_free_gap": 0.0,
            }
            with torch.no_grad():
                if device.type == "cuda" and config.prefetch_cuda:
                    val_prefetcher = CUDAPrefetcher(val_loader, device=device, pin_memory=config.pin_memory)
                    while True:
                        batch = val_prefetcher.next()
                        if batch is None:
                            break
                        with autocast(enabled=use_amp):
                            y_hat, aux = model(batch)
                            val_loss, val_metrics = masked_huber_loss(y_hat=y_hat, y_true=batch["y"], loss_mask=batch["loss_mask"])
                        gate_stats = _compute_gate_stats(gate=aux["fusion"]["gate"], gate_logits=aux["fusion"]["gate_logits"])
                        num_valid = int(val_metrics["num_valid"])
                        val_num_valid_total += num_valid
                        valid_mask = batch["loss_mask"].to(dtype=torch.bool)
                        if bool(valid_mask.any()):
                            val_preds_chunks.append(y_hat[valid_mask].detach().float().cpu())
                            val_targets_chunks.append(batch["y"][valid_mask].detach().float().cpu())
                        val_loss_weighted_sum += _weighted_metric_value(val_loss, num_valid)
                        val_huber_weighted_sum += _weighted_metric_value(val_metrics["huber"], num_valid)
                        val_mae_weighted_sum += _weighted_metric_value(val_metrics["mae"], num_valid)
                        val_mse_weighted_sum += _weighted_metric_value(val_metrics["mse"], num_valid)
                        guided_pool = aux["fusion"]["guided_pool"]
                        free_pool = aux["fusion"]["free_pool"]
                        val_gate_stats_acc["guided_pool_norm"] += _weighted_metric_value(guided_pool.detach().norm(dim=-1).mean(), num_valid)
                        val_gate_stats_acc["free_pool_norm"] += _weighted_metric_value(free_pool.detach().norm(dim=-1).mean(), num_valid)
                        val_gate_stats_acc["guided_free_gap"] += _weighted_metric_value(torch.abs(guided_pool - free_pool).mean(), num_valid)
                        for key in ("gate_mean", "gate_global_std", "gate_logits_mean", "gate_logits_std", "gate_channel_std", "gate_sample_mean_std", "gate_channel_mean_std", "gate_saturation_frac_low", "gate_saturation_frac_high", "gate_mid_frac"):
                            val_gate_stats_acc[key] += _weighted_metric_value(gate_stats[key], num_valid)
                else:
                    for raw_batch in val_loader:
                        batch = _to_device(raw_batch, device, pin_memory=config.pin_memory)
                        with autocast(enabled=use_amp):
                            y_hat, aux = model(batch)
                            val_loss, val_metrics = masked_huber_loss(y_hat=y_hat, y_true=batch["y"], loss_mask=batch["loss_mask"])
                        gate_stats = _compute_gate_stats(gate=aux["fusion"]["gate"], gate_logits=aux["fusion"]["gate_logits"])
                        num_valid = int(val_metrics["num_valid"])
                        val_num_valid_total += num_valid
                        valid_mask = batch["loss_mask"].to(dtype=torch.bool)
                        if bool(valid_mask.any()):
                            val_preds_chunks.append(y_hat[valid_mask].detach().float().cpu())
                            val_targets_chunks.append(batch["y"][valid_mask].detach().float().cpu())
                        val_loss_weighted_sum += _weighted_metric_value(val_loss, num_valid)
                        val_huber_weighted_sum += _weighted_metric_value(val_metrics["huber"], num_valid)
                        val_mae_weighted_sum += _weighted_metric_value(val_metrics["mae"], num_valid)
                        val_mse_weighted_sum += _weighted_metric_value(val_metrics["mse"], num_valid)
                        guided_pool = aux["fusion"]["guided_pool"]
                        free_pool = aux["fusion"]["free_pool"]
                        val_gate_stats_acc["guided_pool_norm"] += _weighted_metric_value(guided_pool.detach().norm(dim=-1).mean(), num_valid)
                        val_gate_stats_acc["free_pool_norm"] += _weighted_metric_value(free_pool.detach().norm(dim=-1).mean(), num_valid)
                        val_gate_stats_acc["guided_free_gap"] += _weighted_metric_value(torch.abs(guided_pool - free_pool).mean(), num_valid)
                        for key in ("gate_mean", "gate_global_std", "gate_logits_mean", "gate_logits_std", "gate_channel_std", "gate_sample_mean_std", "gate_channel_mean_std", "gate_saturation_frac_low", "gate_saturation_frac_high", "gate_mid_frac"):
                            val_gate_stats_acc[key] += _weighted_metric_value(gate_stats[key], num_valid)

            denom = max(1, int(val_num_valid_total))
            if len(val_preds_chunks) > 0:
                y_hat_all = torch.cat(val_preds_chunks, dim=0)
                y_true_all = torch.cat(val_targets_chunks, dim=0)
            else:
                y_hat_all = torch.empty(0, dtype=torch.float32)
                y_true_all = torch.empty(0, dtype=torch.float32)

            global_pearson = _compute_global_pearson(y_hat=y_hat_all, y_true=y_true_all)
            y_true_mean_when_yhat_gt_1, count_yhat_gt_1 = _compute_mean_y_true_when_yhat_gt_threshold(
                y_hat=y_hat_all, y_true=y_true_all, threshold=1.0
            )
            sign_acc_when_abs_yhat_gt_0_5, count_abs_yhat_gt_0_5 = _compute_sign_acc_when_abs_yhat_gt_threshold(
                y_hat=y_hat_all, y_true=y_true_all, threshold=0.5
            )

            val_row = {
                "epoch": int(epoch + 1),
                "loss": float(val_loss_weighted_sum / denom),
                "huber": float(val_huber_weighted_sum / denom),
                "mae": float(val_mae_weighted_sum / denom),
                "mse": float(val_mse_weighted_sum / denom),
                "gate_mean": float(val_gate_stats_acc["gate_mean"] / denom),
                "gate_global_std": float(val_gate_stats_acc["gate_global_std"] / denom),
                "gate_std": float(val_gate_stats_acc["gate_global_std"] / denom),
                "gate_logits_mean": float(val_gate_stats_acc["gate_logits_mean"] / denom),
                "gate_logits_std": float(val_gate_stats_acc["gate_logits_std"] / denom),
                "gate_channel_std": float(val_gate_stats_acc["gate_channel_std"] / denom),
                "gate_sample_mean_std": float(val_gate_stats_acc["gate_sample_mean_std"] / denom),
                "gate_channel_mean_std": float(val_gate_stats_acc["gate_channel_mean_std"] / denom),
                "gate_saturation_frac_low": float(val_gate_stats_acc["gate_saturation_frac_low"] / denom),
                "gate_saturation_frac_high": float(val_gate_stats_acc["gate_saturation_frac_high"] / denom),
                "gate_mid_frac": float(val_gate_stats_acc["gate_mid_frac"] / denom),
                "guided_pool_norm": float(val_gate_stats_acc["guided_pool_norm"] / denom),
                "free_pool_norm": float(val_gate_stats_acc["free_pool_norm"] / denom),
                "guided_free_gap": float(val_gate_stats_acc["guided_free_gap"] / denom),
                "global_pearson": float(global_pearson),
                "y_true_mean_when_yhat_gt_1": float(y_true_mean_when_yhat_gt_1),
                "count_yhat_gt_1": int(count_yhat_gt_1),
                "sign_acc_when_abs_yhat_gt_0_5": float(sign_acc_when_abs_yhat_gt_0_5),
                "count_abs_yhat_gt_0_5": int(count_abs_yhat_gt_0_5),
            }
            required_val_keys = (
                "epoch",
                "loss",
                "huber",
                "mae",
                "mse",
                "gate_mean",
                "gate_global_std",
                "gate_std",
                "gate_logits_mean",
                "gate_logits_std",
                "gate_channel_std",
                "gate_sample_mean_std",
                "gate_channel_mean_std",
                "gate_saturation_frac_low",
                "gate_saturation_frac_high",
                "gate_mid_frac",
                "guided_pool_norm",
                "free_pool_norm",
                "guided_free_gap",
                "count_yhat_gt_1",
                "count_abs_yhat_gt_0_5",
            )
            if not all(_safe_scalar(val_row[k]) is not None for k in required_val_keys):
                logger.warning("skip_non_finite_val epoch=%d values=%s", epoch + 1, val_row)
                continue
            history["val"].append(val_row)
            last_val_row = val_row
            epochs_completed = epoch + 1

            _safe_add_scalar(writer, "val/loss", val_row["loss"], epochs_completed)
            _safe_add_scalar(writer, "val/huber", val_row["huber"], epochs_completed)
            _safe_add_scalar(writer, "val/mae", val_row["mae"], epochs_completed)
            _safe_add_scalar(writer, "val/mse", val_row["mse"], epochs_completed)
            _safe_add_scalar(writer, "val/global_pearson", val_row["global_pearson"], epochs_completed)
            _safe_add_scalar(writer, "val/y_true_mean_when_yhat_gt_1", val_row["y_true_mean_when_yhat_gt_1"], epochs_completed)
            _safe_add_scalar(writer, "val/count_yhat_gt_1", val_row["count_yhat_gt_1"], epochs_completed)
            _safe_add_scalar(writer, "val/sign_acc_when_abs_yhat_gt_0_5", val_row["sign_acc_when_abs_yhat_gt_0_5"], epochs_completed)
            _safe_add_scalar(writer, "val/count_abs_yhat_gt_0_5", val_row["count_abs_yhat_gt_0_5"], epochs_completed)
            _safe_add_scalar(writer, "val_model/gate_mean", val_row["gate_mean"], epochs_completed)
            _safe_add_scalar(writer, "val_model/gate_std", val_row["gate_std"], epochs_completed)
            _safe_add_scalar(writer, "val_model/gate_global_std", val_row["gate_global_std"], epochs_completed)
            _safe_add_scalar(writer, "val_model/gate_logits_mean", val_row["gate_logits_mean"], epochs_completed)
            _safe_add_scalar(writer, "val_model/gate_logits_std", val_row["gate_logits_std"], epochs_completed)
            _safe_add_scalar(writer, "val_model/gate_channel_std", val_row["gate_channel_std"], epochs_completed)
            _safe_add_scalar(writer, "val_model/gate_sample_mean_std", val_row["gate_sample_mean_std"], epochs_completed)
            _safe_add_scalar(writer, "val_model/gate_channel_mean_std", val_row["gate_channel_mean_std"], epochs_completed)
            _safe_add_scalar(writer, "val_model/gate_saturation_frac_low", val_row["gate_saturation_frac_low"], epochs_completed)
            _safe_add_scalar(writer, "val_model/gate_saturation_frac_high", val_row["gate_saturation_frac_high"], epochs_completed)
            _safe_add_scalar(writer, "val_model/gate_mid_frac", val_row["gate_mid_frac"], epochs_completed)
            _safe_add_scalar(writer, "val_model/guided_pool_norm", val_row["guided_pool_norm"], epochs_completed)
            _safe_add_scalar(writer, "val_model/free_pool_norm", val_row["free_pool_norm"], epochs_completed)
            _safe_add_scalar(writer, "val_model/guided_free_gap", val_row["guided_free_gap"], epochs_completed)
            logger.info(
                "epoch=%d val_loss=%.6f val_huber=%.6f val_mae=%.6f val_mse=%.6f pearson=%.6f ytrue@pred>1=%.6f sign_acc@|pred|>0.5=%.6f",
                epochs_completed,
                val_row["loss"],
                val_row["huber"],
                val_row["mae"],
                val_row["mse"],
                val_row["global_pearson"],
                val_row["y_true_mean_when_yhat_gt_1"],
                val_row["sign_acc_when_abs_yhat_gt_0_5"],
            )
            curve_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

            if scheduler is not None and math.isfinite(val_row["loss"]):
                scheduler.step(val_row["loss"])

            is_best = val_row["loss"] <= best_val["loss"]
            for key in ("loss", "huber", "mae", "mse"):
                best_val[key] = min(best_val[key], val_row[key])

            checkpoint_payload = _build_checkpoint_payload(
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                history=history,
                best_val=best_val,
                scheduler=scheduler,
            )
            _save_checkpoint(latest_ckpt_path, checkpoint_payload)
            if is_best:
                _save_checkpoint(best_ckpt_path, checkpoint_payload)
            if epochs_completed % max(1, int(config.save_every)) == 0:
                epoch_ckpt_path = checkpoints_dir / f"epoch_{epochs_completed:04d}.ckpt"
                _save_checkpoint(epoch_ckpt_path, checkpoint_payload)

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

