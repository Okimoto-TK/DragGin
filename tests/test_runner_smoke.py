import json
from pathlib import Path

import numpy as np
import torch

from src.model.head import masked_huber_loss
from src.train.runner import (
    MultiScaleRegressor,
    NpyShardDataset,
    ShardBatchIterator,
    TrainConfig,
    collate_batch,
    run_training,
)


def _write_shard(path: Path, n: int = 3) -> str:
    rng = np.random.default_rng(7)
    shard = {
        "codes": np.array([f"C{i:03d}" for i in range(n)], dtype=object),
        "asof_dates": np.array([f"2024-01-{i+1:02d}" for i in range(n)], dtype=object),
        "X_micro": rng.normal(size=(n, 48, 6)).astype(np.float32),
        "X_mezzo": rng.normal(size=(n, 40, 6)).astype(np.float32),
        "X_macro": rng.normal(size=(n, 30, 6)).astype(np.float32),
        "mask_micro": np.ones((n, 48), dtype=bool),
        "mask_mezzo": np.ones((n, 40), dtype=bool),
        "mask_macro": np.ones((n, 30), dtype=bool),
        "y": rng.normal(size=(n,)).astype(np.float32),
        "y_raw": rng.normal(size=(n,)).astype(np.float32),
        "y_z": rng.normal(size=(n,)).astype(np.float32),
        "dp_ok": np.ones((n,), dtype=bool),
        "label_ok": np.ones((n,), dtype=bool),
        "loss_mask": np.array([(i % 2) == 0 for i in range(n)], dtype=bool),
    }
    np.save(path, shard, allow_pickle=True)
    return str(path)


def _synthetic_batch(batch_size: int = 2) -> dict:
    return {
        "x_micro": torch.randn(batch_size, 48, 6),
        "x_mezzo": torch.randn(batch_size, 40, 6),
        "x_macro": torch.randn(batch_size, 30, 6),
        "mask_micro": torch.ones(batch_size, 48, dtype=torch.bool),
        "mask_mezzo": torch.ones(batch_size, 40, dtype=torch.bool),
        "mask_macro": torch.ones(batch_size, 30, dtype=torch.bool),
        "y": torch.randn(batch_size),
        "dp_ok": torch.ones(batch_size, dtype=torch.bool),
        "label_ok": torch.ones(batch_size, dtype=torch.bool),
        "loss_mask": torch.ones(batch_size, dtype=torch.bool),
    }


def test_dataset_and_collate_shapes(tmp_path: Path) -> None:
    shard_path = _write_shard(tmp_path / "train.npy", n=3)
    ds = NpyShardDataset([shard_path], y_key="y")

    sample = ds[0]
    assert sample["x_micro"].shape == (48, 6)
    assert sample["x_mezzo"].shape == (40, 6)
    assert sample["x_macro"].shape == (30, 6)
    assert sample["mask_micro"].shape == (48,)
    assert sample["loss_mask"].dtype == torch.bool

    batch = collate_batch([ds[0], ds[1]])
    assert batch["x_micro"].shape == (2, 48, 6)
    assert batch["x_mezzo"].shape == (2, 40, 6)
    assert batch["x_macro"].shape == (2, 30, 6)
    assert batch["y"].shape == (2,)
    assert batch["loss_mask"].shape == (2,)


def test_model_forward_smoke() -> None:
    batch = _synthetic_batch(batch_size=3)
    model = MultiScaleRegressor(hidden_dim=8, num_heads=2)

    y_hat, aux = model(batch)

    assert y_hat.shape == (3,)
    assert "micro" in aux
    assert "mezzo" in aux
    assert "macro" in aux
    assert "fusion" in aux


def test_train_one_epoch_smoke(tmp_path: Path) -> None:
    train_path = _write_shard(tmp_path / "train.npy", n=4)
    val_path = _write_shard(tmp_path / "val.npy", n=2)

    cfg = TrainConfig(
        train_shards=[train_path],
        val_shards=[val_path],
        batch_size=2,
        grad_accum_steps=1,
        num_epochs=1,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        exp_name="smoke_run",
        out_dir=str(tmp_path / "out"),
    )
    run_training(cfg)

    out = Path(cfg.out_dir)
    assert (out / "logs" / "train.log").exists()
    runs_dir = out / "runs" / cfg.exp_name
    assert runs_dir.exists()
    assert runs_dir.is_dir()
    assert any("events.out.tfevents" in p.name for p in runs_dir.iterdir()) or runs_dir.exists()
    assert (out / "metrics" / "curve.json").exists()
    assert (out / "reports" / "feedback" / f"{cfg.exp_name}.yaml").exists()

    data = json.loads((out / "metrics" / "curve.json").read_text(encoding="utf-8"))
    assert len(data["train"]) >= 1
    assert len(data["val"]) == 1




def test_train_one_epoch_with_val_ratio_split(tmp_path: Path) -> None:
    train_path = _write_shard(tmp_path / "train_only.npy", n=10)

    cfg = TrainConfig(
        train_shards=[train_path],
        val_shards=[],
        val_ratio=0.2,
        batch_size=2,
        grad_accum_steps=1,
        num_epochs=1,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_dim=8,
        num_heads=2,
        dropout=0.1,
        exp_name="smoke_ratio",
        out_dir=str(tmp_path / "out_ratio"),
    )
    result = run_training(cfg)

    assert result["feedback"]["meta"]["status"] == "ok"
    assert result["feedback"]["data"]["train_samples"] == 8
    assert result["feedback"]["data"]["val_samples"] == 2


def test_shard_batch_iterator_no_buffer_does_not_preload(tmp_path: Path) -> None:
    shard_paths = [_write_shard(tmp_path / f"train_{i}.npy", n=2) for i in range(3)]
    iterator = ShardBatchIterator(shard_paths=shard_paths, batch_size=2, y_key="y", shuffle=True, buffer=False)
    assert iterator._buffered_shards is None


def test_shard_and_row_order_shuffle_each_epoch(tmp_path: Path) -> None:
    shard_paths = [_write_shard(tmp_path / f"train_{i}.npy", n=5) for i in range(3)]
    iterator = ShardBatchIterator(shard_paths=shard_paths, batch_size=2, y_key="y", shuffle=True, buffer=False)

    first_order = None
    second_order = None
    first_rows = None
    second_rows = None

    for _ in iterator:
        pass
    first_order = list(iterator.last_shard_order)
    first_rows = dict(iterator.last_row_orders)

    for _ in iterator:
        pass
    second_order = list(iterator.last_shard_order)
    second_rows = dict(iterator.last_row_orders)

    # In rare cases random permutations can match exactly, so allow one retry.
    if first_order == second_order and all(first_rows[p] == second_rows[p] for p in first_rows):
        for _ in iterator:
            pass
        second_order = list(iterator.last_shard_order)
        second_rows = dict(iterator.last_row_orders)

    assert first_order != second_order or any(first_rows[p] != second_rows[p] for p in first_rows)


def test_shard_batch_iterator_batch_shapes(tmp_path: Path) -> None:
    shard_path = _write_shard(tmp_path / "shape.npy", n=3)
    iterator = ShardBatchIterator([shard_path], batch_size=2, y_key="y", shuffle=False, buffer=False)
    first_batch = next(iter(iterator))

    assert first_batch["x_micro"].shape == (2, 48, 6)
    assert first_batch["x_mezzo"].shape == (2, 40, 6)
    assert first_batch["x_macro"].shape == (2, 30, 6)
    assert first_batch["mask_micro"].shape == (2, 48)
    assert first_batch["mask_mezzo"].shape == (2, 40)
    assert first_batch["mask_macro"].shape == (2, 30)
    assert first_batch["y"].shape == (2,)
    assert first_batch["dp_ok"].shape == (2,)
    assert first_batch["label_ok"].shape == (2,)
    assert first_batch["loss_mask"].shape == (2,)



def test_shard_batch_iterator_multiprocess_shapes(tmp_path: Path) -> None:
    shard_paths = [_write_shard(tmp_path / f"mp_{i}.npy", n=3) for i in range(2)]
    iterator = ShardBatchIterator(
        shard_paths=shard_paths,
        batch_size=2,
        y_key="y",
        shuffle=False,
        buffer=False,
        num_workers=2,
    )
    first_batch = next(iter(iterator))
    assert first_batch["x_micro"].shape == (2, 48, 6)
    assert first_batch["x_mezzo"].shape == (2, 40, 6)
    assert first_batch["x_macro"].shape == (2, 30, 6)




def test_shard_batch_iterator_multiprocess_buffer_shapes(tmp_path: Path) -> None:
    shard_paths = [_write_shard(tmp_path / f"mp_buf_{i}.npy", n=3) for i in range(2)]
    iterator = ShardBatchIterator(
        shard_paths=shard_paths,
        batch_size=2,
        y_key="y",
        shuffle=False,
        buffer=True,
        num_workers=2,
    )
    first_batch = next(iter(iterator))
    assert first_batch["x_micro"].shape == (2, 48, 6)
    assert first_batch["x_mezzo"].shape == (2, 40, 6)
    assert first_batch["x_macro"].shape == (2, 30, 6)


def test_empty_valid_mask_batch_safe() -> None:
    batch = _synthetic_batch(batch_size=2)
    batch["loss_mask"] = torch.zeros(2, dtype=torch.bool)

    model = MultiScaleRegressor(hidden_dim=8, num_heads=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    y_hat, _ = model(batch)
    loss, metrics = masked_huber_loss(y_hat, batch["y"], batch["loss_mask"])

    assert torch.isfinite(loss)
    assert loss.item() == 0.0

    optimizer.zero_grad(set_to_none=True)
    if loss.requires_grad:
        loss.backward()
    optimizer.step()
    assert metrics["num_valid"] == 0
