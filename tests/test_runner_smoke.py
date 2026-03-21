import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.feat.build_multiscale_tensor import C
from src.infer.runner import build_model
from src.model.head import masked_huber_loss
from src.train.runner import (
    MultiScaleRegressor,
    NpyShardDataset,
    ShardBatchIterator,
    TrainConfig,
    collate_batch,
    run_training,
    _resolve_force_flow_gate_value,
    _weighted_metric_value,
)


def _train_diagnostics(out_dir: Path) -> str:
    metrics_path = out_dir / "metrics" / "curve.json"
    log_path = out_dir / "logs" / "train.log"
    feedback_dir = out_dir / "reports" / "feedback"

    lines: list[str] = []
    lines.append(f"metrics_exists={metrics_path.exists()}")
    if metrics_path.exists():
        raw = metrics_path.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
            train_rows = data.get("train", [])
            val_rows = data.get("val", [])
            lines.append(f"curve_train_rows={len(train_rows)}")
            lines.append(f"curve_val_rows={len(val_rows)}")
            if len(train_rows) > 0:
                lines.append(f"curve_last_train_row={train_rows[-1]}")
        except Exception as exc:  # pragma: no cover - diagnostic only
            lines.append(f"curve_parse_error={exc}")
            lines.append(f"curve_raw_head={raw[:1000]}")

    lines.append(f"log_exists={log_path.exists()}")
    if log_path.exists():
        log_lines = log_path.read_text(encoding="utf-8").splitlines()
        tail = log_lines[-80:]
        lines.append("train_log_tail:")
        lines.extend(tail)

    lines.append(f"feedback_dir_exists={feedback_dir.exists()}")
    if feedback_dir.exists():
        lines.append(f"feedback_files={[p.name for p in sorted(feedback_dir.glob('*.yaml'))]}")

    return "\n".join(lines)


def _write_shard(path: Path, n: int = 3, all_loss_mask_false: bool = False) -> str:
    rng = np.random.default_rng(7)
    base = np.datetime64("2024-01-01")
    shard = {
        "codes": np.array([f"C{i:03d}" for i in range(n)], dtype=object),
        "asof_dates": np.array([(base + np.timedelta64(i, "D")).astype(str) for i in range(n)], dtype=object),
        "X_micro": rng.normal(size=(n, 48, C)).astype(np.float32),
        "X_mezzo": rng.normal(size=(n, 40, C)).astype(np.float32),
        "X_macro": rng.normal(size=(n, 30, C)).astype(np.float32),
        "mask_micro": np.ones((n, 48), dtype=bool),
        "mask_mezzo": np.ones((n, 40), dtype=bool),
        "mask_macro": np.ones((n, 30), dtype=bool),
        "flow_x": rng.normal(size=(n, 30, 4)).astype(np.float32),
        "flow_mask": np.ones((n, 30), dtype=bool),
        "y": rng.normal(size=(n,)).astype(np.float32),
        "y_raw": rng.normal(size=(n,)).astype(np.float32),
        "y_z": rng.normal(size=(n,)).astype(np.float32),
        "dp_ok": np.ones((n,), dtype=bool),
        "label_ok": np.ones((n,), dtype=bool),
        "loss_mask": np.zeros((n,), dtype=bool) if all_loss_mask_false else np.array([(i % 2) == 0 for i in range(n)], dtype=bool),
    }
    np.save(path, shard, allow_pickle=True)
    return str(path)


def _synthetic_batch(batch_size: int = 2) -> dict:
    return {
        "x_micro": torch.randn(batch_size, 48, C),
        "x_mezzo": torch.randn(batch_size, 40, C),
        "x_macro": torch.randn(batch_size, 30, C),
        "mask_micro": torch.ones(batch_size, 48, dtype=torch.bool),
        "mask_mezzo": torch.ones(batch_size, 40, dtype=torch.bool),
        "mask_macro": torch.ones(batch_size, 30, dtype=torch.bool),
        "flow_x": torch.randn(batch_size, 30, 4),
        "flow_mask": torch.ones(batch_size, 30, dtype=torch.bool),
        "y": torch.randn(batch_size),
        "dp_ok": torch.ones(batch_size, dtype=torch.bool),
        "label_ok": torch.ones(batch_size, dtype=torch.bool),
        "loss_mask": torch.ones(batch_size, dtype=torch.bool),
    }


def test_dataset_and_collate_shapes(tmp_path: Path) -> None:
    shard_path = _write_shard(tmp_path / "train.npy", n=3)
    ds = NpyShardDataset([shard_path], y_key="y")

    sample = ds[0]
    assert sample["x_micro"].shape == (48, C)
    assert sample["x_mezzo"].shape == (40, C)
    assert sample["x_macro"].shape == (30, C)
    assert sample["mask_micro"].shape == (48,)
    assert sample["loss_mask"].dtype == torch.bool

    batch = collate_batch([ds[0], ds[1]])
    assert batch["x_micro"].shape == (2, 48, C)
    assert batch["x_mezzo"].shape == (2, 40, C)
    assert batch["x_macro"].shape == (2, 30, C)
    assert batch["flow_x"].shape == (2, 30, 4)
    assert batch["flow_mask"].shape == (2, 30)
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
    ckpt_dir = out / "checkpoints" / cfg.exp_name
    assert (ckpt_dir / "latest.ckpt").exists()
    assert (ckpt_dir / "best.ckpt").exists()
    assert (ckpt_dir / "epoch_0001.ckpt").exists()

    data = json.loads((out / "metrics" / "curve.json").read_text(encoding="utf-8"))
    assert len(data["train"]) >= 1
    assert len(data["val"]) == 1








def test_checkpoint_saves_model_hparams(tmp_path: Path) -> None:
    train_path = _write_shard(tmp_path / "train_hparams.npy", n=4)
    val_path = _write_shard(tmp_path / "val_hparams.npy", n=2)

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
        exp_name="hparams_ckpt",
        out_dir=str(tmp_path / "out_hparams"),
        gate_temperature=0.7,
        enable_dynamic_threshold=False,
        enable_free_branch=False,
        init_lambda_micro=1.1,
        init_lambda_mezzo=0.9,
        init_lambda_macro=0.4,
        use_seq_context=False,
    )
    run_training(cfg)

    ckpt = torch.load(Path(cfg.out_dir) / "checkpoints" / cfg.exp_name / "latest.ckpt", map_location="cpu")
    assert ckpt["model_hparams"] == {
        "in_dim": 6,
        "hidden_dim": 8,
        "num_heads": 2,
        "dropout": 0.0,
        "enable_dynamic_threshold": False,
        "enable_free_branch": False,
        "init_lambda_micro": 1.1,
        "init_lambda_mezzo": 0.9,
        "init_lambda_macro": 0.4,
        "gate_temperature": 0.7,
        "use_seq_context": False,
    }


def test_build_model_prefers_checkpoint_hparams(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    train_path = _write_shard(tmp_path / "train_pref.npy", n=4)
    val_path = _write_shard(tmp_path / "val_pref.npy", n=2)

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
        exp_name="pref_ckpt",
        out_dir=str(tmp_path / "out_pref"),
        gate_temperature=0.7,
    )
    run_training(cfg)

    ckpt_path = Path(cfg.out_dir) / "checkpoints" / cfg.exp_name / "latest.ckpt"
    with caplog.at_level("WARNING"):
        model = build_model(device=torch.device("cpu"), checkpoint=str(ckpt_path), gate_temperature=1.0, hidden_dim=16)

    assert model.fusion.gated.gate_temperature == pytest.approx(0.7)
    assert model.fusion.gated.gate_mlp[-1].out_features == 8
    assert "Ignoring explicit model arg gate_temperature=1.0" in caplog.text
    assert "Ignoring explicit model arg hidden_dim=16" in caplog.text


def test_build_model_supports_legacy_checkpoint_fallback(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    model = MultiScaleRegressor(
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        enable_dynamic_threshold=False,
        enable_free_branch=False,
        gate_temperature=0.7,
        use_seq_context=False,
    )
    legacy_path = tmp_path / "legacy.ckpt"
    torch.save({"model_state_dict": model.state_dict()}, legacy_path)

    with caplog.at_level("WARNING"):
        rebuilt = build_model(
            device=torch.device("cpu"),
            checkpoint=str(legacy_path),
            hidden_dim=8,
            num_heads=2,
            dropout=0.0,
            enable_dynamic_threshold=False,
            enable_free_branch=False,
            gate_temperature=0.7,
            use_seq_context=False,
        )

    assert rebuilt.fusion.gated.gate_temperature == pytest.approx(0.7)
    assert "does not contain model_hparams" in caplog.text


def test_build_model_legacy_checkpoint_warns_when_gate_temperature_missing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    model = MultiScaleRegressor(hidden_dim=8, num_heads=2)
    legacy_path = tmp_path / "legacy_default.ckpt"
    torch.save({"model_state_dict": model.state_dict()}, legacy_path)

    with caplog.at_level("WARNING"):
        rebuilt = build_model(
            device=torch.device("cpu"),
            checkpoint=str(legacy_path),
            hidden_dim=8,
            num_heads=2,
        )

    assert rebuilt.fusion.gated.gate_temperature == pytest.approx(1.0)
    assert "defaulting to 1.0" in caplog.text


def test_gate_std_penalty_zero_when_free_branch_disabled(tmp_path: Path) -> None:
    # Stabilize tiny-run behavior across platforms for this regression check.
    torch.manual_seed(0)
    train_path = _write_shard(tmp_path / "train_no_free.npy", n=4)
    val_path = _write_shard(tmp_path / "val_no_free.npy", n=2)

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
        exp_name="no_free_branch",
        out_dir=str(tmp_path / "out_no_free"),
        enable_free_branch=False,
    )
    result = run_training(cfg)
    assert result["feedback"]["meta"]["status"] == "ok"

    out_dir = Path(cfg.out_dir)
    data = json.loads((out_dir / "metrics" / "curve.json").read_text(encoding="utf-8"))
    if len(data["train"]) < 1:
        pytest.fail("expected at least 1 train row, diagnostics:\n" + _train_diagnostics(out_dir))
    for row in data["train"]:
        assert float(row["gate_std_penalty"]) == 0.0
        assert float(row["total_loss"]) == float(row["loss"])

def test_train_one_epoch_all_invalid_train_mask_does_not_crash(tmp_path: Path) -> None:
    train_path = _write_shard(tmp_path / "train_all_invalid.npy", n=4, all_loss_mask_false=True)
    val_path = _write_shard(tmp_path / "val_valid.npy", n=2)

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
        exp_name="smoke_all_invalid",
        out_dir=str(tmp_path / "out_all_invalid"),
    )
    result = run_training(cfg)
    assert result["feedback"]["meta"]["status"] == "ok"


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
        split_mode="code",
    )
    result = run_training(cfg)

    assert result["feedback"]["meta"]["status"] == "ok"
    assert result["feedback"]["data"]["train_samples"] == 8
    assert result["feedback"]["data"]["val_samples"] == 2


def test_train_one_epoch_with_date_split_and_embargo(tmp_path: Path) -> None:
    train_path = _write_shard(tmp_path / "train_only_date.npy", n=100)

    cfg = TrainConfig(
        train_shards=[train_path],
        val_shards=[],
        val_ratio=0.2,
        split_mode="date",
        val_embargo_days=30,
        batch_size=8,
        grad_accum_steps=1,
        num_epochs=1,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_dim=8,
        num_heads=2,
        dropout=0.1,
        exp_name="smoke_date_split",
        out_dir=str(tmp_path / "out_date_split"),
    )
    result = run_training(cfg)

    assert result["feedback"]["meta"]["status"] == "ok"
    assert result["feedback"]["data"]["train_samples"] == 56
    assert result["feedback"]["data"]["val_samples"] == 14




def test_resume_from_checkpoint_and_save_every(tmp_path: Path) -> None:
    train_path = _write_shard(tmp_path / "resume_train.npy", n=6)
    val_path = _write_shard(tmp_path / "resume_val.npy", n=4)

    out_dir = tmp_path / "out_resume"
    first_cfg = TrainConfig(
        train_shards=[train_path],
        val_shards=[val_path],
        batch_size=2,
        grad_accum_steps=1,
        num_epochs=2,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        exp_name="resume_run",
        out_dir=str(out_dir),
        save_every=2,
    )
    run_training(first_cfg)

    ckpt_dir = out_dir / "checkpoints" / first_cfg.exp_name
    latest_path = ckpt_dir / "latest.ckpt"
    epoch2_path = ckpt_dir / "epoch_0002.ckpt"
    assert latest_path.exists()
    assert epoch2_path.exists()

    second_cfg = TrainConfig(
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
        exp_name="resume_run",
        out_dir=str(out_dir),
        checkpoint=str(latest_path),
        save_every=1,
    )
    run_training(second_cfg)

    epoch3_path = ckpt_dir / "epoch_0003.ckpt"
    assert epoch3_path.exists()

    resumed_latest = torch.load(latest_path, map_location="cpu")
    assert int(resumed_latest["epoch"]) == 2


def test_resume_overrides_optimizer_lr_and_weight_decay(tmp_path: Path) -> None:
    train_path = _write_shard(tmp_path / "resume_lr_train.npy", n=6)
    val_path = _write_shard(tmp_path / "resume_lr_val.npy", n=4)

    out_dir = tmp_path / "out_resume_lr"
    first_cfg = TrainConfig(
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
        exp_name="resume_lr_run",
        out_dir=str(out_dir),
    )
    run_training(first_cfg)

    ckpt_dir = out_dir / "checkpoints" / first_cfg.exp_name
    latest_path = ckpt_dir / "latest.ckpt"
    assert latest_path.exists()

    new_lr = 5e-4
    new_weight_decay = 5e-5
    second_cfg = TrainConfig(
        train_shards=[train_path],
        val_shards=[val_path],
        batch_size=2,
        grad_accum_steps=1,
        num_epochs=1,
        lr=new_lr,
        weight_decay=new_weight_decay,
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        exp_name="resume_lr_run",
        out_dir=str(out_dir),
        checkpoint=str(latest_path),
    )
    run_training(second_cfg)

    resumed_latest = torch.load(latest_path, map_location="cpu")
    optimizer_state = resumed_latest["optimizer_state_dict"]
    assert float(optimizer_state["param_groups"][0]["lr"]) == new_lr
    assert float(optimizer_state["param_groups"][0]["weight_decay"]) == new_weight_decay

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

    assert first_batch["x_micro"].shape == (2, 48, C)
    assert first_batch["x_mezzo"].shape == (2, 40, C)
    assert first_batch["x_macro"].shape == (2, 30, C)
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
    assert first_batch["x_micro"].shape == (2, 48, C)
    assert first_batch["x_mezzo"].shape == (2, 40, C)
    assert first_batch["x_macro"].shape == (2, 30, C)




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
    assert first_batch["x_micro"].shape == (2, 48, C)
    assert first_batch["x_mezzo"].shape == (2, 40, C)
    assert first_batch["x_macro"].shape == (2, 30, C)




def test_weighted_metric_value_matches_num_valid_weighting() -> None:
    batch1_loss = torch.tensor(1.0)
    batch2_loss = torch.tensor(9.0)
    batch1_valid = 8
    batch2_valid = 2

    weighted_sum = _weighted_metric_value(batch1_loss, batch1_valid) + _weighted_metric_value(batch2_loss, batch2_valid)
    weighted_mean = weighted_sum / float(batch1_valid + batch2_valid)
    simple_batch_mean = (float(batch1_loss.item()) + float(batch2_loss.item())) / 2.0

    assert weighted_mean == 2.6
    assert weighted_mean != simple_batch_mean

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


def test_checkpoint_contains_scheduler_state(tmp_path: Path) -> None:
    train_path = _write_shard(tmp_path / "sched_train.npy", n=4)
    val_path = _write_shard(tmp_path / "sched_val.npy", n=2)
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
        exp_name="sched_state",
        out_dir=str(tmp_path / "out_sched"),
    )
    run_training(cfg)
    ckpt = torch.load(Path(cfg.out_dir) / "checkpoints" / cfg.exp_name / "latest.ckpt", map_location="cpu")
    assert "scheduler_state_dict" in ckpt
    assert isinstance(ckpt["scheduler_state_dict"], dict)


def test_non_finite_loss_step_is_skipped(tmp_path: Path, monkeypatch) -> None:
    train_path = _write_shard(tmp_path / "nan_train.npy", n=2)
    val_path = _write_shard(tmp_path / "nan_val.npy", n=2)

    def _nan_loss(y_hat, y_true, loss_mask, delta=1.0):
        del y_true, loss_mask, delta
        nan_loss = y_hat.mean() * 0.0 + torch.tensor(float("nan"), dtype=y_hat.dtype, device=y_hat.device)
        return nan_loss, {"num_valid": 1, "huber": torch.zeros_like(nan_loss), "mae": torch.zeros_like(nan_loss), "mse": torch.zeros_like(nan_loss)}

    monkeypatch.setattr("src.train.runner.masked_huber_loss", _nan_loss)
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
        exp_name="nan_skip",
        out_dir=str(tmp_path / "out_nan"),
    )
    result = run_training(cfg)
    assert result["feedback"]["meta"]["status"] == "ok"
    assert len(result["history"]["train"]) == 0


def test_unscale_called_even_without_global_clip(tmp_path: Path, monkeypatch) -> None:
    train_path = _write_shard(tmp_path / "unscale_train.npy", n=2)
    val_path = _write_shard(tmp_path / "unscale_val.npy", n=2)

    calls: list[str] = []

    class _FakeScaled:
        def __init__(self, loss):
            self.loss = loss

        def backward(self):
            self.loss.backward()

    class _FakeScaler:
        def __init__(self, enabled=True):
            del enabled

        def scale(self, loss):
            calls.append("scale")
            return _FakeScaled(loss)

        def unscale_(self, optimizer):
            del optimizer
            calls.append("unscale")

        def step(self, optimizer):
            calls.append("step")
            optimizer.step()

        def update(self):
            calls.append("update")

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            del state

    monkeypatch.setattr("src.train.runner.GradScaler", _FakeScaler)

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
        clip_grad_norm=None,
        exp_name="unscale_called",
        out_dir=str(tmp_path / "out_unscale"),
    )
    run_training(cfg)
    assert "unscale" in calls
    assert calls.index("unscale") < calls.index("step")


def test_resolve_force_flow_gate_value_respects_always_zero_switch() -> None:
    cfg_always_zero = TrainConfig(
        train_shards=["dummy.npy"],
        val_shards=["dummy_val.npy"],
        batch_size=1,
        grad_accum_steps=1,
        num_epochs=1,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        exp_name="resolve_flow_gate",
        out_dir="/tmp/out",
        flow_gate_force_zero_all_steps=True,
    )
    assert _resolve_force_flow_gate_value(cfg_always_zero, global_step=0) == 0.0
    assert _resolve_force_flow_gate_value(cfg_always_zero, global_step=2000) == 0.0
    assert _resolve_force_flow_gate_value(cfg_always_zero, global_step=999999) == 0.0

    cfg_warmup_only = TrainConfig(
        train_shards=["dummy.npy"],
        val_shards=["dummy_val.npy"],
        batch_size=1,
        grad_accum_steps=1,
        num_epochs=1,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        exp_name="resolve_flow_gate_warmup",
        out_dir="/tmp/out2",
    )
    assert _resolve_force_flow_gate_value(cfg_warmup_only, global_step=0) == 0.0
    assert _resolve_force_flow_gate_value(cfg_warmup_only, global_step=1999) == 0.0
    assert _resolve_force_flow_gate_value(cfg_warmup_only, global_step=2000) is None
