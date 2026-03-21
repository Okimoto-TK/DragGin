from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.train.runner import MultiScaleRegressor

LOGGER = logging.getLogger(__name__)

_MODEL_HPARAM_KEYS = (
    "in_dim",
    "hidden_dim",
    "num_heads",
    "dropout",
    "enable_dynamic_threshold",
    "enable_free_branch",
    "init_lambda_micro",
    "init_lambda_mezzo",
    "init_lambda_macro",
    "gate_temperature",
    "use_seq_context",
)


def _normalize_checkpoint_payload(ckpt: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt, ckpt["model_state_dict"]
    return {}, ckpt


def _resolve_model_hparams(
    *,
    checkpoint: str,
    checkpoint_payload: dict[str, Any],
    explicit_hparams: dict[str, Any],
) -> dict[str, Any]:
    ckpt_hparams_raw = checkpoint_payload.get("model_hparams")
    resolved: dict[str, Any] = {}

    if isinstance(ckpt_hparams_raw, dict):
        for key in _MODEL_HPARAM_KEYS:
            if key in ckpt_hparams_raw and ckpt_hparams_raw[key] is not None:
                resolved[key] = ckpt_hparams_raw[key]
        for key, explicit_value in explicit_hparams.items():
            if explicit_value is None:
                continue
            if key in resolved:
                if resolved[key] != explicit_value:
                    LOGGER.warning(
                        "Ignoring explicit model arg %s=%r for checkpoint %s; using checkpoint model_hparams value %r.",
                        key,
                        explicit_value,
                        checkpoint,
                        resolved[key],
                    )
                continue
            resolved[key] = explicit_value

        if resolved.get("gate_temperature") is None:
            resolved["gate_temperature"] = 1.0
            LOGGER.warning(
                "Checkpoint %s has incomplete model_hparams and no gate_temperature; defaulting to 1.0. "
                "Behavior may differ from the original training run.",
                checkpoint,
            )
        return resolved

    LOGGER.warning(
        "Checkpoint %s does not contain model_hparams; falling back to explicit args/defaults. "
        "Old checkpoints may not preserve gate_temperature and other architecture-sensitive settings exactly.",
        checkpoint,
    )
    for key, explicit_value in explicit_hparams.items():
        if explicit_value is not None:
            resolved[key] = explicit_value

    if resolved.get("gate_temperature") is None:
        resolved["gate_temperature"] = 1.0
        LOGGER.warning(
            "Checkpoint %s is legacy and no explicit gate_temperature was provided; defaulting to 1.0. "
            "Inference/backtest/trader behavior may drift from training.",
            checkpoint,
        )
    return resolved


def build_model(
    device: torch.device,
    checkpoint: str,
    in_dim: int | None = None,
    hidden_dim: int | None = None,
    num_heads: int | None = None,
    dropout: float | None = None,
    use_seq_context: bool | None = None,
    enable_dynamic_threshold: bool | None = None,
    enable_free_branch: bool | None = None,
    init_lambda_micro: float | None = None,
    init_lambda_mezzo: float | None = None,
    init_lambda_macro: float | None = None,
    gate_temperature: float | None = None,
) -> MultiScaleRegressor:
    ckpt = torch.load(checkpoint, map_location=device)
    checkpoint_payload, model_state_dict = _normalize_checkpoint_payload(ckpt)
    explicit_hparams = {
        "in_dim": in_dim,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "dropout": dropout,
        "enable_dynamic_threshold": enable_dynamic_threshold,
        "enable_free_branch": enable_free_branch,
        "init_lambda_micro": init_lambda_micro,
        "init_lambda_mezzo": init_lambda_mezzo,
        "init_lambda_macro": init_lambda_macro,
        "gate_temperature": gate_temperature,
        "use_seq_context": use_seq_context,
    }
    model_hparams = _resolve_model_hparams(
        checkpoint=checkpoint,
        checkpoint_payload=checkpoint_payload,
        explicit_hparams=explicit_hparams,
    )

    model = MultiScaleRegressor(
        in_dim=int(model_hparams["in_dim"]) if "in_dim" in model_hparams else 6,
        hidden_dim=int(model_hparams["hidden_dim"]) if "hidden_dim" in model_hparams else 64,
        num_heads=int(model_hparams["num_heads"]) if "num_heads" in model_hparams else 4,
        dropout=float(model_hparams["dropout"]) if "dropout" in model_hparams else 0.0,
        enable_dynamic_threshold=(
            bool(model_hparams["enable_dynamic_threshold"]) if "enable_dynamic_threshold" in model_hparams else True
        ),
        enable_free_branch=bool(model_hparams["enable_free_branch"]) if "enable_free_branch" in model_hparams else True,
        init_lambda_micro=float(model_hparams["init_lambda_micro"]) if "init_lambda_micro" in model_hparams else 1.5,
        init_lambda_mezzo=float(model_hparams["init_lambda_mezzo"]) if "init_lambda_mezzo" in model_hparams else 0.8,
        init_lambda_macro=float(model_hparams["init_lambda_macro"]) if "init_lambda_macro" in model_hparams else 0.3,
        gate_temperature=float(model_hparams["gate_temperature"]),
        use_seq_context=bool(model_hparams["use_seq_context"]) if "use_seq_context" in model_hparams else True,
    ).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model


def infer_from_feature_shard(model: MultiScaleRegressor, device: torch.device, shard_path: Path, infer_batch_size: int) -> pd.DataFrame:
    shard = np.load(shard_path, allow_pickle=True)
    code = str(shard["code"].item() if np.ndim(shard["code"]) == 0 else shard["code"][0])
    asof_dates = shard["asof_dates"].astype(str)
    sample_ok = shard["sample_ok"].astype(bool)
    reasons = shard["reason"].astype(object)

    recs: list[dict[str, Any]] = []
    valid_idx = np.flatnonzero(sample_ok)
    if len(valid_idx) > 0:
        with torch.inference_mode():
            for start in range(0, len(valid_idx), infer_batch_size):
                idxs = valid_idx[start : start + infer_batch_size]
                batch = {
                    "x_micro": torch.from_numpy(np.ascontiguousarray(shard["x_micro"][idxs], dtype=np.float32)).to(device),
                    "x_mezzo": torch.from_numpy(np.ascontiguousarray(shard["x_mezzo"][idxs], dtype=np.float32)).to(device),
                    "x_macro": torch.from_numpy(np.ascontiguousarray(shard["x_macro"][idxs], dtype=np.float32)).to(device),
                    "mask_micro": torch.from_numpy(np.ascontiguousarray(shard["mask_micro"][idxs])).to(torch.bool).to(device),
                    "mask_mezzo": torch.from_numpy(np.ascontiguousarray(shard["mask_mezzo"][idxs])).to(torch.bool).to(device),
                    "mask_macro": torch.from_numpy(np.ascontiguousarray(shard["mask_macro"][idxs])).to(torch.bool).to(device),
                    "flow_x": torch.from_numpy(np.ascontiguousarray(shard["flow_x"][idxs], dtype=np.float32)).to(device),
                    "flow_mask": torch.from_numpy(np.ascontiguousarray(shard["flow_mask"][idxs])).to(torch.bool).to(device),
                }
                y_hat, _ = model(batch)
                scores = y_hat.detach().cpu().numpy().reshape(-1)
                for j, row_idx in enumerate(idxs):
                    recs.append(
                        {
                            "code": code,
                            "asof_date": str(asof_dates[row_idx]),
                            "yhat": float(scores[j]),
                            "tensor_ok": bool(shard["tensor_ok"][row_idx]),
                            "flow_ok": bool(shard["flow_ok"][row_idx]),
                            "sample_ok": True,
                            "reason": "ok",
                        }
                    )

    invalid_idx = np.flatnonzero(~sample_ok)
    for row_idx in invalid_idx:
        recs.append(
            {
                "code": code,
                "asof_date": str(asof_dates[row_idx]),
                "yhat": np.nan,
                "tensor_ok": bool(shard["tensor_ok"][row_idx]),
                "flow_ok": bool(shard["flow_ok"][row_idx]),
                "sample_ok": False,
                "reason": str(reasons[row_idx]),
            }
        )

    return pd.DataFrame(recs).sort_values("asof_date").reset_index(drop=True)


def infer_from_feature_shards(
    model: MultiScaleRegressor,
    device: torch.device,
    shard_paths: list[Path],
    infer_batch_size: int,
) -> pd.DataFrame:
    recs: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def flush_pending() -> None:
        nonlocal pending, recs
        if not pending:
            return

        batch = {
            "x_micro": torch.from_numpy(
                np.ascontiguousarray(np.stack([x["x_micro"] for x in pending]), dtype=np.float32)
            ).to(device),
            "x_mezzo": torch.from_numpy(
                np.ascontiguousarray(np.stack([x["x_mezzo"] for x in pending]), dtype=np.float32)
            ).to(device),
            "x_macro": torch.from_numpy(
                np.ascontiguousarray(np.stack([x["x_macro"] for x in pending]), dtype=np.float32)
            ).to(device),
            "mask_micro": torch.from_numpy(
                np.ascontiguousarray(np.stack([x["mask_micro"] for x in pending]))
            ).to(torch.bool).to(device),
            "mask_mezzo": torch.from_numpy(
                np.ascontiguousarray(np.stack([x["mask_mezzo"] for x in pending]))
            ).to(torch.bool).to(device),
            "mask_macro": torch.from_numpy(
                np.ascontiguousarray(np.stack([x["mask_macro"] for x in pending]))
            ).to(torch.bool).to(device),
            "flow_x": torch.from_numpy(
                np.ascontiguousarray(np.stack([x["flow_x"] for x in pending]), dtype=np.float32)
            ).to(device),
            "flow_mask": torch.from_numpy(
                np.ascontiguousarray(np.stack([x["flow_mask"] for x in pending]))
            ).to(torch.bool).to(device),
        }

        with torch.inference_mode():
            y_hat, _ = model(batch)
        scores = y_hat.detach().cpu().numpy().reshape(-1)

        for meta, score in zip(pending, scores, strict=False):
            recs.append(
                {
                    "code": meta["code"],
                    "asof_date": meta["asof_date"],
                    "yhat": float(score),
                    "tensor_ok": meta["tensor_ok"],
                    "flow_ok": meta["flow_ok"],
                    "sample_ok": True,
                    "reason": "ok",
                }
            )

        pending = []

    for shard_path in shard_paths:
        shard = np.load(shard_path, allow_pickle=True)
        code = str(shard["code"].item() if np.ndim(shard["code"]) == 0 else shard["code"][0])
        asof_dates = shard["asof_dates"].astype(str)
        sample_ok = shard["sample_ok"].astype(bool)
        reasons = shard["reason"].astype(object)

        valid_idx = np.flatnonzero(sample_ok)
        for row_idx in valid_idx:
            pending.append(
                {
                    "code": code,
                    "asof_date": str(asof_dates[row_idx]),
                    "tensor_ok": bool(shard["tensor_ok"][row_idx]),
                    "flow_ok": bool(shard["flow_ok"][row_idx]),
                    "x_micro": shard["x_micro"][row_idx],
                    "x_mezzo": shard["x_mezzo"][row_idx],
                    "x_macro": shard["x_macro"][row_idx],
                    "mask_micro": shard["mask_micro"][row_idx],
                    "mask_mezzo": shard["mask_mezzo"][row_idx],
                    "mask_macro": shard["mask_macro"][row_idx],
                    "flow_x": shard["flow_x"][row_idx],
                    "flow_mask": shard["flow_mask"][row_idx],
                }
            )
            if len(pending) >= infer_batch_size:
                flush_pending()

        invalid_idx = np.flatnonzero(~sample_ok)
        for row_idx in invalid_idx:
            recs.append(
                {
                    "code": code,
                    "asof_date": str(asof_dates[row_idx]),
                    "yhat": np.nan,
                    "tensor_ok": bool(shard["tensor_ok"][row_idx]),
                    "flow_ok": bool(shard["flow_ok"][row_idx]),
                    "sample_ok": False,
                    "reason": str(reasons[row_idx]),
                }
            )

    flush_pending()
    return pd.DataFrame(recs).sort_values(["asof_date", "code"]).reset_index(drop=True)
