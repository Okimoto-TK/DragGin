from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.train.runner import MultiScaleRegressor


def build_model(
    device: torch.device,
    checkpoint: str,
    hidden_dim: int,
    num_heads: int,
    dropout: float,
    use_seq_context: bool,
    enable_dynamic_threshold: bool,
    enable_free_branch: bool,
) -> MultiScaleRegressor:
    model = MultiScaleRegressor(
        hidden_dim=int(hidden_dim),
        num_heads=int(num_heads),
        dropout=float(dropout),
        use_seq_context=bool(use_seq_context),
        enable_dynamic_threshold=bool(enable_dynamic_threshold),
        enable_free_branch=bool(enable_free_branch),
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
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
