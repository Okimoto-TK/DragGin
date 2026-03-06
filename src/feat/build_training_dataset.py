from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.feat.build_multiscale_tensor import build_calendar_from_daily_filenames
from src.feat.build_multiscale_tensor import build_multiscale_tensors
from src.feat.labels_risk_adj import build_label_from_data_dir


@dataclass
class TrainDatasetBundle:
    codes: np.ndarray
    asof_dates: np.ndarray
    X_micro: np.ndarray
    X_mezzo: np.ndarray
    X_macro: np.ndarray
    mask_micro: np.ndarray
    mask_mezzo: np.ndarray
    mask_macro: np.ndarray
    y: np.ndarray
    y_raw: np.ndarray
    y_z: np.ndarray
    dp_ok: np.ndarray
    label_ok: np.ndarray
    loss_mask: np.ndarray


def resolve_codes(data_dir: str | Path, codes: list[str] | None = None) -> list[str]:
    if codes:
        return [c for c in codes if c]
    root = Path(data_dir)
    out: list[str] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "daily.parquet").exists():
            out.append(p.name)
    return out


def resolve_asof_dates(data_dir: str | Path, asof_dates: list[str] | None = None) -> list[str]:
    if asof_dates:
        return [d for d in asof_dates if d]
    return build_calendar_from_daily_filenames(data_dir)


def build_train_dataset(
    data_dir: str | Path,
    codes: list[str] | None = None,
    asof_dates: list[str] | None = None,
    include_invalid: bool = False,
) -> TrainDatasetBundle:
    selected_codes = resolve_codes(data_dir, codes)
    selected_asof_dates = resolve_asof_dates(data_dir, asof_dates)

    rows: list[dict] = []
    for code in selected_codes:
        for asof in selected_asof_dates:
            dp = build_multiscale_tensors(data_dir, code, asof)
            lb = build_label_from_data_dir(data_dir, code, asof, dp_ok=dp.dp_ok)
            if (not include_invalid) and (not lb.loss_mask):
                continue
            rows.append(
                {
                    "code": code,
                    "asof": asof,
                    "X_micro": dp.X_micro.astype(np.float32),
                    "X_mezzo": dp.X_mezzo.astype(np.float32),
                    "X_macro": dp.X_macro.astype(np.float32),
                    "mask_micro": dp.mask_micro.astype(np.uint8),
                    "mask_mezzo": dp.mask_mezzo.astype(np.uint8),
                    "mask_macro": dp.mask_macro.astype(np.uint8),
                    "y": np.float32(lb.y),
                    "y_raw": np.float32(lb.y_raw),
                    "y_z": np.float32(lb.y_z),
                    "dp_ok": bool(dp.dp_ok),
                    "label_ok": bool(lb.label_ok),
                    "loss_mask": bool(lb.loss_mask),
                }
            )

    if not rows:
        return TrainDatasetBundle(
            codes=np.asarray([], dtype=object),
            asof_dates=np.asarray([], dtype=object),
            X_micro=np.zeros((0, 48, 6), dtype=np.float32),
            X_mezzo=np.zeros((0, 40, 6), dtype=np.float32),
            X_macro=np.zeros((0, 30, 6), dtype=np.float32),
            mask_micro=np.zeros((0, 48), dtype=np.uint8),
            mask_mezzo=np.zeros((0, 40), dtype=np.uint8),
            mask_macro=np.zeros((0, 30), dtype=np.uint8),
            y=np.zeros((0,), dtype=np.float32),
            y_raw=np.zeros((0,), dtype=np.float32),
            y_z=np.zeros((0,), dtype=np.float32),
            dp_ok=np.zeros((0,), dtype=np.bool_),
            label_ok=np.zeros((0,), dtype=np.bool_),
            loss_mask=np.zeros((0,), dtype=np.bool_),
        )

    return TrainDatasetBundle(
        codes=np.asarray([r["code"] for r in rows], dtype=object),
        asof_dates=np.asarray([r["asof"] for r in rows], dtype=object),
        X_micro=np.stack([r["X_micro"] for r in rows], axis=0),
        X_mezzo=np.stack([r["X_mezzo"] for r in rows], axis=0),
        X_macro=np.stack([r["X_macro"] for r in rows], axis=0),
        mask_micro=np.stack([r["mask_micro"] for r in rows], axis=0),
        mask_mezzo=np.stack([r["mask_mezzo"] for r in rows], axis=0),
        mask_macro=np.stack([r["mask_macro"] for r in rows], axis=0),
        y=np.asarray([r["y"] for r in rows], dtype=np.float32),
        y_raw=np.asarray([r["y_raw"] for r in rows], dtype=np.float32),
        y_z=np.asarray([r["y_z"] for r in rows], dtype=np.float32),
        dp_ok=np.asarray([r["dp_ok"] for r in rows], dtype=np.bool_),
        label_ok=np.asarray([r["label_ok"] for r in rows], dtype=np.bool_),
        loss_mask=np.asarray([r["loss_mask"] for r in rows], dtype=np.bool_),
    )


def save_train_dataset(bundle: TrainDatasetBundle, out_npz: str | Path) -> None:
    np.savez_compressed(
        out_npz,
        codes=bundle.codes,
        asof_dates=bundle.asof_dates,
        X_micro=bundle.X_micro,
        X_mezzo=bundle.X_mezzo,
        X_macro=bundle.X_macro,
        mask_micro=bundle.mask_micro,
        mask_mezzo=bundle.mask_mezzo,
        mask_macro=bundle.mask_macro,
        y=bundle.y,
        y_raw=bundle.y_raw,
        y_z=bundle.y_z,
        dp_ok=bundle.dp_ok,
        label_ok=bundle.label_ok,
        loss_mask=bundle.loss_mask,
    )
