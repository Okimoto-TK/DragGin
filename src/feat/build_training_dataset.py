from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.feat.build_multiscale_tensor import build_calendar_from_daily_filenames
from src.feat.build_multiscale_tensor import L_MACRO, L_MEZZO, L_MICRO
from src.feat.build_multiscale_tensor import _build_tensor_context, get_tensor_valid_asof_dates
from src.feat.labels_risk_adj import _build_label_context, get_label_valid_asof_dates

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


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


def _empty_bundle() -> TrainDatasetBundle:
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


def _iter_progress(iterable, total: int, show_progress: bool, desc: str):
    if show_progress and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def _rows_from_code_task(
    data_dir: str | Path,
    code: str,
    selected_asof_dates: tuple[str, ...],
    include_invalid: bool,
    shard_dir: str,
) -> dict:
    data_dir_key = str(Path(data_dir).resolve())
    asof_parsed = [pd.to_datetime(x).date() for x in selected_asof_dates]
    n = len(selected_asof_dates)
    codes = np.empty((n,), dtype=object)
    asof_dates = np.empty((n,), dtype=object)
    X_micro = np.zeros((n, 48, 6), dtype=np.float32)
    X_mezzo = np.zeros((n, 40, 6), dtype=np.float32)
    X_macro = np.zeros((n, 30, 6), dtype=np.float32)
    mask_micro = np.zeros((n, 48), dtype=np.uint8)
    mask_mezzo = np.zeros((n, 40), dtype=np.uint8)
    mask_macro = np.zeros((n, 30), dtype=np.uint8)
    y = np.zeros((n,), dtype=np.float32)
    y_raw = np.zeros((n,), dtype=np.float32)
    y_z = np.zeros((n,), dtype=np.float32)
    dp_ok = np.zeros((n,), dtype=np.bool_)
    label_ok = np.zeros((n,), dtype=np.bool_)
    loss_mask = np.zeros((n,), dtype=np.bool_)

    tensor_context = _build_tensor_context(data_dir_key, code)
    valid_tensor_dates = set(get_tensor_valid_asof_dates(data_dir_key, code))
    tensor_valid_pos = np.asarray([i for i, asof in enumerate(selected_asof_dates) if asof in valid_tensor_dates], dtype=np.int64)
    if tensor_context is not None and tensor_valid_pos.size > 0:
        tensor_valid_dates = [asof_parsed[i] for i in tensor_valid_pos]
        micro_end = np.asarray([tensor_context.asof_to_micro_end[d] for d in tensor_valid_dates], dtype=np.int64)
        mezzo_end = np.asarray([tensor_context.asof_to_mezzo_end[d] for d in tensor_valid_dates], dtype=np.int64)
        macro_end = np.asarray([tensor_context.asof_to_macro_idx[d] for d in tensor_valid_dates], dtype=np.int64)

        micro_offsets = np.arange(-L_MICRO + 1, 1, dtype=np.int64)
        mezzo_offsets = np.arange(-L_MEZZO + 1, 1, dtype=np.int64)
        macro_offsets = np.arange(-L_MACRO + 1, 1, dtype=np.int64)

        micro_idx2d = micro_end[:, None] + micro_offsets[None, :]
        mezzo_idx2d = mezzo_end[:, None] + mezzo_offsets[None, :]
        macro_idx2d = macro_end[:, None] + macro_offsets[None, :]

        X_micro[tensor_valid_pos] = tensor_context.micro_z[micro_idx2d].astype(np.float32)
        X_mezzo[tensor_valid_pos] = tensor_context.mezzo_z[mezzo_idx2d].astype(np.float32)
        X_macro[tensor_valid_pos] = tensor_context.macro_z[macro_idx2d].astype(np.float32)
        mask_micro[tensor_valid_pos] = np.ones((tensor_valid_pos.size, L_MICRO), dtype=np.uint8)
        mask_mezzo[tensor_valid_pos] = np.ones((tensor_valid_pos.size, L_MEZZO), dtype=np.uint8)
        mask_macro[tensor_valid_pos] = np.ones((tensor_valid_pos.size, L_MACRO), dtype=np.uint8)
        dp_ok[tensor_valid_pos] = True

    label_context = _build_label_context(data_dir_key, code)
    valid_label_dates = set(get_label_valid_asof_dates(data_dir_key, code))
    label_valid_pos = np.asarray([i for i, asof in enumerate(selected_asof_dates) if asof in valid_label_dates], dtype=np.int64)
    if label_context is not None and label_valid_pos.size > 0:
        label_valid_dates = [asof_parsed[i] for i in label_valid_pos]
        idx_arr = np.asarray([label_context.asof_to_trading_idx[d] for d in label_valid_dates], dtype=np.int64)
        valid_pos_arr = np.asarray([label_context.raw_idx_to_valid_pos[idx] for idx in idx_arr], dtype=np.int64)
        valid_raw_values = np.asarray(label_context.valid_raw_values, dtype=np.float32)
        label_values = np.asarray(label_context.label_value_by_idx, dtype=np.float32)

        y_raw[label_valid_pos] = valid_raw_values[valid_pos_arr]
        y_z[label_valid_pos] = label_values[idx_arr]
        y[label_valid_pos] = label_values[idx_arr]
        label_ok[label_valid_pos] = True

    loss_mask = dp_ok & label_ok

    codes[:] = code
    asof_dates[:] = np.asarray(selected_asof_dates, dtype=object)

    keep = np.arange(n, dtype=np.int64) if include_invalid else np.flatnonzero(loss_mask)
    write_idx = int(keep.size)

    shard_path = Path(shard_dir) / f"{code}.npz"
    np.savez(
        shard_path,
        codes=codes[keep],
        asof_dates=asof_dates[keep],
        X_micro=X_micro[keep],
        X_mezzo=X_mezzo[keep],
        X_macro=X_macro[keep],
        mask_micro=mask_micro[keep],
        mask_mezzo=mask_mezzo[keep],
        mask_macro=mask_macro[keep],
        y=y[keep],
        y_raw=y_raw[keep],
        y_z=y_z[keep],
        dp_ok=dp_ok[keep],
        label_ok=label_ok[keep],
        loss_mask=loss_mask[keep],
    )
    return {"path": str(shard_path), "rows": int(write_idx)}


def _merge_shards(shard_infos: list[dict]) -> TrainDatasetBundle:
    if not shard_infos:
        return _empty_bundle()

    if sum(int(x.get("rows", 0)) for x in shard_infos) == 0:
        return _empty_bundle()

    parts: dict[str, list[np.ndarray]] = {
        "codes": [],
        "asof_dates": [],
        "X_micro": [],
        "X_mezzo": [],
        "X_macro": [],
        "mask_micro": [],
        "mask_mezzo": [],
        "mask_macro": [],
        "y": [],
        "y_raw": [],
        "y_z": [],
        "dp_ok": [],
        "label_ok": [],
        "loss_mask": [],
    }
    for info in shard_infos:
        if int(info.get("rows", 0)) <= 0:
            continue
        with np.load(info["path"], allow_pickle=True) as d:
            for k in parts:
                parts[k].append(d[k])

    if not parts["y"]:
        return _empty_bundle()

    return TrainDatasetBundle(
        codes=np.concatenate(parts["codes"]).astype(object),
        asof_dates=np.concatenate(parts["asof_dates"]).astype(object),
        X_micro=np.concatenate(parts["X_micro"], axis=0).astype(np.float32),
        X_mezzo=np.concatenate(parts["X_mezzo"], axis=0).astype(np.float32),
        X_macro=np.concatenate(parts["X_macro"], axis=0).astype(np.float32),
        mask_micro=np.concatenate(parts["mask_micro"], axis=0).astype(np.uint8),
        mask_mezzo=np.concatenate(parts["mask_mezzo"], axis=0).astype(np.uint8),
        mask_macro=np.concatenate(parts["mask_macro"], axis=0).astype(np.uint8),
        y=np.concatenate(parts["y"]).astype(np.float32),
        y_raw=np.concatenate(parts["y_raw"]).astype(np.float32),
        y_z=np.concatenate(parts["y_z"]).astype(np.float32),
        dp_ok=np.concatenate(parts["dp_ok"]).astype(np.bool_),
        label_ok=np.concatenate(parts["label_ok"]).astype(np.bool_),
        loss_mask=np.concatenate(parts["loss_mask"]).astype(np.bool_),
    )


def build_train_dataset(
    data_dir: str | Path,
    codes: list[str] | None = None,
    asof_dates: list[str] | None = None,
    include_invalid: bool = False,
    num_workers: int = 1,
    show_progress: bool = True,
    shard_tmp_dir: str | Path | None = None,
) -> TrainDatasetBundle:
    selected_codes = resolve_codes(data_dir, codes)
    selected_asof_dates = resolve_asof_dates(data_dir, asof_dates)

    asof_tuple = tuple(selected_asof_dates)
    tmp_root = None if shard_tmp_dir is None else str(Path(shard_tmp_dir))
    with tempfile.TemporaryDirectory(prefix="train_dataset_shards_", dir=tmp_root) as shard_dir:
        shard_infos: list[dict] = []
        if num_workers <= 1:
            iterator = _iter_progress(selected_codes, total=len(selected_codes), show_progress=show_progress, desc="building train dataset")
            for code in iterator:
                shard_infos.append(_rows_from_code_task(data_dir, code, asof_tuple, include_invalid, shard_dir))
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(_rows_from_code_task, data_dir, code, asof_tuple, include_invalid, shard_dir)
                    for code in selected_codes
                ]
                progress_iter = _iter_progress(as_completed(futures), total=len(futures), show_progress=show_progress, desc="building train dataset")
                for fut in progress_iter:
                    shard_infos.append(fut.result())

        return _merge_shards(shard_infos)


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
