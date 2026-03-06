from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np

from src.feat.build_multiscale_tensor import build_calendar_from_daily_filenames
from src.feat.build_multiscale_tensor import build_multiscale_tensors, get_tensor_valid_asof_dates
from src.feat.labels_risk_adj import build_label_from_data_dir, get_label_valid_asof_dates

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
    filtered_asofs = selected_asof_dates
    if not include_invalid:
        tensor_valid = set(get_tensor_valid_asof_dates(data_dir_key, code))
        label_valid = set(get_label_valid_asof_dates(data_dir_key, code))
        both_valid = tensor_valid & label_valid
        if both_valid:
            filtered_asofs = tuple(a for a in selected_asof_dates if a in both_valid)

    n = len(filtered_asofs)
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

    write_idx = 0
    for asof in filtered_asofs:
        dp = build_multiscale_tensors(data_dir, code, asof)
        lb = build_label_from_data_dir(data_dir, code, asof, dp_ok=dp.dp_ok)
        if (not include_invalid) and (not lb.loss_mask):
            continue

        codes[write_idx] = code
        asof_dates[write_idx] = asof
        X_micro[write_idx] = dp.X_micro.astype(np.float32)
        X_mezzo[write_idx] = dp.X_mezzo.astype(np.float32)
        X_macro[write_idx] = dp.X_macro.astype(np.float32)
        mask_micro[write_idx] = dp.mask_micro.astype(np.uint8)
        mask_mezzo[write_idx] = dp.mask_mezzo.astype(np.uint8)
        mask_macro[write_idx] = dp.mask_macro.astype(np.uint8)
        y[write_idx] = np.float32(lb.y)
        y_raw[write_idx] = np.float32(lb.y_raw)
        y_z[write_idx] = np.float32(lb.y_z)
        dp_ok[write_idx] = bool(dp.dp_ok)
        label_ok[write_idx] = bool(lb.label_ok)
        loss_mask[write_idx] = bool(lb.loss_mask)
        write_idx += 1

    shard_path = Path(shard_dir) / f"{code}.npz"
    np.savez(
        shard_path,
        codes=codes[:write_idx],
        asof_dates=asof_dates[:write_idx],
        X_micro=X_micro[:write_idx],
        X_mezzo=X_mezzo[:write_idx],
        X_macro=X_macro[:write_idx],
        mask_micro=mask_micro[:write_idx],
        mask_mezzo=mask_mezzo[:write_idx],
        mask_macro=mask_macro[:write_idx],
        y=y[:write_idx],
        y_raw=y_raw[:write_idx],
        y_z=y_z[:write_idx],
        dp_ok=dp_ok[:write_idx],
        label_ok=label_ok[:write_idx],
        loss_mask=loss_mask[:write_idx],
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
) -> TrainDatasetBundle:
    selected_codes = resolve_codes(data_dir, codes)
    selected_asof_dates = resolve_asof_dates(data_dir, asof_dates)

    asof_tuple = tuple(selected_asof_dates)
    with tempfile.TemporaryDirectory(prefix="train_dataset_shards_") as shard_dir:
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
