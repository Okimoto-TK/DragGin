from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.feat.build_multiscale_tensor import build_calendar_from_daily_filenames
from src.feat.build_multiscale_tensor import build_multiscale_tensors
from src.feat.build_multiscale_tensor import clear_tensor_worker_cache
from src.feat.labels_risk_adj import build_label_from_data_dir
from src.feat.labels_risk_adj import clear_label_worker_cache

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
    flow_x: np.ndarray
    mask_micro: np.ndarray
    mask_mezzo: np.ndarray
    mask_macro: np.ndarray
    flow_mask: np.ndarray
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
        flow_x=np.zeros((0, 30, 4), dtype=np.float32),
        mask_micro=np.zeros((0, 48), dtype=np.uint8),
        mask_mezzo=np.zeros((0, 40), dtype=np.uint8),
        mask_macro=np.zeros((0, 30), dtype=np.uint8),
        flow_mask=np.zeros((0, 30), dtype=np.uint8),
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


@lru_cache(maxsize=512)
def _load_daily_volume(data_dir: str, code: str) -> pd.DataFrame | None:
    path = Path(data_dir) / code / "daily.parquet"
    if not path.exists():
        return None
    try:
        d1 = pd.read_parquet(path, columns=["trade_date", "volume"])
    except Exception:
        return None
    if "trade_date" not in d1.columns or "volume" not in d1.columns:
        return None
    d1["trade_date"] = pd.to_datetime(d1["trade_date"], errors="coerce").dt.date
    d1["volume"] = pd.to_numeric(d1["volume"], errors="coerce")
    d1 = d1.dropna(subset=["trade_date", "volume"]).sort_values("trade_date")
    d1 = d1.drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)
    return d1


@lru_cache(maxsize=512)
def _load_moneyflow(data_dir: str, code: str) -> pd.DataFrame | None:
    path = Path(data_dir) / code / "moneyflow.parquet"
    if not path.exists():
        return None
    cols = [
        "trade_date",
        "net_mf_vol",
        "buy_lg_vol",
        "sell_lg_vol",
        "buy_elg_vol",
        "sell_elg_vol",
    ]
    try:
        mf = pd.read_parquet(path, columns=cols)
    except Exception:
        return None
    if not set(cols).issubset(set(mf.columns)):
        return None
    mf["trade_date"] = pd.to_datetime(mf["trade_date"], errors="coerce").dt.date
    for c in cols[1:]:
        mf[c] = pd.to_numeric(mf[c], errors="coerce")
    mf = mf.dropna(subset=cols).sort_values("trade_date")
    mf = mf.drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)
    return mf


def _invalid_flow() -> tuple[np.ndarray, np.ndarray, bool]:
    return np.zeros((30, 4), dtype=np.float32), np.zeros((30,), dtype=np.uint8), False


def _build_flow_features(data_dir: str | Path, code: str, asof: str) -> tuple[np.ndarray, np.ndarray, bool]:
    data_dir_key = str(Path(data_dir).resolve())
    d1 = _load_daily_volume(data_dir_key, code)
    mf = _load_moneyflow(data_dir_key, code)
    if d1 is None or mf is None or d1.empty or mf.empty:
        return _invalid_flow()

    asof_date = pd.to_datetime(asof).date()
    dates = d1["trade_date"].tolist()
    if asof_date not in set(dates):
        return _invalid_flow()
    idx = dates.index(asof_date)
    if idx + 1 < 30:
        return _invalid_flow()

    tail_dates = dates[idx + 1 - 30 : idx + 1]
    w = d1[d1["trade_date"].isin(set(tail_dates))].merge(mf, on="trade_date", how="inner").sort_values("trade_date")
    if len(w) != 30:
        return _invalid_flow()

    # daily.volume unit: shares(股); moneyflow *_vol unit: lots(手, 100 shares)
    # normalize to same unit before ratio computation.
    volume_shares = w["volume"].to_numpy(dtype=np.float64)
    volume_lots = volume_shares / 100.0
    if (~np.isfinite(volume_lots)).any() or (volume_lots <= 0).any():
        return _invalid_flow()

    net = w["net_mf_vol"].to_numpy(dtype=np.float64)
    lg = (w["buy_lg_vol"] - w["sell_lg_vol"]).to_numpy(dtype=np.float64)
    elg = (w["buy_elg_vol"] - w["sell_elg_vol"]).to_numpy(dtype=np.float64)
    lg_elg = (w["buy_lg_vol"] + w["buy_elg_vol"] - w["sell_lg_vol"] - w["sell_elg_vol"]).to_numpy(dtype=np.float64)

    flow_x = np.stack([net / volume_lots, lg / volume_lots, elg / volume_lots, lg_elg / volume_lots], axis=1).astype(np.float32)
    if flow_x.shape != (30, 4) or (not np.isfinite(flow_x).all()):
        return _invalid_flow()

    return flow_x, np.ones((30,), dtype=np.uint8), True


def _rows_from_code_task(
    data_dir: str | Path,
    code: str,
    selected_asof_dates: tuple[str, ...],
    include_invalid: bool,
    shard_dir: str,
) -> dict:
    try:
        n = len(selected_asof_dates)
        codes = np.empty((n,), dtype=object)
        asof_dates = np.empty((n,), dtype=object)
        X_micro = np.zeros((n, 48, 6), dtype=np.float32)
        X_mezzo = np.zeros((n, 40, 6), dtype=np.float32)
        X_macro = np.zeros((n, 30, 6), dtype=np.float32)
        flow_x = np.zeros((n, 30, 4), dtype=np.float32)
        mask_micro = np.zeros((n, 48), dtype=np.uint8)
        mask_mezzo = np.zeros((n, 40), dtype=np.uint8)
        mask_macro = np.zeros((n, 30), dtype=np.uint8)
        flow_mask = np.zeros((n, 30), dtype=np.uint8)
        y = np.zeros((n,), dtype=np.float32)
        y_raw = np.zeros((n,), dtype=np.float32)
        y_z = np.zeros((n,), dtype=np.float32)
        dp_ok = np.zeros((n,), dtype=np.bool_)
        label_ok = np.zeros((n,), dtype=np.bool_)
        loss_mask = np.zeros((n,), dtype=np.bool_)

        write_idx = 0
        for asof in selected_asof_dates:
            dp = build_multiscale_tensors(data_dir, code, asof)
            lb = build_label_from_data_dir(data_dir, code, asof, dp_ok=dp.dp_ok)
            fx, fm, flow_ok = _build_flow_features(data_dir, code, asof)
            effective_loss_mask = bool(lb.loss_mask and flow_ok)
            if (not include_invalid) and (not effective_loss_mask):
                continue

            codes[write_idx] = code
            asof_dates[write_idx] = asof
            X_micro[write_idx] = dp.X_micro.astype(np.float32)
            X_mezzo[write_idx] = dp.X_mezzo.astype(np.float32)
            X_macro[write_idx] = dp.X_macro.astype(np.float32)
            flow_x[write_idx] = fx.astype(np.float32)
            mask_micro[write_idx] = dp.mask_micro.astype(np.uint8)
            mask_mezzo[write_idx] = dp.mask_mezzo.astype(np.uint8)
            mask_macro[write_idx] = dp.mask_macro.astype(np.uint8)
            flow_mask[write_idx] = fm.astype(np.uint8)
            y[write_idx] = np.float32(lb.y)
            y_raw[write_idx] = np.float32(lb.y_raw)
            y_z[write_idx] = np.float32(lb.y_z)
            dp_ok[write_idx] = bool(dp.dp_ok and flow_ok)
            label_ok[write_idx] = bool(lb.label_ok)
            loss_mask[write_idx] = effective_loss_mask
            write_idx += 1

        shard_path = Path(shard_dir) / f"{code}.npy"
        shard_payload = {
            "codes": codes[:write_idx],
            "asof_dates": asof_dates[:write_idx],
            "X_micro": X_micro[:write_idx],
            "X_mezzo": X_mezzo[:write_idx],
            "X_macro": X_macro[:write_idx],
            "flow_x": flow_x[:write_idx],
            "mask_micro": mask_micro[:write_idx],
            "mask_mezzo": mask_mezzo[:write_idx],
            "mask_macro": mask_macro[:write_idx],
            "flow_mask": flow_mask[:write_idx],
            "y": y[:write_idx],
            "y_raw": y_raw[:write_idx],
            "y_z": y_z[:write_idx],
            "dp_ok": dp_ok[:write_idx],
            "label_ok": label_ok[:write_idx],
            "loss_mask": loss_mask[:write_idx],
        }
        np.save(shard_path, shard_payload, allow_pickle=True)
        return {"path": str(shard_path), "rows": int(write_idx)}
    finally:
        clear_tensor_worker_cache()
        clear_label_worker_cache()
        _load_daily_volume.cache_clear()
        _load_moneyflow.cache_clear()


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
        "flow_x": [],
        "mask_micro": [],
        "mask_mezzo": [],
        "mask_macro": [],
        "flow_mask": [],
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
        data = np.load(info["path"], allow_pickle=True).item()
        for k in parts:
            parts[k].append(data[k])

    if not parts["y"]:
        return _empty_bundle()

    return TrainDatasetBundle(
        codes=np.concatenate(parts["codes"]).astype(object),
        asof_dates=np.concatenate(parts["asof_dates"]).astype(object),
        X_micro=np.concatenate(parts["X_micro"], axis=0).astype(np.float32),
        X_mezzo=np.concatenate(parts["X_mezzo"], axis=0).astype(np.float32),
        X_macro=np.concatenate(parts["X_macro"], axis=0).astype(np.float32),
        flow_x=np.concatenate(parts["flow_x"], axis=0).astype(np.float32),
        mask_micro=np.concatenate(parts["mask_micro"], axis=0).astype(np.uint8),
        mask_mezzo=np.concatenate(parts["mask_mezzo"], axis=0).astype(np.uint8),
        mask_macro=np.concatenate(parts["mask_macro"], axis=0).astype(np.uint8),
        flow_mask=np.concatenate(parts["flow_mask"], axis=0).astype(np.uint8),
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


def build_train_dataset_shards(
    data_dir: str | Path,
    out_dir: str | Path,
    codes: list[str] | None = None,
    asof_dates: list[str] | None = None,
    include_invalid: bool = False,
    num_workers: int = 1,
    show_progress: bool = True,
) -> list[dict]:
    selected_codes = resolve_codes(data_dir, codes)
    selected_asof_dates = resolve_asof_dates(data_dir, asof_dates)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    asof_tuple = tuple(selected_asof_dates)
    shard_infos: list[dict] = []
    if num_workers <= 1:
        iterator = _iter_progress(selected_codes, total=len(selected_codes), show_progress=show_progress, desc="building train dataset")
        for code in iterator:
            shard_infos.append(_rows_from_code_task(data_dir, code, asof_tuple, include_invalid, str(out_path)))
        return shard_infos

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_rows_from_code_task, data_dir, code, asof_tuple, include_invalid, str(out_path))
            for code in selected_codes
        ]
        progress_iter = _iter_progress(as_completed(futures), total=len(futures), show_progress=show_progress, desc="building train dataset")
        for fut in progress_iter:
            shard_infos.append(fut.result())
    return shard_infos


def save_train_dataset(bundle: TrainDatasetBundle, out_npz: str | Path) -> None:
    np.savez_compressed(
        out_npz,
        codes=bundle.codes,
        asof_dates=bundle.asof_dates,
        X_micro=bundle.X_micro,
        X_mezzo=bundle.X_mezzo,
        X_macro=bundle.X_macro,
        flow_x=bundle.flow_x,
        mask_micro=bundle.mask_micro,
        mask_mezzo=bundle.mask_mezzo,
        mask_macro=bundle.mask_macro,
        flow_mask=bundle.flow_mask,
        y=bundle.y,
        y_raw=bundle.y_raw,
        y_z=bundle.y_z,
        dp_ok=bundle.dp_ok,
        label_ok=bundle.label_ok,
        loss_mask=bundle.loss_mask,
    )
