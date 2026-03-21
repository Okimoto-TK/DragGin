from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feat.build_multiscale_tensor import build_multiscale_tensors
from src.infer import build_model, infer_from_feature_shard, merge_score_shards

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@dataclass
class FlowContext:
    daily_dates: list[str]
    daily_date_to_idx: dict[str, int]
    flow_by_date: dict[str, np.ndarray]


def _iter_progress(iterable, total: int, show_progress: bool, desc: str):
    if show_progress and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def _to_datestr(v: Any) -> str:
    ts_ = pd.to_datetime(v, errors="coerce")
    if pd.isna(ts_):
        raise ValueError(f"invalid date value: {v}")
    return ts_.strftime("%Y-%m-%d")


def _read_calendar_dates(data_dir: Path) -> list[str]:
    cal_path = data_dir / "calendar.parquet"
    if not cal_path.exists():
        raise FileNotFoundError(f"calendar file not found: {cal_path}")
    df = pd.read_parquet(cal_path)
    for col in ["trade_date", "cal_date", "date"]:
        if col in df.columns:
            values = sorted({_to_datestr(x) for x in df[col].dropna().tolist()})
            if values:
                return values
    raise ValueError(f"cannot find date column in {cal_path}")


def _resolve_codes(data_dir: Path) -> list[str]:
    out: list[str] = []
    for p in sorted(data_dir.iterdir()):
        if p.is_dir() and (p / "daily.parquet").exists() and (p / "5min.parquet").exists() and (p / "moneyflow.parquet").exists():
            out.append(p.name)
    return out


def _expand_paths(raw_paths: list[str]) -> list[str]:
    expanded: list[str] = []
    for raw in raw_paths:
        if any(ch in raw for ch in "*?[]"):
            expanded.extend(str(p) for p in sorted(Path().glob(raw)))
        else:
            expanded.append(raw)
    return expanded


def _resolve_val_dates(calendar_dates: list[str], val_ratio: float, val_embargo_days: int, val_shards: list[str]) -> list[str]:
    if val_shards:
        out: set[str] = set()
        for path in val_shards:
            payload = np.load(path, allow_pickle=True).item()
            if "asof_dates" not in payload:
                raise KeyError(f"missing asof_dates in shard: {path}")
            out.update(str(x) for x in payload["asof_dates"].astype(str).tolist())
        return sorted(out)

    unique_dates = sorted(set(calendar_dates))
    embargo_days = max(0, int(val_embargo_days))
    ratio = float(min(max(val_ratio, 0.0), 0.99))
    usable_for_train_val = len(unique_dates) - embargo_days
    if usable_for_train_val <= 1:
        raise ValueError("insufficient calendar dates for val split")
    val_dates_count = int(round(usable_for_train_val * ratio))
    val_dates_count = max(1, min(usable_for_train_val - 1, val_dates_count))
    train_dates_count = usable_for_train_val - val_dates_count
    return sorted(unique_dates[train_dates_count + embargo_days :])


def _build_flow_context(data_dir: Path, code: str) -> FlowContext | None:
    d1_path = data_dir / code / "daily.parquet"
    mf_path = data_dir / code / "moneyflow.parquet"
    if not d1_path.exists() or not mf_path.exists():
        return None

    d1 = pd.read_parquet(d1_path, columns=["trade_date", "volume"])
    mf = pd.read_parquet(mf_path, columns=["trade_date", "net_mf_vol", "buy_lg_vol", "sell_lg_vol", "buy_elg_vol", "sell_elg_vol"])

    d1["trade_date"] = pd.to_datetime(d1["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    d1["volume"] = pd.to_numeric(d1["volume"], errors="coerce")
    d1 = d1.dropna().sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    if d1.empty:
        return None

    mf["trade_date"] = pd.to_datetime(mf["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for c in ["net_mf_vol", "buy_lg_vol", "sell_lg_vol", "buy_elg_vol", "sell_elg_vol"]:
        mf[c] = pd.to_numeric(mf[c], errors="coerce")
    mf = mf.dropna().sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    if mf.empty:
        return None

    mf_map: dict[str, np.ndarray] = {}
    for _, row in mf.iterrows():
        mf_map[str(row["trade_date"])] = np.asarray(
            [
                float(row["net_mf_vol"]),
                float(row["buy_lg_vol"] - row["sell_lg_vol"]),
                float(row["buy_elg_vol"] - row["sell_elg_vol"]),
                float(row["buy_lg_vol"] + row["buy_elg_vol"] - row["sell_lg_vol"] - row["sell_elg_vol"]),
            ],
            dtype=np.float64,
        )

    daily_dates = d1["trade_date"].astype(str).tolist()
    flow_by_date: dict[str, np.ndarray] = {}
    for _, row in d1.iterrows():
        d = str(row["trade_date"])
        if d not in mf_map:
            continue
        volume_lots = float(row["volume"]) / 100.0
        if (not np.isfinite(volume_lots)) or volume_lots <= 0:
            continue
        vec = mf_map[d] / volume_lots
        if np.isfinite(vec).all():
            flow_by_date[d] = vec.astype(np.float32)

    return FlowContext(
        daily_dates=daily_dates,
        daily_date_to_idx={d: i for i, d in enumerate(daily_dates)},
        flow_by_date=flow_by_date,
    )


def _flow_window_from_context(ctx: FlowContext | None, asof: str) -> tuple[np.ndarray, np.ndarray, bool]:
    if ctx is None:
        return np.zeros((30, 4), dtype=np.float32), np.zeros((30,), dtype=np.uint8), False
    idx = ctx.daily_date_to_idx.get(asof)
    if idx is None or idx + 1 < 30:
        return np.zeros((30, 4), dtype=np.float32), np.zeros((30,), dtype=np.uint8), False
    tail_dates = ctx.daily_dates[idx + 1 - 30 : idx + 1]
    rows: list[np.ndarray] = []
    for d in tail_dates:
        vec = ctx.flow_by_date.get(d)
        if vec is None:
            return np.zeros((30, 4), dtype=np.float32), np.zeros((30,), dtype=np.uint8), False
        rows.append(vec)
    flow_x = np.stack(rows, axis=0).astype(np.float32)
    if flow_x.shape != (30, 4) or (not np.isfinite(flow_x).all()):
        return np.zeros((30, 4), dtype=np.float32), np.zeros((30,), dtype=np.uint8), False
    return flow_x, np.ones((30,), dtype=np.uint8), True


def _build_feature_shard(data_dir: str, code: str, asof_dates: list[str], out_dir: str) -> dict[str, Any]:
    flow_ctx = _build_flow_context(Path(data_dir), code)
    n = len(asof_dates)
    payload = {
        "code": code,
        "asof_dates": np.asarray(asof_dates, dtype=object),
        "x_micro": np.zeros((n, 48, 6), dtype=np.float32),
        "x_mezzo": np.zeros((n, 40, 6), dtype=np.float32),
        "x_macro": np.zeros((n, 30, 6), dtype=np.float32),
        "mask_micro": np.zeros((n, 48), dtype=np.uint8),
        "mask_mezzo": np.zeros((n, 40), dtype=np.uint8),
        "mask_macro": np.zeros((n, 30), dtype=np.uint8),
        "flow_x": np.zeros((n, 30, 4), dtype=np.float32),
        "flow_mask": np.zeros((n, 30), dtype=np.uint8),
        "tensor_ok": np.zeros((n,), dtype=np.bool_),
        "flow_ok": np.zeros((n,), dtype=np.bool_),
        "sample_ok": np.zeros((n,), dtype=np.bool_),
        "reason": np.empty((n,), dtype=object),
    }

    for idx, asof in enumerate(asof_dates):
        dp = build_multiscale_tensors(data_dir, code, asof)
        flow_x, flow_mask, flow_ok = _flow_window_from_context(flow_ctx, asof)
        tensor_ok = bool(dp.dp_ok)
        sample_ok = bool(tensor_ok and flow_ok)

        payload["tensor_ok"][idx] = tensor_ok
        payload["flow_ok"][idx] = bool(flow_ok)
        payload["sample_ok"][idx] = sample_ok

        if not sample_ok:
            if not tensor_ok and not flow_ok:
                payload["reason"][idx] = "tensor_and_flow_invalid"
            elif not tensor_ok:
                payload["reason"][idx] = "tensor_invalid"
            else:
                payload["reason"][idx] = "flow_invalid"
            continue

        payload["x_micro"][idx] = dp.X_micro.astype(np.float32)
        payload["x_mezzo"][idx] = dp.X_mezzo.astype(np.float32)
        payload["x_macro"][idx] = dp.X_macro.astype(np.float32)
        payload["mask_micro"][idx] = dp.mask_micro.astype(np.uint8)
        payload["mask_mezzo"][idx] = dp.mask_mezzo.astype(np.uint8)
        payload["mask_macro"][idx] = dp.mask_macro.astype(np.uint8)
        payload["flow_x"][idx] = flow_x
        payload["flow_mask"][idx] = flow_mask.astype(np.uint8)
        payload["reason"][idx] = "ok"

    shard_path = Path(out_dir) / f"{code}.npz"
    np.savez_compressed(shard_path, **payload)
    return {"code": code, "path": str(shard_path), "rows": n}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline backtest features and score shards")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--val-embargo-days", type=int, default=30)
    parser.add_argument("--val-shards", nargs="+", default=None)
    parser.add_argument("--hidden-dim", type=int, default=320)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-seq-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-dynamic-threshold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-free-branch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--infer-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--show-progress", type=int, choices=[0, 1], default=1)
    parser.add_argument("--merge-scores", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = out_dir / "feature_shards"
    score_dir = out_dir / "score_shards"
    feature_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    calendar_dates = _read_calendar_dates(data_dir)
    val_shards = _expand_paths(args.val_shards or [])
    val_dates = _resolve_val_dates(calendar_dates, args.val_ratio, args.val_embargo_days, val_shards)
    if len(val_dates) < 6:
        raise ValueError("validation dates must be >= 6 to backtest until last-5 day")
    asof_dates = val_dates[:-5]

    codes = _resolve_codes(data_dir)
    if not codes:
        raise ValueError("no stock folders with required parquet files found")

    num_workers = max(1, int(args.num_workers))

    feature_jobs = [(str(data_dir), code, asof_dates, str(feature_dir)) for code in codes]
    if num_workers <= 1:
        iterator = _iter_progress(feature_jobs, total=len(feature_jobs), show_progress=bool(args.show_progress), desc="building feature shards")
        for job in iterator:
            _build_feature_shard(*job)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(_build_feature_shard, *job): job[1] for job in feature_jobs}
            progress_iter = _iter_progress(as_completed(futures), total=len(futures), show_progress=bool(args.show_progress), desc="building feature shards")
            for fut in progress_iter:
                fut.result()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Legacy CLI args remain as fallback for old checkpoints without model_hparams.
    model = build_model(
        device=device,
        checkpoint=str(args.checkpoint),
        hidden_dim=int(args.hidden_dim),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        use_seq_context=bool(args.use_seq_context),
        enable_dynamic_threshold=bool(args.enable_dynamic_threshold),
        enable_free_branch=bool(args.enable_free_branch),
    )

    feature_paths = sorted(feature_dir.glob("*.npz"))
    infer_iter = _iter_progress(feature_paths, total=len(feature_paths), show_progress=bool(args.show_progress), desc="inferring score shards")
    for shard_path in infer_iter:
        df = infer_from_feature_shard(model, device, shard_path, max(1, int(args.infer_batch_size)))
        df.to_parquet(score_dir / f"{shard_path.stem}.parquet", index=False)

    scores_path: str | None = None
    if bool(args.merge_scores):
        merged = merge_score_shards(score_dir)
        merged_path = out_dir / "scores.parquet"
        merged.to_parquet(merged_path, index=False)
        scores_path = str(merged_path.resolve())
        print(f"saved scores: {merged_path}")

    meta = {
        "codes": len(codes),
        "asof_dates": len(asof_dates),
        "feature_shards": str(feature_dir.resolve()),
        "score_shards": str(score_dir.resolve()),
        "scores": scores_path,
        "merge_scores": bool(args.merge_scores),
        "feature_num_workers": num_workers,
        "inference_device": str(device),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved feature_shards: {feature_dir}")
    print(f"saved score_shards: {score_dir}")
    print(f"saved meta: {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
