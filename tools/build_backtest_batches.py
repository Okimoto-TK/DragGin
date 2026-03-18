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
from src.train.runner import MultiScaleRegressor

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@dataclass
class FlowContext:
    daily_dates: list[str]
    daily_date_to_idx: dict[str, int]
    flow_by_date: dict[str, np.ndarray]


@dataclass
class WorkerContext:
    device: torch.device
    model: MultiScaleRegressor
    data_dir: str
    asof_dates: list[str]
    infer_batch_size: int


_WORKER_CONTEXT: WorkerContext | None = None


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


def _build_model(
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


def _init_worker(
    data_dir: str,
    asof_dates: list[str],
    checkpoint: str,
    hidden_dim: int,
    num_heads: int,
    dropout: float,
    use_seq_context: bool,
    enable_dynamic_threshold: bool,
    enable_free_branch: bool,
    infer_batch_size: int,
) -> None:
    global _WORKER_CONTEXT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(
        device=device,
        checkpoint=checkpoint,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        use_seq_context=use_seq_context,
        enable_dynamic_threshold=enable_dynamic_threshold,
        enable_free_branch=enable_free_branch,
    )
    _WORKER_CONTEXT = WorkerContext(
        device=device,
        model=model,
        data_dir=data_dir,
        asof_dates=list(asof_dates),
        infer_batch_size=max(1, int(infer_batch_size)),
    )


def _infer_one_code(code: str) -> pd.DataFrame:
    global _WORKER_CONTEXT
    if _WORKER_CONTEXT is None:
        raise RuntimeError("worker context is not initialized")

    ctx = _WORKER_CONTEXT
    flow_ctx = _build_flow_context(Path(ctx.data_dir), code)
    recs: list[dict[str, Any]] = []
    feats: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for asof in ctx.asof_dates:
        dp = build_multiscale_tensors(ctx.data_dir, code, asof)
        flow_x, flow_mask, flow_ok = _flow_window_from_context(flow_ctx, asof)
        tensor_ok = bool(dp.dp_ok)
        sample_ok = bool(tensor_ok and flow_ok)
        if not sample_ok:
            reason = "ok"
            if not tensor_ok and not flow_ok:
                reason = "tensor_and_flow_invalid"
            elif not tensor_ok:
                reason = "tensor_invalid"
            elif not flow_ok:
                reason = "flow_invalid"
            recs.append(
                {
                    "code": code,
                    "asof_date": asof,
                    "yhat": np.nan,
                    "tensor_ok": tensor_ok,
                    "flow_ok": bool(flow_ok),
                    "sample_ok": sample_ok,
                    "reason": reason,
                }
            )
            continue
        feats.append(
            (
                asof,
                dp.X_micro.astype(np.float32),
                dp.X_mezzo.astype(np.float32),
                dp.X_macro.astype(np.float32),
                dp.mask_micro.astype(np.uint8),
                dp.mask_mezzo.astype(np.uint8),
                dp.mask_macro.astype(np.uint8),
                flow_x,
                flow_mask.astype(np.uint8),
            )
        )

    with torch.inference_mode():
        for start in range(0, len(feats), ctx.infer_batch_size):
            chunk = feats[start : start + ctx.infer_batch_size]
            if not chunk:
                continue
            asof_chunk = [x[0] for x in chunk]
            batch = {
                "x_micro": torch.from_numpy(np.stack([x[1] for x in chunk], axis=0)).to(ctx.device),
                "x_mezzo": torch.from_numpy(np.stack([x[2] for x in chunk], axis=0)).to(ctx.device),
                "x_macro": torch.from_numpy(np.stack([x[3] for x in chunk], axis=0)).to(ctx.device),
                "mask_micro": torch.from_numpy(np.stack([x[4] for x in chunk], axis=0)).to(torch.bool).to(ctx.device),
                "mask_mezzo": torch.from_numpy(np.stack([x[5] for x in chunk], axis=0)).to(torch.bool).to(ctx.device),
                "mask_macro": torch.from_numpy(np.stack([x[6] for x in chunk], axis=0)).to(torch.bool).to(ctx.device),
                "flow_x": torch.from_numpy(np.stack([x[7] for x in chunk], axis=0)).to(ctx.device),
                "flow_mask": torch.from_numpy(np.stack([x[8] for x in chunk], axis=0)).to(torch.bool).to(ctx.device),
            }
            y_hat, _ = ctx.model(batch)
            scores = y_hat.detach().to("cpu").numpy().reshape(-1)
            for i, asof in enumerate(asof_chunk):
                recs.append(
                    {
                        "code": code,
                        "asof_date": asof,
                        "yhat": float(scores[i]),
                        "tensor_ok": True,
                        "flow_ok": True,
                        "sample_ok": True,
                        "reason": "ok",
                    }
                )

    df = pd.DataFrame(recs)
    if df.empty:
        df = pd.DataFrame(
            {
                "code": [code],
                "asof_date": [None],
                "yhat": [np.nan],
                "tensor_ok": [False],
                "flow_ok": [False],
                "sample_ok": [False],
                "reason": ["empty"],
            }
        )
    return df.sort_values("asof_date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline backtest batches/scores")
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

    calendar_dates = _read_calendar_dates(data_dir)
    val_shards = _expand_paths(args.val_shards or [])
    val_dates = _resolve_val_dates(calendar_dates, args.val_ratio, args.val_embargo_days, val_shards)
    if len(val_dates) < 6:
        raise ValueError("validation dates must be >= 6 to backtest until last-5 day")
    asof_dates = val_dates[:-5]

    codes = _resolve_codes(data_dir)
    if not codes:
        raise ValueError("no stock folders with required parquet files found")

    requested_workers = max(1, int(args.num_workers))
    has_cuda = torch.cuda.is_available()
    if has_cuda and requested_workers > 1:
        print("[warn] CUDA detected; forcing --num-workers=1 to avoid multi-process GPU inference contention.")
    num_workers = 1 if has_cuda else requested_workers

    shard_dir = out_dir / "score_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    init_kwargs = dict(
        data_dir=str(data_dir),
        asof_dates=asof_dates,
        checkpoint=str(args.checkpoint),
        hidden_dim=int(args.hidden_dim),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        use_seq_context=bool(args.use_seq_context),
        enable_dynamic_threshold=bool(args.enable_dynamic_threshold),
        enable_free_branch=bool(args.enable_free_branch),
        infer_batch_size=int(args.infer_batch_size),
    )

    if num_workers <= 1:
        _init_worker(**init_kwargs)
        iterator = _iter_progress(codes, total=len(codes), show_progress=bool(args.show_progress), desc="building backtest shards")
        for code in iterator:
            df = _infer_one_code(code)
            df.to_parquet(shard_dir / f"{code}.parquet", index=False)
    else:
        initargs = (
            init_kwargs["data_dir"],
            init_kwargs["asof_dates"],
            init_kwargs["checkpoint"],
            init_kwargs["hidden_dim"],
            init_kwargs["num_heads"],
            init_kwargs["dropout"],
            init_kwargs["use_seq_context"],
            init_kwargs["enable_dynamic_threshold"],
            init_kwargs["enable_free_branch"],
            init_kwargs["infer_batch_size"],
        )
        with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker, initargs=initargs) as ex:
            futures = {ex.submit(_infer_one_code, code): code for code in codes}
            progress_iter = _iter_progress(as_completed(futures), total=len(futures), show_progress=bool(args.show_progress), desc="building backtest shards")
            for fut in progress_iter:
                code = futures[fut]
                df = fut.result()
                df.to_parquet(shard_dir / f"{code}.parquet", index=False)

    scores_path: str | None = None
    if bool(args.merge_scores):
        merged = pd.concat([pd.read_parquet(p) for p in sorted(shard_dir.glob("*.parquet"))], ignore_index=True)
        merged_path = out_dir / "scores.parquet"
        merged.to_parquet(merged_path, index=False)
        scores_path = str(merged_path.resolve())
        print(f"saved scores: {merged_path}")

    meta = {
        "codes": len(codes),
        "asof_dates": len(asof_dates),
        "score_shards": str(shard_dir.resolve()),
        "scores": scores_path,
        "merge_scores": bool(args.merge_scores),
        "requested_num_workers": requested_workers,
        "effective_num_workers": num_workers,
        "cuda_detected": has_cuda,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved score_shards: {shard_dir}")
    print(f"saved meta: {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
