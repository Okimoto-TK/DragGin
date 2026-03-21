from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import tushare as ts

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import DailyUpdateConfig, update_daily_market_data
from src.infer import (
    build_model,
    infer_from_feature_shard,
    infer_from_feature_shards,
    load_daily_map,
    load_latest_position_state,
    merge_score_shards,
    read_calendar_dates,
    resolve_codes,
    run_trade_simulation,
)
from tools.build_backtest_batches import _build_feature_shard
from tools.preprocess_data import _normalize_5min, _normalize_daily, _normalize_moneyflow


def _vlog(verbose: bool, message: str) -> None:
    if bool(verbose):
        print(f"[trader] {message}")


_UPDATE_DAILY_DAILY_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "adj_factor",
    "vol",
    "amount",
    "pct_chg",
]

_UPDATE_DAILY_MONEYFLOW_COLUMNS = [
    "ts_code",
    "trade_date",
    "buy_sm_vol", "buy_sm_amount",
    "sell_sm_vol", "sell_sm_amount",
    "buy_md_vol", "buy_md_amount",
    "sell_md_vol", "sell_md_amount",
    "buy_lg_vol", "buy_lg_amount",
    "sell_lg_vol", "sell_lg_amount",
    "buy_elg_vol", "buy_elg_amount",
    "sell_elg_vol", "sell_elg_amount",
    "net_mf_vol", "net_mf_amount",
]


def _read_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        return pd.read_parquet(path)



def _canonicalize_update_daily_daily(df: pd.DataFrame) -> pd.DataFrame:
    raw = df.copy()

    if "ts_code" in raw.columns and "code" not in raw.columns:
        raw = raw.rename(columns={"ts_code": "code"})
    if "vol" in raw.columns and "volume" not in raw.columns:
        raw = raw.rename(columns={"vol": "volume"})

    if "code" not in raw.columns:
        raise ValueError("daily data missing required column: code/ts_code")
    if "trade_date" not in raw.columns:
        raise ValueError("daily data missing required column: trade_date")

    raw["code"] = raw["code"].astype(str)
    raw["trade_date"] = pd.to_datetime(
        raw["trade_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    value_cols = ["open", "high", "low", "close", "volume", "adj_factor"]
    missing = [c for c in value_cols if c not in raw.columns]
    if missing:
        raise ValueError(f"daily data missing required columns: {missing}")

    out = raw[["code", "trade_date", "open", "high", "low", "close", "volume", "adj_factor"]].copy()

    for col in value_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.sort_values(["code", "trade_date"]).reset_index(drop=True)
    return _normalize_daily(out)



def _canonicalize_update_daily_moneyflow(df: pd.DataFrame) -> pd.DataFrame:
    raw = df.copy()
    if "ts_code" in raw.columns and "code" not in raw.columns:
        raw = raw.rename(columns={"ts_code": "code"})

    if "code" not in raw.columns:
        raise ValueError("moneyflow data missing required column: code/ts_code")
    if "trade_date" not in raw.columns:
        raise ValueError("moneyflow data missing required column: trade_date")

    raw["code"] = raw["code"].astype(str)
    raw["trade_date"] = pd.to_datetime(raw["trade_date"].astype(str), format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")

    value_cols = [
        "buy_sm_vol", "buy_sm_amount",
        "sell_sm_vol", "sell_sm_amount",
        "buy_md_vol", "buy_md_amount",
        "sell_md_vol", "sell_md_amount",
        "buy_lg_vol", "buy_lg_amount",
        "sell_lg_vol", "sell_lg_amount",
        "buy_elg_vol", "buy_elg_amount",
        "sell_elg_vol", "sell_elg_amount",
        "net_mf_vol", "net_mf_amount",
    ]

    missing = [c for c in value_cols if c not in raw.columns]
    for col in missing:
        raw[col] = 0.0

    out = raw[["code", "trade_date", *value_cols]].copy()

    for col in value_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.sort_values(["code", "trade_date"]).reset_index(drop=True)
    return _normalize_moneyflow(out)



def _canonicalize_update_daily_5min(df: pd.DataFrame, code_hint: str) -> pd.DataFrame:
    raw = df.copy()

    if "ts_code" in raw.columns and "code" not in raw.columns:
        raw = raw.rename(columns={"ts_code": "code"})
    if "t" in raw.columns and "dt" not in raw.columns:
        raw = raw.rename(columns={"t": "dt"})
    if "trade_time" in raw.columns and "dt" not in raw.columns:
        raw = raw.rename(columns={"trade_time": "dt"})
    if "o" in raw.columns and "open" not in raw.columns:
        raw = raw.rename(columns={"o": "open"})
    if "h" in raw.columns and "high" not in raw.columns:
        raw = raw.rename(columns={"h": "high"})
    if "l" in raw.columns and "low" not in raw.columns:
        raw = raw.rename(columns={"l": "low"})
    if "c" in raw.columns and "close" not in raw.columns:
        raw = raw.rename(columns={"c": "close"})
    if "v" in raw.columns and "volume" not in raw.columns:
        raw = raw.rename(columns={"v": "volume"})

    if "code" not in raw.columns:
        raw["code"] = str(code_hint)
    else:
        raw["code"] = raw["code"].fillna(str(code_hint)).astype(str)
        raw.loc[raw["code"].str.strip() == "", "code"] = str(code_hint)

    raw["dt"] = pd.to_datetime(raw["dt"])
    raw["trade_date"] = raw["dt"].dt.strftime("%Y-%m-%d")
    raw["time"] = raw["dt"].dt.strftime("%H:%M")

    out = raw[["code", "trade_date", "open", "high", "low", "close", "volume", "dt", "time"]].copy()

    out["code"] = out["code"].astype(str)
    out["trade_date"] = out["trade_date"].astype(str)
    out["open"] = pd.to_numeric(out["open"], errors="coerce")
    out["high"] = pd.to_numeric(out["high"], errors="coerce")
    out["low"] = pd.to_numeric(out["low"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out["dt"] = pd.to_datetime(out["dt"])
    out["time"] = out["time"].astype(str)

    out = out.sort_values(["code", "dt"]).reset_index(drop=True)
    return _normalize_5min(out)


def _write_processed_code_frames(processed_dir: Path, payloads: dict[str, dict[str, list[pd.DataFrame]]], verbose: bool) -> list[str]:
    written_codes: list[str] = []
    for code in sorted(payloads):
        code_dir = processed_dir / code
        code_dir.mkdir(parents=True, exist_ok=True)
        daily_parts = payloads[code].get("daily", [])
        min5_parts = payloads[code].get("5min", [])
        moneyflow_parts = payloads[code].get("moneyflow", [])
        if not daily_parts or not min5_parts or not moneyflow_parts:
            continue

        daily_df = pd.concat(daily_parts, ignore_index=True)
        daily_df["volume"] = pd.to_numeric(daily_df["volume"], errors="coerce")
        daily_df = daily_df.dropna(subset=["trade_date", "volume"])
        daily_df = daily_df[daily_df["volume"] > 0]
        daily_df = daily_df.sort_values("trade_date").drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)
        if daily_df.empty:
            continue
        valid_trade_dates = set(daily_df["trade_date"].tolist())
        daily_df.to_parquet(code_dir / "daily.parquet", index=False)

        min5_df = pd.concat(min5_parts, ignore_index=True)
        min5_df = min5_df[min5_df["trade_date"].isin(valid_trade_dates)].copy()
        min5_df = min5_df.sort_values("dt").reset_index(drop=True)
        if min5_df.empty:
            continue
        min5_df.to_parquet(code_dir / "5min.parquet", index=False)

        moneyflow_df = pd.concat(moneyflow_parts, ignore_index=True)
        moneyflow_df = moneyflow_df[moneyflow_df["trade_date"].isin(valid_trade_dates)].copy()
        moneyflow_df = moneyflow_df.sort_values("trade_date").drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)
        if moneyflow_df.empty:
            continue
        moneyflow_df.to_parquet(code_dir / "moneyflow.parquet", index=False)
        written_codes.append(code)
        _vlog(verbose, f"wrote processed code data for {code}: daily={len(daily_df)} 5min={len(min5_df)} moneyflow={len(moneyflow_df)}")
    return written_codes



def _build_breakpoints_from_st_snapshots(st_dir: Path, processed_dir: Path, verbose: bool) -> None:
    if not st_dir.exists():
        return
    prev_state: dict[str, int] = {}
    break_dates_by_code: dict[str, list[str]] = defaultdict(list)
    for path in sorted(st_dir.glob("*.parquet")):
        trade_date = path.stem[:8]
        if len(trade_date) != 8 or not trade_date.isdigit():
            continue
        df = pd.read_parquet(path)
        if "ts_code" not in df.columns:
            continue
        state_col = "is_st" if "is_st" in df.columns else None
        if state_col is None and "name" in df.columns:
            names = df["name"].astype(str).str.strip().str.upper()
            df = df.assign(is_st=(names.str.startswith("ST") | names.str.startswith("*ST")).astype(int))
            state_col = "is_st"
        if state_col is None:
            continue
        current_state = {
            str(row["ts_code"]): int(row[state_col])
            for _, row in df[["ts_code", state_col]].dropna(subset=["ts_code"]).iterrows()
            if str(row["ts_code"])
        }
        for code, state in current_state.items():
            if code in prev_state and prev_state[code] != state:
                break_dates_by_code[code].append(pd.to_datetime(trade_date).strftime("%Y-%m-%d"))
        prev_state.update(current_state)

    for code, break_dates in break_dates_by_code.items():
        code_dir = processed_dir / code
        if not code_dir.exists() or not (code_dir / "daily.parquet").exists():
            continue
        pd.DataFrame({"break_date": sorted(set(break_dates))}).to_parquet(code_dir / "breakpoints.parquet", index=False)
        _vlog(verbose, f"wrote breakpoints for {code}: {len(set(break_dates))} dates")



def _build_processed_dataset_from_update_daily(data_dir: Path, processed_dir: Path, verbose: bool) -> Path:
    raw_dir = data_dir / "raw"
    daily_dir = raw_dir / "daily"
    moneyflow_dir = raw_dir / "moneyflow"
    min5_dir = raw_dir / "5min"
    st_dir = raw_dir / "st"
    _vlog(verbose, f"building processed dataset from {raw_dir}")
    if not daily_dir.exists() or not moneyflow_dir.exists() or not min5_dir.exists():
        raise FileNotFoundError(f"update-daily raw outputs are incomplete under {raw_dir}")

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    payloads: dict[str, dict[str, list[pd.DataFrame]]] = defaultdict(lambda: {"daily": [], "5min": [], "moneyflow": []})
    all_calendar_dates: set[pd.Timestamp] = set()

    for path in sorted(daily_dir.glob("*.parquet")):
        df = _canonicalize_update_daily_daily(_read_parquet_columns(path, _UPDATE_DAILY_DAILY_COLUMNS))
        if df.empty:
            continue
        for code, sub in df.groupby("code", sort=False):
            payloads[str(code)]["daily"].append(sub.reset_index(drop=True))
        all_calendar_dates.update(pd.to_datetime(df["trade_date"], errors="coerce").dropna().tolist())

    for path in sorted(moneyflow_dir.glob("*.parquet")):
        df = _canonicalize_update_daily_moneyflow(_read_parquet_columns(path, _UPDATE_DAILY_MONEYFLOW_COLUMNS))
        if df.empty:
            continue
        for code, sub in df.groupby("code", sort=False):
            payloads[str(code)]["moneyflow"].append(sub.reset_index(drop=True))

    count = 0
    acount = len(sorted(min5_dir.iterdir()))
    for code_dir in sorted(min5_dir.iterdir()):
        count += 1
        _vlog(verbose, f"{code_dir}: {count}/{acount}")
        if not code_dir.is_dir():
            continue
        code_hint = code_dir.name
        for path in sorted(code_dir.glob("*.csv")):
            df = pd.read_csv(path, low_memory=False)
            norm = _canonicalize_update_daily_5min(df, code_hint=code_hint)
            if norm.empty:
                continue
            payloads[str(code_hint)]["5min"].append(norm.reset_index(drop=True))

    written_codes = _write_processed_code_frames(processed_dir, payloads, verbose=verbose)
    if all_calendar_dates:
        pd.DataFrame({"trade_date": sorted(all_calendar_dates)}).to_parquet(processed_dir / "calendar.parquet", index=False)
    _build_breakpoints_from_st_snapshots(st_dir, processed_dir, verbose=verbose)
    _vlog(verbose, f"processed dataset ready: codes={len(written_codes)} calendar_dates={len(all_calendar_dates)}")
    if not written_codes:
        raise ValueError("no usable processed code data was built from update-daily outputs")
    return processed_dir



def _infer_feature_shard_worker(
    shard_path: str,
    checkpoint: str,
    hidden_dim: int,
    num_heads: int,
    dropout: float,
    use_seq_context: bool,
    enable_dynamic_threshold: bool,
    enable_free_branch: bool,
    infer_batch_size: int,
    score_dir: str,
    device_str: str,
) -> str:
    device = torch.device(device_str)
    # Keep explicit args only as fallback for legacy checkpoints without model_hparams.
    model = build_model(
        device=device,
        checkpoint=checkpoint,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        use_seq_context=use_seq_context,
        enable_dynamic_threshold=enable_dynamic_threshold,
        enable_free_branch=enable_free_branch,
    )
    model.eval()
    df = infer_from_feature_shard(
        model,
        device,
        Path(shard_path),
        max(1, int(infer_batch_size)),
    )
    out_path = Path(score_dir) / f"{Path(shard_path).stem}.parquet"
    df.to_parquet(out_path, index=False)
    return str(out_path)



def _run_trade(args: argparse.Namespace) -> None:
    device_str = str(args.device).strip()
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but not available: {device_str}")
    data_dir = Path(args.data_dir)
    position_dir = Path(args.position_dir)
    position_dir.mkdir(parents=True, exist_ok=True)

    update_meta: dict[str, Any] | None = None
    if not args.no_update:
        update_meta = update_daily_market_data(
            DailyUpdateConfig(
                data_dir=data_dir,
                lookback_trading_days=max(2, int(args.lookback_trading_days)),
                request_sleep_seconds=max(0.0, float(args.sleep)),
                refresh_latest=not bool(args.no_refresh_latest),
                verbose=bool(args.verbose),
            )
        )

    work_dir = data_dir / "trade_workspace"
    feature_dir = work_dir / "feature_shards"
    score_dir = work_dir / "score_shards"
    processed_dir = work_dir / "processed"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    _build_processed_dataset_from_update_daily(data_dir, processed_dir, verbose=bool(args.verbose))
    calendar_dates = read_calendar_dates(processed_dir)
    if len(calendar_dates) < 2:
        raise ValueError("trade requires at least 2 processed trading dates")
    token = str(os.environ.get("TUSHARE_TOKEN", "")).strip()
    pro = ts.pro_api(token) if token else None
    asof_date = calendar_dates[-1]
    trade_date = pd.to_datetime(
            pro.trade_cal(exchange="", start_date=asof_date.replace("-", ""), is_open=1)
            .sort_values("cal_date")
            .query("cal_date > @asof_date.replace('-', '')")
            .iloc[0]["cal_date"]
        ).strftime("%Y-%m-%d")
    _vlog(verbose=args.verbose, message=f"{asof_date}, {trade_date}")
    codes = resolve_codes(processed_dir)
    if not codes:
        raise ValueError("no stock folders with required parquet files found after preprocessing")

    feature_jobs = [(str(processed_dir), code, [asof_date], str(feature_dir)) for code in codes]
    _vlog(bool(args.verbose), f"building feature shards: codes={len(feature_jobs)} asof={asof_date}")
    if int(args.num_workers) <= 1:
        for job in feature_jobs:
            _build_feature_shard(*job)
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.num_workers))) as ex:
            futures = [ex.submit(_build_feature_shard, *job) for job in feature_jobs]
            for fut in as_completed(futures):
                fut.result()

    feature_paths = sorted(feature_dir.glob("*.npz"))
    _vlog(bool(args.verbose), f"feature shards ready: {len(feature_paths)}")
    infer_workers = max(1, int(args.infer_workers or args.num_workers))
    worker_args = dict(
        checkpoint=str(args.checkpoint),
        hidden_dim=int(args.hidden_dim),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        use_seq_context=bool(args.use_seq_context),
        enable_dynamic_threshold=bool(args.enable_dynamic_threshold),
        enable_free_branch=bool(args.enable_free_branch),
        infer_batch_size=max(1, int(args.infer_batch_size)),
        score_dir=str(score_dir),
    )

    def _chunk_list(items: list[Path], n_chunks: int) -> list[list[Path]]:
        if not items:
            return []
        n_chunks = max(1, min(n_chunks, len(items)))
        buckets: list[list[Path]] = [[] for _ in range(n_chunks)]
        for i, item in enumerate(items):
            buckets[i % n_chunks].append(item)
        return [b for b in buckets if b]

    grouped_feature_paths = _chunk_list(feature_paths, infer_workers)
    _vlog(
        bool(args.verbose),
        f"running grouped inference: shards={len(feature_paths)} groups={len(grouped_feature_paths)} workers={infer_workers} batch={args.infer_batch_size}",
    )

    if infer_workers <= 1:
        for worker_id, group in enumerate(grouped_feature_paths):
            _infer_feature_shards_worker(
                [str(p) for p in group],
                **worker_args,
                device_str=device_str,
                worker_id=worker_id,
            )
    else:
        with ProcessPoolExecutor(max_workers=infer_workers) as ex:
            futures = [
                ex.submit(
                    _infer_feature_shards_worker,
                    [str(p) for p in group],
                    **worker_args,
                    device_str=device_str,
                    worker_id=worker_id,
                )
                for worker_id, group in enumerate(grouped_feature_paths)
            ]
            for fut in as_completed(futures):
                fut.result()

    score_df = merge_score_shards(score_dir)
    _vlog(bool(args.verbose), f"score shards merged: rows={len(score_df)}")
    score_by_date = {asof_date: [(str(row["code"]), float(row["yhat"])) for _, row in score_df.iterrows() if pd.notna(row["yhat"])]}

    daily_cache = {code: load_daily_map(processed_dir, code) for code in codes}
    if args.init is None:
        latest_position_date, initial_cash, initial_positions = load_latest_position_state(position_dir, daily_cache)
    else:
        latest_position_date, initial_cash, initial_positions = None, float(args.init), {}

    _vlog(bool(args.verbose), "loading initial state")
    _vlog(bool(args.verbose), f"executing trade step: asof={asof_date} trade_date={trade_date} topk={int(args.topk)}")
    payloads = run_trade_simulation(
        data_dir=processed_dir,
        score_by_date=score_by_date,
        asof_dates=[asof_date],
        trading_dates=[trade_date],
        topk=int(args.topk),
        buy_gate=float(args.buy_gate),
        sell_gate=float(args.sell_gate),
        initial_cash=float(initial_cash),
        initial_positions=initial_positions,
        pro=pro,
        st_dir=data_dir / "raw" / "st",
        out_dir=position_dir,
        verbose=bool(args.verbose),
    )
    meta = {
        "update_meta": update_meta,
        "asof_date": asof_date,
        "trade_date": trade_date,
        "position_source_date": latest_position_date,
        "position_record": str((position_dir / f"{trade_date.replace('-', '')}.json").resolve()),
    }
    (work_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    # print(json.dumps({"trade": payloads[-1], "meta": meta}, ensure_ascii=False, indent=2))


def _infer_feature_shards_worker(
    shard_paths: list[str],
    checkpoint: str,
    hidden_dim: int,
    num_heads: int,
    dropout: float,
    use_seq_context: bool,
    enable_dynamic_threshold: bool,
    enable_free_branch: bool,
    infer_batch_size: int,
    score_dir: str,
    device_str: str,
    worker_id: int,
) -> str:
    device = torch.device(device_str)
    # Keep explicit args only as fallback for legacy checkpoints without model_hparams.
    model = build_model(
        device=device,
        checkpoint=checkpoint,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        use_seq_context=use_seq_context,
        enable_dynamic_threshold=enable_dynamic_threshold,
        enable_free_branch=enable_free_branch,
    )
    model.eval()

    df = infer_from_feature_shards(
        model=model,
        device=device,
        shard_paths=[Path(p) for p in shard_paths],
        infer_batch_size=max(1, int(infer_batch_size)),
    )
    out_path = Path(score_dir) / f"group_{worker_id:04d}.parquet"
    df.to_parquet(out_path, index=False)
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trader tool entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    update_daily = subparsers.add_parser("update-daily", help="Update raw daily market data cache under ./data")
    update_daily.add_argument("--data-dir", default="data")
    update_daily.add_argument("--lookback-trading-days", type=int, default=150)
    update_daily.add_argument("--sleep", type=float, default=0.0)
    update_daily.add_argument("--verbose", action="store_true")
    update_daily.add_argument("--no-refresh-latest", action="store_true")

    trade = subparsers.add_parser("trade", help="Update daily data, infer scores on CPU, and execute one trading step")
    trade.add_argument("--data-dir", default="data")
    trade.add_argument("--position-dir", default="data/position")
    trade.add_argument("--checkpoint", required=True)
    trade.add_argument("--topk", type=int, default=7)
    trade.add_argument("--buy-gate", type=float, default=0.75)
    trade.add_argument("--sell-gate", type=float, default=0.7)
    trade.add_argument("--lookback-trading-days", type=int, default=150)
    trade.add_argument("--sleep", type=float, default=0.0)
    trade.add_argument("--verbose", action="store_true")
    trade.add_argument("--no-refresh-latest", action="store_true")
    trade.add_argument("--no-update", action="store_true")
    trade.add_argument("--hidden-dim", type=int, default=256)
    trade.add_argument("--num-heads", type=int, default=8)
    trade.add_argument("--dropout", type=float, default=0.1)
    trade.add_argument("--use-seq-context", action=argparse.BooleanOptionalAction, default=True)
    trade.add_argument("--enable-dynamic-threshold", action=argparse.BooleanOptionalAction, default=True)
    trade.add_argument("--enable-free-branch", action=argparse.BooleanOptionalAction, default=True)
    trade.add_argument("--device", default="cuda", help="inference device, e.g. cuda, cuda:0, cpu")
    trade.add_argument("--infer-batch-size", type=int, default=8192)
    trade.add_argument("--num-workers", type=int, default=8)
    trade.add_argument("--infer-workers", type=int, default=1)
    trade.add_argument("--init", type=float, default=None, help="initial cash; when set, ignore latest position record")

    args = parser.parse_args()

    if args.command == "update-daily":
        meta = update_daily_market_data(
            DailyUpdateConfig(
                data_dir=Path(args.data_dir),
                lookback_trading_days=max(1, int(args.lookback_trading_days)),
                request_sleep_seconds=max(0.0, float(args.sleep)),
                refresh_latest=not bool(args.no_refresh_latest),
                verbose=bool(args.verbose),
            )
        )
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return

    if args.command == "trade":
        _run_trade(args)
        return

    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
