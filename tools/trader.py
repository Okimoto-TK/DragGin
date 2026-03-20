from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import torch
import tushare as ts

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import DailyUpdateConfig, update_daily_market_data
from src.infer import (
    Position,
    build_model,
    infer_from_feature_shard,
    load_daily_map,
    load_latest_position_state,
    merge_score_shards,
    read_calendar_dates,
    resolve_codes,
    run_trade_simulation,
)
from tools.build_backtest_batches import _build_feature_shard
from tools.preprocess_data import preprocess


def _symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        if src.is_file():
            shutil.copy2(src, dst)
        else:
            raise


def _build_processed_dataset_from_update_daily(data_dir: Path, processed_dir: Path, num_workers: int) -> Path:
    raw_dir = data_dir / "raw"
    daily_dir = raw_dir / "daily"
    moneyflow_dir = raw_dir / "moneyflow"
    min5_dir = raw_dir / "5min"
    if not daily_dir.exists() or not moneyflow_dir.exists() or not min5_dir.exists():
        raise FileNotFoundError(f"update-daily raw outputs are incomplete under {raw_dir}")

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="trader_raw_flat_", dir=str(data_dir)) as tmp_dir:
        flat_dir = Path(tmp_dir)
        for path in sorted(daily_dir.glob("*.parquet")):
            _symlink_or_copy(path, flat_dir / path.name)
        for path in sorted(moneyflow_dir.glob("*.parquet")):
            _symlink_or_copy(path, flat_dir / f"{path.stem}_mf.parquet")
        for code_dir in sorted(min5_dir.iterdir()):
            if not code_dir.is_dir():
                continue
            for path in sorted(code_dir.glob("*.csv")):
                _symlink_or_copy(path, flat_dir / f"{code_dir.name}_{path.name}")
        preprocess(flat_dir, processed_dir, max_workers=max(1, int(num_workers)))

    return processed_dir



def _infer_feature_shard_cpu_worker(
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
) -> str:
    device = torch.device("cpu")
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
    df = infer_from_feature_shard(model, device, Path(shard_path), max(1, int(infer_batch_size)))
    out_path = Path(score_dir) / f"{Path(shard_path).stem}.parquet"
    df.to_parquet(out_path, index=False)
    return str(out_path)



def _run_trade(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    position_dir = Path(args.position_dir)
    position_dir.mkdir(parents=True, exist_ok=True)

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

    _build_processed_dataset_from_update_daily(data_dir, processed_dir, max(1, int(args.num_workers)))
    calendar_dates = read_calendar_dates(processed_dir)
    if len(calendar_dates) < 2:
        raise ValueError("trade requires at least 2 processed trading dates")
    asof_date = calendar_dates[-2]
    trade_date = calendar_dates[-1]

    codes = resolve_codes(processed_dir)
    if not codes:
        raise ValueError("no stock folders with required parquet files found after preprocessing")

    feature_jobs = [(str(processed_dir), code, [asof_date], str(feature_dir)) for code in codes]
    if int(args.num_workers) <= 1:
        for job in feature_jobs:
            _build_feature_shard(*job)
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.num_workers))) as ex:
            futures = [ex.submit(_build_feature_shard, *job) for job in feature_jobs]
            for fut in as_completed(futures):
                fut.result()

    feature_paths = sorted(feature_dir.glob("*.npz"))
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
    if infer_workers <= 1:
        for shard_path in feature_paths:
            _infer_feature_shard_cpu_worker(str(shard_path), **worker_args)
    else:
        with ProcessPoolExecutor(max_workers=infer_workers) as ex:
            futures = [ex.submit(_infer_feature_shard_cpu_worker, str(shard_path), **worker_args) for shard_path in feature_paths]
            for fut in as_completed(futures):
                fut.result()

    score_df = merge_score_shards(score_dir)
    score_by_date = {asof_date: [(str(row["code"]), float(row["yhat"])) for _, row in score_df.iterrows() if pd.notna(row["yhat"])]}

    daily_cache = {code: load_daily_map(processed_dir, code) for code in codes}
    if args.init is None:
        latest_position_date, initial_cash, initial_positions = load_latest_position_state(position_dir, daily_cache)
    else:
        latest_position_date, initial_cash, initial_positions = None, float(args.init), {}

    token = str(os.environ.get("TUSHARE_TOKEN", "")).strip()
    pro = ts.pro_api(token) if token else None
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
    print(json.dumps({"trade": payloads[-1], "meta": meta}, ensure_ascii=False, indent=2))



def main() -> None:
    parser = argparse.ArgumentParser(description="Trader tool entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    update_daily = subparsers.add_parser("update-daily", help="Update raw daily market data cache under ./data")
    update_daily.add_argument("--data-dir", default="data")
    update_daily.add_argument("--lookback-trading-days", type=int, default=120)
    update_daily.add_argument("--sleep", type=float, default=0.0)
    update_daily.add_argument("--verbose", action="store_true")
    update_daily.add_argument("--no-refresh-latest", action="store_true")

    trade = subparsers.add_parser("trade", help="Update daily data, infer scores on CPU, and execute one trading step")
    trade.add_argument("--data-dir", default="data")
    trade.add_argument("--position-dir", default="data/position")
    trade.add_argument("--checkpoint", required=True)
    trade.add_argument("--topk", type=int, default=20)
    trade.add_argument("--buy-gate", type=float, default=1.0)
    trade.add_argument("--sell-gate", type=float, default=0.5)
    trade.add_argument("--lookback-trading-days", type=int, default=120)
    trade.add_argument("--sleep", type=float, default=0.0)
    trade.add_argument("--verbose", action="store_true")
    trade.add_argument("--no-refresh-latest", action="store_true")
    trade.add_argument("--hidden-dim", type=int, default=320)
    trade.add_argument("--num-heads", type=int, default=8)
    trade.add_argument("--dropout", type=float, default=0.1)
    trade.add_argument("--use-seq-context", action=argparse.BooleanOptionalAction, default=True)
    trade.add_argument("--enable-dynamic-threshold", action=argparse.BooleanOptionalAction, default=True)
    trade.add_argument("--enable-free-branch", action=argparse.BooleanOptionalAction, default=True)
    trade.add_argument("--infer-batch-size", type=int, default=512)
    trade.add_argument("--num-workers", type=int, default=1)
    trade.add_argument("--infer-workers", type=int, default=0)
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
