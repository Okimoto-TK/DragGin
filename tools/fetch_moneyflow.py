from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


MONEYFLOW_FIELDS = [
    "ts_code",
    "trade_date",
    "buy_sm_vol",
    "buy_sm_amount",
    "sell_sm_vol",
    "sell_sm_amount",
    "buy_md_vol",
    "buy_md_amount",
    "sell_md_vol",
    "sell_md_amount",
    "buy_lg_vol",
    "buy_lg_amount",
    "sell_lg_vol",
    "sell_lg_amount",
    "buy_elg_vol",
    "buy_elg_amount",
    "sell_elg_vol",
    "sell_elg_amount",
    "net_mf_vol",
    "net_mf_amount",
]

MAX_RETRIES = 10


def _get_pro_client(token: str):
    import tushare as ts

    ts.set_token(token)
    return ts.pro_api(token)


def _iter_trade_dates(pro, start_date: str, end_date: str) -> list[str]:
    cal = pro.trade_cal(
        exchange="",
        start_date=start_date,
        end_date=end_date,
        is_open="1",
        fields="cal_date,is_open",
    )
    if cal is None or cal.empty or "cal_date" not in cal.columns:
        return []
    out = cal["cal_date"].astype(str).sort_values().drop_duplicates().tolist()
    return out


def _fetch_moneyflow_by_date(pro, trade_date: str, sleep_seconds: float = 0.0, max_retries: int = MAX_RETRIES) -> pd.DataFrame:
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            parts: list[pd.DataFrame] = []
            offset = 0
            limit = 6000

            while True:
                df = pro.moneyflow(
                    trade_date=trade_date,
                    offset=offset,
                    limit=limit,
                    fields=",".join(MONEYFLOW_FIELDS),
                )
                if df is None or df.empty:
                    break
                parts.append(df)
                if len(df) < limit:
                    break
                offset += limit
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

            if not parts:
                raise RuntimeError(f"Empty moneyflow response for trade_date={trade_date} on attempt {attempt}")

            out = pd.concat(parts, ignore_index=True)
            out = out.reindex(columns=MONEYFLOW_FIELDS)
            out = out.drop_duplicates(subset=["ts_code", "trade_date"], keep="last").sort_values(["ts_code", "trade_date"])
            if out.empty:
                raise RuntimeError(f"Empty moneyflow dataframe after concat for trade_date={trade_date} on attempt {attempt}")
            return out.reset_index(drop=True)
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to fetch moneyflow for trade_date={trade_date} after {max_retries} retries") from last_error


def _fetch_and_write_single_date(
    token: str,
    out_dir: Path,
    trade_date: str,
    sleep_seconds: float = 0.0,
    max_retries: int = MAX_RETRIES,
) -> None:
    pro = _get_pro_client(token)
    out_file = out_dir / f"{trade_date}_mf.parquet"
    df = _fetch_moneyflow_by_date(
        pro=pro,
        trade_date=trade_date,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
    )
    df.to_parquet(out_file, index=False)


def fetch_moneyflow_range(start_date: str, end_date: str, out_dir: Path, sleep_seconds: float = 0.0, max_workers: int = 4) -> None:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required")

    out_dir.mkdir(parents=True, exist_ok=True)

    pro = _get_pro_client(token)
    trade_dates = _iter_trade_dates(pro=pro, start_date=start_date, end_date=end_date)
    if not trade_dates:
        raise RuntimeError("No open trade dates found from trade_cal in the requested range")

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = [
            executor.submit(_fetch_and_write_single_date, token, out_dir, trade_date, sleep_seconds, MAX_RETRIES)
            for trade_date in trade_dates
        ]
        future_set = set(futures)

        with tqdm(total=len(futures), desc="Phase 1: Fetching Moneyflow chunks") as pbar:
            for future in as_completed(futures):
                pbar.update(1)
                exc = future.exception()
                if exc is not None:
                    for pending in future_set:
                        if pending is not future:
                            pending.cancel()
                    raise exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Tushare moneyflow data and save as YYYYmmdd_mf.parquet files")
    parser.add_argument("--st", required=True, help="Start date in YYYYmmdd")
    parser.add_argument("--et", required=True, help="End date in YYYYmmdd")
    parser.add_argument("--out-dir", required=True, help="Output directory path")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between paginated requests")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum worker threads for fetching dates")
    args = parser.parse_args()

    fetch_moneyflow_range(
        start_date=args.st,
        end_date=args.et,
        out_dir=Path(args.out_dir),
        sleep_seconds=max(0.0, args.sleep),
        max_workers=max(1, args.max_workers),
    )


if __name__ == "__main__":
    main()
