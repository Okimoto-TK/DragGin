from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd

MAX_RETRIES = 5
STK_LIMIT_FIELDS = "ts_code,trade_date,pre_close,up_limit,down_limit"


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
    return cal["cal_date"].astype(str).sort_values().drop_duplicates().tolist()


def _fetch_stk_limit_by_date(pro, trade_date: str, sleep_seconds: float = 0.0) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = pro.stk_limit(trade_date=trade_date, fields=STK_LIMIT_FIELDS)
            if df is None or df.empty:
                raise RuntimeError(f"Empty stk_limit response for trade_date={trade_date}")
            out = (
                df.reindex(columns=STK_LIMIT_FIELDS.split(","))
                .drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
                .sort_values(["ts_code", "trade_date"])
                .reset_index(drop=True)
            )
            if out.empty:
                raise RuntimeError(f"Empty stk_limit dataframe after cleanup for trade_date={trade_date}")
            return out
        except Exception as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to fetch stk_limit for trade_date={trade_date} after {MAX_RETRIES} retries") from last_error


def fetch_stk_limit_range(start_date: str, end_date: str, out_dir: Path, sleep_seconds: float = 0.0) -> None:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required")

    out_dir.mkdir(parents=True, exist_ok=True)

    pro = _get_pro_client(token)
    trade_dates = _iter_trade_dates(pro=pro, start_date=start_date, end_date=end_date)
    if not trade_dates:
        raise RuntimeError("No open trade dates found from trade_cal in the requested range")

    for trade_date in trade_dates:
        df = _fetch_stk_limit_by_date(pro=pro, trade_date=trade_date, sleep_seconds=sleep_seconds)
        out_file = out_dir / f"{trade_date}_stk_limit.parquet"
        df.to_parquet(out_file, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Tushare stk_limit data for each open day and save as YYYYmmdd_stk_limit.parquet",
    )
    parser.add_argument("--st", required=True, help="Start date in YYYYmmdd")
    parser.add_argument("--et", required=True, help="End date in YYYYmmdd")
    parser.add_argument("--out-dir", required=True, help="Output directory path")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between retry attempts")
    args = parser.parse_args()

    fetch_stk_limit_range(
        start_date=args.st,
        end_date=args.et,
        out_dir=Path(args.out_dir),
        sleep_seconds=max(0.0, args.sleep),
    )


if __name__ == "__main__":
    main()
