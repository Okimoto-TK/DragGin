from __future__ import annotations

import argparse
import os
import time
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


def _get_pro_client(token: str):
    import tushare as ts

    ts.set_token(token)
    return ts.pro_api(token)


def _iter_trade_dates(start_date: str, end_date: str) -> list[str]:
    dates = pd.date_range(
        start=pd.to_datetime(start_date, format="%Y%m%d"),
        end=pd.to_datetime(end_date, format="%Y%m%d"),
        freq="D",
    )
    return [d.strftime("%Y%m%d") for d in dates]


def _fetch_moneyflow_by_date(pro, trade_date: str, sleep_seconds: float = 0.0) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    offset = 0
    limit = 6000

    while True:
        df = pro.moneyflow(trade_date=trade_date, offset=offset, limit=limit, fields=",".join(MONEYFLOW_FIELDS))
        if df is None or df.empty:
            break
        parts.append(df)
        if len(df) < limit:
            break
        offset += limit
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if not parts:
        return pd.DataFrame(columns=MONEYFLOW_FIELDS)

    out = pd.concat(parts, ignore_index=True)
    out = out.reindex(columns=MONEYFLOW_FIELDS)
    out = out.drop_duplicates(subset=["ts_code", "trade_date"], keep="last").sort_values(["ts_code", "trade_date"])
    return out.reset_index(drop=True)


def fetch_moneyflow_range(start_date: str, end_date: str, out_dir: Path, sleep_seconds: float = 0.0) -> None:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required")

    out_dir.mkdir(parents=True, exist_ok=True)
    pro = _get_pro_client(token)

    trade_dates = _iter_trade_dates(start_date, end_date)
    for trade_date in tqdm(trade_dates, total=len(trade_dates), desc="Fetching Moneyflow daily parquets"):
        out_file = out_dir / f"{trade_date}_mf.parquet"
        df = _fetch_moneyflow_by_date(pro=pro, trade_date=trade_date, sleep_seconds=sleep_seconds)
        df.to_parquet(out_file, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Tushare moneyflow data and save as YYYYmmdd_mf.parquet files")
    parser.add_argument("--st", required=True, help="Start date in YYYYmmdd")
    parser.add_argument("--et", required=True, help="End date in YYYYmmdd")
    parser.add_argument("--out-dir", required=True, help="Output directory path")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between paginated requests")
    args = parser.parse_args()

    fetch_moneyflow_range(
        start_date=args.st,
        end_date=args.et,
        out_dir=Path(args.out_dir),
        sleep_seconds=max(0.0, args.sleep),
    )


if __name__ == "__main__":
    main()
