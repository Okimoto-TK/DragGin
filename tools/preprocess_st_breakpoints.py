from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tools.preprocess_data import _build_st_breakpoints, _fetch_namechange, _normalize_trade_date, _write_breakpoint_files


def _find_date_range_from_calendar(out_dir: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    calendar_file = out_dir / "calendar.parquet"
    if not calendar_file.exists():
        return None

    try:
        cal = pd.read_parquet(calendar_file, columns=["trade_date"])
    except Exception:
        try:
            cal = pd.read_parquet(calendar_file)
        except Exception:
            return None

    if "trade_date" not in cal.columns or cal.empty:
        return None

    trade_date = _normalize_trade_date(cal["trade_date"])
    trade_date = trade_date.dropna()
    if trade_date.empty:
        return None
    return pd.to_datetime(trade_date.min()), pd.to_datetime(trade_date.max())


def _find_date_range_from_daily(out_dir: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    min_date = None
    max_date = None

    for daily_file in out_dir.glob("*/daily.parquet"):
        try:
            df = pd.read_parquet(daily_file, columns=["trade_date"])
        except Exception:
            try:
                df = pd.read_parquet(daily_file)
            except Exception:
                continue

        if "trade_date" not in df.columns or df.empty:
            continue

        td = _normalize_trade_date(df["trade_date"]).dropna()
        if td.empty:
            continue

        cur_min = pd.to_datetime(td.min())
        cur_max = pd.to_datetime(td.max())
        min_date = cur_min if min_date is None else min(min_date, cur_min)
        max_date = cur_max if max_date is None else max(max_date, cur_max)

    if min_date is None or max_date is None:
        return None
    return min_date, max_date


def build_st_breakpoints(out_dir: Path, start_date: str | None = None, end_date: str | None = None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    if start_date and end_date:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
    else:
        date_range = _find_date_range_from_calendar(out_dir)
        if date_range is None:
            date_range = _find_date_range_from_daily(out_dir)
        if date_range is None:
            print("[ST] No usable calendar.parquet or */daily.parquet found, skipping.")
            return 0
        start, end = date_range

    namechange = _fetch_namechange(start, end)
    breakpoints = _build_st_breakpoints(namechange)
    _write_breakpoint_files(breakpoints, out_dir)

    print(f"[ST] Date range: {start.date()} ~ {end.date()}")
    print(f"[ST] Generated breakpoints rows: {len(breakpoints)}")
    print(f"[ST] Affected codes: {breakpoints['code'].nunique() if not breakpoints.empty else 0}")
    return len(breakpoints)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="../data")
    parser.add_argument("--start-date", default=None, help="Optional, YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="Optional, YYYY-MM-DD")
    args = parser.parse_args()

    build_st_breakpoints(
        Path(args.out_dir),
        start_date=args.start_date,
        end_date=args.end_date,
    )
