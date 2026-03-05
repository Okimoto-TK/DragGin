from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


def _normalize_trade_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def _normalize_5min(df: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "time", "open", "high", "low", "close", "volume", "code"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    out = df[["code", "trade_date", "time", "open", "high", "low", "close", "volume"]].copy()
    out["code"] = out["code"].astype(str)
    out["trade_date"] = _normalize_trade_date(out["trade_date"])
    out = out.dropna(subset=["code", "trade_date", "time"]).reset_index(drop=True)
    out["dt"] = pd.to_datetime(out["trade_date"].astype(str) + " " + out["time"].astype(str), errors="coerce")
    out = out.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    return out


def _normalize_daily(df: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "open", "high", "low", "close", "volume", "adj_factor", "code"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    out = df[["code", "trade_date", "open", "high", "low", "close", "volume", "adj_factor"]].copy()
    out["code"] = out["code"].astype(str)
    out["trade_date"] = _normalize_trade_date(out["trade_date"])
    out = out.dropna(subset=["code", "trade_date"])
    out = out.sort_values(["code", "trade_date"]).drop_duplicates(subset=["code", "trade_date"], keep="last")
    out = out.reset_index(drop=True)
    return out


def preprocess(raw_dir: Path, out_dir: Path) -> None:
    per_code_5min: dict[str, list[pd.DataFrame]] = defaultdict(list)
    per_code_daily: dict[str, list[pd.DataFrame]] = defaultdict(list)

    for csv_file in raw_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            continue
        if "code" not in df.columns:
            continue
        norm = _normalize_5min(df)
        if norm.empty:
            continue
        for code, g in norm.groupby("code", sort=False):
            per_code_5min[code].append(g.drop(columns=["code"]).reset_index(drop=True))

    for pq_file in raw_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(pq_file)
        except Exception:
            continue
        if "code" not in df.columns:
            continue
        norm = _normalize_daily(df)
        if norm.empty:
            continue
        for code, g in norm.groupby("code", sort=False):
            per_code_daily[code].append(g.drop(columns=["code"]).reset_index(drop=True))

    all_codes = sorted(set(per_code_5min) | set(per_code_daily))
    all_calendar_dates: set = set()

    for code in all_codes:
        code_dir = out_dir / code
        code_dir.mkdir(parents=True, exist_ok=True)

        chunks_5m = per_code_5min.get(code, [])
        if chunks_5m:
            m5 = pd.concat(chunks_5m, ignore_index=True)
            m5 = m5.sort_values("dt").reset_index(drop=True)
            m5.to_parquet(code_dir / "5min.parquet", index=False)

        chunks_daily = per_code_daily.get(code, [])
        if chunks_daily:
            d1 = pd.concat(chunks_daily, ignore_index=True)
            d1 = d1.sort_values("trade_date").drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)
            d1.to_parquet(code_dir / "daily.parquet", index=False)
            all_calendar_dates.update(d1["trade_date"].tolist())

    if all_calendar_dates:
        calendar = pd.DataFrame({"trade_date": sorted(all_calendar_dates)})
        calendar.to_parquet(out_dir / "calendar.parquet", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", default="./data")
    parser.add_argument("--out-dir", default="./data")
    args = parser.parse_args()
    preprocess(Path(args.raw_data_dir), Path(args.out_dir))
