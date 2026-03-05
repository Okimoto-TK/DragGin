from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATE_PATTERN = r"(\d{4}-\d{2}-\d{2}|\d{8})"


def _normalize_trade_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def build_5min(raw_dir: Path, code: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for csv_file in raw_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        if "code" in df.columns:
            df = df[df["code"] == code]
        elif code not in csv_file.stem:
            continue
        if df.empty:
            continue
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    required = {"trade_date", "time", "open", "high", "low", "close", "volume"}
    if not required.issubset(out.columns):
        return pd.DataFrame()

    out = out[["trade_date", "time", "open", "high", "low", "close", "volume"]].copy()
    out["trade_date"] = _normalize_trade_date(out["trade_date"])
    out = out.dropna(subset=["trade_date", "time"]).reset_index(drop=True)
    out["dt"] = pd.to_datetime(out["trade_date"].astype(str) + " " + out["time"].astype(str), errors="coerce")
    out = out.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    return out


def build_daily(raw_dir: Path, code: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for pq_file in raw_dir.glob("*.parquet"):
        df = pd.read_parquet(pq_file)
        if "code" in df.columns:
            df = df[df["code"] == code]
        elif code not in pq_file.stem:
            continue
        if df.empty:
            continue
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    required = {"trade_date", "open", "high", "low", "close", "volume", "adj_factor"}
    if not required.issubset(out.columns):
        return pd.DataFrame()

    out = out[["trade_date", "open", "high", "low", "close", "volume", "adj_factor"]].copy()
    out["trade_date"] = _normalize_trade_date(out["trade_date"])
    out = out.dropna(subset=["trade_date"]).drop_duplicates(subset=["trade_date"], keep="last")
    out = out.sort_values("trade_date").reset_index(drop=True)
    return out


def preprocess(raw_dir: Path, out_dir: Path) -> None:
    codes: set[str] = set()
    for csv_file in raw_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, usecols=["code"])
        except Exception:
            continue
        if "code" in df.columns:
            codes.update(df["code"].dropna().astype(str).unique().tolist())
        else:
            parts = csv_file.stem.split("_")
            if parts:
                codes.add(parts[0])

    for pq_file in raw_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(pq_file, columns=["code"])
        except Exception:
            df = pd.DataFrame()
        if "code" in df.columns and not df.empty:
            codes.update(df["code"].dropna().astype(str).unique().tolist())
        else:
            parts = pq_file.stem.split("_")
            if parts:
                codes.add(parts[0])

    for code in sorted(codes):
        m5 = build_5min(raw_dir, code)
        d1 = build_daily(raw_dir, code)
        if m5.empty and d1.empty:
            continue
        code_dir = out_dir / code
        code_dir.mkdir(parents=True, exist_ok=True)
        if not m5.empty:
            m5.to_parquet(code_dir / "5min.parquet", index=False)
        if not d1.empty:
            d1.to_parquet(code_dir / "daily.parquet", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", default="./data")
    parser.add_argument("--out-dir", default="./data")
    args = parser.parse_args()
    preprocess(Path(args.raw_data_dir), Path(args.out_dir))
