from __future__ import annotations

import argparse
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


def _normalize_trade_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%Y%m%d", errors="coerce").dt.date


def _pick_column(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _normalize_5min(df: pd.DataFrame) -> pd.DataFrame:
    code_col = _pick_column(df, ["code"])
    date_col = _pick_column(df, ["trade_date", "date"])
    time_col = _pick_column(df, ["time", "trade_time"])
    vol_col = _pick_column(df, ["volume", "vol"])
    required_price = {"open", "high", "low", "close"}
    if code_col is None or date_col is None or time_col is None or vol_col is None or not required_price.issubset(df.columns):
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "code": df[code_col],
            "trade_date": df[date_col],
            "time": df[time_col],
            "open": df["open"],
            "high": df["high"],
            "low": df["low"],
            "close": df["close"],
            "volume": df[vol_col],
        }
    )
    out["code"] = out["code"].astype(str)
    out["trade_date"] = _normalize_trade_date(out["trade_date"])
    out["time"] = out["time"].astype(str)
    out["time"] = out["time"].str.extract(r"(\d{2}:\d{2})", expand=False).fillna(out["time"])
    out = out.dropna(subset=["code", "trade_date", "time"]).reset_index(drop=True)
    # 09:30 bar includes opening auction data; exclude it from intraday bars.
    out = out[out["time"] != "09:30"].reset_index(drop=True)
    out["dt"] = pd.to_datetime(out["trade_date"].astype(str) + " " + out["time"].astype(str), errors="coerce")
    out = out.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    return out


def _normalize_daily(df: pd.DataFrame) -> pd.DataFrame:
    code_col = _pick_column(df, ["code"])
    date_col = _pick_column(df, ["trade_date", "date"])
    volume_col = _pick_column(df, ["volume", "vol"])
    required = {"open", "high", "low", "close", "adj_factor"}
    if code_col is None or date_col is None or not required.issubset(df.columns):
        return pd.DataFrame()

    if volume_col is not None:
        volume = pd.to_numeric(df[volume_col], errors="coerce")
    elif "pct_chg" in df.columns:
        pct = pd.to_numeric(df["pct_chg"], errors="coerce").abs().fillna(0.0)
        volume = (pct + 1.0) * 1_000_000.0
    else:
        spread = pd.to_numeric(df["high"], errors="coerce") - pd.to_numeric(df["low"], errors="coerce")
        volume = spread.abs().fillna(0.0) + 1.0

    out = pd.DataFrame(
        {
            "code": df[code_col],
            "trade_date": df[date_col],
            "open": df["open"],
            "high": df["high"],
            "low": df["low"],
            "close": df["close"],
            "volume": volume,
            "adj_factor": df["adj_factor"],
        }
    )
    out["code"] = out["code"].astype(str)
    out["trade_date"] = _normalize_trade_date(out["trade_date"])
    out = out.dropna(subset=["code", "trade_date"])
    out = out.sort_values(["code", "trade_date"]).drop_duplicates(subset=["code", "trade_date"], keep="last")
    out = out.reset_index(drop=True)
    return out


def _process_5min_file(csv_file: Path) -> dict[str, list[pd.DataFrame]]:
    try:
        df = pd.read_csv(csv_file)
    except Exception:
        return {}
    if "code" not in df.columns:
        return {}
    norm = _normalize_5min(df)
    if norm.empty:
        return {}

    return _split_by_code(norm)


def _split_by_code(norm: pd.DataFrame) -> dict[str, list[pd.DataFrame]]:
    out: dict[str, list[pd.DataFrame]] = defaultdict(list)
    codes = norm["code"].to_numpy()
    values = norm.drop(columns=["code"])

    if len(codes) == 0:
        return out

    # Fast-path for files that only contain one code.
    first_code = codes[0]
    if np.all(codes == first_code):
        out[str(first_code)].append(values.reset_index(drop=True))
        return out

    labels, uniques = pd.factorize(codes, sort=False)
    order = labels.argsort(kind="mergesort")
    sorted_labels = labels[order]
    split_at = np.flatnonzero(np.diff(sorted_labels)) + 1

    for code, idx in zip(uniques.tolist(), np.split(order, split_at)):
        out[str(code)].append(values.iloc[idx].reset_index(drop=True))
    return out


def _process_daily_file(pq_file: Path) -> dict[str, list[pd.DataFrame]]:
    try:
        df = pd.read_parquet(pq_file)
    except Exception:
        return {}
    if "code" not in df.columns:
        return {}
    norm = _normalize_daily(df)
    if norm.empty:
        return {}

    return _split_by_code(norm)


def _extend_chunks(dst: dict[str, list[pd.DataFrame]], src: dict[str, list[pd.DataFrame]]) -> None:
    for code, chunks in src.items():
        dst[code].extend(chunks)


def _fetch_namechange(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        return pd.DataFrame()

    try:
        import tushare as ts
    except ImportError:
        return pd.DataFrame()

    pro = ts.pro_api(token)
    try:
        df = pro.namechange(
            ts_code="",
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            limit="",
            offset="",
            fields=["ts_code", "name", "start_date", "end_date"],
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()
    return df


def _build_st_markers(namechange: pd.DataFrame) -> pd.DataFrame:
    if namechange.empty:
        return pd.DataFrame()

    df = namechange.copy()
    df = df.rename(columns={"ts_code": "code"})
    df["name"] = df["name"].astype(str)
    df["start_date"] = _normalize_trade_date(df["start_date"])
    df["end_date"] = _normalize_trade_date(df["end_date"])
    df = df.dropna(subset=["code", "start_date"]).reset_index(drop=True)

    normalized_name = df["name"].str.strip()
    is_star_st = normalized_name.str.startswith("*ST", na=False)
    is_st = normalized_name.str.startswith("ST", na=False) | is_star_st
    st_df = df[is_st].copy()
    if st_df.empty:
        return pd.DataFrame()

    st_df["st_type"] = np.where(is_star_st.loc[st_df.index], "*ST", "ST")
    st_df["revoke_st_date"] = st_df["end_date"]
    st_df = st_df[["code", "name", "start_date", "end_date", "st_type", "revoke_st_date"]]
    st_df = st_df.sort_values(["code", "start_date", "end_date"], na_position="last").reset_index(drop=True)
    return st_df


def _write_st_files(st_df: pd.DataFrame, out_dir: Path) -> None:
    if st_df.empty:
        return
    for code, g in st_df.groupby("code", sort=False):
        code_dir = out_dir / str(code)
        code_dir.mkdir(parents=True, exist_ok=True)
        g.reset_index(drop=True).to_parquet(code_dir / "st.parquet", index=False)


def preprocess(raw_dir: Path, out_dir: Path, max_workers: int = 4) -> None:
    per_code_5min: dict[str, list[pd.DataFrame]] = defaultdict(list)
    per_code_daily: dict[str, list[pd.DataFrame]] = defaultdict(list)

    csv_files = sorted(raw_dir.glob("*.csv"))
    pq_files = sorted(raw_dir.glob("*.parquet"))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_5min_file, csv_file) for csv_file in csv_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing 5min CSV"):
            _extend_chunks(per_code_5min, future.result())

        futures = [executor.submit(_process_daily_file, pq_file) for pq_file in pq_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing daily parquet"):
            _extend_chunks(per_code_daily, future.result())

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

        min_trade_date = pd.to_datetime(calendar["trade_date"].min())
        max_trade_date = pd.to_datetime(calendar["trade_date"].max())
        namechange = _fetch_namechange(min_trade_date, max_trade_date)
        st_df = _build_st_markers(namechange)
        _write_st_files(st_df, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", default="./data")
    parser.add_argument("--out-dir", default="./data")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    preprocess(Path(args.raw_data_dir), Path(args.out_dir), max_workers=max(1, args.max_workers))
