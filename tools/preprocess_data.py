from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


def _normalize_trade_date(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    missing = dt.isna()
    if missing.any():
        dt2 = pd.to_datetime(s[missing], errors="coerce")
        dt.loc[missing] = dt2
    return dt


def _detect_csv_sep(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(4096)
        if not sample:
            return ","
        try:
            return csv.Sniffer().sniff(sample, delimiters=[",", "\t", "|", ";"]).delimiter
        except Exception:
            counts = {d: sample.count(d) for d in [",", "\t", "|", ";"]}
            return max(counts, key=counts.get) if max(counts.values()) > 0 else ","
    except Exception:
        return ","


def _time_series_to_minutes(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.replace("：", ":", regex=False)
    has_date = s.str.contains(r"[/-]", regex=True, na=False)
    minutes = pd.Series(pd.NA, index=s.index, dtype="Int32")

    if has_date.any():
        dt = pd.to_datetime(s[has_date], errors="coerce")
        mins = (dt.dt.hour.astype("Int32") * 60 + dt.dt.minute.astype("Int32")).astype("Int32")
        minutes.loc[has_date] = mins

    remain = minutes.isna()
    if remain.any():
        ss = s[remain]
        has_colon = ss.str.contains(":", na=False)
        if has_colon.any():
            parts = ss[has_colon].str.split(":", n=2, expand=True)
            hh = pd.to_numeric(parts[0], errors="coerce")
            mm = pd.to_numeric(parts[1], errors="coerce")
            ok = hh.notna() & mm.notna() & hh.between(0, 23) & mm.between(0, 59)
            mins = (hh[ok].astype("int32") * 60 + mm[ok].astype("int32")).astype("int32")
            minutes.loc[mins.index] = mins.astype("Int32")

        remain2 = minutes.isna() & remain
        if remain2.any():
            ss2 = s[remain2]
            digits = ss2.str.fullmatch(r"\d{3,4}", na=False)
            if digits.any():
                z = ss2[digits].str.zfill(4)
                hh = pd.to_numeric(z.str.slice(0, 2), errors="coerce")
                mm = pd.to_numeric(z.str.slice(2, 4), errors="coerce")
                ok = hh.notna() & mm.notna() & hh.between(0, 23) & mm.between(0, 59)
                mins = (hh[ok].astype("int32") * 60 + mm[ok].astype("int32")).astype("int32")
                minutes.loc[mins.index] = mins.astype("Int32")

    return minutes


def _minutes_to_hhmm(minutes: pd.Series) -> pd.Series:
    m = pd.to_numeric(minutes, errors="coerce")
    hh = (m // 60).astype("Int32")
    mm = (m % 60).astype("Int32")
    out = hh.astype("string").str.zfill(2) + ":" + mm.astype("string").str.zfill(2)
    return out


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
    if code_col is None or vol_col is None or not required_price.issubset(df.columns):
        return pd.DataFrame()

    dt = None
    if "trade_time" in df.columns:
        dt = pd.to_datetime(df["trade_time"], errors="coerce")

    if dt is not None and dt.notna().any():
        trade_date = dt.dt.normalize()
        minutes = (dt.dt.hour.astype("Int32") * 60 + dt.dt.minute.astype("Int32")).astype("Int32")

        missing = dt.isna()
        if missing.any() and date_col is not None and time_col is not None:
            trade_date_fallback = _normalize_trade_date(df.loc[missing, date_col])
            minutes_fallback = _time_series_to_minutes(df.loc[missing, time_col])
            trade_date.loc[missing] = trade_date_fallback
            minutes.loc[missing] = minutes_fallback
    else:
        if date_col is None or time_col is None:
            return pd.DataFrame()
        trade_date = _normalize_trade_date(df[date_col])
        minutes = _time_series_to_minutes(df[time_col])

    out = pd.DataFrame(
        {
            "code": df[code_col].astype(str),
            "trade_date": trade_date,
            "minutes": minutes,
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df[vol_col], errors="coerce"),
        }
    )

    out = out.dropna(subset=["code", "trade_date", "minutes"]).reset_index(drop=True)
    out["minutes"] = out["minutes"].astype("int32")
    out = out[out["minutes"] != 9 * 60 + 30].reset_index(drop=True)

    out["dt"] = out["trade_date"] + pd.to_timedelta(out["minutes"], unit="m")
    out = out.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    out["time"] = _minutes_to_hhmm(out["minutes"])
    out = out.drop(columns=["minutes"])
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
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out.dropna(subset=["code", "trade_date", "volume"])
    out = out[out["volume"] > 0]
    out = out.sort_values(["code", "trade_date"]).drop_duplicates(subset=["code", "trade_date"], keep="last")
    out = out.reset_index(drop=True)
    return out


def _split_by_code(norm: pd.DataFrame) -> dict[str, list[pd.DataFrame]]:
    out = defaultdict(list)
    if norm.empty:
        return out
    for code, group in norm.groupby("code", sort=False):
        out[str(code)].append(group)
    return out


def _process_5min_file(csv_file: Path) -> dict[str, list[pd.DataFrame]]:
    wanted = {"code", "trade_date", "date", "time", "trade_time", "open", "high", "low", "close", "volume", "vol"}
    sep = _detect_csv_sep(csv_file)

    try:
        header = pd.read_csv(csv_file, sep=sep, engine="c", nrows=0)
        usecols = [c for c in header.columns if c in wanted]
    except Exception:
        usecols = list(wanted)

    if not usecols or "code" not in usecols:
        usecols = list(wanted)

    try:
        df = pd.read_csv(csv_file, sep=sep, engine="c", usecols=usecols, low_memory=False)
    except Exception:
        df = None
        for sep2 in [",", "\t", "|", ";"]:
            try:
                header = pd.read_csv(csv_file, sep=sep2, engine="c", nrows=0)
                usecols2 = [c for c in header.columns if c in wanted]
                if not usecols2:
                    usecols2 = list(wanted)
                df = pd.read_csv(csv_file, sep=sep2, engine="c", usecols=usecols2, low_memory=False)
                break
            except Exception:
                continue
        if df is None: return {}

    if "code" not in df.columns: return {}
    norm = _normalize_5min(df)
    return _split_by_code(norm)


_DAILY_PARQUET_COLUMNS = [
    "code", "trade_date", "date", "open", "high", "low", "close", "adj_factor", "volume", "vol", "pct_chg",
]


def _process_daily_file(pq_file: Path) -> dict[str, list[pd.DataFrame]]:
    try:
        df = pd.read_parquet(pq_file, columns=_DAILY_PARQUET_COLUMNS)
    except Exception:
        try:
            df = pd.read_parquet(pq_file)
        except Exception:
            return {}
    if "code" not in df.columns: return {}
    norm = _normalize_daily(df)
    return _split_by_code(norm)


# =====================================================================
# Phase 1: 进程内部切块 (Map) - 无IPC通讯，无临时巨无霸文件
# =====================================================================
def _process_csv_chunk(chunk_files: list[Path], chunk_id: int, out_dir: Path) -> list[str]:
    per_code_5m = defaultdict(list)
    for f in chunk_files:
        out = _process_5min_file(f)
        for code, dfs in out.items():
            per_code_5m[code].extend(dfs)

    processed_codes = []
    # 写出本批次的切片小文件，写完自动释放内存
    for code, dfs in per_code_5m.items():
        if dfs:
            code_dir = out_dir / str(code)
            code_dir.mkdir(parents=True, exist_ok=True)
            m5 = pd.concat(dfs, ignore_index=True)
            m5.to_parquet(code_dir / f"5m_part_{chunk_id}.parquet", index=False)
            processed_codes.append(code)

    return processed_codes


def _process_daily_chunk(chunk_files: list[Path], chunk_id: int, out_dir: Path) -> list[str]:
    per_code_1d = defaultdict(list)
    for f in chunk_files:
        out = _process_daily_file(f)
        for code, dfs in out.items():
            per_code_1d[code].extend(dfs)

    processed_codes = []
    for code, dfs in per_code_1d.items():
        if dfs:
            code_dir = out_dir / str(code)
            code_dir.mkdir(parents=True, exist_ok=True)
            d1 = pd.concat(dfs, ignore_index=True)
            d1.to_parquet(code_dir / f"1d_part_{chunk_id}.parquet", index=False)
            processed_codes.append(code)

    return processed_codes


# =====================================================================
# Phase 2: 直接合并各股票目录内的切片小文件 (Reduce)
# =====================================================================
def _combine_and_write_code(code: str, out_dir: Path) -> set[pd.Timestamp]:
    code_dir = out_dir / str(code)
    cal_dates = set()

    valid_trade_dates: set[pd.Timestamp] | None = None

    parts_1d = list(code_dir.glob("1d_part_*.parquet"))
    if parts_1d:
        dfs = [pd.read_parquet(p) for p in parts_1d]
        if dfs:
            d1 = pd.concat(dfs, ignore_index=True)
            d1["volume"] = pd.to_numeric(d1["volume"], errors="coerce")
            d1 = d1.dropna(subset=["trade_date", "volume"])
            d1 = d1[d1["volume"] > 0]
            d1 = d1.sort_values("trade_date").drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)
            d1.to_parquet(code_dir / "daily.parquet", index=False)
            valid_trade_dates = set(d1["trade_date"].tolist())
            cal_dates.update(valid_trade_dates)
        for p in parts_1d:
            try:
                p.unlink()
            except OSError:
                pass

    parts_5m = list(code_dir.glob("5m_part_*.parquet"))
    if parts_5m:
        dfs = [pd.read_parquet(p) for p in parts_5m]
        if dfs:
            m5 = pd.concat(dfs, ignore_index=True)
            if valid_trade_dates is not None:
                m5 = m5[m5["trade_date"].isin(valid_trade_dates)]
            m5 = m5.sort_values("dt").reset_index(drop=True)
            m5.to_parquet(code_dir / "5min.parquet", index=False)
        for p in parts_5m:
            try:
                p.unlink()
            except OSError:
                pass

    return cal_dates


def _fetch_namechange(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        return pd.DataFrame()
    try:
        import tushare as ts

        pro = ts.pro_api(token)
        df = pro.namechange(
            ts_code="",
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            limit="",
            offset="",
            fields=["ts_code", "name", "start_date", "end_date"],
        )
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _build_st_breakpoints(namechange: pd.DataFrame) -> pd.DataFrame:
    if namechange.empty:
        return pd.DataFrame(columns=["code", "break_date"])

    df = namechange.copy().rename(columns={"ts_code": "code"})
    if "code" not in df.columns or "name" not in df.columns or "start_date" not in df.columns:
        return pd.DataFrame(columns=["code", "break_date"])

    df["code"] = df["code"].astype(str)
    df["name"] = df["name"].astype(str).str.strip()
    df["start_date"] = _normalize_trade_date(df["start_date"])
    df = df.dropna(subset=["code", "start_date"]).sort_values(["code", "start_date"]).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["code", "break_date"])

    df["is_st"] = df["name"].str.startswith("ST", na=False) | df["name"].str.startswith("*ST", na=False)
    df["prev_is_st"] = df.groupby("code", sort=False)["is_st"].shift(1)
    changed = df[(df["prev_is_st"].notna()) & (df["is_st"] != df["prev_is_st"])].copy()
    if changed.empty:
        return pd.DataFrame(columns=["code", "break_date"])

    out = changed[["code", "start_date"]].rename(columns={"start_date": "break_date"})
    out = out.drop_duplicates(subset=["code", "break_date"]).sort_values(["code", "break_date"]).reset_index(drop=True)
    return out


def _write_breakpoint_files(breakpoints: pd.DataFrame, out_dir: Path) -> None:
    if breakpoints.empty:
        return
    for code, g in breakpoints.groupby("code", sort=False):
        code_dir = out_dir / str(code)
        code_dir.mkdir(parents=True, exist_ok=True)
        g[["break_date"]].reset_index(drop=True).to_parquet(code_dir / "breakpoints.parquet", index=False)


def preprocess(raw_dir: Path, out_dir: Path, max_workers: int = 4) -> None:
    csv_files = sorted(raw_dir.glob("*.csv"))
    pq_files = sorted(raw_dir.glob("*.parquet"))

    all_codes: set[str] = set()

    # 1. Map 阶段：每次处理 100 天，原地合并为本地切片碎片，极度省内存且避开了跨进程数据传输
    chunk_size_5m = 100
    csv_chunks = [csv_files[i:i + chunk_size_5m] for i in range(0, len(csv_files), chunk_size_5m)]
    if csv_chunks:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_csv_chunk, chunk, i, out_dir) for i, chunk in enumerate(csv_chunks)]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Phase 1: Parsing 5min CSV chunks"):
                all_codes.update(future.result())

    chunk_size_1d = 200
    pq_chunks = [pq_files[i:i + chunk_size_1d] for i in range(0, len(pq_files), chunk_size_1d)]
    if pq_chunks:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_daily_chunk, chunk, i, out_dir) for i, chunk in enumerate(pq_chunks)]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Phase 1: Parsing Daily chunks"):
                all_codes.update(future.result())

    # 2. Reduce 阶段：每只股票只要读取自身目录下的数十个切片即可 (抛弃全局跨文件搜寻，速度起飞)
    codes_list = sorted(list(all_codes))
    all_calendar_dates: set[pd.Timestamp] = set()

    if codes_list:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_combine_and_write_code, code, out_dir) for code in codes_list]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Phase 2: Writing final parquets"):
                all_calendar_dates.update(future.result())

    if all_calendar_dates:
        calendar = pd.DataFrame({"trade_date": sorted(all_calendar_dates)})
        calendar.to_parquet(out_dir / "calendar.parquet", index=False)

        min_trade_date = pd.to_datetime(calendar["trade_date"].min())
        max_trade_date = pd.to_datetime(calendar["trade_date"].max())
        namechange = _fetch_namechange(min_trade_date, max_trade_date)
        breakpoints = _build_st_breakpoints(namechange)
        _write_breakpoint_files(breakpoints, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", default="./raw_data")
    parser.add_argument("--out-dir", default="../data")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    preprocess(
        Path(args.raw_data_dir),
        Path(args.out_dir),
        max_workers=max(1, args.max_workers),
    )