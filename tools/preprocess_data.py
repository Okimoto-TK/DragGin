from __future__ import annotations

import argparse
import csv
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
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
    # Keep datetime64 dtype (instead of Python date objects) for faster vectorized ops.
    return pd.to_datetime(series, format="%Y%m%d", errors="coerce")


def _detect_csv_sep(path: Path) -> str:
    # Fast delimiter detection (comma vs tab vs pipe). Read a small sample only.
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(4096)
        if not sample:
            return ","
        try:
            return csv.Sniffer().sniff(sample, delimiters=[",", "\t", "|", ";"]).delimiter
        except Exception:
            # Heuristic: choose the most frequent delimiter in the sample.
            counts = {d: sample.count(d) for d in [",", "\t", "|", ";"]}
            return max(counts, key=counts.get) if max(counts.values()) > 0 else ","
    except Exception:
        return ","


def _time_series_to_minutes(series: pd.Series) -> pd.Series:
    """Convert a variety of time representations to minutes-from-midnight (Int32 with NA)."""
    s = series.astype("string").str.strip().str.replace("：", ":", regex=False)

    # Case A: datetime-like strings (e.g., '2026/2/13 9:30' or '2026-02-13 09:30:00')
    has_date = s.str.contains(r"[/-]", regex=True, na=False)
    minutes = pd.Series(pd.NA, index=s.index, dtype="Int32")
    if has_date.any():
        dt = pd.to_datetime(s[has_date], errors="coerce")
        mins = (dt.dt.hour.astype("Int32") * 60 + dt.dt.minute.astype("Int32")).astype("Int32")
        minutes.loc[has_date] = mins

    # Case B: 'H:MM' / 'HH:MM' / 'HH:MM:SS'
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

        # Case C: compact digits '935' / '0935'
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

    # Fast path: if we have full datetime strings in trade_time, parse dt directly.
    dt = None
    if "trade_time" in df.columns:
        dt = pd.to_datetime(df["trade_time"], errors="coerce")
    elif time_col is not None and date_col is not None and time_col == "trade_time":
        dt = pd.to_datetime(df[time_col], errors="coerce")

    if dt is not None and dt.notna().any():
        trade_date = dt.dt.normalize()
        minutes = (dt.dt.hour.astype("Int32") * 60 + dt.dt.minute.astype("Int32")).astype("Int32")
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

    # 09:30 bar includes opening auction data; exclude it from intraday bars.
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
    out = out.dropna(subset=["code", "trade_date"])
    out = out.sort_values(["code", "trade_date"]).drop_duplicates(subset=["code", "trade_date"], keep="last")
    out = out.reset_index(drop=True)
    return out


def _process_5min_file(csv_file: Path) -> dict[str, list[pd.DataFrame]]:
    # NOTE: This path is extremely hot; keep it in C-engine and avoid Python regex-heavy parsing.
    wanted = {
        "code",
        "trade_date",
        "date",
        "time",
        "trade_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vol",
    }

    sep = _detect_csv_sep(csv_file)

    # Read header only to compute concrete usecols list (avoid python-side per-column lambda).
    try:
        header = pd.read_csv(csv_file, sep=sep, engine="c", nrows=0)
        usecols = [c for c in header.columns if c in wanted]
    except Exception:
        usecols = list(wanted)

    if not usecols or "code" not in usecols:
        # If header read failed, we still attempt full read; some files may have BOM/oddities.
        usecols = list(wanted)

    try:
        df = pd.read_csv(
            csv_file,
            sep=sep,
            engine="c",
            usecols=usecols,
            low_memory=False,
        )
    except Exception:
        # Fallback: try common separators explicitly.
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
        if df is None:
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


_DAILY_PARQUET_COLUMNS = [
    "code",
    "trade_date",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_factor",
    "volume",
    "vol",
    "pct_chg",
]


def _process_daily_file(pq_file: Path) -> dict[str, list[pd.DataFrame]]:
    try:
        df = pd.read_parquet(pq_file, columns=_DAILY_PARQUET_COLUMNS)
    except Exception:
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


def _write_code_files(code: str, out_dir: Path, chunks_5m: list[pd.DataFrame], chunks_daily: list[pd.DataFrame]) -> list[pd.Timestamp]:
    code_dir = out_dir / code
    code_dir.mkdir(parents=True, exist_ok=True)

    calendar_dates: list[pd.Timestamp] = []

    if chunks_5m:
        m5 = pd.concat(chunks_5m, ignore_index=True)
        m5 = m5.sort_values("dt").reset_index(drop=True)
        m5.to_parquet(code_dir / "5min.parquet", index=False)

    if chunks_daily:
        d1 = pd.concat(chunks_daily, ignore_index=True)
        d1 = d1.sort_values("trade_date").drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)
        d1.to_parquet(code_dir / "daily.parquet", index=False)
        calendar_dates = d1["trade_date"].tolist()

    return calendar_dates


class SyncTimer:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._records: list[tuple[str, float]] = []

    @contextmanager
    def section(self, name: str) -> Iterable[None]:
        if not self.enabled:
            yield
            return

        started_at = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started_at
            self._records.append((name, elapsed))
            print(f"[timer] {name}: {elapsed:.3f}s")

    def summary(self) -> None:
        if not self.enabled or not self._records:
            return

        total = sum(elapsed for _, elapsed in self._records)
        print("[timer] ==== preprocess timing summary ====")
        for name, elapsed in sorted(self._records, key=lambda x: x[1], reverse=True):
            ratio = (elapsed / total * 100.0) if total > 0 else 0.0
            print(f"[timer] {name:<28} {elapsed:>8.3f}s  ({ratio:>5.1f}%)")
        print(f"[timer] {'TOTAL':<28} {total:>8.3f}s")


def preprocess(raw_dir: Path, out_dir: Path, max_workers: int = 4, enable_timer: bool = True) -> None:
    timer = SyncTimer(enabled=enable_timer)
    per_code_5min: dict[str, list[pd.DataFrame]] = defaultdict(list)
    per_code_daily: dict[str, list[pd.DataFrame]] = defaultdict(list)

    with timer.section("discover raw files"):
        csv_files = sorted(raw_dir.glob("*.csv"))
        pq_files = sorted(raw_dir.glob("*.parquet"))

    with timer.section("process 5min CSV"):
        # 5m CSV parsing is CPU+GIL heavy in Python space (strings) and benefits from multiprocessing.
        # Using processes also isolates pandas parser state and avoids thread contention.
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_5min_file, csv_file) for csv_file in csv_files]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing 5min CSV"):
                _extend_chunks(per_code_5min, future.result())

    with timer.section("process daily parquet"):
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_daily_file, pq_file) for pq_file in pq_files]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing daily parquet"):
                _extend_chunks(per_code_daily, future.result())

    with timer.section("build code list"):
        all_codes = sorted(set(per_code_5min) | set(per_code_daily))
    all_calendar_dates: set[pd.Timestamp] = set()

    with timer.section("write per-code parquet"):
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _write_code_files,
                    code,
                    out_dir,
                    per_code_5min.get(code, []),
                    per_code_daily.get(code, []),
                )
                for code in all_codes
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Writing per-code parquet"):
                all_calendar_dates.update(future.result())

    if all_calendar_dates:
        with timer.section("write calendar parquet"):
            calendar = pd.DataFrame({"trade_date": sorted(all_calendar_dates)})
            calendar.to_parquet(out_dir / "calendar.parquet", index=False)

        with timer.section("fetch and write ST markers"):
            min_trade_date = pd.to_datetime(calendar["trade_date"].min())
            max_trade_date = pd.to_datetime(calendar["trade_date"].max())
            namechange = _fetch_namechange(min_trade_date, max_trade_date)
            st_df = _build_st_markers(namechange)
            _write_st_files(st_df, out_dir)

    timer.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", default="./data")
    parser.add_argument("--out-dir", default="./data")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--disable-timer", action="store_true")
    args = parser.parse_args()
    preprocess(
        Path(args.raw_data_dir),
        Path(args.out_dir),
        max_workers=max(1, args.max_workers),
        enable_timer=not args.disable_timer,
    )