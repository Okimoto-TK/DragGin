from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

L_MICRO = 48
L_MEZZO = 40
L_MACRO = 30
Z_DIM = 6
STRUCT_SHAPE_DIM = 12
C = Z_DIM + STRUCT_SHAPE_DIM
RAW_WARMUP = 20
EPS = 1e-8
TANH_K = 5.0

W_MICRO = 120
W_MEZZO = 80
W_MACRO = 45

DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2}|\d{8})")


@dataclass
class DPResult:
    code: str
    asof_date: str
    dp_ok: bool
    reason: str
    X_micro: np.ndarray
    X_mezzo: np.ndarray
    X_macro: np.ndarray
    mask_micro: np.ndarray
    mask_mezzo: np.ndarray
    mask_macro: np.ndarray


@dataclass
class _TensorContext:
    micro_z: np.ndarray
    mezzo_z: np.ndarray
    macro_z: np.ndarray
    micro_ss: np.ndarray
    mezzo_ss: np.ndarray
    macro_ss: np.ndarray
    asof_to_micro_end: dict[date, int]
    asof_to_mezzo_end: dict[date, int]
    asof_to_macro_idx: dict[date, int]
    asof_to_stock_idx: dict[date, int]
    stock_calendar: tuple[date, ...]
    breakpoints: frozenset[date]
    asof_adj_valid: frozenset[date]


def empty_result(code: str, asof_date: str, reason: str) -> DPResult:
    return DPResult(
        code=code,
        asof_date=asof_date,
        dp_ok=False,
        reason=reason,
        X_micro=np.zeros((L_MICRO, C), dtype=np.float32),
        X_mezzo=np.zeros((L_MEZZO, C), dtype=np.float32),
        X_macro=np.zeros((L_MACRO, C), dtype=np.float32),
        mask_micro=np.zeros((L_MICRO,), dtype=np.uint8),
        mask_mezzo=np.zeros((L_MEZZO,), dtype=np.uint8),
        mask_macro=np.zeros((L_MACRO,), dtype=np.uint8),
    )


def parse_date_from_filename(path: Path) -> Optional[date]:
    match = DATE_PATTERN.search(path.name)
    if not match:
        return None
    raw = match.group(1)
    fmt = "%Y-%m-%d" if "-" in raw else "%Y%m%d"
    try:
        return pd.to_datetime(raw, format=fmt).date()
    except Exception:
        return None


def build_calendar_from_daily_filenames(data_dir: str | Path) -> list[str]:
    root = Path(data_dir)
    calendar_file = root / "calendar.parquet"
    if calendar_file.exists():
        try:
            cal = pd.read_parquet(calendar_file, columns=["trade_date"])
            parsed = pd.to_datetime(cal["trade_date"], errors="coerce").dropna()
            return [d.isoformat() for d in sorted(set(parsed.dt.date.tolist()))]
        except Exception:
            pass

    dates: set[date] = set()
    for daily_file in root.glob("*/daily.parquet"):
        try:
            df = pd.read_parquet(daily_file, columns=["trade_date"])
        except Exception:
            continue
        if "trade_date" not in df.columns:
            continue
        parsed = pd.to_datetime(df["trade_date"], errors="coerce").dropna()
        dates.update(parsed.dt.date.tolist())
    return [d.isoformat() for d in sorted(dates)]


def load_5m_data(data_dir: str | Path, code: str) -> pd.DataFrame:
    file_path = Path(data_dir) / code / "5min.parquet"
    if not file_path.exists():
        return pd.DataFrame()
    out = pd.read_parquet(file_path)
    required = {"trade_date", "time", "open", "high", "low", "close", "volume"}
    if not required.issubset(set(out.columns)):
        return pd.DataFrame()
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.date
    out["dt"] = pd.to_datetime(out["trade_date"].astype(str) + " " + out["time"].astype(str))
    out = out.sort_values(["dt"]).reset_index(drop=True)
    return out


def load_daily_data(data_dir: str | Path, code: str) -> pd.DataFrame:
    file_path = Path(data_dir) / code / "daily.parquet"
    if not file_path.exists():
        return pd.DataFrame()
    out = pd.read_parquet(file_path)
    required = {"trade_date", "open", "high", "low", "close", "volume", "adj_factor"}
    if not required.issubset(set(out.columns)):
        return pd.DataFrame()
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.date
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out[np.isfinite(out["volume"])]
    out = out.drop_duplicates(subset=["trade_date"], keep="last").sort_values("trade_date").reset_index(drop=True)
    return out


@lru_cache(maxsize=512)
def _load_5m_data_cached(data_dir: str, code: str) -> pd.DataFrame:
    return load_5m_data(data_dir, code)


@lru_cache(maxsize=512)
def _load_daily_data_cached(data_dir: str, code: str) -> pd.DataFrame:
    return load_daily_data(data_dir, code)


def load_limit_data(data_dir: str | Path, code: str) -> pd.DataFrame:
    file_path = Path(data_dir) / code / "limit.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"limit data unavailable for {code}: {file_path}")

    out = pd.read_parquet(file_path)
    required = {"trade_date", "up_limit", "down_limit", "limit_pct"}
    missing = sorted(required.difference(set(out.columns)))
    if missing:
        raise ValueError(f"limit data missing required columns for {code}: {missing}")

    out = out[["trade_date", "up_limit", "down_limit", "limit_pct"]].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.date
    for col in ["up_limit", "down_limit", "limit_pct"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["trade_date"]).drop_duplicates(subset=["trade_date"], keep="last").sort_values("trade_date").reset_index(drop=True)
    return out


@lru_cache(maxsize=512)
def _load_limit_data_cached(data_dir: str, code: str) -> pd.DataFrame:
    return load_limit_data(data_dir, code)


def load_breakpoints(data_dir: str | Path, code: str) -> set[date]:
    file_path = Path(data_dir) / code / "breakpoints.parquet"
    if not file_path.exists():
        return set()
    try:
        bp = pd.read_parquet(file_path, columns=["break_date"])
    except Exception:
        return set()
    parsed = pd.to_datetime(bp.get("break_date"), errors="coerce").dropna()
    return set(parsed.dt.date.tolist())


@lru_cache(maxsize=512)
def _load_breakpoints_cached(data_dir: str, code: str) -> tuple[date, ...]:
    return tuple(sorted(load_breakpoints(data_dir, code)))


@lru_cache(maxsize=16)
def _load_market_calendar_dates(data_dir: str) -> tuple[date, ...]:
    return tuple(pd.to_datetime(x).date() for x in build_calendar_from_daily_filenames(data_dir))


def aggregate_30m_from_5m(df_5m: pd.DataFrame) -> pd.DataFrame:
    if df_5m.empty:
        return pd.DataFrame()

    sorted_df = df_5m.sort_values(["trade_date", "dt"]).reset_index(drop=True)
    counts = sorted_df.groupby("trade_date", sort=False).size().to_numpy()
    if len(counts) == 0 or np.any(counts != 48):
        return pd.DataFrame()
    if sorted_df.duplicated(subset=["trade_date", "dt"]).any():
        return pd.DataFrame()

    num_days = len(counts)
    buckets_per_day = 8
    bars_per_bucket = 6

    trade_dates = sorted_df["trade_date"].to_numpy().reshape(num_days, 48)[:, 0]
    dt_arr = sorted_df["dt"].to_numpy().reshape(num_days, buckets_per_day, bars_per_bucket)
    open_arr = sorted_df["open"].to_numpy(dtype=np.float64).reshape(num_days, buckets_per_day, bars_per_bucket)
    high_arr = sorted_df["high"].to_numpy(dtype=np.float64).reshape(num_days, buckets_per_day, bars_per_bucket)
    low_arr = sorted_df["low"].to_numpy(dtype=np.float64).reshape(num_days, buckets_per_day, bars_per_bucket)
    close_arr = sorted_df["close"].to_numpy(dtype=np.float64).reshape(num_days, buckets_per_day, bars_per_bucket)
    volume_arr = sorted_df["volume"].to_numpy(dtype=np.float64).reshape(num_days, buckets_per_day, bars_per_bucket)

    out_dict = {
        "trade_date": np.repeat(trade_dates, buckets_per_day),
        "bucket": np.tile(np.arange(buckets_per_day, dtype=np.int64), num_days),
        "dt": dt_arr[:, :, 0].reshape(-1),
        "open": open_arr[:, :, 0].reshape(-1),
        "high": high_arr.max(axis=2).reshape(-1),
        "low": low_arr.min(axis=2).reshape(-1),
        "close": close_arr[:, :, -1].reshape(-1),
        "volume": volume_arr.sum(axis=2).reshape(-1),
    }
    if "amount" in sorted_df.columns:
        amount_arr = sorted_df["amount"].to_numpy(dtype=np.float64).reshape(num_days, buckets_per_day, bars_per_bucket)
        out_dict["amount"] = amount_arr.sum(axis=2).reshape(-1)

    return pd.DataFrame(out_dict).sort_values("dt").reset_index(drop=True)


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    z["prev_close"] = z["close"].shift(1)
    z["ret"] = np.log(z["close"] / z["prev_close"])
    z["C1"] = z["ret"]
    z["C2"] = np.log(z["open"] / z["prev_close"])
    z["C3"] = np.log(z["high"] / z["low"])
    z["C4"] = np.where(z["high"] == z["low"], 0.5, (z["close"] - z["low"]) / (z["high"] - z["low"]))
    z["vol_mean20"] = z["volume"].rolling(20, min_periods=20).mean().shift(1)
    z["C5"] = z["volume"] / z["vol_mean20"]
    z["ret_std5"] = z["ret"].rolling(5, min_periods=5).std().shift(1)
    z["C6"] = z["ret_std5"]
    return z


def _validate_raw(df: pd.DataFrame) -> bool:
    needed = ["open", "high", "low", "close", "volume"]
    if df.empty:
        return False
    if df[needed].isna().any().any():
        return False
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        return False
    return True


def _build_adj_factor_map(df_daily: pd.DataFrame) -> tuple[Optional[pd.Series], str]:
    if "adj_factor" not in df_daily.columns:
        return None, "daily missing adj_factor"
    s = df_daily.set_index("trade_date")["adj_factor"].astype(float)
    if s.isna().any():
        return None, "daily adj_factor NaN"
    if (~np.isfinite(s.to_numpy()) | (s.to_numpy() <= 0)).any():
        return None, "daily adj_factor invalid"
    return s, ""


def _apply_base_price_adjustment(df: pd.DataFrame, adj_factor_by_date: pd.Series) -> tuple[Optional[pd.DataFrame], str]:
    trade_dates = df["trade_date"]
    mapped = trade_dates.map(adj_factor_by_date)
    if mapped.isna().any():
        return None, "missing adj_factor in required history"
    factors = mapped.astype(float).to_numpy()
    if (~np.isfinite(factors) | (factors <= 0)).any():
        return None, "invalid adj_factor ratio"

    out = df.copy()
    for col in ["open", "high", "low", "close"]:
        out[col] = out[col].astype(float) * factors
    return out, ""


def _compute_feature_matrix(df: pd.DataFrame, z_window: int) -> np.ndarray:
    work = _compute_features(df)
    cols = ["C1", "C2", "C3", "C4", "C5", "C6"]
    raw = work[cols]
    past_only = raw.shift(1)
    mu = past_only.rolling(window=z_window, min_periods=z_window).mean()
    sd = past_only.rolling(window=z_window, min_periods=z_window).std(ddof=0)
    z = (raw - mu) / sd.clip(lower=EPS)
    z = TANH_K * np.tanh(z / TANH_K)
    return z.to_numpy(dtype=np.float32)


def _compute_struct_shape_features(df: pd.DataFrame, short: int, mid: int, long: int) -> np.ndarray:
    close = df["close"].astype(np.float64)
    open_ = df["open"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    prev_close = close.shift(1)

    ma_short = close.rolling(short, min_periods=short).mean()
    ma_long = close.rolling(long, min_periods=long).mean()
    rolling_low_short = low.rolling(short, min_periods=short).min()
    rolling_high_short = high.rolling(short, min_periods=short).max()
    rolling_low_long = low.rolling(long, min_periods=long).min()
    rolling_high_long = high.rolling(long, min_periods=long).max()
    intrabar_range = high - low

    feats = pd.DataFrame(
        {
            "ret_short_raw": np.tanh(4.0 * np.log(close / close.shift(short))),
            "ret_mid_raw": np.tanh(3.0 * np.log(close / close.shift(mid))),
            "ma_short_gap": np.tanh(5.0 * (close / ma_short - 1.0)),
            "ma_long_gap": np.tanh(5.0 * (close / ma_long - 1.0)),
            "close_rank_short": (close - rolling_low_short) / (rolling_high_short - rolling_low_short + EPS),
            "close_rank_long": (close - rolling_low_long) / (rolling_high_long - rolling_low_long + EPS),
            "body_ratio": (close - open_) / (intrabar_range + EPS),
            "upper_shadow_ratio": (high - np.maximum(open_, close)) / (intrabar_range + EPS),
            "lower_shadow_ratio": (np.minimum(open_, close) - low) / (intrabar_range + EPS),
            "close_pos_raw": (close - low) / (intrabar_range + EPS),
            "gap_raw": np.tanh(6.0 * np.log(open_ / prev_close)),
            "drawdown_long": np.tanh(4.0 * (close / rolling_high_long - 1.0)),
        }
    )
    return feats.to_numpy(dtype=np.float32)


def _has_breakpoint_crossing(dates: list[date], start_idx: int, end_idx: int, breakpoints: set[date]) -> bool:
    if not breakpoints or end_idx <= start_idx:
        return False
    window_dates = dates[start_idx : end_idx + 1]
    if len(window_dates) < 2:
        return False
    return any(window_dates[0] < b <= window_dates[-1] for b in breakpoints)


def _build_day_end_index(df: pd.DataFrame, expected_per_day: int) -> dict[date, int]:
    out: dict[date, int] = {}
    for _, g in df.groupby("trade_date"):
        if len(g) == expected_per_day:
            d = g.iloc[-1]["trade_date"]
            out[d] = int(g.index[-1])
    return out


@lru_cache(maxsize=512)
def _build_tensor_context(data_dir: str, code: str) -> _TensorContext | None:
    m5 = _load_5m_data_cached(data_dir, code).copy()
    d1 = _load_daily_data_cached(data_dir, code).copy()
    if not _validate_raw(m5) or not _validate_raw(d1):
        return None

    trade_dates = set(d1["trade_date"].tolist())
    m5 = m5[m5["trade_date"].isin(trade_dates)].copy()
    if not _validate_raw(m5):
        return None

    adj_factor_by_date, _ = _build_adj_factor_map(d1)
    if adj_factor_by_date is None:
        return None

    m5_adj, _ = _apply_base_price_adjustment(m5, adj_factor_by_date)
    if m5_adj is None:
        return None
    m30 = aggregate_30m_from_5m(m5_adj)
    if m30.empty:
        return None

    d1_adj, _ = _apply_base_price_adjustment(d1, adj_factor_by_date)
    if d1_adj is None:
        return None

    market_calendar = _load_market_calendar_dates(data_dir)
    stock_trade_dates = set(d1["trade_date"].tolist())
    stock_calendar = tuple(d for d in market_calendar if d in stock_trade_dates)

    micro_z = _compute_feature_matrix(m5_adj, W_MICRO)
    mezzo_z = _compute_feature_matrix(m30, W_MEZZO)
    macro_z = _compute_feature_matrix(d1_adj, W_MACRO)
    micro_ss = _compute_struct_shape_features(m5_adj, short=8, mid=24, long=48)
    mezzo_ss = _compute_struct_shape_features(m30, short=4, mid=12, long=40)
    macro_ss = _compute_struct_shape_features(d1_adj, short=5, mid=10, long=20)

    return _TensorContext(
        micro_z=micro_z,
        mezzo_z=mezzo_z,
        macro_z=macro_z,
        micro_ss=micro_ss,
        mezzo_ss=mezzo_ss,
        macro_ss=macro_ss,
        asof_to_micro_end=_build_day_end_index(m5_adj, 48),
        asof_to_mezzo_end=_build_day_end_index(m30, 8),
        asof_to_macro_idx={d: i for i, d in enumerate(d1_adj["trade_date"].tolist())},
        asof_to_stock_idx={d: i for i, d in enumerate(stock_calendar)},
        stock_calendar=stock_calendar,
        breakpoints=frozenset(_load_breakpoints_cached(data_dir, code)),
        asof_adj_valid=frozenset(adj_factor_by_date.index.tolist()),
    )


def _slice_tensor(
    z: np.ndarray,
    end_idx: int,
    L: int,
    req_z: int,
    label: str,
    *,
    require_finite: bool = True,
) -> tuple[Optional[np.ndarray], str]:
    if end_idx + 1 < req_z:
        return None, f"{label}: insufficient zscore warmup/history"
    if end_idx + 1 < L:
        return None, f"{label}: insufficient raw warmup/history"
    tail = z[end_idx + 1 - L : end_idx + 1]
    if tail.shape[0] != L:
        return None, f"{label}: insufficient zscore warmup/history"
    if require_finite and not np.isfinite(tail).all():
        return None, f"{label}: zscore NaN from strict rolling"
    return tail.astype(np.float32), ""


@lru_cache(maxsize=512)
def get_tensor_valid_asof_dates(data_dir: str, code: str) -> tuple[str, ...]:
    context = _build_tensor_context(data_dir, code)
    if context is None:
        return tuple()
    out: list[str] = []
    stock_calendar = context.stock_calendar
    breakpoints = set(context.breakpoints)
    req_macro = L_MACRO + W_MACRO + RAW_WARMUP
    req_micro = L_MICRO + W_MICRO + RAW_WARMUP
    req_mezzo = L_MEZZO + W_MEZZO + RAW_WARMUP

    for idx, asof in enumerate(stock_calendar):
        if asof not in context.asof_adj_valid:
            continue
        micro_end = context.asof_to_micro_end.get(asof)
        mezzo_end = context.asof_to_mezzo_end.get(asof)
        macro_idx = context.asof_to_macro_idx.get(asof)
        if micro_end is None or mezzo_end is None or macro_idx is None:
            continue
        if micro_end + 1 < req_micro or mezzo_end + 1 < req_mezzo or idx + 1 < req_macro:
            continue
        if _has_breakpoint_crossing(list(stock_calendar), idx + 1 - req_macro, idx, breakpoints):
            continue
        if not np.isfinite(context.micro_z[micro_end + 1 - L_MICRO : micro_end + 1]).all():
            continue
        if not np.isfinite(context.mezzo_z[mezzo_end + 1 - L_MEZZO : mezzo_end + 1]).all():
            continue
        if not np.isfinite(context.macro_z[macro_idx + 1 - L_MACRO : macro_idx + 1]).all():
            continue
        out.append(asof.isoformat())
    return tuple(out)


def clear_tensor_worker_cache() -> None:
    _load_5m_data_cached.cache_clear()
    _load_daily_data_cached.cache_clear()
    _load_limit_data_cached.cache_clear()
    _load_breakpoints_cached.cache_clear()
    _load_market_calendar_dates.cache_clear()
    _build_tensor_context.cache_clear()
    get_tensor_valid_asof_dates.cache_clear()


def build_multiscale_tensors(data_dir: str | Path, code: str, asof_date: str) -> DPResult:
    data_dir_key = str(Path(data_dir).resolve())
    asof = pd.to_datetime(asof_date).date()
    context = _build_tensor_context(data_dir_key, code)
    if context is None:
        m5 = _load_5m_data_cached(data_dir_key, code).copy()
        d1 = _load_daily_data_cached(data_dir_key, code).copy()
        if not _validate_raw(m5):
            return empty_result(code, asof_date, "missing/invalid 5m raw schema")
        if not _validate_raw(d1):
            return empty_result(code, asof_date, "missing/invalid daily raw schema")
        return empty_result(code, asof_date, "30m aggregation failure")

    if asof not in context.asof_adj_valid:
        return empty_result(code, asof_date, "missing adj_factor on asof date")

    micro_end = context.asof_to_micro_end.get(asof)
    if micro_end is None:
        return empty_result(code, asof_date, "micro day must have exactly 48 bars")

    x_micro, err = _slice_tensor(context.micro_z, micro_end, L_MICRO, L_MICRO + W_MICRO + RAW_WARMUP, "micro")
    if x_micro is None:
        return empty_result(code, asof_date, err)
    s_micro, err = _slice_tensor(context.micro_ss, micro_end, L_MICRO, L_MICRO, "micro", require_finite=False)
    if s_micro is None:
        return empty_result(code, asof_date, err)
    s_micro = np.nan_to_num(s_micro, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    x_micro = np.concatenate([x_micro, s_micro], axis=1).astype(np.float32, copy=False)

    mezzo_end = context.asof_to_mezzo_end.get(asof)
    if mezzo_end is None:
        return empty_result(code, asof_date, "30m aggregation failure")
    x_mezzo, err = _slice_tensor(context.mezzo_z, mezzo_end, L_MEZZO, L_MEZZO + W_MEZZO + RAW_WARMUP, "mezzo")
    if x_mezzo is None:
        return empty_result(code, asof_date, err)
    s_mezzo, err = _slice_tensor(context.mezzo_ss, mezzo_end, L_MEZZO, L_MEZZO, "mezzo", require_finite=False)
    if s_mezzo is None:
        return empty_result(code, asof_date, err)
    s_mezzo = np.nan_to_num(s_mezzo, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    x_mezzo = np.concatenate([x_mezzo, s_mezzo], axis=1).astype(np.float32, copy=False)

    stock_calendar = context.stock_calendar
    idx = context.asof_to_stock_idx.get(asof)
    if idx is None:
        return empty_result(code, asof_date, "asof date not in calendar")

    req_macro = L_MACRO + W_MACRO + RAW_WARMUP
    if idx + 1 < req_macro:
        return empty_result(code, asof_date, "macro: insufficient zscore warmup/history")
    if _has_breakpoint_crossing(list(stock_calendar), idx + 1 - req_macro, idx, set(context.breakpoints)):
        return empty_result(code, asof_date, "macro: history crosses st breakpoint")

    macro_idx = context.asof_to_macro_idx.get(asof)
    if macro_idx is None:
        return empty_result(code, asof_date, "macro: missing adj_factor in required history")
    x_macro, err = _slice_tensor(context.macro_z, macro_idx, L_MACRO, req_macro, "macro")
    if x_macro is None:
        return empty_result(code, asof_date, err)
    s_macro, err = _slice_tensor(context.macro_ss, macro_idx, L_MACRO, L_MACRO, "macro", require_finite=False)
    if s_macro is None:
        return empty_result(code, asof_date, err)
    s_macro = np.nan_to_num(s_macro, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    x_macro = np.concatenate([x_macro, s_macro], axis=1).astype(np.float32, copy=False)

    return DPResult(
        code=code,
        asof_date=asof_date,
        dp_ok=True,
        reason="",
        X_micro=x_micro,
        X_mezzo=x_mezzo,
        X_macro=x_macro,
        mask_micro=np.ones((L_MICRO,), dtype=np.uint8),
        mask_mezzo=np.ones((L_MEZZO,), dtype=np.uint8),
        mask_macro=np.ones((L_MACRO,), dtype=np.uint8),
    )


def print_calendar_summary(data_dir: str | Path) -> None:
    files = list(Path(data_dir).glob("*/daily.parquet"))
    cal = build_calendar_from_daily_filenames(data_dir)
    print(f"total files scanned: {len(files)}")
    print(f"total valid dates: {len(cal)}")
    if cal:
        print(f"first date: {cal[0]}")
        print(f"last date: {cal[-1]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--code", required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--print-calendar", action="store_true")
    parser.add_argument("--dump-out", default=None)
    args = parser.parse_args()

    if args.print_calendar:
        print_calendar_summary(args.data_dir)

    result = build_multiscale_tensors(args.data_dir, args.code, args.asof)
    print(f"dp_ok: {result.dp_ok}")
    print(f"reason: {result.reason or 'ok'}")
    print(f"X_micro: {result.X_micro.shape}, mask_micro: {result.mask_micro.shape}")
    print(f"X_mezzo: {result.X_mezzo.shape}, mask_mezzo: {result.mask_mezzo.shape}")
    print(f"X_macro: {result.X_macro.shape}, mask_macro: {result.mask_macro.shape}")

    if args.dump_out:
        np.savez(
            args.dump_out,
            X_micro=result.X_micro,
            X_mezzo=result.X_mezzo,
            X_macro=result.X_macro,
            mask_micro=result.mask_micro,
            mask_mezzo=result.mask_mezzo,
            mask_macro=result.mask_macro,
            dp_ok=np.array(result.dp_ok),
        )


if __name__ == "__main__":
    main()
