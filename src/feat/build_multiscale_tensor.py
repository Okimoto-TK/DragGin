from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

L_MICRO = 48
L_MEZZO = 40
L_MACRO = 30
C = 6
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
def aggregate_30m_from_5m(df_5m: pd.DataFrame) -> pd.DataFrame:
    agg_rows = []
    for d, g in df_5m.groupby("trade_date"):
        g = g.sort_values("dt")
        if len(g) != 48:
            return pd.DataFrame()
        if g["dt"].duplicated().any():
            return pd.DataFrame()
        for i in range(8):
            chunk = g.iloc[i * 6 : (i + 1) * 6]
            vol_sum = chunk["volume"].sum()
            agg_rows.append(
                {
                    "trade_date": d,
                    "bucket": i,
                    "dt": chunk.iloc[0]["dt"],
                    "open": chunk.iloc[0]["open"],
                    "high": chunk["high"].max(),
                    "low": chunk["low"].min(),
                    "close": chunk.iloc[-1]["close"],
                    "volume": vol_sum,
                }
            )
    if not agg_rows:
        return pd.DataFrame()
    return pd.DataFrame(agg_rows).sort_values("dt").reset_index(drop=True)


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


def _apply_asof_price_adjustment(
    df: pd.DataFrame,
    asof_date: date,
    adj_factor_by_date: pd.Series,
) -> tuple[Optional[pd.DataFrame], str]:
    if asof_date not in adj_factor_by_date.index:
        return None, "missing adj_factor on asof date"
    asof_factor = float(adj_factor_by_date.loc[asof_date])
    if (not np.isfinite(asof_factor)) or asof_factor <= 0:
        return None, "invalid adj_factor on asof date"

    trade_dates = df["trade_date"]
    mapped = trade_dates.map(adj_factor_by_date)
    if mapped.isna().any():
        return None, "missing adj_factor in required history"
    factors = mapped.astype(float).to_numpy() / asof_factor
    if (~np.isfinite(factors) | (factors <= 0)).any():
        return None, "invalid adj_factor ratio"

    out = df.copy()
    for col in ["open", "high", "low", "close"]:
        out[col] = out[col].astype(float) * factors
    return out, ""


def _extract_tail_tensor(
    df: pd.DataFrame,
    L: int,
    asof_date: date,
    by_daily: bool,
    z_window: int,
    expected_daily_dates: Optional[list[date]] = None,
) -> tuple[Optional[np.ndarray], str]:
    if by_daily:
        hist = df[df["trade_date"] <= asof_date].copy()
    else:
        hist = df[df["dt"] <= pd.Timestamp(str(asof_date) + " 23:59:59")].copy()

    req_raw = L + RAW_WARMUP
    if len(hist) < req_raw:
        return None, f"insufficient raw warmup/history: need {req_raw}, got {len(hist)}"

    req_z = L + z_window + RAW_WARMUP
    if len(hist) < req_z:
        return None, f"insufficient zscore warmup/history: need {req_z}, got {len(hist)}"

    if by_daily:
        dates = hist["trade_date"].tolist()
        if expected_daily_dates is not None and dates[-req_z:] != expected_daily_dates:
            return None, "missing daily bar in required history"

    work = _compute_features(hist)
    cols = ["C1", "C2", "C3", "C4", "C5", "C6"]

    raw_region = work.iloc[-(L + z_window) :][cols]
    if raw_region.isna().any().any():
        return None, "feature NaN from strict rolling"
    if not np.isfinite(raw_region.to_numpy()).all():
        return None, "feature inf/non-finite"

    raw = work[cols]
    past_only = raw.shift(1)
    mu = past_only.rolling(window=z_window, min_periods=z_window).mean()
    sd = past_only.rolling(window=z_window, min_periods=z_window).std(ddof=0)

    sd_tail = sd.iloc[-L:]
    if (~np.isfinite(sd_tail.to_numpy()) | (sd_tail.to_numpy() < 0)).any():
        return None, "zscore sd invalid (<0 or non-finite)"

    z = (raw - mu) / sd.clip(lower=EPS)
    z = TANH_K * np.tanh(z / TANH_K)

    tail = z.iloc[-L:]
    if tail.isna().any().any():
        return None, "zscore NaN from strict rolling"
    if not np.isfinite(tail.to_numpy()).all():
        return None, "zscore inf/non-finite"

    return tail.to_numpy(dtype=np.float32), ""




def _has_breakpoint_crossing(dates: list[date], start_idx: int, end_idx: int, breakpoints: set[date]) -> bool:
    if not breakpoints or end_idx <= start_idx:
        return False
    window_dates = dates[start_idx : end_idx + 1]
    if len(window_dates) < 2:
        return False
    return any(window_dates[0] < b <= window_dates[-1] for b in breakpoints)
def build_multiscale_tensors(data_dir: str | Path, code: str, asof_date: str) -> DPResult:
    asof = pd.to_datetime(asof_date).date()
    m5 = load_5m_data(data_dir, code)
    d1 = load_daily_data(data_dir, code)
    if not _validate_raw(m5):
        return empty_result(code, asof_date, "missing/invalid 5m raw schema")
    if not _validate_raw(d1):
        return empty_result(code, asof_date, "missing/invalid daily raw schema")

    trade_dates = set(d1["trade_date"].tolist())
    m5 = m5[m5["trade_date"].isin(trade_dates)].copy()
    if not _validate_raw(m5):
        return empty_result(code, asof_date, "missing/invalid 5m raw schema")

    adj_factor_by_date, adj_err = _build_adj_factor_map(d1)
    if adj_factor_by_date is None:
        return empty_result(code, asof_date, adj_err)

    m5_asof = m5[m5["trade_date"] == asof]
    if len(m5_asof) != 48:
        return empty_result(code, asof_date, "micro day must have exactly 48 bars")

    m5_adj, err = _apply_asof_price_adjustment(m5, asof, adj_factor_by_date)
    if m5_adj is None:
        return empty_result(code, asof_date, f"micro: {err}")

    m30 = aggregate_30m_from_5m(m5_adj)
    if m30.empty:
        return empty_result(code, asof_date, "30m aggregation failure")

    X_micro, err = _extract_tail_tensor(m5_adj, L_MICRO, asof, by_daily=False, z_window=W_MICRO)
    if X_micro is None:
        return empty_result(code, asof_date, f"micro: {err}")

    X_mezzo, err = _extract_tail_tensor(m30, L_MEZZO, asof, by_daily=False, z_window=W_MEZZO)
    if X_mezzo is None:
        return empty_result(code, asof_date, f"mezzo: {err}")

    market_calendar = [pd.to_datetime(x).date() for x in build_calendar_from_daily_filenames(data_dir)]
    stock_trade_dates = set(d1["trade_date"].tolist())
    stock_calendar = [d for d in market_calendar if d in stock_trade_dates]
    if asof not in stock_calendar:
        return empty_result(code, asof_date, "asof date not in calendar")
    idx = stock_calendar.index(asof)
    breakpoints = load_breakpoints(data_dir, code)
    req = L_MACRO + W_MACRO + RAW_WARMUP
    if idx + 1 < req:
        return empty_result(code, asof_date, "macro: insufficient zscore warmup/history")
    expected_daily_dates = stock_calendar[idx + 1 - req : idx + 1]
    if _has_breakpoint_crossing(stock_calendar, idx + 1 - req, idx, breakpoints):
        return empty_result(code, asof_date, "macro: history crosses st breakpoint")

    d1_adj, err = _apply_asof_price_adjustment(d1, asof, adj_factor_by_date)
    if d1_adj is None:
        return empty_result(code, asof_date, f"macro: {err}")

    X_macro, err = _extract_tail_tensor(
        d1_adj,
        L_MACRO,
        asof,
        by_daily=True,
        z_window=W_MACRO,
        expected_daily_dates=expected_daily_dates,
    )
    if X_macro is None:
        return empty_result(code, asof_date, f"macro: {err}")

    return DPResult(
        code=code,
        asof_date=asof_date,
        dp_ok=True,
        reason="",
        X_micro=X_micro,
        X_mezzo=X_mezzo,
        X_macro=X_macro,
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
