from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.feat.build_multiscale_tensor import build_calendar_from_daily_filenames, load_daily_data

EPS = 1e-6
ZSCORE_EPS = 1e-8
VOL_WINDOW = 30
LABEL_Z_WINDOW = 252
TANH_K = 5.0


@dataclass
class LabelBundle:
    code: str
    asof_date: str
    y: np.float32
    y_raw: np.float32
    y_z: np.float32
    label_ok: bool
    loss_mask: bool
    entry_date: str | None
    exit_date: str | None
    entry_open: float | None
    exit_close: float | None
    vol30: float | None
    ret_log: float | None
    fail_reason: str | None


def compute_daily_log_returns(df_daily_sorted: pd.DataFrame) -> pd.Series:
    close = df_daily_sorted["close"].astype(float)
    return np.log(close / close.shift(1))


def rolling_std_last_n(returns: pd.Series, n: int = VOL_WINDOW) -> pd.Series:
    return returns.rolling(window=n, min_periods=n).std()


def _fail(code: str, asof_date: str, dp_ok: bool, reason: str) -> LabelBundle:
    return LabelBundle(
        code=code,
        asof_date=asof_date,
        y=np.float32(0.0),
        y_raw=np.float32(0.0),
        y_z=np.float32(0.0),
        label_ok=False,
        loss_mask=bool(dp_ok and False),
        entry_date=None,
        exit_date=None,
        entry_open=None,
        exit_close=None,
        vol30=None,
        ret_log=None,
        fail_reason=reason,
    )


def _compute_raw_label_for_idx(
    idx: int,
    calendar_dates: list,
    by_date: pd.DataFrame,
) -> tuple[bool, float | None, dict, str | None]:
    if idx + 3 >= len(calendar_dates):
        return False, None, {}, "future calendar coverage missing for T+3"
    if idx - VOL_WINDOW < 0:
        return False, None, {}, "insufficient calendar history for vol30"

    entry_d = calendar_dates[idx + 1]
    exit_d = calendar_dates[idx + 3]
    asof_d = calendar_dates[idx]

    if "adj_factor" not in by_date.columns:
        return False, None, {}, "daily missing adj_factor"
    if asof_d not in by_date.index:
        return False, None, {}, f"missing asof daily row on {asof_d.isoformat()}"
    asof_factor = float(by_date.at[asof_d, "adj_factor"])
    if not np.isfinite(asof_factor) or asof_factor <= 0:
        return False, None, {}, f"invalid asof adj_factor on {asof_d.isoformat()}"

    required_close_dates = calendar_dates[idx - VOL_WINDOW : idx + 1]
    for d in required_close_dates:
        if d not in by_date.index:
            return False, None, {}, f"missing close for vol30 date {d.isoformat()}"
        f = float(by_date.at[d, "adj_factor"])
        if not np.isfinite(f) or f <= 0:
            return False, None, {}, f"invalid adj_factor for vol30 date {d.isoformat()}"
        c = by_date.at[d, "close"]
        if not np.isfinite(c) or c <= 0:
            return False, None, {}, f"invalid close for vol30 date {d.isoformat()}"

    if entry_d not in by_date.index:
        return False, None, {}, f"missing entry open on {entry_d.isoformat()}"
    entry_factor = float(by_date.at[entry_d, "adj_factor"])
    if not np.isfinite(entry_factor) or entry_factor <= 0:
        return False, None, {}, f"invalid entry adj_factor on {entry_d.isoformat()}"
    entry_open = float(by_date.at[entry_d, "open"]) * (entry_factor / asof_factor)
    if not np.isfinite(entry_open) or entry_open <= 0:
        return False, None, {}, f"invalid entry open on {entry_d.isoformat()}"

    if exit_d not in by_date.index:
        return False, None, {}, f"missing exit close on {exit_d.isoformat()}"
    exit_factor = float(by_date.at[exit_d, "adj_factor"])
    if not np.isfinite(exit_factor) or exit_factor <= 0:
        return False, None, {}, f"invalid exit adj_factor on {exit_d.isoformat()}"
    exit_close = float(by_date.at[exit_d, "close"]) * (exit_factor / asof_factor)
    if not np.isfinite(exit_close) or exit_close <= 0:
        return False, None, {}, f"invalid exit close on {exit_d.isoformat()}"

    closes = by_date.loc[required_close_dates, ["close", "adj_factor"]].reset_index()
    closes["close"] = closes["close"].astype(float) * (closes["adj_factor"].astype(float) / asof_factor)
    closes = closes[["trade_date", "close"]]
    returns = compute_daily_log_returns(closes)
    vol30 = float(rolling_std_last_n(returns, n=VOL_WINDOW).iloc[-1])
    if not np.isfinite(vol30) or vol30 <= 0:
        return False, None, {}, "vol30 unavailable/invalid"

    ret_log = float(np.log(exit_close / entry_open))
    y_raw = float(ret_log / (vol30 + EPS))
    details = {
        "entry_date": entry_d.isoformat(),
        "exit_date": exit_d.isoformat(),
        "entry_open": entry_open,
        "exit_close": exit_close,
        "vol30": vol30,
        "ret_log": ret_log,
    }
    return True, y_raw, details, None


def build_label_for_sample(
    code: str,
    asof_date: str,
    calendar: list[str],
    daily_loader,
    dp_ok: bool = True,
) -> LabelBundle:
    asof = pd.to_datetime(asof_date).date()
    calendar_dates = [pd.to_datetime(x).date() for x in calendar]

    if asof not in calendar_dates:
        return _fail(code, asof_date, dp_ok, "asof date not in calendar")

    daily = daily_loader(code)
    if daily.empty:
        return _fail(code, asof_date, dp_ok, "daily data unavailable")

    daily = daily.copy()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"]).dt.date
    if "volume" in daily.columns:
        daily["volume"] = pd.to_numeric(daily["volume"], errors="coerce")
        daily = daily[(daily["volume"] > 0) & np.isfinite(daily["volume"])]
    daily = daily.drop_duplicates(subset=["trade_date"], keep="last").sort_values("trade_date")
    by_date = daily.set_index("trade_date")

    trading_calendar_dates = [d for d in calendar_dates if d in by_date.index]
    if asof not in trading_calendar_dates:
        return _fail(code, asof_date, dp_ok, "asof date not in trading calendar")

    idx = trading_calendar_dates.index(asof)
    ok_raw, y_raw_val, details, fail_reason = _compute_raw_label_for_idx(idx, trading_calendar_dates, by_date)
    if not ok_raw or y_raw_val is None:
        return _fail(code, asof_date, dp_ok, fail_reason or "raw label build failed")

    past_raw_values: list[float] = []
    for pidx in range(idx):
        ok_past, y_past, _, _ = _compute_raw_label_for_idx(pidx, trading_calendar_dates, by_date)
        if ok_past and y_past is not None and np.isfinite(y_past):
            past_raw_values.append(float(y_past))

    if len(past_raw_values) < LABEL_Z_WINDOW:
        return _fail(code, asof_date, dp_ok, f"insufficient y_raw history for label zscore: need {LABEL_Z_WINDOW}, got {len(past_raw_values)}")

    past_window = np.asarray(past_raw_values[-LABEL_Z_WINDOW:], dtype=np.float64)
    mu = float(np.mean(past_window))
    sd = float(np.std(past_window, ddof=0))
    if (not np.isfinite(mu)) or (not np.isfinite(sd)) or sd <= 0:
        return _fail(code, asof_date, dp_ok, "label zscore sd invalid (<=0 or non-finite)")

    y_z = (float(y_raw_val) - mu) / max(sd, ZSCORE_EPS)
    y_z = float(TANH_K * np.tanh(y_z / TANH_K))
    if not np.isfinite(y_z):
        return _fail(code, asof_date, dp_ok, "label zscore non-finite")

    label_ok = True
    return LabelBundle(
        code=code,
        asof_date=asof_date,
        y=np.float32(y_z),
        y_raw=np.float32(y_raw_val),
        y_z=np.float32(y_z),
        label_ok=label_ok,
        loss_mask=bool(dp_ok and label_ok),
        entry_date=details["entry_date"],
        exit_date=details["exit_date"],
        entry_open=details["entry_open"],
        exit_close=details["exit_close"],
        vol30=details["vol30"],
        ret_log=details["ret_log"],
        fail_reason=None,
    )


def build_label_from_data_dir(
    data_dir: str | Path,
    code: str,
    asof_date: str,
    dp_ok: bool = True,
) -> LabelBundle:
    calendar = build_calendar_from_daily_filenames(data_dir)

    def _loader(target_code: str) -> pd.DataFrame:
        return load_daily_data(data_dir, target_code)

    return build_label_for_sample(
        code=code,
        asof_date=asof_date,
        calendar=calendar,
        daily_loader=_loader,
        dp_ok=dp_ok,
    )
