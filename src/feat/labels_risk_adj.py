from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.feat.build_multiscale_tensor import build_calendar_from_daily_filenames, load_daily_data

EPS = 1e-6
VOL_WINDOW = 30


@dataclass
class LabelBundle:
    code: str
    asof_date: str
    y: np.float32
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

    idx = calendar_dates.index(asof)
    if idx + 3 >= len(calendar_dates):
        return _fail(code, asof_date, dp_ok, "future calendar coverage missing for T+3")

    if idx - VOL_WINDOW < 0:
        return _fail(code, asof_date, dp_ok, "insufficient calendar history for vol30")

    entry_d = calendar_dates[idx + 1]
    exit_d = calendar_dates[idx + 3]

    daily = daily_loader(code)
    if daily.empty:
        return _fail(code, asof_date, dp_ok, "daily data unavailable")

    daily = daily.copy()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"]).dt.date
    daily = daily.drop_duplicates(subset=["trade_date"], keep="last").sort_values("trade_date")
    by_date = daily.set_index("trade_date")

    required_close_dates = calendar_dates[idx - VOL_WINDOW : idx + 1]
    for d in required_close_dates:
        if d not in by_date.index:
            return _fail(code, asof_date, dp_ok, f"missing close for vol30 date {d.isoformat()}")
        c = by_date.at[d, "close"]
        if not np.isfinite(c) or c <= 0:
            return _fail(code, asof_date, dp_ok, f"invalid close for vol30 date {d.isoformat()}")

    if entry_d not in by_date.index:
        return _fail(code, asof_date, dp_ok, f"missing entry open on {entry_d.isoformat()}")
    entry_open = float(by_date.at[entry_d, "open"])
    if not np.isfinite(entry_open) or entry_open <= 0:
        return _fail(code, asof_date, dp_ok, f"invalid entry open on {entry_d.isoformat()}")

    if exit_d not in by_date.index:
        return _fail(code, asof_date, dp_ok, f"missing exit close on {exit_d.isoformat()}")
    exit_close = float(by_date.at[exit_d, "close"])
    if not np.isfinite(exit_close) or exit_close <= 0:
        return _fail(code, asof_date, dp_ok, f"invalid exit close on {exit_d.isoformat()}")

    closes = by_date.loc[required_close_dates, ["close"]].reset_index()
    returns = compute_daily_log_returns(closes)
    vol30 = float(rolling_std_last_n(returns, n=VOL_WINDOW).iloc[-1])
    if not np.isfinite(vol30) or vol30 <= 0:
        return _fail(code, asof_date, dp_ok, "vol30 unavailable/invalid")

    ret_log = float(np.log(exit_close / entry_open))
    y = np.float32(ret_log / (vol30 + EPS))

    label_ok = True
    return LabelBundle(
        code=code,
        asof_date=asof_date,
        y=y,
        label_ok=label_ok,
        loss_mask=bool(dp_ok and label_ok),
        entry_date=entry_d.isoformat(),
        exit_date=exit_d.isoformat(),
        entry_open=entry_open,
        exit_close=exit_close,
        vol30=vol30,
        ret_log=ret_log,
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
