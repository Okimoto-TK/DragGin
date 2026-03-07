from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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


@dataclass
class _LabelContext:
    calendar_dates: tuple
    trading_calendar_dates: tuple
    by_date: pd.DataFrame
    breakpoints: frozenset
    valid_raw_indices: tuple[int, ...]
    valid_raw_values: tuple[float, ...]
    raw_details_by_index: dict[int, dict]
    raw_fail_by_index: dict[int, str]
    z_mu_by_valid_pos: tuple[float, ...]
    z_sd_by_valid_pos: tuple[float, ...]
    asof_to_trading_idx: dict[date, int]
    raw_idx_to_valid_pos: dict[int, int]
    label_ok_by_idx: tuple[bool, ...]
    label_value_by_idx: tuple[float, ...]


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
    adj_open_np: np.ndarray,
    adj_close_np: np.ndarray,
    daily_log_ret: np.ndarray,
    vol30_by_idx: np.ndarray,
    date_to_pos: dict,
) -> tuple[bool, float | None, dict, str | None]:
    if idx + 3 >= len(calendar_dates):
        return False, None, {}, "future calendar coverage missing for T+3"
    if idx - VOL_WINDOW < 0:
        return False, None, {}, "insufficient calendar history for vol30"

    entry_d = calendar_dates[idx + 1]
    exit_d = calendar_dates[idx + 3]
    asof_d = calendar_dates[idx]

    asof_pos = date_to_pos.get(asof_d)
    if asof_pos is None:
        return False, None, {}, f"missing asof daily row on {asof_d.isoformat()}"

    for d in calendar_dates[idx - VOL_WINDOW : idx + 1]:
        if date_to_pos.get(d) is None:
            return False, None, {}, f"missing close for vol30 date {d.isoformat()}"

    entry_pos = date_to_pos.get(entry_d)
    if entry_pos is None:
        return False, None, {}, f"missing entry open on {entry_d.isoformat()}"
    entry_open = float(adj_open_np[entry_pos])
    if not np.isfinite(entry_open) or entry_open <= 0:
        return False, None, {}, f"invalid entry open on {entry_d.isoformat()}"

    exit_pos = date_to_pos.get(exit_d)
    if exit_pos is None:
        return False, None, {}, f"missing exit close on {exit_d.isoformat()}"
    exit_close = float(adj_close_np[exit_pos])
    if not np.isfinite(exit_close) or exit_close <= 0:
        return False, None, {}, f"invalid exit close on {exit_d.isoformat()}"

    _ = daily_log_ret  # precomputed in context for consistency and future reuse
    vol30 = float(vol30_by_idx[idx])
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


def _precompute_label_arrays(calendar_dates: list, by_date: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    n = len(calendar_dates)
    adj_open_np = np.full((n,), np.nan, dtype=np.float64)
    adj_close_np = np.full((n,), np.nan, dtype=np.float64)
    date_to_pos = {d: i for i, d in enumerate(calendar_dates)}

    if "adj_factor" not in by_date.columns:
        return adj_open_np, adj_close_np, np.full((n,), np.nan, dtype=np.float64), np.full((n,), np.nan, dtype=np.float64), date_to_pos

    for i, d in enumerate(calendar_dates):
        if d not in by_date.index:
            continue
        row = by_date.loc[d]
        adj_factor = float(row["adj_factor"])
        if not np.isfinite(adj_factor) or adj_factor <= 0:
            continue
        open_v = float(row["open"])
        close_v = float(row["close"])
        if np.isfinite(open_v) and open_v > 0:
            adj_open_np[i] = open_v * adj_factor
        if np.isfinite(close_v) and close_v > 0:
            adj_close_np[i] = close_v * adj_factor

    daily_log_ret = np.full((n,), np.nan, dtype=np.float64)
    if n > 1:
        prev = adj_close_np[:-1]
        curr = adj_close_np[1:]
        valid = np.isfinite(prev) & np.isfinite(curr) & (prev > 0) & (curr > 0)
        daily_log_ret[1:] = np.where(valid, np.log(curr / prev), np.nan)
    vol30_by_idx = rolling_std_last_n(pd.Series(daily_log_ret), n=VOL_WINDOW).to_numpy(dtype=np.float64)

    return adj_open_np, adj_close_np, daily_log_ret, vol30_by_idx, date_to_pos




def _has_breakpoint_crossing(calendar_dates: list, start_idx: int, end_idx: int, breakpoints: set) -> bool:
    if not breakpoints or end_idx <= start_idx:
        return False
    window_dates = calendar_dates[start_idx : end_idx + 1]
    if len(window_dates) < 2:
        return False
    return any(window_dates[0] < b <= window_dates[-1] for b in breakpoints)


def _load_breakpoints(data_dir: str | Path, code: str) -> set:
    file_path = Path(data_dir) / code / "breakpoints.parquet"
    if not file_path.exists():
        return set()
    try:
        bp = pd.read_parquet(file_path, columns=["break_date"])
    except Exception:
        return set()
    parsed = pd.to_datetime(bp.get("break_date"), errors="coerce").dropna()
    return set(parsed.dt.date.tolist())


@lru_cache(maxsize=16)
def _calendar_dates_cached(data_dir: str) -> tuple:
    return tuple(pd.to_datetime(x).date() for x in build_calendar_from_daily_filenames(data_dir))


@lru_cache(maxsize=512)
def _daily_data_cached(data_dir: str, code: str) -> pd.DataFrame:
    return load_daily_data(data_dir, code)


@lru_cache(maxsize=512)
def _breakpoints_cached(data_dir: str, code: str) -> frozenset:
    return frozenset(_load_breakpoints(data_dir, code))


@lru_cache(maxsize=512)
def _build_label_context(data_dir: str, code: str) -> _LabelContext | None:
    calendar_dates = _calendar_dates_cached(data_dir)
    daily = _daily_data_cached(data_dir, code)
    if daily.empty:
        return None

    daily = daily.copy()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"]).dt.date
    suspended_dates: set = set()
    if "volume" in daily.columns:
        daily["volume"] = pd.to_numeric(daily["volume"], errors="coerce")
        suspended_dates = set(daily.loc[(daily["volume"] <= 0) | (~np.isfinite(daily["volume"])), "trade_date"].tolist())
        daily = daily[(daily["volume"] > 0) & np.isfinite(daily["volume"])]
    daily = daily.drop_duplicates(subset=["trade_date"], keep="last").sort_values("trade_date")
    by_date = daily.set_index("trade_date")

    trading_calendar_dates = tuple(d for d in calendar_dates if d not in suspended_dates)
    adj_open_np, adj_close_np, daily_log_ret, vol30_by_idx, date_to_pos = _precompute_label_arrays(list(trading_calendar_dates), by_date)

    valid_raw_indices: list[int] = []
    valid_raw_values: list[float] = []
    raw_details_by_index: dict[int, dict] = {}
    raw_fail_by_index: dict[int, str] = {}
    for idx in range(len(trading_calendar_dates)):
        ok_raw, y_raw_val, details, fail_reason = _compute_raw_label_for_idx(
            idx,
            trading_calendar_dates,
            by_date,
            adj_open_np,
            adj_close_np,
            daily_log_ret,
            vol30_by_idx,
            date_to_pos,
        )
        if ok_raw and y_raw_val is not None and np.isfinite(y_raw_val):
            valid_raw_indices.append(idx)
            valid_raw_values.append(float(y_raw_val))
            raw_details_by_index[idx] = details
        else:
            raw_fail_by_index[idx] = fail_reason or "raw label build failed"

    valid_raw_values_np = np.asarray(valid_raw_values, dtype=np.float64)
    if len(valid_raw_values_np) > 0:
        csum = np.concatenate(([0.0], np.cumsum(valid_raw_values_np)))
        csum2 = np.concatenate(([0.0], np.cumsum(valid_raw_values_np * valid_raw_values_np)))
        mu = np.full((len(valid_raw_values_np),), np.nan, dtype=np.float64)
        sd = np.full((len(valid_raw_values_np),), np.nan, dtype=np.float64)
        if len(valid_raw_values_np) > LABEL_Z_WINDOW:
            ends = np.arange(LABEL_Z_WINDOW, len(valid_raw_values_np), dtype=np.int64)
            sums = csum[ends] - csum[ends - LABEL_Z_WINDOW]
            sums2 = csum2[ends] - csum2[ends - LABEL_Z_WINDOW]
            m = sums / LABEL_Z_WINDOW
            var = np.maximum(sums2 / LABEL_Z_WINDOW - m * m, 0.0)
            mu[ends] = m
            sd[ends] = np.sqrt(var)
    else:
        mu = np.asarray([], dtype=np.float64)
        sd = np.asarray([], dtype=np.float64)

    asof_to_trading_idx = {d: i for i, d in enumerate(trading_calendar_dates)}
    raw_idx_to_valid_pos = {idx: pos for pos, idx in enumerate(valid_raw_indices)}
    label_ok_by_idx = np.zeros((len(trading_calendar_dates),), dtype=np.bool_)
    label_value_by_idx = np.zeros((len(trading_calendar_dates),), dtype=np.float32)
    for pos, idx in enumerate(valid_raw_indices):
        if pos < LABEL_Z_WINDOW:
            continue
        if idx not in raw_details_by_index:
            continue
        mu_pos = float(mu[pos])
        sd_pos = float(sd[pos])
        if (not np.isfinite(mu_pos)) or (not np.isfinite(sd_pos)) or sd_pos <= 0:
            continue
        y_raw_pos = float(valid_raw_values[pos])
        y_z_pos = (y_raw_pos - mu_pos) / max(sd_pos, ZSCORE_EPS)
        y_z_pos = float(TANH_K * np.tanh(y_z_pos / TANH_K))
        if not np.isfinite(y_z_pos):
            continue
        label_ok_by_idx[idx] = True
        label_value_by_idx[idx] = np.float32(y_z_pos)

    return _LabelContext(
        calendar_dates=calendar_dates,
        trading_calendar_dates=trading_calendar_dates,
        by_date=by_date,
        breakpoints=_breakpoints_cached(data_dir, code),
        valid_raw_indices=tuple(valid_raw_indices),
        valid_raw_values=tuple(valid_raw_values),
        raw_details_by_index=raw_details_by_index,
        raw_fail_by_index=raw_fail_by_index,
        z_mu_by_valid_pos=tuple(mu.tolist()),
        z_sd_by_valid_pos=tuple(sd.tolist()),
        asof_to_trading_idx=asof_to_trading_idx,
        raw_idx_to_valid_pos=raw_idx_to_valid_pos,
        label_ok_by_idx=tuple(bool(x) for x in label_ok_by_idx.tolist()),
        label_value_by_idx=tuple(float(x) for x in label_value_by_idx.tolist()),
    )

def build_label_for_sample(
    code: str,
    asof_date: str,
    calendar: list[str],
    daily_loader,
    dp_ok: bool = True,
    breakpoints: set | None = None,
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
    suspended_dates: set[pd.Timestamp | object] = set()
    if "volume" in daily.columns:
        daily["volume"] = pd.to_numeric(daily["volume"], errors="coerce")
        suspended_dates = set(daily.loc[(daily["volume"] <= 0) | (~np.isfinite(daily["volume"])), "trade_date"].tolist())
        daily = daily[(daily["volume"] > 0) & np.isfinite(daily["volume"])]
    daily = daily.drop_duplicates(subset=["trade_date"], keep="last").sort_values("trade_date")
    by_date = daily.set_index("trade_date")

    trading_calendar_dates = [d for d in calendar_dates if d not in suspended_dates]
    adj_open_np, adj_close_np, daily_log_ret, vol30_by_idx, date_to_pos = _precompute_label_arrays(trading_calendar_dates, by_date)
    if asof not in trading_calendar_dates:
        return _fail(code, asof_date, dp_ok, "asof date not in trading calendar")

    idx = trading_calendar_dates.index(asof)
    effective_breakpoints = breakpoints or set()
    window_start_idx = max(0, idx - LABEL_Z_WINDOW - VOL_WINDOW)
    if _has_breakpoint_crossing(trading_calendar_dates, window_start_idx, idx + 3, effective_breakpoints):
        return _fail(code, asof_date, dp_ok, "label window crosses st breakpoint")

    ok_raw, y_raw_val, details, fail_reason = _compute_raw_label_for_idx(
        idx,
        trading_calendar_dates,
        by_date,
        adj_open_np,
        adj_close_np,
        daily_log_ret,
        vol30_by_idx,
        date_to_pos,
    )
    if not ok_raw or y_raw_val is None:
        return _fail(code, asof_date, dp_ok, fail_reason or "raw label build failed")

    past_raw_values: list[float] = []
    for pidx in range(idx):
        ok_past, y_past, _, _ = _compute_raw_label_for_idx(
            pidx,
            trading_calendar_dates,
            by_date,
            adj_open_np,
            adj_close_np,
            daily_log_ret,
            vol30_by_idx,
            date_to_pos,
        )
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



@lru_cache(maxsize=512)
def get_label_valid_asof_dates(data_dir: str, code: str) -> tuple[str, ...]:
    context = _build_label_context(data_dir, code)
    if context is None:
        return tuple()

    trading_calendar_dates = context.trading_calendar_dates
    valid_indices = context.valid_raw_indices
    out: list[str] = []
    for pos, idx in enumerate(valid_indices):
        if pos < LABEL_Z_WINDOW:
            continue
        if idx not in context.raw_details_by_index:
            continue
        mu = float(context.z_mu_by_valid_pos[pos])
        sd = float(context.z_sd_by_valid_pos[pos])
        if (not np.isfinite(mu)) or (not np.isfinite(sd)) or sd <= 0:
            continue
        window_start_idx = max(0, idx - LABEL_Z_WINDOW - VOL_WINDOW)
        if _has_breakpoint_crossing(trading_calendar_dates, window_start_idx, idx + 3, set(context.breakpoints)):
            continue
        out.append(trading_calendar_dates[idx].isoformat())
    return tuple(out)


def clear_label_worker_cache() -> None:
    _calendar_dates_cached.cache_clear()
    _daily_data_cached.cache_clear()
    _breakpoints_cached.cache_clear()
    _build_label_context.cache_clear()
    get_label_valid_asof_dates.cache_clear()

def build_label_from_data_dir(
    data_dir: str | Path,
    code: str,
    asof_date: str,
    dp_ok: bool = True,
    breakpoints: set | None = None,
) -> LabelBundle:
    data_dir_key = str(Path(data_dir).resolve())
    context = _build_label_context(data_dir_key, code)
    if context is None:
        return _fail(code, asof_date, dp_ok, "daily data unavailable")

    asof = pd.to_datetime(asof_date).date()
    calendar_dates = context.calendar_dates
    if asof not in calendar_dates:
        return _fail(code, asof_date, dp_ok, "asof date not in calendar")

    trading_calendar_dates = context.trading_calendar_dates
    idx = context.asof_to_trading_idx.get(asof)
    if idx is None:
        return _fail(code, asof_date, dp_ok, "asof date not in trading calendar")

    effective_breakpoints = set(breakpoints) if breakpoints is not None else set(context.breakpoints)
    window_start_idx = max(0, idx - LABEL_Z_WINDOW - VOL_WINDOW)
    if _has_breakpoint_crossing(trading_calendar_dates, window_start_idx, idx + 3, effective_breakpoints):
        return _fail(code, asof_date, dp_ok, "label window crosses st breakpoint")

    if idx not in context.raw_details_by_index:
        return _fail(code, asof_date, dp_ok, context.raw_fail_by_index.get(idx, "raw label build failed"))

    valid_values = context.valid_raw_values
    pos = context.raw_idx_to_valid_pos.get(idx)
    if pos is None:
        return _fail(code, asof_date, dp_ok, context.raw_fail_by_index.get(idx, "raw label build failed"))
    if pos < LABEL_Z_WINDOW:
        return _fail(code, asof_date, dp_ok, f"insufficient y_raw history for label zscore: need {LABEL_Z_WINDOW}, got {pos}")

    if not context.label_ok_by_idx[idx]:
        mu = float(context.z_mu_by_valid_pos[pos])
        sd = float(context.z_sd_by_valid_pos[pos])
        if (not np.isfinite(mu)) or (not np.isfinite(sd)) or sd <= 0:
            return _fail(code, asof_date, dp_ok, "label zscore sd invalid (<=0 or non-finite)")
        return _fail(code, asof_date, dp_ok, "label zscore non-finite")

    y_raw_val = float(valid_values[pos])
    y_z = float(context.label_value_by_idx[idx])

    details = context.raw_details_by_index[idx]
    return LabelBundle(
        code=code,
        asof_date=asof_date,
        y=np.float32(y_z),
        y_raw=np.float32(y_raw_val),
        y_z=np.float32(y_z),
        label_ok=True,
        loss_mask=bool(dp_ok),
        entry_date=details["entry_date"],
        exit_date=details["exit_date"],
        entry_open=details["entry_open"],
        exit_close=details["exit_close"],
        vol30=details["vol30"],
        ret_log=details["ret_log"],
        fail_reason=None,
    )
