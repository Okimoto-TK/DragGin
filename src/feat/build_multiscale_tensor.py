from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
import os
import re
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple


DATE_PATTERN = re.compile(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])")

PAD_VALUE = 0.0
FEATURE_COUNT = 7
PRE_ROLL_BARS = 20


@dataclass(frozen=True)
class ScaleSpec:
    name: str
    interval: str
    window_days: int
    length: int
    bars_per_day: int


SCALE_SPECS = {
    "micro": ScaleSpec("micro", "5m", 1, 48, 48),
    "mezzo": ScaleSpec("mezzo", "30m", 5, 40, 8),
    "macro": ScaleSpec("macro", "1d", 30, 30, 1),
}


class BarProvider(Protocol):
    def get_bars(self, code: str, interval: str, end_date: date, total_bars: int) -> Sequence[Mapping[str, float]]:
        """Return up to `total_bars` bars ending at `end_date` (inclusive), chronological order."""


def build_trading_calendar(data_dir: str) -> Tuple[List[date], Dict[date, int]]:
    dates = set()
    for filename in os.listdir(data_dir):
        match = DATE_PATTERN.search(filename)
        if not match:
            continue
        y, m, d = match.groups()
        dates.add(date(int(y), int(m), int(d)))

    trading_calendar = sorted(dates)
    calendar_index = {d: i for i, d in enumerate(trading_calendar)}
    return trading_calendar, calendar_index


def _zero_tensor(length: int) -> List[List[float]]:
    return [[PAD_VALUE] * FEATURE_COUNT for _ in range(length)]


def _zero_mask(length: int) -> List[int]:
    return [0] * length


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _extract_window_dates(asof_date: date, calendar: Sequence[date], window_days: int) -> Optional[List[date]]:
    idx_map = {d: i for i, d in enumerate(calendar)}
    if asof_date not in idx_map:
        return None
    end_idx = idx_map[asof_date]
    start_idx = end_idx - (window_days - 1)
    if start_idx < 0:
        return None
    return list(calendar[start_idx : end_idx + 1])


def _validate_final_window_dates(final_window: Sequence[Mapping[str, float]], expected_dates: Sequence[date], bars_per_day: int) -> bool:
    if bars_per_day == 1:
        return [bar.get("date") for bar in final_window] == list(expected_dates)

    expected = []
    for d in expected_dates:
        expected.extend([d] * bars_per_day)
    actual = [bar.get("date") for bar in final_window]
    return actual == expected


def _compute_features(bars: Sequence[Mapping[str, float]], length: int) -> Optional[List[List[float]]]:
    # bars length should be PRE_ROLL_BARS + length
    start = len(bars) - length
    out: List[List[float]] = []

    for i in range(start, len(bars)):
        bar = bars[i]
        required = ["open", "high", "low", "close", "volume", "vwap"]
        if not all(_is_finite_number(bar.get(k)) for k in required):
            return None

        o = float(bar["open"])
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
        v = float(bar["volume"])
        vw = float(bar["vwap"])

        if i - 1 < 0:
            return None
        prev_close = bars[i - 1].get("close")
        if not _is_finite_number(prev_close):
            return None
        prev_close = float(prev_close)

        if prev_close <= 0 or o <= 0 or h <= 0 or l <= 0 or c <= 0:
            return None

        if h == l:
            c4 = 0.5
        else:
            c4 = (c - l) / (h - l)

        vol_hist = [bars[j].get("volume") for j in range(i - 20, i)]
        if len(vol_hist) != 20 or any(not _is_finite_number(x) for x in vol_hist):
            return None
        vol_hist = [float(x) for x in vol_hist]
        vol_mean = sum(vol_hist) / 20.0
        if vol_mean <= 0:
            return None

        returns_hist: List[float] = []
        for j in range(i - 5, i):
            close_j = bars[j].get("close")
            close_prev = bars[j - 1].get("close") if j - 1 >= 0 else None
            if not _is_finite_number(close_j) or not _is_finite_number(close_prev):
                return None
            close_j = float(close_j)
            close_prev = float(close_prev)
            if close_j <= 0 or close_prev <= 0:
                return None
            returns_hist.append(math.log(close_j / close_prev))
        if len(returns_hist) != 5:
            return None
        mean_ret = sum(returns_hist) / 5.0
        c6 = math.sqrt(sum((x - mean_ret) ** 2 for x in returns_hist) / 5.0)

        c1 = math.log(c / prev_close)
        c2 = math.log(o / prev_close)
        c3 = math.log(h / l)
        c5 = v / vol_mean
        c7 = vw / c

        feats = [c1, c2, c3, c4, c5, c6, c7]
        if not all(math.isfinite(x) for x in feats):
            return None
        out.append(feats)

    if len(out) != length:
        return None
    return out


def _build_scale_tensor(code: str, asof_date: date, provider: BarProvider, calendar: Sequence[date], spec: ScaleSpec) -> Optional[Tuple[List[List[float]], List[int]]]:
    window_dates = _extract_window_dates(asof_date, calendar, spec.window_days)
    if window_dates is None:
        return None

    total_bars = spec.length + PRE_ROLL_BARS
    bars = provider.get_bars(code=code, interval=spec.interval, end_date=asof_date, total_bars=total_bars)
    if len(bars) != total_bars:
        return None

    final_window = bars[-spec.length :]
    if not _validate_final_window_dates(final_window, window_dates, spec.bars_per_day):
        return None

    features = _compute_features(bars, spec.length)
    if features is None:
        return None

    return features, [1] * spec.length


def build_multiscale_tensor(code: str, asof_date: date, provider: BarProvider, calendar: Sequence[date]) -> Dict[str, object]:
    micro0 = _zero_tensor(SCALE_SPECS["micro"].length)
    mezzo0 = _zero_tensor(SCALE_SPECS["mezzo"].length)
    macro0 = _zero_tensor(SCALE_SPECS["macro"].length)
    out = {
        "X_micro": micro0,
        "mask_micro": _zero_mask(SCALE_SPECS["micro"].length),
        "X_mezzo": mezzo0,
        "mask_mezzo": _zero_mask(SCALE_SPECS["mezzo"].length),
        "X_macro": macro0,
        "mask_macro": _zero_mask(SCALE_SPECS["macro"].length),
        "dp_ok": False,
        "meta": {"code": code, "asof_date": asof_date},
    }

    if not calendar:
        return out

    scale_results = {}
    for scale_name, spec in SCALE_SPECS.items():
        tensor = _build_scale_tensor(code, asof_date, provider, calendar, spec)
        if tensor is None:
            return out
        scale_results[scale_name] = tensor

    out["X_micro"], out["mask_micro"] = scale_results["micro"]
    out["X_mezzo"], out["mask_mezzo"] = scale_results["mezzo"]
    out["X_macro"], out["mask_macro"] = scale_results["macro"]
    out["dp_ok"] = True
    return out
