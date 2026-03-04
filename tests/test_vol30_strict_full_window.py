from __future__ import annotations

import numpy as np
import pandas as pd

from src.feat.labels_risk_adj import LABEL_Z_WINDOW, TANH_K, build_label_for_sample


def _daily_df(dates: list[str]) -> pd.DataFrame:
    n = len(dates)
    close = 100.0 + np.arange(n) + 0.2 * np.sin(np.arange(n))
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1000.0),
            "vwap": close - 0.1,
            "adj_factor": 1.0 + 0.001 * np.arange(n),
        }
    )


def test_label_zscore_strict_full_window() -> None:
    calendar = pd.date_range("2023-01-01", periods=320, freq="D").date
    calendar_str = [d.isoformat() for d in calendar]

    df = _daily_df(calendar_str)

    asof_insufficient = calendar[281].isoformat()
    bad_result = build_label_for_sample("AAA", asof_insufficient, calendar_str, lambda _: df)
    assert not bad_result.label_ok
    assert "label zscore" in (bad_result.fail_reason or "")

    asof_ok = calendar[282].isoformat()
    ok_result = build_label_for_sample("AAA", asof_ok, calendar_str, lambda _: df)
    assert ok_result.label_ok
    assert np.isfinite(ok_result.y_z)
    assert abs(float(ok_result.y_z)) <= TANH_K + 1e-4

    flat_df = df.copy()
    flat_df["open"] = 100.0
    flat_df["high"] = 100.0
    flat_df["low"] = 100.0
    flat_df["close"] = 100.0
    flat_df["adj_factor"] = 1.0
    zero_sd_result = build_label_for_sample("AAA", asof_ok, calendar_str, lambda _: flat_df)
    assert not zero_sd_result.label_ok
    assert zero_sd_result.y == np.float32(0.0)
    assert LABEL_Z_WINDOW == 252
