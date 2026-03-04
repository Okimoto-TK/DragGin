from __future__ import annotations

import numpy as np
import pandas as pd

from src.feat.labels_risk_adj import build_label_for_sample


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
        }
    )


def test_vol30_strict_full_window() -> None:
    calendar = pd.date_range("2024-01-01", periods=34, freq="D").date
    calendar_str = [d.isoformat() for d in calendar]
    asof = calendar[30].isoformat()

    ok_df = _daily_df(calendar_str)
    ok_result = build_label_for_sample("AAA", asof, calendar_str, lambda _: ok_df)
    assert ok_result.label_ok
    assert ok_result.vol30 is not None and ok_result.vol30 > 0

    bad_df = ok_df[ok_df["trade_date"] != calendar[0].isoformat()].reset_index(drop=True)
    bad_result = build_label_for_sample("AAA", asof, calendar_str, lambda _: bad_df)
    assert not bad_result.label_ok
    assert bad_result.y == np.float32(0.0)
