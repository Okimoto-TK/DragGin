from __future__ import annotations

import numpy as np
import pandas as pd

from src.feat.labels_risk_adj import build_label_for_sample


def _base_df(calendar: list[str]) -> pd.DataFrame:
    n = len(calendar)
    close = 200.0 + np.arange(n) + 0.3 * np.cos(np.arange(n))
    return pd.DataFrame(
        {
            "trade_date": calendar,
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1000.0),
            "adj_factor": 1.0 + 0.001 * np.arange(n),
        }
    )


def test_label_ok_false_when_future_prices_missing() -> None:
    calendar = [d.date().isoformat() for d in pd.date_range("2023-01-01", periods=320, freq="D")]
    asof_idx = 290
    asof = calendar[asof_idx]

    full = _base_df(calendar)

    missing_entry = full[full["trade_date"] != calendar[asof_idx + 1]].reset_index(drop=True)
    res_entry = build_label_for_sample("AAA", asof, calendar, lambda _: missing_entry)
    assert not res_entry.label_ok
    assert res_entry.fail_reason is not None

    missing_exit = full[full["trade_date"] != calendar[asof_idx + 3]].reset_index(drop=True)
    res_exit = build_label_for_sample("AAA", asof, calendar, lambda _: missing_exit)
    assert not res_exit.label_ok
    assert res_exit.fail_reason is not None
