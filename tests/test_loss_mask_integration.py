from __future__ import annotations

import numpy as np
import pandas as pd

from src.feat.labels_risk_adj import build_label_for_sample


def _make_df(calendar: list[str]) -> pd.DataFrame:
    n = len(calendar)
    close = 300.0 + np.arange(n) + 0.4 * np.sin(np.arange(n) / 2)
    return pd.DataFrame(
        {
            "trade_date": calendar,
            "open": close - 0.2,
            "high": close + 0.7,
            "low": close - 0.7,
            "close": close,
            "volume": np.full(n, 1000.0),
            "vwap": close,
        }
    )


def test_loss_mask_integration_rule() -> None:
    calendar = [d.date().isoformat() for d in pd.date_range("2024-01-01", periods=40, freq="D")]
    asof = calendar[34]
    df = _make_df(calendar)

    label_ok_dp_false = build_label_for_sample("AAA", asof, calendar, lambda _: df, dp_ok=False)
    assert label_ok_dp_false.label_ok
    assert not label_ok_dp_false.loss_mask

    label_bad_dp_true = build_label_for_sample("AAA", calendar[-2], calendar, lambda _: df, dp_ok=True)
    assert not label_bad_dp_true.label_ok
    assert not label_bad_dp_true.loss_mask

    label_ok_dp_true = build_label_for_sample("AAA", asof, calendar, lambda _: df, dp_ok=True)
    assert label_ok_dp_true.label_ok
    assert label_ok_dp_true.loss_mask
