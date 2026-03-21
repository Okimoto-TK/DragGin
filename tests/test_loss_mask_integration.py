from __future__ import annotations

import numpy as np
import pandas as pd

from src.feat.labels_risk_adj import build_label_for_sample


def _limit_df(calendar: list[str], limit_pct: float = 0.1) -> pd.DataFrame:
    return pd.DataFrame({
        "trade_date": calendar,
        "up_limit": np.full(len(calendar), 110.0),
        "down_limit": np.full(len(calendar), 90.0),
        "limit_pct": np.full(len(calendar), limit_pct),
    })


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
            "adj_factor": 1.0 + 0.001 * np.arange(n),
        }
    )


def test_loss_mask_integration_rule() -> None:
    calendar = [d.date().isoformat() for d in pd.date_range("2023-01-01", periods=320, freq="D")]
    asof = calendar[290]
    df = _make_df(calendar)
    limit_df = _limit_df(calendar)

    label_ok_dp_false = build_label_for_sample("AAA", asof, calendar, lambda _: df, limit_loader=lambda _: limit_df, dp_ok=False)
    assert label_ok_dp_false.label_ok
    assert not label_ok_dp_false.loss_mask

    label_bad_dp_true = build_label_for_sample("AAA", calendar[30], calendar, lambda _: df, limit_loader=lambda _: limit_df, dp_ok=True)
    assert not label_bad_dp_true.label_ok
    assert not label_bad_dp_true.loss_mask

    label_ok_dp_true = build_label_for_sample("AAA", asof, calendar, lambda _: df, limit_loader=lambda _: limit_df, dp_ok=True)
    assert label_ok_dp_true.label_ok
    assert label_ok_dp_true.loss_mask
