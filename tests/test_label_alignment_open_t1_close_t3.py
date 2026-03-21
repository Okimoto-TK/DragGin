from __future__ import annotations

import numpy as np
import pandas as pd

from src.feat.labels_risk_adj import TANH_K, LIMIT_Y_K, build_label_for_sample


def _limit_df(calendar: list[str], limit_pct: float = 0.1) -> pd.DataFrame:
    return pd.DataFrame({
        "trade_date": calendar,
        "up_limit": np.full(len(calendar), 110.0),
        "down_limit": np.full(len(calendar), 90.0),
        "limit_pct": np.full(len(calendar), limit_pct),
    })


def test_label_alignment_open_t1_close_t3() -> None:
    calendar = pd.date_range("2023-01-01", periods=320, freq="D").date
    calendar_str = [d.isoformat() for d in calendar]
    n = len(calendar_str)
    close = np.linspace(100.5, 420.5, 320)
    df = pd.DataFrame(
        {
            "trade_date": calendar_str,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(320, 1000.0),
            "adj_factor": 1.0 + 0.001 * np.arange(n),
        }
    )

    asof_idx = 290
    asof = calendar[asof_idx].isoformat()

    limit_df = _limit_df(calendar_str)
    result = build_label_for_sample(
        code="AAA",
        asof_date=asof,
        calendar=calendar_str,
        daily_loader=lambda _: df,
        limit_loader=lambda _: limit_df,
        dp_ok=True,
    )

    assert result.label_ok
    assert result.entry_date == calendar[asof_idx + 1].isoformat()
    assert result.exit_date == calendar[asof_idx + 3].isoformat()

    asof_date = calendar[asof_idx].isoformat()
    f_entry = float(df.loc[df["trade_date"] == result.entry_date, "adj_factor"].iloc[0])
    f_exit = float(df.loc[df["trade_date"] == result.exit_date, "adj_factor"].iloc[0])
    entry_adj = float(df.loc[df["trade_date"] == result.entry_date, "open"].iloc[0]) * f_entry
    exit_adj = float(df.loc[df["trade_date"] == result.exit_date, "open"].iloc[0]) * f_exit
    expected_ret = exit_adj / entry_adj - 1.0
    expected_y = np.tanh(LIMIT_Y_K * (expected_ret / (0.1 * 3.0 + 1e-6)))
    assert np.isclose(result.ret_log, expected_ret)
    assert np.isclose(float(result.limit_pct), 0.1)
    assert result.y == result.y_z
    assert np.isclose(float(result.y), expected_y)
    assert abs(float(result.y_z)) <= 1.0 + 1e-4
