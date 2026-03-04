from __future__ import annotations

import numpy as np
import pandas as pd

from src.feat.labels_risk_adj import TANH_K, build_label_for_sample


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
            "vwap": close - 0.1,
            "adj_factor": 1.0 + 0.001 * np.arange(n),
        }
    )

    asof_idx = 290
    asof = calendar[asof_idx].isoformat()

    result = build_label_for_sample(
        code="AAA",
        asof_date=asof,
        calendar=calendar_str,
        daily_loader=lambda _: df,
        dp_ok=True,
    )

    assert result.label_ok
    assert result.entry_date == calendar[asof_idx + 1].isoformat()
    assert result.exit_date == calendar[asof_idx + 3].isoformat()

    asof_date = calendar[asof_idx].isoformat()
    f_asof = float(df.loc[df["trade_date"] == asof_date, "adj_factor"].iloc[0])
    f_entry = float(df.loc[df["trade_date"] == result.entry_date, "adj_factor"].iloc[0])
    f_exit = float(df.loc[df["trade_date"] == result.exit_date, "adj_factor"].iloc[0])
    entry_adj = float(df.loc[df["trade_date"] == result.entry_date, "open"].iloc[0]) * (f_entry / f_asof)
    exit_adj = float(df.loc[df["trade_date"] == result.exit_date, "close"].iloc[0]) * (f_exit / f_asof)
    expected_ret = np.log(exit_adj / entry_adj)
    assert np.isclose(result.ret_log, expected_ret)
    assert result.y == result.y_z
    assert abs(float(result.y_z)) <= TANH_K + 1e-4
