from __future__ import annotations

import numpy as np
import pandas as pd

from src.feat.labels_risk_adj import build_label_for_sample


def test_label_alignment_open_t1_close_t3() -> None:
    calendar = pd.date_range("2024-01-01", periods=40, freq="D").date
    calendar_str = [d.isoformat() for d in calendar]
    df = pd.DataFrame(
        {
            "trade_date": calendar_str,
            "open": np.linspace(100.0, 139.0, 40),
            "high": np.linspace(101.0, 140.0, 40),
            "low": np.linspace(99.0, 138.0, 40),
            "close": np.linspace(100.5, 140.5, 40),
            "volume": np.full(40, 1000.0),
            "vwap": np.linspace(100.2, 140.2, 40),
        }
    )

    asof_idx = 34
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

    expected_ret = np.log(
        float(df.loc[df["trade_date"] == result.exit_date, "close"].iloc[0])
        / float(df.loc[df["trade_date"] == result.entry_date, "open"].iloc[0])
    )
    assert np.isclose(result.ret_log, expected_ret)
