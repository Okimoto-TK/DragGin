import pandas as pd

from src.feat.build_multiscale_tensor import aggregate_30m_from_5m


def test_30m_aggregation() -> None:
    base = pd.Timestamp("2024-01-02 09:30")
    rows = []
    for i in range(48):
        p = 100 + i
        rows.append(
            {
                "trade_date": pd.Timestamp("2024-01-02").date(),
                "dt": base + pd.Timedelta(minutes=5 * i),
                "open": p,
                "high": p + 2,
                "low": p - 1,
                "close": p + 1,
                "volume": 10,
            }
        )
    df = pd.DataFrame(rows)
    out = aggregate_30m_from_5m(df)
    assert len(out) == 8
    first = out.iloc[0]
    assert first["open"] == 100
    assert first["close"] == 106
    assert first["high"] == 107
    assert first["low"] == 99
    assert first["volume"] == 60
