from pathlib import Path

import pandas as pd

from tools.preprocess_data import preprocess


def test_preprocess_outputs_per_code_layout_and_calendar(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    raw.mkdir()
    out.mkdir()

    pd.DataFrame(
        {
            "code": ["AAA", "AAA", "BBB"],
            "trade_date": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "time": ["09:30", "09:35", "09:30"],
            "open": [1.0, 1.1, 2.0],
            "high": [1.2, 1.3, 2.2],
            "low": [0.9, 1.0, 1.8],
            "close": [1.1, 1.2, 2.1],
            "volume": [100, 110, 200],
        }
    ).to_csv(raw / "bars.csv", index=False)

    pd.DataFrame(
        {
            "code": ["AAA", "BBB"],
            "trade_date": ["2024-01-02", "2024-01-03"],
            "open": [10.0, 20.0],
            "high": [11.0, 21.0],
            "low": [9.0, 19.0],
            "close": [10.5, 20.5],
            "volume": [1000, 2000],
            "adj_factor": [1.0, 1.1],
        }
    ).to_parquet(raw / "daily.parquet", index=False)

    preprocess(raw, out)

    assert (out / "AAA" / "5min.parquet").exists()
    assert (out / "AAA" / "daily.parquet").exists()
    assert (out / "BBB" / "5min.parquet").exists()
    assert (out / "BBB" / "daily.parquet").exists()
    assert (out / "calendar.parquet").exists()

    cal = pd.read_parquet(out / "calendar.parquet")
    assert sorted(cal["trade_date"].astype(str).tolist()) == ["2024-01-02", "2024-01-03"]
