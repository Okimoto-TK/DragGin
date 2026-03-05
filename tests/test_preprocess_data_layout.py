from pathlib import Path

import pandas as pd

from tools import preprocess_data
from tools.preprocess_data import preprocess


def test_preprocess_outputs_per_code_layout_and_calendar(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    raw.mkdir()
    out.mkdir()

    pd.DataFrame(
        {
            "code": ["AAA", "AAA", "BBB"],
            "trade_time": ["09:30", "09:35", "09:35"],
            "close": [1.1, 1.2, 2.1],
            "open": [1.0, 1.1, 2.0],
            "high": [1.2, 1.3, 2.2],
            "low": [0.9, 1.0, 1.8],
            "vol": [100, 110, 200],
            "amount": [1000, 1100, 2000],
            "date": ["20240102", "20240102", "20240103"],
            "pre_close": [1.0, 1.1, 2.0],
            "change": [0.1, 0.1, 0.1],
            "pct_chg": [10.0, 9.1, 5.0],
        }
    ).to_csv(raw / "bars.csv", index=False)

    pd.DataFrame(
        {
            "code": ["AAA", "BBB"],
            "date": ["20240102", "20240103"],
            "adj_factor": [1.0, 1.1],
            "open": [10.0, 20.0],
            "high": [11.0, 21.0],
            "low": [9.0, 19.0],
            "close": [10.5, 20.5],
            "pre_close": [10.0, 20.0],
            "pct_chg": [5.0, 2.5],
        }
    ).to_parquet(raw / "daily.parquet", index=False)

    preprocess(raw, out)

    assert (out / "AAA" / "5min.parquet").exists()
    assert (out / "AAA" / "daily.parquet").exists()
    assert (out / "BBB" / "5min.parquet").exists()
    assert (out / "BBB" / "daily.parquet").exists()
    assert (out / "calendar.parquet").exists()

    m5 = pd.read_parquet(out / "AAA" / "5min.parquet")
    assert {"trade_date", "time", "open", "high", "low", "close", "volume", "dt"}.issubset(m5.columns)
    assert (m5["time"] != "09:30").all()

    d1 = pd.read_parquet(out / "AAA" / "daily.parquet")
    assert {"trade_date", "open", "high", "low", "close", "volume", "adj_factor"}.issubset(d1.columns)
    assert (d1["volume"] > 0).all()

    cal = pd.read_parquet(out / "calendar.parquet")
    assert sorted(cal["trade_date"].astype(str).tolist()) == ["2024-01-02", "2024-01-03"]


def test_preprocess_writes_st_parquet_from_namechange(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    raw.mkdir()
    out.mkdir()

    pd.DataFrame(
        {
            "code": ["AAA"],
            "trade_time": ["09:35"],
            "close": [1.2],
            "open": [1.1],
            "high": [1.3],
            "low": [1.0],
            "vol": [110],
            "date": ["20240102"],
        }
    ).to_csv(raw / "bars.csv", index=False)

    pd.DataFrame(
        {
            "code": ["AAA", "AAA"],
            "date": ["20240102", "20240103"],
            "adj_factor": [1.0, 1.0],
            "open": [10.0, 10.0],
            "high": [11.0, 11.0],
            "low": [9.0, 9.0],
            "close": [10.5, 10.5],
            "pct_chg": [5.0, 2.5],
        }
    ).to_parquet(raw / "daily.parquet", index=False)

    def fake_fetch(start_date, end_date):
        assert start_date.strftime("%Y%m%d") == "20240102"
        assert end_date.strftime("%Y%m%d") == "20240103"
        return pd.DataFrame(
            {
                "ts_code": ["AAA", "AAA", "BBB"],
                "name": ["AAA*ST", "AAA", "BBBST"],
                "start_date": ["20240102", "20240201", "20240103"],
                "end_date": ["20240110", None, "20240105"],
            }
        )

    monkeypatch.setattr(preprocess_data, "_fetch_namechange", fake_fetch)

    preprocess(raw, out)

    st = pd.read_parquet(out / "AAA" / "st.parquet")
    assert st["st_type"].tolist() == ["*ST"]
    assert st["start_date"].astype(str).tolist() == ["2024-01-02"]
    assert st["revoke_st_date"].astype(str).tolist() == ["2024-01-10"]
    assert not (out / "BBB" / "st.parquet").exists()
