from pathlib import Path

import pandas as pd

from tools import preprocess_st_breakpoints as st_script


def test_build_st_breakpoints_from_calendar(monkeypatch, tmp_path: Path) -> None:
    pd.DataFrame({"trade_date": ["2024-01-02", "2024-01-03"]}).to_parquet(tmp_path / "calendar.parquet", index=False)

    called = {}

    def _fake_fetch(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        called["start"] = start_date
        called["end"] = end_date
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000001.SZ"],
                "name": ["平安银行", "ST平安银行"],
                "start_date": ["20240101", "20240103"],
                "end_date": ["", ""],
            }
        )

    monkeypatch.setattr(st_script, "_fetch_namechange", _fake_fetch)

    n = st_script.build_st_breakpoints(tmp_path)

    assert n == 1
    assert called["start"].date().isoformat() == "2024-01-02"
    assert called["end"].date().isoformat() == "2024-01-03"

    bp = pd.read_parquet(tmp_path / "000001.SZ" / "breakpoints.parquet")
    assert bp["break_date"].dt.date.iloc[0].isoformat() == "2024-01-03"
