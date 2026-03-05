from pathlib import Path

import pandas as pd

from src.feat.build_multiscale_tensor import build_calendar_from_daily_filenames


def test_calendar_from_filenames(tmp_path: Path) -> None:
    aaa = tmp_path / "AAA"
    bbb = tmp_path / "BBB"
    aaa.mkdir()
    bbb.mkdir()

    pd.DataFrame({"trade_date": ["2024-01-02", "2024-01-03"]}).to_parquet(aaa / "daily.parquet", index=False)
    pd.DataFrame({"trade_date": ["2024-01-01"]}).to_parquet(bbb / "daily.parquet", index=False)

    calendar = build_calendar_from_daily_filenames(tmp_path)
    assert calendar == ["2024-01-01", "2024-01-02", "2024-01-03"]
