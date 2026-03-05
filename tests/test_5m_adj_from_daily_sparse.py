from pathlib import Path

import pandas as pd

from src.feat.build_multiscale_tensor import build_multiscale_tensors
from tests.test_strict_c5_c6_warmup import _write_data


def test_5m_adjustment_uses_daily_factors_only_on_overlap(tmp_path: Path) -> None:
    code = "AAA"
    asof = _write_data(tmp_path, code=code, days=130)

    code_dir = tmp_path / code
    d1 = pd.read_parquet(code_dir / "daily.parquet")
    min_5m_date = pd.read_parquet(code_dir / "5min.parquet")["trade_date"].min()
    d1 = d1[pd.to_datetime(d1["trade_date"]).dt.date >= pd.to_datetime(min_5m_date).date()].reset_index(drop=True)
    d1.to_parquet(code_dir / "daily.parquet", index=False)

    res = build_multiscale_tensors(tmp_path, code, asof)
    assert res.dp_ok, res.reason
