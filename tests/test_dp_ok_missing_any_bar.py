from pathlib import Path

from src.feat.build_multiscale_tensor import build_multiscale_tensors
from tests.test_strict_c5_c6_warmup import _write_data


def test_dp_ok_missing_any_bar(tmp_path: Path) -> None:
    code = "AAA"

    bad_5m = tmp_path / "bad5m"
    bad_5m.mkdir()
    asof = _write_data(bad_5m, code=code, days=60, missing_5m=True)
    res_5m = build_multiscale_tensors(bad_5m, code, asof)
    assert not res_5m.dp_ok

    bad_daily = tmp_path / "badd1"
    bad_daily.mkdir()
    asof2 = _write_data(bad_daily, code=code, days=60, missing_daily_for_code_date="2024-02-20")
    res_daily = build_multiscale_tensors(bad_daily, code, asof2)
    assert not res_daily.dp_ok
