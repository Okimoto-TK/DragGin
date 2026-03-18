from pathlib import Path

import pandas as pd

from tools import fetch_namechange


def test_iter_ts_codes_from_data_dir_uses_folder_names(tmp_path: Path) -> None:
    (tmp_path / "600848.SH").mkdir()
    (tmp_path / "000001.SZ").mkdir()
    (tmp_path / ".cache").mkdir()
    (tmp_path / "calendar.parquet").write_text("x", encoding="utf-8")

    assert fetch_namechange._iter_ts_codes_from_data_dir(tmp_path) == ["000001.SZ", "600848.SH"]


def test_fetch_namechange_from_data_dir_writes_combined_parquet(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "600848.SH").mkdir()
    (data_dir / "000001.SZ").mkdir()
    out_file = tmp_path / "namechange.parquet"

    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")

    class FakePro:
        def namechange(self, ts_code: str, fields: str) -> pd.DataFrame:
            assert fields == "ts_code,name,start_date,end_date"
            if ts_code == "000001.SZ":
                return pd.DataFrame(
                    {
                        "ts_code": ["000001.SZ"],
                        "name": ["平安银行"],
                        "start_date": ["19910403"],
                        "end_date": [None],
                    }
                )
            return pd.DataFrame(
                {
                    "ts_code": ["600848.SH", "600848.SH"],
                    "name": ["自仪股份", "上海临港"],
                    "start_date": ["19940324", "20151118"],
                    "end_date": ["20151117", None],
                }
            )

    monkeypatch.setattr(fetch_namechange, "_get_pro_client", lambda token: FakePro())

    result = fetch_namechange.fetch_namechange_from_data_dir(data_dir=data_dir, out_file=out_file, max_workers=2)

    assert out_file.exists()
    saved = pd.read_parquet(out_file)
    assert saved.equals(result)
    assert saved["ts_code"].tolist() == ["000001.SZ", "600848.SH", "600848.SH"]
    assert saved["name"].tolist() == ["平安银行", "自仪股份", "上海临港"]
