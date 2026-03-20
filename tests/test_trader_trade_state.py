import json
from pathlib import Path

import pandas as pd

from src.infer import load_latest_position_state, load_st_flags


def test_load_st_flags_accepts_update_daily_snapshot_names(tmp_path: Path) -> None:
    pd.DataFrame({"ts_code": ["000001.SZ"]}).to_parquet(tmp_path / "20240102.parquet", index=False)

    got = load_st_flags(tmp_path)

    assert got == {"20240102": {"000001.SZ"}}


def test_load_latest_position_state_uses_latest_json_and_adj_factor(tmp_path: Path) -> None:
    position_dir = tmp_path / "position"
    position_dir.mkdir()
    (position_dir / "20240102.json").write_text(
        json.dumps(
            {
                "date": "2024-01-02",
                "final_cash": 12345.0,
                "final_positions": [{"code": "000001.SZ", "qty": 200.0, "cost": 10.5}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    daily_cache = {"000001.SZ": {"2024-01-02": {"adj_factor": 1.23}}}

    record_date, cash, positions = load_latest_position_state(position_dir, daily_cache)

    assert record_date == "2024-01-02"
    assert cash == 12345.0
    assert positions["000001.SZ"].qty == 200.0
    assert positions["000001.SZ"].cost_per_share == 10.5
    assert positions["000001.SZ"].last_adj_factor == 1.23
