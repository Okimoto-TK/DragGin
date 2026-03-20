import json
from pathlib import Path

import pandas as pd

from src.infer import load_latest_position_state, load_st_flags
from tools.trader import _canonicalize_update_daily_5min, _canonicalize_update_daily_daily, _canonicalize_update_daily_moneyflow


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


def test_canonicalize_update_daily_daily_renames_ts_code_and_vol() -> None:
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20240102"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.5],
            "close": [10.2],
            "adj_factor": [1.1],
            "vol": [1000.0],
        }
    )

    got = _canonicalize_update_daily_daily(df)

    assert got.loc[0, "code"] == "000001.SZ"
    assert float(got.loc[0, "volume"]) == 1000.0


def test_canonicalize_update_daily_5min_uses_ts_code_and_alias_columns() -> None:
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_time": ["2024-01-02 09:35:00"],
            "o": [10.0],
            "h": [10.2],
            "l": [9.9],
            "c": [10.1],
            "v": [5000.0],
        }
    )

    got = _canonicalize_update_daily_5min(df, code_hint="000001.SZ")

    assert got.loc[0, "code"] == "000001.SZ"
    assert got.loc[0, "time"] == "09:35"
    assert float(got.loc[0, "volume"]) == 5000.0


def test_canonicalize_update_daily_moneyflow_renames_code_column() -> None:
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20240102"],
            "net_mf_vol": [1.0],
            "buy_lg_vol": [2.0],
            "sell_lg_vol": [1.0],
            "buy_elg_vol": [1.5],
            "sell_elg_vol": [0.5],
        }
    )

    got = _canonicalize_update_daily_moneyflow(df)

    assert got.loc[0, "code"] == "000001.SZ"
    assert float(got.loc[0, "net_mf_vol"]) == 1.0
