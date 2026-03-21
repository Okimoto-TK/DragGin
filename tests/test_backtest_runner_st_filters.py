import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools import backtest_runner


def _write_code_data(root: Path, code: str, dates: list[str], opens: list[float], closes: list[float]) -> None:
    code_dir = root / code
    code_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.DataFrame(
        {
            "trade_date": dates,
            "open": opens,
            "high": [x + 1.0 for x in opens],
            "low": [max(0.01, x - 1.0) for x in opens],
            "close": closes,
            "adj_factor": [1.0] * len(dates),
        }
    )
    intraday = pd.DataFrame(
        {
            "code": [code] * len(dates),
            "trade_date": dates,
            "time": ["09:30"] * len(dates),
            "open": opens,
            "high": [x + 0.1 for x in opens],
            "low": [max(0.01, x - 0.1) for x in opens],
            "close": closes,
            "volume": [1000] * len(dates),
        }
    )
    moneyflow = pd.DataFrame(
        {
            "trade_date": dates,
            "net_mf_vol": [1.0] * len(dates),
            "buy_lg_vol": [1.0] * len(dates),
            "sell_lg_vol": [1.0] * len(dates),
            "buy_elg_vol": [1.0] * len(dates),
            "sell_elg_vol": [1.0] * len(dates),
        }
    )
    daily.to_parquet(code_dir / "daily.parquet", index=False)
    intraday.to_parquet(code_dir / "5min.parquet", index=False)
    moneyflow.to_parquet(code_dir / "moneyflow.parquet", index=False)


def _write_common_files(root: Path, dates: list[str]) -> tuple[Path, Path, Path, Path]:
    data_dir = root / "data"
    score_dir = root / "scores"
    out_dir = root / "out"
    st_dir = root / "st"
    data_dir.mkdir()
    score_dir.mkdir()
    out_dir.mkdir()
    st_dir.mkdir()
    pd.DataFrame({"trade_date": dates}).to_parquet(data_dir / "calendar.parquet", index=False)
    np.save(root / "val_shard.npy", {"asof_dates": np.array(dates, dtype=object)}, allow_pickle=True)
    return data_dir, score_dir, out_dir, st_dir


def test_backtest_runner_skips_st_and_board_filtered_buys(tmp_path: Path, monkeypatch) -> None:
    dates = [f"2024-01-0{i}" for i in range(1, 8)]
    data_dir, score_dir, out_dir, st_dir = _write_common_files(tmp_path, dates)

    _write_code_data(data_dir, "000001.SZ", dates, [10.0] * 7, [10.0] * 7)
    _write_code_data(data_dir, "920001.SZ", dates, [10.0] * 7, [10.0] * 7)
    _write_code_data(data_dir, "920001.SH", dates, [10.0] * 7, [10.0] * 7)

    pd.DataFrame(
        {
            "code": ["000001.SZ", "920001.SZ", "920001.SH"],
            "asof_date": ["2024-01-01"] * 3,
            "yhat": [0.8, 0.99, 0.98],
        }
    ).to_parquet(score_dir / "scores.parquet", index=False)

    pd.DataFrame({"ts_code": ["000001.SZ"]}).to_parquet(st_dir / "20240102_stock_st.parquet", index=False)

    monkeypatch.setattr(backtest_runner.ts, "pro_api", lambda token: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "backtest_runner.py",
            "--data-dir",
            str(data_dir),
            "--score-dir",
            str(score_dir),
            "--out-dir",
            str(out_dir),
            "--val-shards",
            str(tmp_path / "val_shard.npy"),
            "--topk",
            "2",
            "--st-dir",
            str(st_dir),
        ],
    )

    backtest_runner.main()

    payload = json.loads((out_dir / "20240102.json").read_text(encoding="utf-8"))
    assert payload["buy_records"] == []
    assert payload["final_positions"] == []


class _FakePro:
    def stk_limit(self, trade_date: str) -> pd.DataFrame:
        if trade_date == "20240103":
            return pd.DataFrame({"ts_code": ["000001.SZ"], "up_limit": [11.0], "down_limit": [8.0]})
        return pd.DataFrame(columns=["ts_code", "up_limit", "down_limit"])


def test_backtest_runner_sells_existing_st_holdings_at_open(tmp_path: Path, monkeypatch) -> None:
    dates = [f"2024-01-0{i}" for i in range(1, 8)]
    data_dir, score_dir, out_dir, st_dir = _write_common_files(tmp_path, dates)

    _write_code_data(data_dir, "000001.SZ", dates, [10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 9.0], [10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 9.0])
    _write_code_data(data_dir, "000002.SZ", dates, [10.0] * 7, [10.0] * 7)

    pd.DataFrame(
        {
            "code": ["000001.SZ", "000002.SZ"],
            "asof_date": ["2024-01-01", "2024-01-02"],
            "yhat": [0.9, 0.8],
        }
    ).to_parquet(score_dir / "scores.parquet", index=False)

    pd.DataFrame({"ts_code": ["000001.SZ"]}).to_parquet(st_dir / "20240103_stock_st.parquet", index=False)

    monkeypatch.setattr(backtest_runner.ts, "pro_api", lambda token: _FakePro())
    monkeypatch.setattr(
        "sys.argv",
        [
            "backtest_runner.py",
            "--data-dir",
            str(data_dir),
            "--score-dir",
            str(score_dir),
            "--out-dir",
            str(out_dir),
            "--val-shards",
            str(tmp_path / "val_shard.npy"),
            "--topk",
            "1",
            "--ts-token",
            "dummy",
            "--st-dir",
            str(st_dir),
        ],
    )

    backtest_runner.main()

    day1 = json.loads((out_dir / "20240102.json").read_text(encoding="utf-8"))
    assert [row["code"] for row in day1["buy_records"]] == ["000001.SZ"]

    day2 = json.loads((out_dir / "20240103.json").read_text(encoding="utf-8"))
    assert [row["code"] for row in day2["sell_records"]] == ["000001.SZ"]
    assert day2["sell_records"][0]["price"] == 9.0
    assert [row["code"] for row in day2["buy_records"]] == ["000002.SZ"]
    assert [row["code"] for row in day2["final_positions"]] == ["000002.SZ"]
