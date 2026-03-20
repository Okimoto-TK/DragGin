from pathlib import Path

from src.data.update_daily import DailyUpdateConfig, _compute_code_open_dates, _select_pending_5min_code_windows
import pandas as pd


def test_compute_code_open_dates_excludes_suspended_days() -> None:
    stock_basic = pd.DataFrame({"ts_code": ["AAA", "BBB"]})
    target_dates = ["20240102", "20240103", "20240104"]
    suspend_map = {
        "20240102": set(),
        "20240103": {"AAA"},
        "20240104": {"BBB"},
    }

    got = _compute_code_open_dates(stock_basic, target_dates, suspend_map)

    assert got == {
        "AAA": ["20240102", "20240104"],
        "BBB": ["20240102", "20240103"],
    }


def test_select_pending_5min_code_windows_skips_complete_codes(tmp_path: Path) -> None:
    config = DailyUpdateConfig(data_dir=tmp_path, refresh_latest=False)
    (config.min5_dir / "AAA").mkdir(parents=True, exist_ok=True)
    for trade_date in ["20240102", "20240103", "20240104"]:
        (config.min5_dir / "AAA" / f"{trade_date}.csv").write_text("ok\n", encoding="utf-8")

    got = _select_pending_5min_code_windows(
        config,
        {
            "AAA": ["20240102", "20240103", "20240104"],
            "BBB": ["20240102", "20240103", "20240104"],
        },
    )

    assert got == {"BBB": ("20240102", "20240104")}


def test_select_pending_5min_code_windows_only_fetches_missing_suffix(tmp_path: Path) -> None:
    config = DailyUpdateConfig(data_dir=tmp_path, refresh_latest=False)
    code_dir = config.min5_dir / "AAA"
    code_dir.mkdir(parents=True, exist_ok=True)
    for trade_date in ["20240102", "20240103"]:
        (code_dir / f"{trade_date}.csv").write_text("ok\n", encoding="utf-8")

    got = _select_pending_5min_code_windows(config, {"AAA": ["20240102", "20240103", "20240104", "20240105"]})

    assert got == {"AAA": ("20240104", "20240105")}


def test_select_pending_5min_code_windows_refreshes_latest_even_if_present(tmp_path: Path) -> None:
    config = DailyUpdateConfig(data_dir=tmp_path, refresh_latest=True)
    code_dir = config.min5_dir / "AAA"
    code_dir.mkdir(parents=True, exist_ok=True)
    for trade_date in ["20240102", "20240103", "20240104"]:
        (code_dir / f"{trade_date}.csv").write_text("ok\n", encoding="utf-8")

    got = _select_pending_5min_code_windows(config, {"AAA": ["20240102", "20240103", "20240104"]})

    assert got == {"AAA": ("20240104", "20240104")}
