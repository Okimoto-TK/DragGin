from pathlib import Path

import pandas as pd

from src.feat.build_multiscale_tensor import build_multiscale_tensors
from src.feat.labels_risk_adj import build_label_for_sample


def _write_base_data(root: Path, code: str, days: int, zero_vol_date: str) -> str:
    start = pd.Timestamp("2024-01-01")
    dates = [start + pd.Timedelta(days=i) for i in range(days)]

    rows_5m = []
    for d in dates:
        for i in range(48):
            t = pd.Timestamp(d.date().isoformat() + " 09:30") + pd.Timedelta(minutes=5 * i)
            px = 100.0 + (d - start).days * 0.2 + i * 0.01
            rows_5m.append(
                {
                    "code": code,
                    "trade_date": d.date().isoformat(),
                    "time": t.strftime("%H:%M"),
                    "open": px,
                    "high": px + 0.5,
                    "low": px - 0.5,
                    "close": px + 0.1,
                    "volume": 100 + i,
                }
            )

    rows_daily = []
    for d in dates:
        day_idx = (d - start).days
        rows_daily.append(
            {
                "code": code,
                "trade_date": d.date().isoformat(),
                "open": 100.0 + day_idx * 0.2,
                "high": 101.0 + day_idx * 0.2,
                "low": 99.0 + day_idx * 0.2,
                "close": 100.5 + day_idx * 0.2,
                "volume": 0 if d.date().isoformat() == zero_vol_date else 1000,
                "adj_factor": 1.0 + 0.0001 * day_idx,
            }
        )

    code_dir = root / code
    code_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_5m).to_parquet(code_dir / "5min.parquet", index=False)
    pd.DataFrame(rows_daily).to_parquet(code_dir / "daily.parquet", index=False)
    pd.DataFrame({"trade_date": [d.date().isoformat() for d in dates]}).to_parquet(root / "calendar.parquet", index=False)
    return dates[-1].date().isoformat()


def test_multiscale_ignores_zero_volume_daily_in_trading_calendar(tmp_path: Path) -> None:
    code = "AAA"
    asof = _write_base_data(tmp_path, code=code, days=130, zero_vol_date="2024-03-01")

    res = build_multiscale_tensors(tmp_path, code, asof)

    assert res.dp_ok, res.reason


def test_label_ignores_zero_volume_daily_in_trading_calendar() -> None:
    calendar = [d.date().isoformat() for d in pd.date_range("2023-01-01", periods=330, freq="D")]
    n = len(calendar)
    df = pd.DataFrame(
        {
            "trade_date": calendar,
            "open": [100.0 + 0.1 * i for i in range(n)],
            "high": [101.0 + 0.1 * i for i in range(n)],
            "low": [99.0 + 0.1 * i for i in range(n)],
            "close": [100.5 + 0.1 * i for i in range(n)],
            "volume": [0.0 if d == "2023-08-01" else 1000.0 for d in calendar],
            "adj_factor": [1.0 + 0.0001 * i for i in range(n)],
        }
    )

    asof = calendar[320]
    res = build_label_for_sample("AAA", asof, calendar, lambda _: df)

    assert res.label_ok, res.fail_reason
