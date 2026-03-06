from pathlib import Path

import pandas as pd

from src.feat.build_multiscale_tensor import build_multiscale_tensors
from src.feat.labels_risk_adj import build_label_from_data_dir


def _write_data_with_breakpoint(root: Path, code: str = "AAA", days: int = 180) -> str:
    start = pd.Timestamp("2024-01-01")
    dates = [start + pd.Timedelta(days=i) for i in range(days)]

    rows_5m = []
    for d in dates:
        for i in range(48):
            t = pd.Timestamp(d.date().isoformat() + " 09:30") + pd.Timedelta(minutes=5 * i)
            day_idx = (d - start).days
            px = 100.0 + day_idx * 0.2 + i * 0.01 + 0.05 * ((day_idx + i) % 7)
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
        base = 100.0 + day_idx * 0.2 + 0.1 * ((day_idx % 5) - 2)
        rows_daily.append(
            {
                "code": code,
                "trade_date": d.date().isoformat(),
                "open": base,
                "high": base + 1.0,
                "low": base - 1.0,
                "close": base + 0.5,
                "volume": 1000,
                "adj_factor": 1.0 + 0.0001 * day_idx,
            }
        )

    code_dir = root / code
    code_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_5m).to_parquet(code_dir / "5min.parquet", index=False)
    pd.DataFrame(rows_daily).to_parquet(code_dir / "daily.parquet", index=False)
    pd.DataFrame({"trade_date": [d.date().isoformat() for d in dates]}).to_parquet(root / "calendar.parquet", index=False)

    bp_date = (start + pd.Timedelta(days=140)).date().isoformat()
    pd.DataFrame({"break_date": [bp_date]}).to_parquet(code_dir / "breakpoints.parquet", index=False)

    return dates[-1].date().isoformat()


def test_multiscale_invalid_when_macro_history_crosses_breakpoint(tmp_path: Path) -> None:
    asof = _write_data_with_breakpoint(tmp_path)
    res = build_multiscale_tensors(tmp_path, "AAA", asof)
    assert not res.dp_ok
    assert "breakpoint" in res.reason


def test_label_invalid_when_window_crosses_breakpoint(tmp_path: Path) -> None:
    asof = _write_data_with_breakpoint(tmp_path, days=220)
    res = build_label_from_data_dir(tmp_path, "AAA", asof)
    assert not res.label_ok
    assert res.fail_reason is not None
    assert "breakpoint" in res.fail_reason
