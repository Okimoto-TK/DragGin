from pathlib import Path

import numpy as np
import pandas as pd

from src.feat.build_multiscale_tensor import build_multiscale_tensors


def _write_data(root: Path, code: str, days: int, missing_5m: bool = False, missing_daily_date: str | None = None, missing_daily_for_code_date: str | None = None, flat: bool = False) -> str:
    start = pd.Timestamp("2024-01-01")
    dates = [start + pd.Timedelta(days=i) for i in range(days)]
    rows_5m = []
    for d in dates:
        for i in range(48):
            if missing_5m and d == dates[-1] and i == 10:
                continue
            t = pd.Timestamp(d.date().isoformat() + " 09:30") + pd.Timedelta(minutes=5 * i)
            base_idx = len(rows_5m)
            px = 100.0 if flat else 100 + base_idx * 0.01 + 0.2 * np.sin(base_idx / 11)
            rows_5m.append(
                {
                    "code": code,
                    "trade_date": d.date().isoformat(),
                    "time": t.strftime("%H:%M"),
                    "open": px,
                    "high": px + (0.4 if flat else 0.4 + 0.05 * np.sin(base_idx / 7)),
                    "low": px - (0.4 if flat else 0.4 + 0.05 * np.cos(base_idx / 9)),
                    "close": px + (0.05 if flat else 0.05 * np.sin(base_idx / 5)),
                    "volume": 100 if flat else 100 + (base_idx % 13),
                    "vwap": px + (0.02 if flat else 0.02 * np.cos(base_idx / 6)),
                }
            )
    pd.DataFrame(rows_5m).to_csv(root / f"{code}_5m.csv", index=False)

    for d in dates:
        if missing_daily_date and d.date().isoformat() == missing_daily_date:
            continue
        day_idx = (d - dates[0]).days
        px = 100.0 if flat else 100 + day_idx * 0.2 + 0.6 * np.sin(day_idx / 6)
        rows = []
        if not (missing_daily_for_code_date and d.date().isoformat() == missing_daily_for_code_date):
            rows.append(
                {
                    "code": code,
                    "trade_date": d.date().isoformat(),
                    "open": px,
                    "high": px + (1 if flat else 1 + 0.2 * np.sin(day_idx / 4)),
                    "low": px - (1 if flat else 1 + 0.2 * np.cos(day_idx / 5)),
                    "close": px + (0.3 if flat else 0.3 * np.sin(day_idx / 3)),
                    "volume": 1000 if flat else 1000 + (day_idx % 17) * 10,
                    "vwap": px + (0.1 if flat else 0.1 * np.cos(day_idx / 2)),
                    "adj_factor": 1.0 + 0.0005 * day_idx,
                }
            )
        pd.DataFrame(rows).to_parquet(root / f"{code}_{d.date().isoformat()}.parquet", index=False)

    return dates[-1].date().isoformat()


def test_strict_c5_c6_warmup(tmp_path: Path) -> None:
    code = "AAA"
    asof = _write_data(tmp_path, code=code, days=90)
    bad = build_multiscale_tensors(tmp_path, code, asof)
    assert not bad.dp_ok
    assert "zscore warmup" in bad.reason

    tmp2 = tmp_path / "ok"
    tmp2.mkdir()
    asof2 = _write_data(tmp2, code=code, days=130)
    ok = build_multiscale_tensors(tmp2, code, asof2)
    assert ok.dp_ok
    assert ok.X_micro.shape == (48, 7)
    assert ok.X_mezzo.shape == (40, 7)
    assert ok.X_macro.shape == (30, 7)
    assert np.all(ok.mask_micro == 1)
    assert np.all(ok.mask_mezzo == 1)
    assert np.all(ok.mask_macro == 1)
