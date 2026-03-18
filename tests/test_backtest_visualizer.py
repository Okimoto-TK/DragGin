import json
from pathlib import Path

import pytest

from tools.backtest_visualizer import _build_html, _load_backtest_rows, main


def _write_day(path: Path, day: str, asof: str, initial: float, final: float) -> None:
    payload = {
        "date": day,
        "asof_date": asof,
        "initial_total_asset": initial,
        "final_total_asset": final,
        "initial_holding_amount": initial * 0.4,
        "initial_cash": initial * 0.6,
        "final_holding_amount": final * 0.5,
        "final_cash": final * 0.5,
        "initial_positions": [{"code": "000001.SZ", "qty": 100, "cost": 10.0, "float_pnl": 20.0}],
        "final_positions": [{"code": "000001.SZ", "qty": 100, "cost": 10.0, "float_pnl": 30.0}],
        "buy_records": [{"code": "000001.SZ", "qty": 100, "cost": 10.1}],
        "sell_records": [],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_load_backtest_rows_computes_returns(tmp_path: Path) -> None:
    _write_day(tmp_path / "20240102.json", "2024-01-02", "2024-01-01", 1000.0, 1100.0)
    _write_day(tmp_path / "20240103.json", "2024-01-03", "2024-01-02", 1100.0, 1210.0)

    rows = _load_backtest_rows(tmp_path)

    assert [row["date"] for row in rows] == ["2024-01-02", "2024-01-03"]
    assert rows[0]["daily_return"] == pytest.approx(0.1)
    assert rows[0]["start_cum_return"] == pytest.approx(0.0)
    assert rows[0]["cum_return"] == pytest.approx(0.1)
    assert rows[1]["daily_return"] == pytest.approx(0.1)
    assert rows[1]["start_cum_return"] == pytest.approx(0.1)
    assert rows[1]["cum_return"] == pytest.approx(0.21)


def test_build_html_contains_adaptive_plot_and_structured_sidebar(tmp_path: Path) -> None:
    _write_day(tmp_path / "20240102.json", "2024-01-02", "2024-01-01", 1000.0, 1100.0)
    rows = _load_backtest_rows(tmp_path)

    html = _build_html(rows, "My Backtest")

    assert "SHORT_WINDOW_THRESHOLD" in html
    assert "type: 'candlestick'" in html
    assert "renderChart(event)" in html
    assert "资产摘要" in html
    assert "section-buttons" in html
    assert "detail-back" in html
    assert "trade-card" in html
    assert "My Backtest" in html
    assert "backtest_report.html" not in html


def test_main_writes_html_report(tmp_path: Path, monkeypatch) -> None:
    backtest_dir = tmp_path / "bt"
    backtest_dir.mkdir()
    _write_day(backtest_dir / "20240102.json", "2024-01-02", "2024-01-01", 1000.0, 1050.0)

    out_file = tmp_path / "report.html"
    monkeypatch.setattr(
        "sys.argv",
        [
            "backtest_visualizer.py",
            "--backtest-dir",
            str(backtest_dir),
            "--out-file",
            str(out_file),
            "--title",
            "Demo",
        ],
    )

    main()

    assert out_file.exists()
    text = out_file.read_text(encoding="utf-8")
    assert "Demo" in text
    assert "2024-01-02" in text
    assert "累计收益率 K 线" in text
