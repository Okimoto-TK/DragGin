from datetime import date

from src.feat.build_multiscale_tensor import build_multiscale_tensor, build_trading_calendar

from conftest import make_calendar, make_provider


def test_calendar_from_filenames(tmp_path):
    (tmp_path / "foo_20260213.csv").write_text("x")
    (tmp_path / "bar_20260214.parquet").write_text("x")
    (tmp_path / "dup_20260213.txt").write_text("x")
    (tmp_path / "no_date.txt").write_text("x")

    cal, idx = build_trading_calendar(str(tmp_path))
    assert cal == [date(2026, 2, 13), date(2026, 2, 14)]
    assert idx[date(2026, 2, 13)] == 0
    assert idx[date(2026, 2, 14)] == 1


def test_shape_and_masks_when_ok():
    calendar = make_calendar()
    code, provider = make_provider(calendar)
    out = build_multiscale_tensor(code, calendar[-1], provider, calendar)

    assert out["dp_ok"] is True
    assert len(out["X_micro"]) == 48 and len(out["X_micro"][0]) == 7
    assert len(out["X_mezzo"]) == 40 and len(out["X_mezzo"][0]) == 7
    assert len(out["X_macro"]) == 30 and len(out["X_macro"][0]) == 7
    assert set(out["mask_micro"]) == {1}
    assert set(out["mask_mezzo"]) == {1}
    assert set(out["mask_macro"]) == {1}


def test_shape_when_dp_not_ok_is_still_fixed():
    calendar = make_calendar()
    code, provider = make_provider(calendar)

    # break micro daily composition (one extra wrong-day bar in final segment)
    provider.data[(code, "5m")][-1]["date"] = calendar[-2]

    out = build_multiscale_tensor(code, calendar[-1], provider, calendar)

    assert out["dp_ok"] is False
    assert len(out["X_micro"]) == 48 and len(out["X_micro"][0]) == 7
    assert len(out["X_mezzo"]) == 40 and len(out["X_mezzo"][0]) == 7
    assert len(out["X_macro"]) == 30 and len(out["X_macro"][0]) == 7
    assert set(out["mask_micro"]) == {0}
    assert set(out["mask_mezzo"]) == {0}
    assert set(out["mask_macro"]) == {0}
