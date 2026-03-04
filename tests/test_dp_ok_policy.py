from src.feat.build_multiscale_tensor import build_multiscale_tensor

from conftest import make_calendar, make_provider


def test_dp_ok_false_when_preroll_volume_short_by_one():
    calendar = make_calendar()
    code, provider = make_provider(calendar)
    t = calendar[-1]

    # invalidate one pre-roll volume needed by micro t=0 (20 bars history)
    provider.data[(code, "5m")][-49]["volume"] = None

    out = build_multiscale_tensor(code, t, provider, calendar)
    assert out["dp_ok"] is False
    assert set(out["mask_micro"]) == {0}


def test_dp_ok_false_when_returns_history_short_by_one():
    calendar = make_calendar()
    code, provider = make_provider(calendar)
    t = calendar[-1]

    # invalidate close in return history area
    provider.data[(code, "30m")][-41]["close"] = None

    out = build_multiscale_tensor(code, t, provider, calendar)
    assert out["dp_ok"] is False
    assert set(out["mask_mezzo"]) == {0}


def test_dp_ok_false_when_any_bar_missing_in_window():
    calendar = make_calendar()
    code, provider = make_provider(calendar)
    t = calendar[-1]

    # remove one 30m bar from final 5-day window
    provider.data[(code, "30m")].pop()

    out = build_multiscale_tensor(code, t, provider, calendar)
    assert out["dp_ok"] is False
