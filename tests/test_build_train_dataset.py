from __future__ import annotations

import numpy as np

from src.feat.build_multiscale_tensor import DPResult
from src.feat.build_training_dataset import build_train_dataset, get_last_benchmark_report, resolve_asof_dates, resolve_codes
from src.feat.labels_risk_adj import LabelBundle


def _dp(dp_ok: bool) -> DPResult:
    return DPResult(
        code="AAA",
        asof_date="2024-01-02",
        dp_ok=dp_ok,
        reason="",
        X_micro=np.ones((48, 6), dtype=np.float32),
        X_mezzo=np.ones((40, 6), dtype=np.float32),
        X_macro=np.ones((30, 6), dtype=np.float32),
        mask_micro=np.ones((48,), dtype=np.uint8),
        mask_mezzo=np.ones((40,), dtype=np.uint8),
        mask_macro=np.ones((30,), dtype=np.uint8),
    )


def _lb(label_ok: bool, loss_mask: bool) -> LabelBundle:
    return LabelBundle(
        code="AAA",
        asof_date="2024-01-02",
        y=np.float32(0.1),
        y_raw=np.float32(0.2),
        y_z=np.float32(0.1),
        label_ok=label_ok,
        loss_mask=loss_mask,
        entry_date=None,
        exit_date=None,
        entry_open=None,
        exit_close=None,
        vol30=None,
        ret_log=None,
        fail_reason=None,
    )


def test_build_train_dataset_filters_invalid(monkeypatch):
    from src.feat import build_training_dataset as btd

    def fake_build_multiscale_tensors(data_dir, code, asof):
        return _dp(dp_ok=(asof == "2024-01-03"))

    def fake_build_label_from_data_dir(data_dir, code, asof_date, dp_ok=True):
        if asof_date == "2024-01-03":
            return _lb(label_ok=True, loss_mask=True)
        return _lb(label_ok=False, loss_mask=False)

    monkeypatch.setattr(btd, "build_multiscale_tensors", fake_build_multiscale_tensors)
    monkeypatch.setattr(btd, "build_label_from_data_dir", fake_build_label_from_data_dir)

    filtered = build_train_dataset(".", ["AAA"], ["2024-01-02", "2024-01-03"], include_invalid=False, show_progress=False)
    assert filtered.y.shape == (1,)
    assert filtered.asof_dates.tolist() == ["2024-01-03"]

    unfiltered = build_train_dataset(".", ["AAA"], ["2024-01-02", "2024-01-03"], include_invalid=True, show_progress=False)
    assert unfiltered.y.shape == (2,)
    assert unfiltered.loss_mask.tolist() == [False, True]


def test_resolve_defaults(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    (data_dir / "AAA").mkdir(parents=True)
    (data_dir / "BBB").mkdir(parents=True)
    (data_dir / "AAA" / "daily.parquet").write_text("x")
    (data_dir / "BBB" / "daily.parquet").write_text("x")

    codes = resolve_codes(data_dir, None)
    assert codes == ["AAA", "BBB"]

    from src.feat import build_training_dataset as btd

    monkeypatch.setattr(btd, "build_calendar_from_daily_filenames", lambda _: ["2024-01-02", "2024-01-03"])
    asof_dates = resolve_asof_dates(data_dir, None)
    assert asof_dates == ["2024-01-02", "2024-01-03"]


def test_build_train_dataset_multiprocess_path(monkeypatch):
    from src.feat import build_training_dataset as btd

    def fake_build_multiscale_tensors(data_dir, code, asof):
        return _dp(dp_ok=True)

    def fake_build_label_from_data_dir(data_dir, code, asof_date, dp_ok=True):
        return _lb(label_ok=True, loss_mask=True)

    class FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return FakeFuture(fn(*args, **kwargs))

    monkeypatch.setattr(btd, "build_multiscale_tensors", fake_build_multiscale_tensors)
    monkeypatch.setattr(btd, "build_label_from_data_dir", fake_build_label_from_data_dir)
    monkeypatch.setattr(btd, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(btd, "as_completed", lambda futures: futures)

    out = build_train_dataset(
        ".",
        codes=["AAA"],
        asof_dates=["2024-01-02", "2024-01-03"],
        include_invalid=False,
        num_workers=2,
        show_progress=False,
    )
    assert out.y.shape == (2,)


def test_progress_wrapper_uses_tqdm(monkeypatch):
    from src.feat import build_training_dataset as btd

    called = {"v": False}

    def fake_tqdm(iterable, total, desc):
        called["v"] = True
        return iterable

    monkeypatch.setattr(btd, "tqdm", fake_tqdm)

    def fake_build_multiscale_tensors(data_dir, code, asof):
        return _dp(dp_ok=True)

    def fake_build_label_from_data_dir(data_dir, code, asof_date, dp_ok=True):
        return _lb(label_ok=True, loss_mask=True)

    monkeypatch.setattr(btd, "build_multiscale_tensors", fake_build_multiscale_tensors)
    monkeypatch.setattr(btd, "build_label_from_data_dir", fake_build_label_from_data_dir)

    _ = build_train_dataset(
        ".",
        codes=["AAA"],
        asof_dates=["2024-01-02"],
        include_invalid=False,
        num_workers=1,
        show_progress=True,
    )
    assert called["v"]


def test_benchmark_limits_to_first_code(monkeypatch):
    from src.feat import build_training_dataset as btd

    seen_codes = []

    def fake_build_multiscale_tensors(data_dir, code, asof):
        seen_codes.append(code)
        return _dp(dp_ok=True)

    def fake_build_label_from_data_dir(data_dir, code, asof_date, dp_ok=True):
        return _lb(label_ok=True, loss_mask=True)

    monkeypatch.setattr(btd, "build_multiscale_tensors", fake_build_multiscale_tensors)
    monkeypatch.setattr(btd, "build_label_from_data_dir", fake_build_label_from_data_dir)

    _ = build_train_dataset(
        ".",
        codes=["AAA", "BBB"],
        asof_dates=["2024-01-02"],
        include_invalid=False,
        num_workers=4,
        show_progress=False,
        benchmark=True,
    )

    assert seen_codes == ["AAA"]
    report = get_last_benchmark_report()
    assert report is not None
    assert report.code == "AAA"
