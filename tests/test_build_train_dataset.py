from __future__ import annotations

import numpy as np

from src.feat.build_multiscale_tensor import DPResult
from src.feat.build_training_dataset import build_train_dataset, build_train_dataset_shards, resolve_asof_dates, resolve_codes
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


def _flow(ok: bool = True):
    if ok:
        return np.ones((30, 4), dtype=np.float32), np.ones((30,), dtype=np.uint8), True
    return np.zeros((30, 4), dtype=np.float32), np.zeros((30,), dtype=np.uint8), False


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
    monkeypatch.setattr(btd, "_build_flow_features", lambda *_: _flow(True))

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
    monkeypatch.setattr(btd, "_build_flow_features", lambda *_: _flow(True))
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
    monkeypatch.setattr(btd, "_build_flow_features", lambda *_: _flow(True))

    _ = build_train_dataset(
        ".",
        codes=["AAA"],
        asof_dates=["2024-01-02"],
        include_invalid=False,
        num_workers=1,
        show_progress=True,
    )
    assert called["v"]


def test_worker_cache_cleared_after_code_task(monkeypatch):
    from src.feat import build_training_dataset as btd

    calls = {"tensor": 0, "label": 0}

    def fake_build_multiscale_tensors(data_dir, code, asof):
        return _dp(dp_ok=True)

    def fake_build_label_from_data_dir(data_dir, code, asof_date, dp_ok=True):
        return _lb(label_ok=True, loss_mask=True)

    def fake_clear_tensor_worker_cache():
        calls["tensor"] += 1

    def fake_clear_label_worker_cache():
        calls["label"] += 1

    monkeypatch.setattr(btd, "build_multiscale_tensors", fake_build_multiscale_tensors)
    monkeypatch.setattr(btd, "build_label_from_data_dir", fake_build_label_from_data_dir)
    monkeypatch.setattr(btd, "_build_flow_features", lambda *_: _flow(True))
    monkeypatch.setattr(btd, "clear_tensor_worker_cache", fake_clear_tensor_worker_cache)
    monkeypatch.setattr(btd, "clear_label_worker_cache", fake_clear_label_worker_cache)

    out = build_train_dataset(
        ".",
        codes=["AAA", "BBB"],
        asof_dates=["2024-01-02"],
        include_invalid=False,
        num_workers=1,
        show_progress=False,
    )
    assert out.y.shape == (2,)
    assert calls == {"tensor": 2, "label": 2}


def test_build_train_dataset_shards_writes_per_code_files(tmp_path, monkeypatch):
    from src.feat import build_training_dataset as btd

    def fake_build_multiscale_tensors(data_dir, code, asof):
        return _dp(dp_ok=True)

    def fake_build_label_from_data_dir(data_dir, code, asof_date, dp_ok=True):
        return _lb(label_ok=True, loss_mask=True)

    monkeypatch.setattr(btd, "build_multiscale_tensors", fake_build_multiscale_tensors)
    monkeypatch.setattr(btd, "build_label_from_data_dir", fake_build_label_from_data_dir)
    monkeypatch.setattr(btd, "_build_flow_features", lambda *_: _flow(True))

    out_dir = tmp_path / "out"
    infos = build_train_dataset_shards(
        ".",
        out_dir=out_dir,
        codes=["AAA", "BBB"],
        asof_dates=["2024-01-02"],
        include_invalid=False,
        num_workers=1,
        show_progress=False,
    )

    assert sorted(p.name for p in out_dir.glob("*.npy")) == ["AAA.npy", "BBB.npy"]
    assert len(infos) == 2
    assert sorted(int(x["rows"]) for x in infos) == [1, 1]


def test_build_train_dataset_flow_features_formula(tmp_path, monkeypatch):
    from src.feat import build_training_dataset as btd

    def fake_build_multiscale_tensors(data_dir, code, asof):
        return _dp(dp_ok=True)

    def fake_build_label_from_data_dir(data_dir, code, asof_date, dp_ok=True):
        return _lb(label_ok=True, loss_mask=True)

    monkeypatch.setattr(btd, "build_multiscale_tensors", fake_build_multiscale_tensors)
    monkeypatch.setattr(btd, "build_label_from_data_dir", fake_build_label_from_data_dir)

    code_dir = tmp_path / "AAA"
    code_dir.mkdir(parents=True)
    dates = np.array(np.arange("2024-01-01", "2024-01-31", dtype="datetime64[D]"), dtype="datetime64[D]")
    date_str = [str(x) for x in dates]

    import pandas as pd

    pd.DataFrame({"trade_date": date_str, "volume": [10000.0] * 30}).to_parquet(code_dir / "daily.parquet", index=False)
    pd.DataFrame(
        {
            "trade_date": date_str,
            "net_mf_vol": [10.0] * 30,
            "buy_lg_vol": [7.0] * 30,
            "sell_lg_vol": [2.0] * 30,
            "buy_elg_vol": [9.0] * 30,
            "sell_elg_vol": [4.0] * 30,
        }
    ).to_parquet(code_dir / "moneyflow.parquet", index=False)

    out = btd.build_train_dataset(
        tmp_path,
        codes=["AAA"],
        asof_dates=["2024-01-30"],
        include_invalid=False,
        show_progress=False,
    )
    assert out.flow_x.shape == (1, 30, 4)
    assert out.flow_mask.shape == (1, 30)
    np.testing.assert_allclose(out.flow_x[0, :, 0], 0.1)
    np.testing.assert_allclose(out.flow_x[0, :, 1], 0.05)
    np.testing.assert_allclose(out.flow_x[0, :, 2], 0.05)
    np.testing.assert_allclose(out.flow_x[0, :, 3], 0.1)
    assert np.all(out.flow_mask[0] == 1)


def test_build_train_dataset_invalid_when_flow_missing(monkeypatch):
    from src.feat import build_training_dataset as btd

    def fake_build_multiscale_tensors(data_dir, code, asof):
        return _dp(dp_ok=True)

    def fake_build_label_from_data_dir(data_dir, code, asof_date, dp_ok=True):
        return _lb(label_ok=True, loss_mask=True)

    monkeypatch.setattr(btd, "build_multiscale_tensors", fake_build_multiscale_tensors)
    monkeypatch.setattr(btd, "build_label_from_data_dir", fake_build_label_from_data_dir)
    monkeypatch.setattr(btd, "_build_flow_features", lambda *_: _flow(False))

    out = btd.build_train_dataset(
        ".",
        codes=["AAA"],
        asof_dates=["2024-01-02"],
        include_invalid=True,
        show_progress=False,
    )
    assert out.y.shape == (1,)
    assert out.loss_mask.tolist() == [False]
    assert out.dp_ok.tolist() == [False]
