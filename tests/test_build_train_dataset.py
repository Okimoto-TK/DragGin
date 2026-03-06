from __future__ import annotations

import numpy as np

from src.feat.build_multiscale_tensor import DPResult
from src.feat.build_training_dataset import build_train_dataset
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

    filtered = build_train_dataset(".", ["AAA"], ["2024-01-02", "2024-01-03"], include_invalid=False)
    assert filtered.y.shape == (1,)
    assert filtered.asof_dates.tolist() == ["2024-01-03"]

    unfiltered = build_train_dataset(".", ["AAA"], ["2024-01-02", "2024-01-03"], include_invalid=True)
    assert unfiltered.y.shape == (2,)
    assert unfiltered.loss_mask.tolist() == [False, True]
