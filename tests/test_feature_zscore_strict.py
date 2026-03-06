from pathlib import Path

import numpy as np

from src.feat.build_multiscale_tensor import TANH_K, build_multiscale_tensors
from tests.test_strict_c5_c6_warmup import _write_data


def test_feature_zscore_tanh_bound(tmp_path: Path) -> None:
    asof = _write_data(tmp_path, code="AAA", days=130)
    res = build_multiscale_tensors(tmp_path, "AAA", asof)
    assert res.dp_ok
    assert np.max(np.abs(res.X_micro)) <= TANH_K + 1e-4
    assert np.max(np.abs(res.X_mezzo)) <= TANH_K + 1e-4
    assert np.max(np.abs(res.X_macro)) <= TANH_K + 1e-4
