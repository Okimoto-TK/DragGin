from pathlib import Path

import numpy as np
import pandas as pd

from src.feat.build_multiscale_tensor import C, TANH_K, Z_DIM, _build_tensor_context, build_multiscale_tensors
from tests.test_strict_c5_c6_warmup import _write_data


def test_feature_zscore_tanh_bound(tmp_path: Path) -> None:
    asof = _write_data(tmp_path, code="AAA", days=130)
    res = build_multiscale_tensors(tmp_path, "AAA", asof)
    assert res.dp_ok
    assert np.max(np.abs(res.X_micro)) <= TANH_K + 1e-4
    assert np.max(np.abs(res.X_mezzo)) <= TANH_K + 1e-4
    assert np.max(np.abs(res.X_macro)) <= TANH_K + 1e-4


def test_feature_prefix_matches_original_zscore_channels(tmp_path: Path) -> None:
    asof = _write_data(tmp_path, code="AAA", days=130)
    res = build_multiscale_tensors(tmp_path, "AAA", asof)
    assert res.dp_ok
    assert res.X_micro.shape[-1] == C
    assert res.X_mezzo.shape[-1] == C
    assert res.X_macro.shape[-1] == C

    ctx = _build_tensor_context(str(tmp_path.resolve()), "AAA")
    assert ctx is not None
    asof_date = pd.to_datetime(asof).date()
    micro_end = ctx.asof_to_micro_end[asof_date]
    mezzo_end = ctx.asof_to_mezzo_end[asof_date]
    macro_idx = ctx.asof_to_macro_idx[asof_date]

    np.testing.assert_allclose(res.X_micro[:, :Z_DIM], ctx.micro_z[micro_end + 1 - 48 : micro_end + 1], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(res.X_mezzo[:, :Z_DIM], ctx.mezzo_z[mezzo_end + 1 - 40 : mezzo_end + 1], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(res.X_macro[:, :Z_DIM], ctx.macro_z[macro_idx + 1 - 30 : macro_idx + 1], atol=0.0, rtol=0.0)
