import torch

from src.model.wno.soft_threshold_gate import SoftThresholdGate
from src.model.wno.wavelet_ops import CoefPack, wavelet_decompose_1d, wavelet_reconstruct_1d


def test_wavelet_gate_integration_shapes_and_finiteness() -> None:
    x = torch.randn(2, 6, 30)
    pack = wavelet_decompose_1d(x, levels=3)
    gate = SoftThresholdGate(init_lambda=0.1)

    approx_thr, _ = gate(pack.approx)
    details_thr = []
    for detail in pack.details:
        d_thr, _ = gate(detail)
        details_thr.append(d_thr)

    gated_pack = CoefPack(
        approx=approx_thr,
        details=details_thr,
        orig_len=pack.orig_len,
        padded_len=pack.padded_len,
        levels=pack.levels,
        wavelet=pack.wavelet,
        pad_mode=pack.pad_mode,
    )
    x2 = wavelet_reconstruct_1d(gated_pack)

    assert x2.shape == x.shape
    assert torch.isfinite(x2).all()
