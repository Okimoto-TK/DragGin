import torch

from src.model.flow import FlowLegendreProjectionEncoder, WaveletGatedFiLM
from src.model.wno import WNO1DEncoder


def test_flow_lpe_shape() -> None:
    enc = FlowLegendreProjectionEncoder(window=30, in_features=4, order=6)
    flow_x = torch.randn(3, 30, 4)
    flow_mask = torch.ones(3, 30, dtype=torch.bool)
    out = enc(flow_x, flow_mask)
    assert out.shape == (3, 24)


def test_wavelet_gated_film_force_zero_identity() -> None:
    mod = WaveletGatedFiLM(hidden_dim=8, flow_raw_dim=24)
    coeff = torch.randn(2, 8, 16)
    flow_lpe = torch.randn(2, 24)
    out, aux = mod(coeff, flow_lpe, force_gate_value=0.0)
    assert torch.allclose(out, coeff, atol=1e-7)
    assert set(aux.keys()) == {"flow_gate_mean", "flow_gate_std", "flow_gamma_mean", "flow_beta_mean"}


def test_wno_forward_backward_compat_without_flow() -> None:
    x = torch.randn(2, 30, 6)
    mask = torch.ones(2, 30)
    encoder = WNO1DEncoder(in_dim=6, hidden_dim=8)
    h_seq, h_pool, aux = encoder(x, mask)
    assert h_seq.shape == (2, 30, 8)
    assert h_pool.shape == (2, 8)
    assert "flow_gate_approx_mean" in aux


def test_wno_with_flow_and_forced_zero_gate_matches_no_flow() -> None:
    x = torch.randn(2, 30, 6)
    mask = torch.ones(2, 30)
    flow = torch.randn(2, 24)
    encoder = WNO1DEncoder(in_dim=6, hidden_dim=8)
    out_no = encoder(x, mask)
    out_zero = encoder(x, mask, flow_raw=flow, force_flow_gate_value=0.0)
    assert torch.allclose(out_no[0], out_zero[0], atol=1e-6)
    assert torch.allclose(out_no[1], out_zero[1], atol=1e-6)
