import torch

from src.model.wno.soft_threshold_gate import SoftThresholdGate


def test_lambda_nonneg_and_shapes() -> None:
    gate = SoftThresholdGate(init_lambda=0.2)
    w = torch.randn(2, 6, 17)
    w_thr, lam = gate(w)

    assert w_thr.shape == w.shape
    assert lam.shape == (2, 6, 1)
    assert torch.all(lam >= 0)


def test_zero_input_stability() -> None:
    gate = SoftThresholdGate(init_lambda=0.1)
    w = torch.zeros(2, 6, 17)
    w_thr, lam = gate(w)

    assert torch.all(w_thr == 0)
    assert torch.isfinite(lam).all()
    assert torch.isfinite(w_thr).all()


def test_threshold_effect() -> None:
    w = torch.randn(2, 6, 17)

    gate_large = SoftThresholdGate(init_lambda=10.0)
    w_thr_large, _ = gate_large(w)

    gate_small = SoftThresholdGate(init_lambda=1e-6)
    w_thr_small, _ = gate_small(w)

    assert torch.linalg.norm(w_thr_large) < 0.25 * torch.linalg.norm(w)
    assert torch.mean(torch.abs(w_thr_small - w)) < 1e-3
