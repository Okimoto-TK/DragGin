from __future__ import annotations

import torch

from src.model.head.regression import RegressionHead, masked_huber_loss


def test_regression_head_output_shape() -> None:
    hidden_dim = 16
    fused_seq = torch.randn(2, 30, hidden_dim)
    fused_pool = torch.randn(2, hidden_dim)

    head = RegressionHead(hidden_dim=hidden_dim)
    y_hat = head(fused_seq=fused_seq, fused_pool=fused_pool)

    assert y_hat.shape == (2,)
    assert torch.isfinite(y_hat).all()


def test_masked_huber_loss_basic() -> None:
    y_hat = torch.tensor([1.0, 3.0, 4.0])
    y_true = torch.tensor([0.0, 3.5, 2.0])
    mask = torch.tensor([True, True, False])

    loss, metrics = masked_huber_loss(y_hat=y_hat, y_true=y_true, loss_mask=mask, delta=1.0)

    assert metrics["num_valid"] == 2
    assert torch.isfinite(loss)
    assert set(["huber", "mae", "mse"]).issubset(metrics.keys())
    assert float(metrics["huber"]) >= 0.0
    assert float(metrics["mae"]) >= 0.0
    assert float(metrics["mse"]) >= 0.0


def test_masked_huber_loss_ignores_invalid() -> None:
    y_hat = torch.tensor([1.0, 2.0, 10000.0])
    y_true = torch.tensor([1.5, 1.0, -10000.0])
    mask = torch.tensor([1, 1, 0])

    loss_a, metrics_a = masked_huber_loss(y_hat=y_hat, y_true=y_true, loss_mask=mask)

    y_hat_perturbed = y_hat.clone()
    y_true_perturbed = y_true.clone()
    y_hat_perturbed[2] = -1.0e9
    y_true_perturbed[2] = 1.0e9

    loss_b, metrics_b = masked_huber_loss(y_hat=y_hat_perturbed, y_true=y_true_perturbed, loss_mask=mask)

    assert torch.isclose(loss_a, loss_b)
    assert torch.isclose(metrics_a["huber"], metrics_b["huber"])
    assert torch.isclose(metrics_a["mae"], metrics_b["mae"])
    assert torch.isclose(metrics_a["mse"], metrics_b["mse"])


def test_masked_huber_loss_empty_mask() -> None:
    y_hat = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([1.5, 2.5, 3.5])
    mask = torch.tensor([False, False, False])

    loss, metrics = masked_huber_loss(y_hat=y_hat, y_true=y_true, loss_mask=mask)

    assert float(loss) == 0.0
    assert metrics["num_valid"] == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["huber"])
    assert torch.isfinite(metrics["mae"])
    assert torch.isfinite(metrics["mse"])


def test_use_seq_context_flag() -> None:
    hidden_dim = 8
    fused_seq = torch.randn(3, 30, hidden_dim)
    fused_pool = torch.randn(3, hidden_dim)

    head_no_ctx = RegressionHead(hidden_dim=hidden_dim, use_seq_context=False)
    head_ctx = RegressionHead(hidden_dim=hidden_dim, use_seq_context=True)

    y_hat_no_ctx = head_no_ctx(fused_seq=fused_seq, fused_pool=fused_pool)
    y_hat_ctx = head_ctx(fused_seq=fused_seq, fused_pool=fused_pool)

    assert y_hat_no_ctx.shape == (3,)
    assert y_hat_ctx.shape == (3,)
    assert torch.isfinite(y_hat_no_ctx).all()
    assert torch.isfinite(y_hat_ctx).all()
