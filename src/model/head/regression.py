"""Module 6 regression head and masked Huber loss utilities."""

from __future__ import annotations

import torch
from torch import nn


class RegressionHead(nn.Module):
    """Map fused Module 5 representations to scalar predictions."""

    def __init__(
        self,
        hidden_dim: int,
        use_seq_context: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_seq_context = use_seq_context

        input_dim = hidden_dim * 2 if use_seq_context else hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        fused_seq: torch.Tensor,
        fused_pool: torch.Tensor,
    ) -> torch.Tensor:
        head_input = fused_pool
        if self.use_seq_context:
            seq_summary = fused_seq.mean(dim=1)
            head_input = torch.cat([fused_pool, seq_summary], dim=-1)

        y_hat = self.proj(head_input).squeeze(-1)
        return y_hat


def masked_huber_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    loss_mask: torch.Tensor,
    delta: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Compute Huber loss and regression metrics over valid masked entries."""
    valid_mask = loss_mask.to(dtype=torch.bool)
    num_valid = int(valid_mask.sum().item())

    zero = y_hat.new_zeros(())
    if num_valid == 0:
        metrics = {
            "num_valid": 0,
            "huber": zero,
            "mae": zero,
            "mse": zero,
        }
        return zero, metrics

    valid_hat = y_hat[valid_mask]
    valid_true = y_true[valid_mask]

    error = valid_hat - valid_true
    abs_err = error.abs()
    quadratic = torch.minimum(abs_err, abs_err.new_tensor(delta))
    linear = abs_err - quadratic

    huber = (0.5 * quadratic.square() + delta * linear).mean()
    mae = abs_err.mean()
    mse = error.square().mean()

    metrics = {
        "num_valid": num_valid,
        "huber": huber,
        "mae": mae,
        "mse": mse,
    }
    return huber, metrics


class RegressionModelHead(nn.Module):
    """Convenience wrapper exposing prediction and masked-loss methods."""

    def __init__(
        self,
        hidden_dim: int,
        use_seq_context: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.head = RegressionHead(
            hidden_dim=hidden_dim,
            use_seq_context=use_seq_context,
            dropout=dropout,
        )

    def forward(self, fused_seq: torch.Tensor, fused_pool: torch.Tensor) -> torch.Tensor:
        return self.head(fused_seq=fused_seq, fused_pool=fused_pool)

    def compute_loss(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        loss_mask: torch.Tensor,
        delta: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        return masked_huber_loss(y_hat=y_hat, y_true=y_true, loss_mask=loss_mask, delta=delta)
