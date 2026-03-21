"""Module 6 regression head and masked Huber loss utilities."""

from __future__ import annotations

import torch
from torch import nn


def masked_mean(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=seq.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (seq * weights).sum(dim=1) / denom


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
        self.fused_pool_norm = nn.LayerNorm(hidden_dim)
        self.seq_summary_norm = nn.LayerNorm(hidden_dim)

        input_dim = hidden_dim * 2 if use_seq_context else hidden_dim
        self.head_input_norm = nn.LayerNorm(input_dim)
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
        mask_macro: torch.Tensor | None = None,
    ) -> torch.Tensor:
        head_input = self.fused_pool_norm(fused_pool)
        if self.use_seq_context:
            if mask_macro is None:
                seq_summary = fused_seq.mean(dim=1)
            else:
                seq_summary = masked_mean(fused_seq, mask_macro)
            seq_summary = self.seq_summary_norm(seq_summary)
            head_input = torch.cat([head_input, seq_summary], dim=-1)

        y_hat = self.proj(self.head_input_norm(head_input)).squeeze(-1)
        return y_hat


def masked_huber_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    loss_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    confidence_weight: torch.Tensor | None = None,
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
    valid_sample_weight = (
        sample_weight[valid_mask].to(dtype=valid_hat.dtype)
        if sample_weight is not None
        else torch.ones_like(valid_hat, dtype=valid_hat.dtype)
    )
    valid_confidence_weight = (
        confidence_weight[valid_mask].to(dtype=valid_hat.dtype)
        if confidence_weight is not None
        else torch.ones_like(valid_hat, dtype=valid_hat.dtype)
    )
    valid_weight = (valid_sample_weight * valid_confidence_weight).clamp_min(0.0)
    weight_sum = valid_weight.sum().clamp_min(1e-8)

    error = valid_hat - valid_true
    abs_err = error.abs()
    quadratic = torch.minimum(abs_err, abs_err.new_tensor(delta))
    linear = abs_err - quadratic

    huber = ((0.5 * quadratic.square() + delta * linear) * valid_weight).sum() / weight_sum
    mae = (abs_err * valid_weight).sum() / weight_sum
    mse = (error.square() * valid_weight).sum() / weight_sum

    metrics = {
        "num_valid": num_valid,
        "huber": huber,
        "mae": mae,
        "mse": mse,
        "weight_sum": weight_sum,
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

    def forward(self, fused_seq: torch.Tensor, fused_pool: torch.Tensor, mask_macro: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(fused_seq=fused_seq, fused_pool=fused_pool, mask_macro=mask_macro)

    def compute_loss(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        loss_mask: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        confidence_weight: torch.Tensor | None = None,
        delta: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        return masked_huber_loss(
            y_hat=y_hat,
            y_true=y_true,
            loss_mask=loss_mask,
            sample_weight=sample_weight,
            confidence_weight=confidence_weight,
            delta=delta,
        )
