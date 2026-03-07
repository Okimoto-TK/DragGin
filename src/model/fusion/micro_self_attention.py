"""Module 5 free branch: micro self-attention + deterministic compression."""

from __future__ import annotations

import torch
from torch import nn


class MicroSelfAttention(nn.Module):
    """Runs micro self-attention then compresses 48 -> target_len via AdaptiveAvgPool1d."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        micro_seq: torch.Tensor,
        mask_micro: torch.Tensor,
        target_len: int = 30,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_padding_mask = ~mask_micro.bool()
        attn_out, _ = self.attn(
            query=micro_seq,
            key=micro_seq,
            value=micro_seq,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        weights = mask_micro.to(dtype=attn_out.dtype).unsqueeze(-1)
        values = attn_out * weights

        values_perm = values.transpose(1, 2)
        weights_perm = weights.transpose(1, 2)

        pooled_values = nn.functional.adaptive_avg_pool1d(values_perm, target_len)
        pooled_weights = nn.functional.adaptive_avg_pool1d(weights_perm, target_len)

        free_seq_perm = pooled_values / pooled_weights.clamp_min(1e-6)
        free_seq_perm = torch.where(pooled_weights > 0, free_seq_perm, torch.zeros_like(free_seq_perm))

        free_seq = free_seq_perm.transpose(1, 2)
        free_pool = free_seq.mean(dim=1)
        return free_seq, free_pool
