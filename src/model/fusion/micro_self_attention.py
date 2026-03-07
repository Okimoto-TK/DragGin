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

        masked = attn_out * mask_micro.to(dtype=attn_out.dtype).unsqueeze(-1)
        pooled = nn.functional.adaptive_avg_pool1d(masked.transpose(1, 2), target_len)
        free_seq = pooled.transpose(1, 2)
        free_pool = free_seq.mean(dim=1)
        return free_seq, free_pool
