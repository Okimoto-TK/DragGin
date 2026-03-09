"""Module 5 guided branch: macro-guided cross-scale attention."""

from __future__ import annotations

import torch
from torch import nn


def _masked_mean(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=seq.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (seq * weights).sum(dim=1) / denom


class CrossScaleAttention(nn.Module):
    """Cross-attention with macro queries over concatenated micro+mezzo keys/values."""

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
        macro_seq: torch.Tensor,
        micro_seq: torch.Tensor,
        mezzo_seq: torch.Tensor,
        mask_macro: torch.Tensor,
        mask_micro: torch.Tensor,
        mask_mezzo: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv = torch.cat([micro_seq, mezzo_seq], dim=1)
        kv_mask = torch.cat([mask_micro, mask_mezzo], dim=1)
        key_padding_mask = ~kv_mask.bool()

        guided_seq, _ = self.attn(
            query=macro_seq,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        macro_w = mask_macro.to(dtype=guided_seq.dtype).unsqueeze(-1)
        guided_seq = guided_seq * macro_w
        guided_pool = _masked_mean(guided_seq, mask_macro)
        return guided_seq, guided_pool
