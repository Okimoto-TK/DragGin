"""Module 5 gated fusion for guided and free branches."""

from __future__ import annotations

import torch
from torch import nn

from src.model.fusion.cross_scale_attention import CrossScaleAttention
from src.model.fusion.micro_self_attention import MicroSelfAttention


def _masked_mean(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=seq.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (seq * weights).sum(dim=1) / denom


class GatedFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        enable_free_branch: bool = True,
        gate_temperature: float = 2.0,
    ) -> None:
        super().__init__()
        self.enable_free_branch = enable_free_branch
        self.gate_temperature = float(gate_temperature)
        self.gate_norm = nn.LayerNorm(6 * hidden_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(6 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        guided_seq: torch.Tensor,
        guided_pool: torch.Tensor,
        free_seq: torch.Tensor,
        free_pool: torch.Tensor,
        macro_pool: torch.Tensor,
        mezzo_pool: torch.Tensor,
        micro_pool: torch.Tensor,
        mask_macro: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        gate_in = torch.cat(
            [
                macro_pool,
                micro_pool,
                mezzo_pool,
                guided_pool,
                free_pool,
                torch.abs(guided_pool - free_pool),
            ],
            dim=-1,
        )
        gate_logits = self.gate_mlp(self.gate_norm(gate_in))
        g = torch.sigmoid(self.gate_temperature * gate_logits)

        if self.enable_free_branch:
            alpha = g.unsqueeze(1)
            fused_seq = alpha * guided_seq + (1.0 - alpha) * free_seq
        else:
            fused_seq = guided_seq

        macro_w = mask_macro.to(dtype=fused_seq.dtype).unsqueeze(-1)
        fused_seq = fused_seq * macro_w
        fused_pool = _masked_mean(fused_seq, mask_macro)

        aux = {
            "gate": g,
            "gate_logits": gate_logits,
            "guided_pool": guided_pool,
            "free_pool": free_pool,
        }
        return fused_seq, fused_pool, aux


class MultiScaleFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        enable_free_branch: bool = True,
        gate_temperature: float = 2.0,
    ) -> None:
        super().__init__()
        self.guided = CrossScaleAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.free = MicroSelfAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.gated = GatedFusion(
            hidden_dim=hidden_dim,
            enable_free_branch=enable_free_branch,
            gate_temperature=gate_temperature,
        )

    def forward(
        self,
        micro_seq: torch.Tensor,
        mezzo_seq: torch.Tensor,
        macro_seq: torch.Tensor,
        micro_pool: torch.Tensor,
        mezzo_pool: torch.Tensor,
        macro_pool: torch.Tensor,
        mask_micro: torch.Tensor,
        mask_mezzo: torch.Tensor,
        mask_macro: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        guided_seq, guided_pool = self.guided(
            macro_seq=macro_seq,
            micro_seq=micro_seq,
            mezzo_seq=mezzo_seq,
            mask_macro=mask_macro,
            mask_micro=mask_micro,
            mask_mezzo=mask_mezzo,
        )
        free_seq, free_pool = self.free(micro_seq=micro_seq, mask_micro=mask_micro, target_len=macro_seq.shape[1])

        return self.gated(
            guided_seq=guided_seq,
            guided_pool=guided_pool,
            free_seq=free_seq,
            free_pool=free_pool,
            macro_pool=macro_pool,
            mezzo_pool=mezzo_pool,
            micro_pool=micro_pool,
            mask_macro=mask_macro,
        )
