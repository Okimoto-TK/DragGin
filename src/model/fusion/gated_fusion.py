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
        gate_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.enable_free_branch = enable_free_branch
        self.gate_temperature = float(gate_temperature)
        self.macro_pool_norm = nn.LayerNorm(hidden_dim)
        self.micro_pool_norm = nn.LayerNorm(hidden_dim)
        self.mezzo_pool_norm = nn.LayerNorm(hidden_dim)
        self.guided_pool_norm = nn.LayerNorm(hidden_dim)
        self.free_pool_norm = nn.LayerNorm(hidden_dim)
        self.gate_norm = nn.LayerNorm(6 * hidden_dim)
        self.output_pool_norm = nn.LayerNorm(hidden_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(6 * hidden_dim, hidden_dim),
            nn.GELU(),
            # Vector gate: emit one logit per hidden channel for channel-wise fusion.
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

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
        force_gate_value: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        macro_pool_n = self.macro_pool_norm(macro_pool)
        micro_pool_n = self.micro_pool_norm(micro_pool)
        mezzo_pool_n = self.mezzo_pool_norm(mezzo_pool)
        guided_pool_n = self.guided_pool_norm(guided_pool)
        free_pool_n = self.free_pool_norm(free_pool)
        guided_free_gap = torch.abs(guided_pool_n - free_pool_n)

        gate_in = torch.cat(
            [
                macro_pool_n,
                micro_pool_n,
                mezzo_pool_n,
                guided_pool_n,
                free_pool_n,
                guided_free_gap,
            ],
            dim=-1,
        )
        gate_logits = self.gate_mlp(self.gate_norm(gate_in))
        if not torch.isfinite(gate_logits).all():
            raise RuntimeError("non-finite gate_logits")
        # Temperature divides logits so larger temp softens the gate response.
        temp = max(self.gate_temperature, 1e-6)
        g = torch.sigmoid(gate_logits / temp)
        if not torch.isfinite(g).all():
            raise RuntimeError("non-finite gate")
        if force_gate_value is not None:
            g = torch.full_like(g, float(force_gate_value))

        if self.enable_free_branch:
            alpha = g.unsqueeze(1)
            fused_seq = alpha * guided_seq + (1.0 - alpha) * free_seq
        else:
            fused_seq = guided_seq

        macro_w = mask_macro.to(dtype=fused_seq.dtype).unsqueeze(-1)
        fused_seq = fused_seq * macro_w
        fused_pool_raw = _masked_mean(fused_seq, mask_macro)
        fused_pool_stable = self.output_pool_norm(fused_pool_raw)
        if not torch.isfinite(fused_pool_stable).all():
            raise RuntimeError("non-finite fused_pool")
        fused_pool = fused_pool_raw

        aux = {
            "gate": g,
            "gate_logits": gate_logits,
            "guided_pool": guided_pool,
            "free_pool": free_pool,
            "guided_pool_normed": guided_pool_n,
            "free_pool_normed": free_pool_n,
            "macro_pool_normed": macro_pool_n,
            "mezzo_pool_normed": mezzo_pool_n,
            "micro_pool_normed": micro_pool_n,
        }
        return fused_seq, fused_pool, aux


class MultiScaleFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        enable_free_branch: bool = True,
        gate_temperature: float = 1.0,
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
        force_gate_value: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        guided_seq, guided_pool = self.guided(
            macro_seq=macro_seq,
            micro_seq=micro_seq,
            mezzo_seq=mezzo_seq,
            mask_macro=mask_macro,
            mask_micro=mask_micro,
            mask_mezzo=mask_mezzo,
        )
        if self.gated.enable_free_branch:
            free_seq, free_pool = self.free(micro_seq=micro_seq, mask_micro=mask_micro, target_len=macro_seq.shape[1])
        else:
            free_seq = guided_seq
            free_pool = guided_pool

        return self.gated(
            guided_seq=guided_seq,
            guided_pool=guided_pool,
            free_seq=free_seq,
            free_pool=free_pool,
            macro_pool=macro_pool,
            mezzo_pool=mezzo_pool,
            micro_pool=micro_pool,
            mask_macro=mask_macro,
            force_gate_value=force_gate_value,
        )
