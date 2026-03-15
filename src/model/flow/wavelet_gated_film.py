from __future__ import annotations

import torch
from torch import nn


class WaveletGatedFiLM(nn.Module):
    def __init__(self, hidden_dim: int, flow_raw_dim: int = 24, hidden_mid: int | None = None) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.flow_raw_dim = int(flow_raw_dim)
        mid = int(hidden_mid) if hidden_mid is not None else max(128, self.hidden_dim // 2)
        self.param_head = nn.Sequential(
            nn.Linear(self.flow_raw_dim, mid),
            nn.GELU(),
            nn.Linear(mid, 3 * self.hidden_dim),
        )
        self._init_identity()

    def _init_identity(self) -> None:
        last = self.param_head[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        with torch.no_grad():
            last.bias[2 * self.hidden_dim :].fill_(-2.0)

    def forward(
        self,
        coeff: torch.Tensor,
        flow_lpe: torch.Tensor,
        force_gate_value: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if coeff.ndim != 3:
            raise ValueError("coeff must be shaped [B, H, L].")
        if flow_lpe.ndim != 2:
            raise ValueError("flow_lpe must be shaped [B, 24].")
        b, h, _ = coeff.shape
        if h != self.hidden_dim:
            raise ValueError(f"coeff hidden dim mismatch: expected {self.hidden_dim}, got {h}")
        if flow_lpe.shape[0] != b or flow_lpe.shape[1] != self.flow_raw_dim:
            raise ValueError(f"flow_lpe shape mismatch: expected [B, {self.flow_raw_dim}]")

        raw = self.param_head(flow_lpe)
        raw_gamma, raw_beta, raw_gate = raw.split(self.hidden_dim, dim=-1)

        gamma = 1.0 + 0.1 * torch.tanh(raw_gamma)
        beta = raw_beta
        gate = torch.sigmoid(raw_gate)
        if force_gate_value is not None:
            gate = torch.full_like(gate, float(force_gate_value))

        gamma_u = gamma.unsqueeze(-1)
        beta_u = beta.unsqueeze(-1)
        gate_u = gate.unsqueeze(-1)

        film = gamma_u * coeff + beta_u
        delta = film - coeff
        coeff_tilde = coeff + gate_u * delta

        aux = {
            "flow_gate_mean": gate.mean(),
            "flow_gate_std": gate.std(unbiased=False),
            "flow_gamma_mean": gamma.mean(),
            "flow_beta_mean": beta.mean(),
        }
        return coeff_tilde, aux
