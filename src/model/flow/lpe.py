from __future__ import annotations

import torch
from torch import nn


class FlowLegendreProjectionEncoder(nn.Module):
    """Legendre Projection Encoder for 30-day, 4-feature flow sequence.

    Input:
        flow_x: [B, 30, 4]
        flow_mask: [B, 30]
    Output:
        flow_lpe: [B, 24]  # 4 features * 6 orders
    """

    def __init__(self, window: int = 30, in_features: int = 4, order: int = 6) -> None:
        super().__init__()
        self.window = int(window)
        self.in_features = int(in_features)
        self.order = int(order)
        basis = self._build_legendre_basis(self.window, self.order)
        self.register_buffer("basis", basis, persistent=False)

    @staticmethod
    def _build_legendre_basis(window: int, order: int) -> torch.Tensor:
        x = torch.linspace(-1.0, 1.0, steps=window, dtype=torch.float32)
        basis = []
        p0 = torch.ones_like(x)
        basis.append(p0)
        if order > 1:
            p1 = x
            basis.append(p1)
            prev2, prev1 = p0, p1
            for n in range(2, order):
                pn = ((2 * n - 1) * x * prev1 - (n - 1) * prev2) / float(n)
                basis.append(pn)
                prev2, prev1 = prev1, pn
        return torch.stack(basis, dim=0)  # [order, window]

    def forward(self, flow_x: torch.Tensor, flow_mask: torch.Tensor) -> torch.Tensor:
        if flow_x.ndim != 3:
            raise ValueError("flow_x must be [B, 30, 4].")
        if flow_mask.ndim != 2:
            raise ValueError("flow_mask must be [B, 30].")
        b, t, f = flow_x.shape
        if t != self.window or f != self.in_features:
            raise ValueError(f"flow_x shape must be [B, {self.window}, {self.in_features}].")
        if flow_mask.shape[0] != b or flow_mask.shape[1] != t:
            raise ValueError("flow_mask shape must match flow_x on [B, T].")

        mask = flow_mask.to(dtype=flow_x.dtype)
        basis = self.basis.to(dtype=flow_x.dtype, device=flow_x.device)  # [order, T]

        weighted_basis = basis.unsqueeze(0) * mask.unsqueeze(1)  # [B, order, T]
        denom = weighted_basis.abs().sum(dim=2).clamp_min(1e-6)  # [B, order]

        x_bt = flow_x.transpose(1, 2)  # [B, F, T]
        coeff = torch.einsum("bot,bft->bfo", weighted_basis, x_bt) / denom.unsqueeze(1)
        return coeff.reshape(b, self.in_features * self.order)
