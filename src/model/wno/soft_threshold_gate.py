"""Dynamic soft-threshold gate for Module 3.

Axis contract: input/output tensors are shaped ``[B, D, L]``.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SoftThresholdGate(nn.Module):
    """Applies channel-wise dynamic soft-thresholding on [B, D, L] coefficients.

    Lambda is computed as:
      E = mean(w^2, dim=-1, keepdim=True)
      lam = softplus(linear(log(E + eps)))

    ``linear`` is an affine scalar map shared across all channels/samples.
    Bias is initialized so ``lam`` starts close to ``init_lambda``.
    """

    def __init__(self, init_lambda: float, eps: float = 1e-6, per: str = "sample_channel") -> None:
        super().__init__()
        if per != "sample_channel":
            raise ValueError("Only per='sample_channel' is supported.")
        if init_lambda < 0:
            raise ValueError("init_lambda must be non-negative.")
        self.eps = float(eps)
        self.per = per
        self.linear = nn.Linear(1, 1)

        # Choose bias so softplus(bias) ≈ init_lambda at startup.
        target = float(init_lambda)
        if target > 20.0:
            bias = target
        else:
            bias = math.log(math.expm1(max(target, 1e-12)))
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, bias)

    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if w.ndim != 3:
            raise ValueError("w must be shaped [B, D, Lc].")

        energy = torch.mean(w.float().pow(2), dim=-1, keepdim=True)
        log_e = torch.log(energy + self.eps)

        lam_in = log_e.reshape(-1, 1)
        lam = F.softplus(self.linear(lam_in)).reshape_as(log_e).to(dtype=w.dtype)

        w_thr = torch.sign(w) * F.relu(torch.abs(w) - lam)
        return w_thr, lam
