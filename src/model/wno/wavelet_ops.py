"""Wavelet operators for Module 3.

Axis contract: all tensors are shaped ``[B, D, L]`` where ``B`` is batch,
``D`` is channel/latent dimension, and ``L`` is sequence length.

Coefficient packing order in ``CoefPack.details`` is level order
``[detail_level1, detail_level2, ..., detail_levelN]`` where level1 is the
first (highest-resolution) decomposition step.

Padding behavior: decomposition pads circularly so ``padded_len`` is divisible
by ``2**levels``. Reconstruction always crops the final output back to
``orig_len``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_DB4_DEC_LO = [
    -0.010597401785069032,
    0.0328830116668852,
    0.030841381835560764,
    -0.18703481171909309,
    -0.027983769416859854,
    0.6308807679298587,
    0.7148465705529157,
    0.2303778133088965,
]
_DB4_DEC_HI = [
    -0.2303778133088965,
    0.7148465705529157,
    -0.6308807679298587,
    -0.027983769416859854,
    0.18703481171909309,
    0.030841381835560764,
    -0.0328830116668852,
    -0.010597401785069032,
]
_DB4_REC_LO = [
    0.2303778133088965,
    0.7148465705529157,
    0.6308807679298587,
    -0.027983769416859854,
    -0.18703481171909309,
    0.030841381835560764,
    0.0328830116668852,
    -0.010597401785069032,
]
_DB4_REC_HI = [
    -0.010597401785069032,
    -0.0328830116668852,
    0.030841381835560764,
    0.18703481171909309,
    -0.027983769416859854,
    -0.6308807679298587,
    0.7148465705529157,
    -0.2303778133088965,
]


@dataclass
class CoefPack:
    approx: torch.Tensor
    details: list[torch.Tensor]
    orig_len: int
    padded_len: int
    levels: int
    wavelet: str
    pad_mode: str


def _build_filters(dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, ...]:
    dec_lo = torch.tensor(_DB4_DEC_LO, dtype=dtype, device=device)
    dec_hi = torch.tensor(_DB4_DEC_HI, dtype=dtype, device=device)
    rec_lo = torch.tensor(_DB4_REC_LO, dtype=dtype, device=device)
    rec_hi = torch.tensor(_DB4_REC_HI, dtype=dtype, device=device)
    return dec_lo, dec_hi, rec_lo, rec_hi


def _analysis_periodic(x: torch.Tensor, dec_lo: torch.Tensor, dec_hi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, channels, length = x.shape
    if length % 2 != 0:
        raise ValueError("Analysis requires even length per level.")

    filt_len = dec_lo.numel()
    half_len = length // 2

    n = torch.arange(half_len, device=x.device)
    k = torch.arange(filt_len, device=x.device)
    idx = (2 * n[:, None] + k[None, :]) % length

    x_segments = x[:, :, idx]  # [B, D, L/2, K]
    approx = torch.sum(x_segments * dec_lo.view(1, 1, 1, -1), dim=-1)
    detail = torch.sum(x_segments * dec_hi.view(1, 1, 1, -1), dim=-1)
    return approx, detail


def _synthesis_periodic(
    approx: torch.Tensor,
    detail: torch.Tensor,
    rec_lo: torch.Tensor,
    rec_hi: torch.Tensor,
) -> torch.Tensor:
    bsz, channels, half_len = approx.shape
    filt_len = rec_lo.numel()
    out_len = half_len * 2

    n = torch.arange(half_len, device=approx.device)
    k = torch.arange(filt_len, device=approx.device)
    idx = (2 * n[:, None] + k[None, :]) % out_len

    rec_lo = rec_lo.flip(0)
    rec_hi = rec_hi.flip(0)
    src = (
        approx.unsqueeze(-1) * rec_lo.view(1, 1, 1, -1)
        + detail.unsqueeze(-1) * rec_hi.view(1, 1, 1, -1)
    )
    out = torch.zeros((bsz, channels, out_len), dtype=approx.dtype, device=approx.device)
    out.scatter_add_(2, idx.view(1, 1, -1).expand(bsz, channels, -1), src.reshape(bsz, channels, -1))
    return out


def wavelet_decompose_1d(
    x: torch.Tensor,
    wavelet: str = "db4",
    levels: int = 3,
    pad_mode: str = "circular",
) -> CoefPack:
    if x.ndim != 3:
        raise ValueError("x must be shaped [B, D, L].")
    if wavelet != "db4":
        raise ValueError("Only wavelet='db4' is supported.")
    if pad_mode != "circular":
        raise ValueError("Only pad_mode='circular' is supported.")
    if levels < 1:
        raise ValueError("levels must be >= 1.")

    orig_len = int(x.shape[-1])
    block = 2**levels
    pad_needed = (block - (orig_len % block)) % block
    padded_len = orig_len + pad_needed

    if pad_needed:
        x = torch.cat([x, x[..., :pad_needed]], dim=-1)

    dec_lo, dec_hi, _, _ = _build_filters(x.dtype, x.device)

    details: list[torch.Tensor] = []
    approx = x
    for _ in range(levels):
        approx, detail = _analysis_periodic(approx, dec_lo, dec_hi)
        details.append(detail)

    return CoefPack(
        approx=approx,
        details=details,
        orig_len=orig_len,
        padded_len=padded_len,
        levels=levels,
        wavelet=wavelet,
        pad_mode=pad_mode,
    )


def wavelet_reconstruct_1d(pack: CoefPack) -> torch.Tensor:
    if pack.wavelet != "db4":
        raise ValueError("Only wavelet='db4' is supported.")
    if pack.pad_mode != "circular":
        raise ValueError("Only pad_mode='circular' is supported.")
    if len(pack.details) != pack.levels:
        raise ValueError("pack.details length must equal pack.levels.")

    _, _, rec_lo, rec_hi = _build_filters(pack.approx.dtype, pack.approx.device)

    approx = pack.approx
    for detail in reversed(pack.details):
        approx = _synthesis_periodic(approx, detail, rec_lo, rec_hi)

    x_rec = approx[..., : pack.orig_len]
    return x_rec
