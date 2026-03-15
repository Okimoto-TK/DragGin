"""Module 4 WNO1D encoder.

Input contract:
- x: [B, L, C]
- mask: [B, L]

Internal wavelet contract (Module 3):
- [B, D, L]
"""

from __future__ import annotations

import torch
from torch import nn

from src.model.flow import WaveletGatedFiLM

from .soft_threshold_gate import SoftThresholdGate
from .wavelet_ops import CoefPack, wavelet_decompose_1d, wavelet_reconstruct_1d


class WNO1DEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        levels: int = 3,
        wavelet: str = "db4",
        pad_mode: str = "circular",
        enable_dynamic_threshold: bool = True,
        init_lambda: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.levels = int(levels)
        self.wavelet = wavelet
        self.pad_mode = pad_mode
        self.enable_dynamic_threshold = bool(enable_dynamic_threshold)

        self.input_proj = nn.Conv1d(self.in_dim, self.hidden_dim, kernel_size=1)
        self.mix = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1),
            nn.GELU(),
        )

        if self.enable_dynamic_threshold:
            self.approx_gate = SoftThresholdGate(init_lambda=init_lambda)
            self.detail_gates = nn.ModuleList(
                [SoftThresholdGate(init_lambda=init_lambda) for _ in range(self.levels)]
            )
        else:
            self.approx_gate = None
            self.detail_gates = None

        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_states: list[torch.Tensor] | None = None
        if torch.cuda.is_available():
            cuda_rng_states = torch.cuda.get_rng_state_all()
        try:
            self.flow_film_approx = WaveletGatedFiLM(hidden_dim=hidden_dim, flow_raw_dim=24)
            self.flow_film_details = nn.ModuleList([WaveletGatedFiLM(hidden_dim=hidden_dim, flow_raw_dim=24) for _ in range(self.levels)])
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)

    def _mask_to_channel(self, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return mask.unsqueeze(1).to(dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        flow_raw: torch.Tensor | None = None,
        force_flow_gate_value: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        if x.ndim != 3:
            raise ValueError("x must be shaped [B, L, C].")
        if mask.ndim != 2:
            raise ValueError("mask must be shaped [B, L].")
        if x.shape[0] != mask.shape[0] or x.shape[1] != mask.shape[1]:
            raise ValueError("x and mask batch/length dimensions must match.")

        # [B, L, C] -> [B, C, L] for Module 3 compatibility.
        x_perm = x.permute(0, 2, 1)
        mask_c = self._mask_to_channel(mask, x_perm.dtype)
        x_perm = x_perm * mask_c

        z = self.input_proj(x_perm)
        pack = wavelet_decompose_1d(z, wavelet=self.wavelet, levels=self.levels, pad_mode=self.pad_mode)

        flow_approx_aux: dict[str, torch.Tensor] | None = None
        flow_detail_aux_list: list[dict[str, torch.Tensor]] = []
        should_apply_flow_film = (flow_raw is not None) and (force_flow_gate_value != 0.0)
        if should_apply_flow_film:
            assert flow_raw is not None
            pack.approx, flow_approx_aux = self.flow_film_approx(pack.approx, flow_raw, force_flow_gate_value)
            mod_details: list[torch.Tensor] = []
            for i, detail in enumerate(pack.details):
                detail_mod, detail_aux = self.flow_film_details[i](detail, flow_raw, force_flow_gate_value)
                mod_details.append(detail_mod)
                flow_detail_aux_list.append(detail_aux)
            pack.details = mod_details

        if self.enable_dynamic_threshold:
            assert self.approx_gate is not None
            assert self.detail_gates is not None

            approx_thr, lam_approx = self.approx_gate(pack.approx)
            details_thr = []
            lam_all = [lam_approx]
            for i, detail in enumerate(pack.details):
                d_thr, lam_detail = self.detail_gates[i](detail)
                details_thr.append(d_thr)
                lam_all.append(lam_detail)

            gated_pack = CoefPack(
                approx=approx_thr,
                details=details_thr,
                orig_len=pack.orig_len,
                padded_len=pack.padded_len,
                levels=pack.levels,
                wavelet=pack.wavelet,
                pad_mode=pack.pad_mode,
            )
            lam_cat = torch.cat([lam.reshape(lam.shape[0], -1) for lam in lam_all], dim=1)
            lambda_mean = lam_cat.mean(dim=1)
            lambda_max = lam_cat.max(dim=1).values
        else:
            gated_pack = pack
            batch = x.shape[0]
            device = x.device
            dtype = x.dtype
            lambda_mean = torch.zeros(batch, dtype=dtype, device=device)
            lambda_max = torch.zeros(batch, dtype=dtype, device=device)

        z_rec = wavelet_reconstruct_1d(gated_pack)
        out = self.mix(z + z_rec)

        h_seq = out.permute(0, 2, 1)
        h_seq = h_seq * mask.unsqueeze(-1).to(dtype=h_seq.dtype)

        mask_f = mask.to(dtype=h_seq.dtype)
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        h_pool = (h_seq * mask_f.unsqueeze(-1)).sum(dim=1) / denom

        aux = {
            "lambda_mean": lambda_mean,
            "lambda_max": lambda_max,
        }
        if flow_approx_aux is None:
            zero = torch.zeros((), dtype=h_seq.dtype, device=h_seq.device)
            aux.update(
                {
                    "flow_gate_approx_mean": zero,
                    "flow_gate_approx_std": zero,
                    "flow_gamma_approx_mean": zero,
                    "flow_beta_approx_mean": zero,
                    "flow_gate_detail_mean": zero,
                    "flow_gate_detail_std": zero,
                    "flow_gamma_detail_mean": zero,
                    "flow_beta_detail_mean": zero,
                }
            )
        else:
            detail_gate_mean = torch.stack([a["flow_gate_mean"] for a in flow_detail_aux_list]).mean()
            detail_gate_std = torch.stack([a["flow_gate_std"] for a in flow_detail_aux_list]).mean()
            detail_gamma_mean = torch.stack([a["flow_gamma_mean"] for a in flow_detail_aux_list]).mean()
            detail_beta_mean = torch.stack([a["flow_beta_mean"] for a in flow_detail_aux_list]).mean()
            aux.update(
                {
                    "flow_gate_approx_mean": flow_approx_aux["flow_gate_mean"],
                    "flow_gate_approx_std": flow_approx_aux["flow_gate_std"],
                    "flow_gamma_approx_mean": flow_approx_aux["flow_gamma_mean"],
                    "flow_beta_approx_mean": flow_approx_aux["flow_beta_mean"],
                    "flow_gate_detail_mean": detail_gate_mean,
                    "flow_gate_detail_std": detail_gate_std,
                    "flow_gamma_detail_mean": detail_gamma_mean,
                    "flow_beta_detail_mean": detail_beta_mean,
                }
            )
        return h_seq, h_pool, aux
