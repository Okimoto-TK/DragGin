import pytest
import torch
from torch import nn

from src.model.fusion.gated_fusion import GatedFusion, MultiScaleFusion


def _masked_mean(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=seq.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (seq * weights).sum(dim=1) / denom


def _force_gate_constant(module: GatedFusion, bias: float) -> None:
    with torch.no_grad():
        module.gate_norm.weight.fill_(1.0)
        module.gate_norm.bias.zero_()
        for param in module.gate_mlp.parameters():
            param.zero_()
        module.gate_mlp[-1].bias.fill_(bias)


def test_fusion_shapes() -> None:
    torch.manual_seed(0)
    bsz, hidden = 2, 16

    model = MultiScaleFusion(hidden_dim=hidden, num_heads=4)
    micro_seq = torch.randn(bsz, 48, hidden)
    mezzo_seq = torch.randn(bsz, 40, hidden)
    macro_seq = torch.randn(bsz, 30, hidden)
    micro_pool = torch.randn(bsz, hidden)
    mezzo_pool = torch.randn(bsz, hidden)
    macro_pool = torch.randn(bsz, hidden)
    mask_micro = torch.ones(bsz, 48, dtype=torch.bool)
    mask_mezzo = torch.ones(bsz, 40, dtype=torch.bool)
    mask_macro = torch.ones(bsz, 30, dtype=torch.bool)

    fused_seq, fused_pool, aux = model(
        micro_seq,
        mezzo_seq,
        macro_seq,
        micro_pool,
        mezzo_pool,
        macro_pool,
        mask_micro,
        mask_mezzo,
        mask_macro,
    )

    assert fused_seq.shape == (2, 30, hidden)
    assert fused_pool.shape == (2, hidden)
    assert aux["gate"].shape == (2, hidden)
    assert aux["gate_logits"].shape == (2, hidden)


def test_gate_one_returns_guided() -> None:
    bsz, seq_len, hidden = 2, 30, 8
    module = GatedFusion(hidden_dim=hidden, enable_free_branch=True)
    _force_gate_constant(module, bias=50.0)

    guided_seq = torch.randn(bsz, seq_len, hidden)
    free_seq = torch.randn(bsz, seq_len, hidden)
    guided_pool = torch.randn(bsz, hidden)
    free_pool = torch.randn(bsz, hidden)
    macro_pool = torch.randn(bsz, hidden)
    mezzo_pool = torch.randn(bsz, hidden)
    micro_pool = torch.randn(bsz, hidden)
    mask_macro = torch.ones(bsz, seq_len, dtype=torch.bool)

    fused_seq, fused_pool, _ = module(
        guided_seq,
        guided_pool,
        free_seq,
        free_pool,
        macro_pool,
        mezzo_pool,
        micro_pool,
        mask_macro,
    )

    assert torch.allclose(fused_seq, guided_seq, atol=1e-5)
    assert torch.allclose(fused_pool, _masked_mean(guided_seq, mask_macro), atol=1e-5)


def test_gate_zero_returns_free() -> None:
    bsz, seq_len, hidden = 2, 30, 8
    module = GatedFusion(hidden_dim=hidden, enable_free_branch=True)
    _force_gate_constant(module, bias=-50.0)

    guided_seq = torch.randn(bsz, seq_len, hidden)
    free_seq = torch.randn(bsz, seq_len, hidden)
    guided_pool = torch.randn(bsz, hidden)
    free_pool = torch.randn(bsz, hidden)
    macro_pool = torch.randn(bsz, hidden)
    mezzo_pool = torch.randn(bsz, hidden)
    micro_pool = torch.randn(bsz, hidden)
    mask_macro = torch.ones(bsz, seq_len, dtype=torch.bool)

    fused_seq, _, _ = module(
        guided_seq,
        guided_pool,
        free_seq,
        free_pool,
        macro_pool,
        mezzo_pool,
        micro_pool,
        mask_macro,
    )

    assert torch.allclose(fused_seq, free_seq, atol=1e-5)


def test_mask_macro_zero_outputs_zero() -> None:
    bsz, hidden = 2, 12
    model = MultiScaleFusion(hidden_dim=hidden, num_heads=4)

    micro_seq = torch.randn(bsz, 48, hidden)
    mezzo_seq = torch.randn(bsz, 40, hidden)
    macro_seq = torch.randn(bsz, 30, hidden)
    micro_pool = torch.randn(bsz, hidden)
    mezzo_pool = torch.randn(bsz, hidden)
    macro_pool = torch.randn(bsz, hidden)
    mask_micro = torch.ones(bsz, 48, dtype=torch.bool)
    mask_mezzo = torch.ones(bsz, 40, dtype=torch.bool)
    mask_macro = torch.zeros(bsz, 30, dtype=torch.bool)

    fused_seq, fused_pool, _ = model(
        micro_seq,
        mezzo_seq,
        macro_seq,
        micro_pool,
        mezzo_pool,
        macro_pool,
        mask_micro,
        mask_mezzo,
        mask_macro,
    )

    assert torch.allclose(fused_seq, torch.zeros_like(fused_seq))
    assert torch.allclose(fused_pool, torch.zeros_like(fused_pool))
    assert torch.isfinite(fused_seq).all()
    assert torch.isfinite(fused_pool).all()


def test_disable_free_branch() -> None:
    bsz, hidden = 2, 10
    model = MultiScaleFusion(hidden_dim=hidden, num_heads=2, enable_free_branch=False)

    micro_seq = torch.randn(bsz, 48, hidden)
    mezzo_seq = torch.randn(bsz, 40, hidden)
    macro_seq = torch.randn(bsz, 30, hidden)
    micro_pool = torch.randn(bsz, hidden)
    mezzo_pool = torch.randn(bsz, hidden)
    macro_pool = torch.randn(bsz, hidden)
    mask_micro = torch.ones(bsz, 48, dtype=torch.bool)
    mask_mezzo = torch.ones(bsz, 40, dtype=torch.bool)
    mask_macro = torch.ones(bsz, 30, dtype=torch.bool)

    guided_seq, _ = model.guided(
        macro_seq=macro_seq,
        micro_seq=micro_seq,
        mezzo_seq=mezzo_seq,
        mask_macro=mask_macro,
        mask_micro=mask_micro,
        mask_mezzo=mask_mezzo,
    )

    fused_seq, _, _ = model(
        micro_seq,
        mezzo_seq,
        macro_seq,
        micro_pool,
        mezzo_pool,
        macro_pool,
        mask_micro,
        mask_mezzo,
        mask_macro,
    )

    assert torch.allclose(fused_seq, guided_seq, atol=1e-5)




class _IdentityAttention(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, None]:
        del key, value, key_padding_mask, need_weights
        return query, None


def test_free_pool_masked_mean_equals_mean_when_all_bins_valid() -> None:
    bsz, seq_len, hidden = 2, 48, 4
    target_len = 6
    model = MultiScaleFusion(hidden_dim=hidden, num_heads=2)
    model.free.attn = _IdentityAttention()

    micro_seq = torch.randn(bsz, seq_len, hidden)
    mask_micro = torch.ones(bsz, seq_len, dtype=torch.bool)

    free_seq, free_pool = model.free(micro_seq=micro_seq, mask_micro=mask_micro, target_len=target_len)

    assert torch.allclose(free_pool, free_seq.mean(dim=1), atol=1e-6)


def test_free_pool_masked_mean_ignores_invalid_bins() -> None:
    bsz, seq_len, hidden = 1, 48, 2
    target_len = 6
    model = MultiScaleFusion(hidden_dim=hidden, num_heads=2)
    model.free.attn = _IdentityAttention()

    micro_seq = torch.ones(bsz, seq_len, hidden)
    mask_micro = torch.zeros(bsz, seq_len, dtype=torch.bool)
    mask_micro[:, :24] = True

    free_seq, free_pool = model.free(micro_seq=micro_seq, mask_micro=mask_micro, target_len=target_len)

    simple_mean = free_seq.mean(dim=1)
    assert torch.allclose(free_pool, torch.ones_like(free_pool), atol=1e-6)
    assert torch.allclose(simple_mean, torch.full_like(simple_mean, 0.5), atol=1e-6)


def test_free_pool_all_invalid_is_finite() -> None:
    bsz, seq_len, hidden = 2, 48, 3
    target_len = 6
    model = MultiScaleFusion(hidden_dim=hidden, num_heads=1)
    model.free.attn = _IdentityAttention()

    micro_seq = torch.randn(bsz, seq_len, hidden)
    mask_micro = torch.zeros(bsz, seq_len, dtype=torch.bool)

    free_seq, free_pool = model.free(micro_seq=micro_seq, mask_micro=mask_micro, target_len=target_len)

    assert torch.isfinite(free_seq).all()
    assert torch.isfinite(free_pool).all()
    assert torch.allclose(free_pool, torch.zeros_like(free_pool), atol=1e-6)

def test_free_branch_partial_mask_safe() -> None:
    torch.manual_seed(42)
    bsz, hidden = 2, 14
    model = MultiScaleFusion(hidden_dim=hidden, num_heads=2)

    micro_seq = torch.randn(bsz, 48, hidden)
    mezzo_seq = torch.randn(bsz, 40, hidden)
    macro_seq = torch.randn(bsz, 30, hidden)
    micro_pool = torch.randn(bsz, hidden)
    mezzo_pool = torch.randn(bsz, hidden)
    macro_pool = torch.randn(bsz, hidden)

    mask_micro = torch.randint(0, 2, (bsz, 48), dtype=torch.bool)
    mask_mezzo = torch.ones(bsz, 40, dtype=torch.bool)
    mask_macro = torch.ones(bsz, 30, dtype=torch.bool)

    fused_seq, fused_pool, aux = model(
        micro_seq,
        mezzo_seq,
        macro_seq,
        micro_pool,
        mezzo_pool,
        macro_pool,
        mask_micro,
        mask_mezzo,
        mask_macro,
    )

    assert fused_seq.shape == (bsz, 30, hidden)
    assert fused_pool.shape == (bsz, hidden)
    assert aux["free_pool"].shape == (bsz, hidden)
    assert torch.isfinite(fused_seq).all()
    assert torch.isfinite(fused_pool).all()
    assert torch.isfinite(aux["free_pool"]).all()


def test_gated_fusion_has_pool_norm_layers_and_shape_unchanged() -> None:
    hidden = 8
    module = GatedFusion(hidden_dim=hidden)
    assert isinstance(module.macro_pool_norm, nn.LayerNorm)
    assert isinstance(module.micro_pool_norm, nn.LayerNorm)
    assert isinstance(module.mezzo_pool_norm, nn.LayerNorm)
    assert isinstance(module.guided_pool_norm, nn.LayerNorm)
    assert isinstance(module.free_pool_norm, nn.LayerNorm)

    bsz, seq_len = 2, 5
    guided_seq = torch.randn(bsz, seq_len, hidden)
    free_seq = torch.randn(bsz, seq_len, hidden)
    pooled = torch.randn(bsz, hidden)
    mask_macro = torch.ones(bsz, seq_len, dtype=torch.bool)
    fused_seq, fused_pool, _ = module(
        guided_seq=guided_seq,
        guided_pool=pooled,
        free_seq=free_seq,
        free_pool=pooled,
        macro_pool=pooled,
        mezzo_pool=pooled,
        micro_pool=pooled,
        mask_macro=mask_macro,
    )
    assert fused_seq.shape == (bsz, seq_len, hidden)
    assert fused_pool.shape == (bsz, hidden)


def test_gated_fusion_raises_on_non_finite_gate_logits() -> None:
    hidden = 4
    module = GatedFusion(hidden_dim=hidden)
    bsz, seq_len = 1, 3
    guided_seq = torch.randn(bsz, seq_len, hidden)
    free_seq = torch.randn(bsz, seq_len, hidden)
    pooled = torch.randn(bsz, hidden)
    pooled[0, 0] = float("inf")
    mask_macro = torch.ones(bsz, seq_len, dtype=torch.bool)

    with pytest.raises(RuntimeError, match="non-finite gate_logits"):
        module(
            guided_seq=guided_seq,
            guided_pool=pooled,
            free_seq=free_seq,
            free_pool=pooled,
            macro_pool=pooled,
            mezzo_pool=pooled,
            micro_pool=pooled,
            mask_macro=mask_macro,
        )



def test_disable_free_branch_does_not_execute_free_path() -> None:
    bsz, hidden = 2, 10
    model = MultiScaleFusion(hidden_dim=hidden, num_heads=2, enable_free_branch=False)

    def _boom(*args, **kwargs):
        raise AssertionError("free branch should not run when disabled")

    model.free.forward = _boom  # type: ignore[assignment]

    micro_seq = torch.randn(bsz, 48, hidden)
    mezzo_seq = torch.randn(bsz, 40, hidden)
    macro_seq = torch.randn(bsz, 30, hidden)
    micro_pool = torch.randn(bsz, hidden)
    mezzo_pool = torch.randn(bsz, hidden)
    macro_pool = torch.randn(bsz, hidden)
    mask_micro = torch.ones(bsz, 48, dtype=torch.bool)
    mask_mezzo = torch.ones(bsz, 40, dtype=torch.bool)
    mask_macro = torch.ones(bsz, 30, dtype=torch.bool)

    fused_seq, fused_pool, aux = model(
        micro_seq,
        mezzo_seq,
        macro_seq,
        micro_pool,
        mezzo_pool,
        macro_pool,
        mask_micro,
        mask_mezzo,
        mask_macro,
    )

    assert fused_seq.shape == (bsz, 30, hidden)
    assert fused_pool.shape == (bsz, hidden)
    assert torch.isfinite(aux["free_pool"]).all()

