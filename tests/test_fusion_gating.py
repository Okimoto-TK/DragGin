import torch

from src.model.fusion.gated_fusion import GatedFusion, MultiScaleFusion


def _masked_mean(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=seq.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (seq * weights).sum(dim=1) / denom


def _force_gate_constant(module: GatedFusion, bias: float) -> None:
    with torch.no_grad():
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
    assert aux["gate"].shape == (2, 1)


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
