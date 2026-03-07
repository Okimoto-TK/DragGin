import torch

from src.model.wno.wno1d import WNO1DEncoder


def test_wno1d_shapes_all_scales() -> None:
    for length in [48, 40, 30]:
        x = torch.randn(2, length, 6)
        mask = torch.ones(2, length)
        encoder = WNO1DEncoder(in_dim=6, hidden_dim=16, init_lambda=1.0)

        h_seq, h_pool, aux = encoder(x, mask)

        assert h_seq.shape == (2, length, 16)
        assert h_pool.shape == (2, 16)
        assert "lambda_mean" in aux
        assert "lambda_max" in aux


def test_wno1d_zero_mask_outputs_zero() -> None:
    length = 48
    x = torch.randn(2, length, 6)
    mask = torch.zeros(2, length)
    encoder = WNO1DEncoder(in_dim=6, hidden_dim=16, init_lambda=1.0)

    h_seq, h_pool, _ = encoder(x, mask)

    assert torch.isfinite(h_seq).all()
    assert torch.isfinite(h_pool).all()
    assert torch.allclose(h_seq, torch.zeros_like(h_seq), atol=1e-7)
    assert torch.allclose(h_pool, torch.zeros_like(h_pool), atol=1e-7)


def test_wno1d_partial_mask_safe() -> None:
    length = 40
    x = torch.randn(2, length, 6)
    mask = torch.zeros(2, length)
    mask[0, : length // 2] = 1
    mask[1, 5:20] = 1

    encoder = WNO1DEncoder(in_dim=6, hidden_dim=16, init_lambda=1.0)
    h_seq, h_pool, _ = encoder(x, mask)

    assert h_seq.shape == (2, length, 16)
    assert h_pool.shape == (2, 16)
    assert torch.isfinite(h_seq).all()
    assert torch.isfinite(h_pool).all()


def test_wno1d_disable_threshold() -> None:
    length = 30
    x = torch.randn(2, length, 6)
    mask = torch.ones(2, length)

    encoder = WNO1DEncoder(
        in_dim=6,
        hidden_dim=16,
        enable_dynamic_threshold=False,
        init_lambda=1.0,
    )
    h_seq, h_pool, aux = encoder(x, mask)

    assert h_seq.shape == (2, length, 16)
    assert h_pool.shape == (2, 16)
    assert "lambda_mean" in aux
    assert "lambda_max" in aux
