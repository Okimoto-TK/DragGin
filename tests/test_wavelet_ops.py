import torch

from src.model.wno.wavelet_ops import wavelet_decompose_1d, wavelet_reconstruct_1d


def test_roundtrip_lengths() -> None:
    torch.manual_seed(0)
    for length in [48, 40, 30]:
        x = torch.randn(2, 6, length, dtype=torch.float32)
        pack = wavelet_decompose_1d(x, wavelet="db4", levels=3)
        x2 = wavelet_reconstruct_1d(pack)

        assert x2.shape == x.shape
        assert torch.max(torch.abs(x2 - x)).item() < 1e-3


def test_padding_crop_l30() -> None:
    x = torch.randn(1, 6, 30, dtype=torch.float32)
    pack = wavelet_decompose_1d(x, wavelet="db4", levels=3)
    x2 = wavelet_reconstruct_1d(pack)

    assert pack.orig_len == 30
    assert pack.padded_len % 8 == 0
    assert x2.shape[-1] == 30
