import numpy as np
import pytest
from pyfalcons import falcons, falconsConfig

@pytest.mark.parametrize("size", [1, 16, 256])
def test_small_array_roundtrip(size):
    data = np.linspace(0, 1, size, dtype=np.float32)
    config = falconsConfig(data.shape)
    config.absErrorBound = 1e-3
    compressed, _ = falcons.compress(data, config, copy=True)
    reconstructed, _ = falcons.decompress(compressed, data.dtype, data.shape)
    assert np.max(np.abs(data - reconstructed)) <= 1e-3

def test_wrong_shape_is_rejected_before_decompression():
    data = np.arange(256, dtype=np.float32)
    compressed, _ = falcons.compress(data, falconsConfig(data.shape), copy=True)
    with pytest.raises(ValueError, match="shape"):
        falcons.decompress(compressed, data.dtype, (1,))

@pytest.mark.parametrize("data", [np.array([], dtype=np.uint8), np.array([0], dtype=np.uint8)])
def test_truncated_input_is_rejected(data):
    with pytest.raises(ValueError):
        falcons.decompress(data, np.float32, (1,))

def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        falcons.compress(np.array([], dtype=np.float32), falconsConfig(1))

def test_integer_metrics_do_not_overflow():
    a = np.array([0, 100000], dtype=np.int32)
    b = np.array([50000, 50000], dtype=np.int32)
    maximum, psnr, nrmse = falcons.verify(a, b)
    assert maximum == 50000
    assert nrmse == pytest.approx(0.5)
    assert psnr == pytest.approx(-20 * np.log10(0.5))

def test_high_int64_metrics_preserve_small_differences():
    base = 1 << 60
    a = np.array([base, base + 100], dtype=np.int64)
    b = np.array([base + 1, base + 99], dtype=np.int64)
    maximum, psnr, nrmse = falcons.verify(a, b)
    assert maximum == 1
    assert nrmse == pytest.approx(0.01)
    assert psnr == pytest.approx(40.0)
