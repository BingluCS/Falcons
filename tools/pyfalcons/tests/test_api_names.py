"""Exercise the renamed public Python API through the compiled extension."""
import numpy as np
import pytest

from pyfalcons import falcons, falconsConfig, falconsErrorBoundMode


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("shape", [(256, 256), (16, 32, 128)])
def test_falcons_roundtrip(dtype, shape):
    values = np.sin(np.arange(np.prod(shape)) / 17.0)
    integral = np.issubdtype(dtype, np.integer)
    data = (np.rint(values * 1000) if integral else values).astype(dtype).reshape(shape)
    original = data.copy()
    config = falconsConfig(shape)
    config.errorBoundMode = falconsErrorBoundMode.ABS
    config.absErrorBound = 1.0 if integral else 1e-3

    compressed, ratio = falcons.compress(data, config, copy=True)
    restored, _ = falcons.decompress(compressed, dtype, shape)

    assert ratio > 0
    assert np.array_equal(data, original)
    assert restored.shape == shape
    assert restored.dtype == data.dtype
    assert np.max(np.abs(restored.astype(np.float64) - original.astype(np.float64))) <= config.absErrorBound
