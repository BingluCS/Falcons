"""
Test script for pyfalcons - Python interface for Falcons compression

Usage:
    pytest tests/                    # Run with pytest
    python tests/test_pyfalcons.py       # Run standalone
"""

import sys
import numpy as np

from pyfalcons import falcons, falconsConfig, falconsErrorBoundMode, falconsAlgorithm


def test_compression():
    """Test basic compression and decompression"""
    
    print("=" * 70)
    print("pyfalcons Basic Test")
    print("=" * 70)
    
    # Create test data
    print("\n[1] Creating test data...")
    data = np.random.randn(100, 100).astype(np.float32)
    print(f"    ✓ Shape: {data.shape}, dtype: {data.dtype}, size: {data.nbytes} bytes")
    
    # Create config
    print("[2] Creating config...")
    config = falconsConfig(100, 100)
    config.errorBoundMode = falconsErrorBoundMode.ABS
    config.absErrorBound = 0.01
    print(f"    ✓ Mode: {config.errorBoundMode}, Bound: {config.absErrorBound}")
    
    # Compress
    print("[3] Compressing...")
    data_original = data.copy()
    compressed, ratio = falcons.compress(data, config, copy=True)
    print(f"    ✓ Ratio: {ratio:.2f}x ({data.nbytes} → {len(compressed)} bytes)")
    assert np.array_equal(data, data_original), "copy=True should leave input data unchanged"
    
    # Decompress
    print("[4] Decompressing...")
    decompressed, _ = falcons.decompress(compressed, data.dtype, data.shape, config)
    print(f"    ✓ Shape: {decompressed.shape}")
    
    # Verify
    print("[5] Verifying...")
    max_error, psnr, nrmse = falcons.verify(data, decompressed)
    print(f"    ✓ Max error: {max_error:.2e}, PSNR: {psnr:.2f} dB, NRMSE: {nrmse:.2e}")
    
    assert max_error <= 0.01, f"Error {max_error} exceeds bound 0.01"
    
    # Test double precision
    print("[6] Testing double precision...")
    data_double = np.random.randn(50, 50).astype(np.float64)
    config_double = falconsConfig(50, 50)
    config_double.errorBoundMode = falconsErrorBoundMode.ABS
    config_double.absErrorBound = 1e-6
    compressed_d, ratio_d = falcons.compress(data_double, config_double, copy=True)
    decompressed_d, _ = falcons.decompress(compressed_d, data_double.dtype, data_double.shape, config_double)
    max_error_d, _, _ = falcons.verify(data_double, decompressed_d)
    print(f"    ✓ Ratio: {ratio_d:.2f}x, Max error: {max_error_d:.2e}")
    assert max_error_d <= 1e-6, f"Double error {max_error_d} exceeds bound 1e-6"
    
    # Test 3D data
    print("[7] Testing 3D data...")
    data_3d = np.random.randn(20, 30, 40).astype(np.float32)
    config_3d = falconsConfig(20, 30, 40)
    config_3d.errorBoundMode = falconsErrorBoundMode.REL
    config_3d.relErrorBound = 0.001
    compressed_3d, ratio_3d = falcons.compress(data_3d, config_3d, copy=True)
    decompressed_3d, _ = falcons.decompress(compressed_3d, data_3d.dtype, data_3d.shape, config_3d)
    max_error_3d, _, _ = falcons.verify(data_3d, decompressed_3d)
    print(f"    ✓ Ratio: {ratio_3d:.2f}x, Max error: {max_error_3d:.2e}")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)


def test_compress_default_mutates_input():
    """copy=False is the default and passes the NumPy buffer directly to Falcons."""
    data = np.random.default_rng(0).normal(size=(100, 100)).astype(np.float32)
    original = data.copy()

    config = falconsConfig(data.shape)
    config.errorBoundMode = falconsErrorBoundMode.ABS
    config.absErrorBound = 0.01

    compressed, _ = falcons.compress(data, config)
    assert not np.array_equal(data, original), "copy=False should allow Falcons to overwrite input data"

    decompressed, _ = falcons.decompress(compressed, original.dtype, original.shape)
    max_error, _, _ = falcons.verify(original, decompressed)
    assert max_error <= config.absErrorBound


if __name__ == '__main__':
    try:
        test_compression()
        test_compress_default_mutates_input()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
