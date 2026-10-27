# pyfalcons

Python bindings for **Falcons** — ultra-fast, error-bounded lossy compression for scientific data.

## Overview

`pyfalcons` provides a clean Python interface to Falcons, built with Cython for high performance. It compresses NumPy arrays (`float32`, `float64`, `int32`, `int64`) under a user-specified error bound and recovers them to within that bound.

## Installation

### Building from source

Install from the complete source tree. You need:

- **CMake ≥ 3.18**
- a **C++17 compiler** (g++, clang++, or MSVC)
- **Python development headers** (`python3-dev` / `python3-devel`)

From the repository root:

```bash
cd tools/pyfalcons
pip install -e .
```

During a source install, `setup.py` builds Falcons from source (with bundled Zstd), compiles the Cython bindings against it, and packages everything together.

### SIMD acceleration (AVX2 / SVE2)

Falcons's fast paths are gated on the compiler's SIMD flags, so `pyfalcons` decides them **when it compiles the bindings** (the translation unit that includes the Falcons headers). Without a flag, Falcons falls back to the portable scalar path.

- **Default — auto-detect:** builds with `-march=native`, which enables **AVX2** on an AVX2-capable x86 and **SVE2** on an SVE2-capable ARM, and stays scalar on anything else. Ideal when you build and run on the same machine.
- **Explicit:** set the `PYFALCONS_SIMD` environment variable before installing:

  | `PYFALCONS_SIMD` | effect |
  |---|---|
  | *(unset)* / `auto` / `native` | `-march=native` (default) — the build machine's ISA |
  | `avx2` | `-mavx2 -mfma` — a portable AVX2 floor (x86) |
  | `sve2` | `-march=armv8.6-a+sve2` — ARM SVE2 |
  | `none` / `scalar` | no SIMD — portable everywhere, slowest |
  | *(anything else)* | passed verbatim, e.g. `PYFALCONS_SIMD="-march=znver4"` |

  ```bash
  PYFALCONS_SIMD=avx2 pip install -e .     # force a portable AVX2 floor
  PYFALCONS_SIMD=none pip install -e .     # scalar, maximum portability
  ```

> **Portability note:** `-march=native` (and `avx2` / `sve2`) bake those instructions into the `.so`, so it must run on a CPU that supports them — an unsupported CPU raises `SIGILL`. For a redistributable wheel, pick an explicit floor (`avx2`) or `none`; for a local `pip install -e .`, the default `native` is best. The chosen flags are printed during the build: `[pyfalcons] SIMD flags: …`.

## Quick Start

```python
import numpy as np
from pyfalcons import falcons, falconsConfig, falconsErrorBoundMode, falconsAlgorithm

# Create test data
data = np.random.rand(8, 8, 128).astype(np.float32)

# Configure
config = falconsConfig(data.shape)
config.errorBoundMode = falconsErrorBoundMode.ABS
config.absErrorBound  = 1e-3
# optional: config.cmprAlgo = falconsAlgorithm.INTERP_LORENZO

# Compress on a private copy so the original array can still be used for verification
compressed, ratio = falcons.compress(data, config, copy=True)
print(f"Compression ratio: {ratio:.2f}x")

# Decompress (the Falcons config is recovered from the stream, so config is optional)
decompressed, config = falcons.decompress(compressed, np.float32, data.shape)

# Verify
max_err, psnr, nrmse = falcons.verify(data, decompressed)
print(f"Max error: {max_err:.2e}, PSNR: {psnr:.2f} dB, NRMSE: {nrmse:.2e}")
```

## API Reference

### `falconsConfig`

Configuration object mirroring the C++ `Config` class.

```python
config = falconsConfig(data.shape)       # shape tuple (recommended)
config = falconsConfig((100, 200, 300))  # tuple
config = falconsConfig([100, 200, 300])  # list
config = falconsConfig(100, 200, 300)    # individual dimensions
```

Common fields:

- `errorBoundMode` — an `falconsErrorBoundMode` (see below)
- `absErrorBound`, `relErrorBound` — the bound values
- `cmprAlgo` — an `falconsAlgorithm` (see below)
- `interpAlgo` — an `falconsInterpAlgorithm` (`LINEAR` or `CUBIC`)

Load settings from an INI config file:

```python
config = falconsConfig(data.shape)
config.loadcfg('falcons.config')
```

### Enums

```python
from pyfalcons import falconsErrorBoundMode, falconsAlgorithm, falconsInterpAlgorithm

falconsErrorBoundMode.ABS | REL | PSNR | L2NORM | ABS_AND_REL | ABS_OR_REL
falconsAlgorithm.LORENZO_REG | INTERP_LORENZO | INTERP | NOPRED | LOSSLESS
falconsInterpAlgorithm.LINEAR | CUBIC
```

### `falcons.compress()`

```python
falcons.compress(data, config, copy=False) -> (compressed, ratio)
```

Compress a NumPy array. Dimensions are inferred from the array shape. By default (`copy=False`), the NumPy buffer is passed directly to Falcons for maximum speed, and `data` may be overwritten in place. The input must be writable and C-contiguous. Pass `copy=True` to compress a private C-contiguous copy and leave `data` unchanged.

- `data` (ndarray): `float32`, `float64`, `int32`, or `int64`
- `config` (`falconsConfig` or `str`): config object, or a path to a config file
- `copy` (bool): `False` by default for zero-copy compression; `True` preserves the input array
- returns `compressed` (uint8 ndarray) and `ratio` (original / compressed size)

### `falcons.decompress()`

```python
falcons.decompress(compressed, dtype, shape, config=None) -> (data, config)
```

Decompress back to a NumPy array. The Falcons configuration is recovered from the stream, so `config` is optional.

- `compressed` (ndarray): uint8 array from `compress()`
- `dtype` (type): `np.float32`, `np.float64`, `np.int32`, or `np.int64`
- `shape` (tuple): shape of the original data
- returns the decompressed `data` and the recovered `config`

### `falcons.verify()`

```python
falcons.verify(src_data, dec_data) -> (max_diff, psnr, nrmse)
```

Quality metrics between the original and decompressed arrays: maximum absolute difference, PSNR (dB), and NRMSE.

## Usage Examples

### Relative error bound

```python
config = falconsConfig(data.shape)
config.errorBoundMode = falconsErrorBoundMode.REL
config.relErrorBound  = 1e-4          # 0.01% relative error
compressed, ratio = falcons.compress(data, config)
```

### Choosing an algorithm

```python
config.cmprAlgo = falconsAlgorithm.INTERP_LORENZO   # default (best quality)
config.cmprAlgo = falconsAlgorithm.INTERP           # interpolation only
config.cmprAlgo = falconsAlgorithm.LORENZO_REG      # Lorenzo / regression
config.cmprAlgo = falconsAlgorithm.LOSSLESS         # lossless only
```

### Double precision

```python
data = np.random.randn(50, 50, 50).astype(np.float64)
config = falconsConfig(data.shape)
config.errorBoundMode = falconsErrorBoundMode.ABS
config.absErrorBound  = 1e-6
compressed, ratio     = falcons.compress(data, config)
decompressed, config  = falcons.decompress(compressed, np.float64, data.shape)
```

### Save / load compressed data

```python
# Compress and save
compressed, ratio = falcons.compress(data, config)
compressed.tofile('data.falcons')

# Later: load and decompress
compressed = np.fromfile('data.falcons', dtype=np.uint8)
decompressed, config = falcons.decompress(compressed, np.float32, (8, 8, 128))
```

## Troubleshooting

**`ImportError: cannot import name 'falcons' from 'pyfalcons'`** — reinstall the package:

```bash
cd tools/pyfalcons
pip install -e .
```

**`OSError: libzstd.so: cannot open shared object file`** — the wheel should include its bundled Zstd library. Rebuild from the complete source tree:

```bash
pip install --no-cache-dir --force-reinstall .
```

## Tests

After installation, run from `tools/pyfalcons`:

```bash
python -m pip install pytest
python -m pytest tests/
```

The suite covers the `pyfalcons` API with float32, float64, int32, and int64 arrays, including two- and three-dimensional round trips.

## License

See [LICENSE](LICENSE) for the project license and bundled third-party notices.
Falcons-specific code and modifications are authored and maintained by Bing Lu.
Contributions by Bing Lu use the BSD 3-Clause license; third-party components
retain their applicable terms, as included in the package LICENSE.

This product includes software produced by UChicago Argonne, LLC under
Contract No. DE-AC02-06CH11357 with the Department of Energy.

### Source distribution

`python -m build` produces a source archive that includes the C++ headers and
vendored FSE sources required to build the extension independently of a Git checkout.
Building the extension still requires CMake, a C++17 compiler, and access to the
bundled Zstd download on the first build.

Decompression validates the requested element count before writing into the NumPy
array. Pass the original dtype and shape. Empty input and truncated headers raise
Python exceptions.
