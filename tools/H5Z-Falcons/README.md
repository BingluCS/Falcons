# H5Z-Falcons Filter Integration

H5Z-Falcons is an HDF5 filter for error-bounded lossy compression. It can be
used from C/C++, HDF5 command-line tools and h5py. Reading a compressed dataset
requires the filter and a compatible HDF5 environment on the reading machine.

## Contents

- [Build and configure](#build-and-configure)
- [h5py compression and error bounds](#default-compression)
- [ERA5 dimensions and chunks](#choosing-an-era5-chunk-shape)
- [Slurm and OpenMP](#running-with-slurm)
- [Compress and decompress HDF5 files](#converting-an-existing-era5-file)
- [PyTorch DataLoader](#reading-with-a-pytorch-dataloader)
- [HDF5 command-line tools and C/C++](#hdf5-command-line-tools-and-cc)

## Build and configure

Start in the Falcons repository root with CMake, a C++17 compiler and HDF5
development files. The plugin and the application using it, including h5py,
must use the **same actual shared HDF5 library**. Configure CMake to find
that library and its matching headers.

```bash
cmake -S . -B build -DBUILD_H5Z_FILTER=ON -DBUILD_SHARED_LIBS=ON
cmake --build build --parallel
export HDF5_PLUGIN_PATH="$PWD/build/tools/H5Z-Falcons${HDF5_PLUGIN_PATH:+:$HDF5_PLUGIN_PATH}"
```

For the Python examples, install NumPy, h5py, Cython and setuptools in the
Python environment, then install the Falcons bindings:

```bash
python -m pip install --no-build-isolation -e tools/pyfalcons
```

The plugin directory is `build/tools/H5Z-Falcons` for this build. Set
`HDF5_PLUGIN_PATH` **before importing h5py** and start a fresh Python process
after changing it. On x86 CPUs with AVX2 and FMA support, optionally add
`-DENABLE_AVX2=ON` to the CMake configuration; do not enable it on other
architectures. If changing the HDF5 installation, use a fresh build directory
to avoid cached CMake paths and adjust the paths below accordingly.

`HDF5_PLUGIN_PATH` locates the filter; it does not locate its shared-library
dependencies. If the loader cannot find those dependencies, add their
directories to `LD_LIBRARY_PATH` or use your system's equivalent. Python and
`h5repack` do not require a separate linker command for the plugin.

For a conda environment in which both `h5py` and HDF5 were installed by
conda, pass `-DHDF5_ROOT="$CONDA_PREFIX"` when configuring CMake, then verify
the resolved libraries below. A pip h5py wheel may instead bundle a separate
HDF5 library and needs a plugin built against that same library.

### Check the HDF5 libraries

On Linux, inspect the h5py build and the resolved dependencies:

```bash
python -c 'import h5py; print(h5py.version.info)'
ldd "$(python -c 'import h5py.h5z; print(h5py.h5z.__file__)')"
ldd build/tools/H5Z-Falcons/libhdf5falcons.so
```

Check that both use the same physical `libhdf5` shared library, resolving
symlinks as necessary, and that no dependencies are `not found`. `ldd` is a
starting point; if it does not show a shared HDF5 dependency or runtime loading
differs, inspect the libraries actually loaded by the Python process. Matching
version numbers alone is insufficient. If the libraries differ, rebuild the
plugin against the HDF5 used by h5py or use an environment where they match.

`h5py.h5z.filter_avail(40024)` checks filter availability, but does not prove
that creating or reading a dataset will work. On one server, h5py 3.6 and
HDF5 1.10.7 using the same HDF5 shared object successfully wrote and read
default-compressed `float32` data, with maximum absolute error `0.0009996891`.
In a separate test, a h5py 3.16 wheel carrying HDF5 2.0 and a plugin linked to
system HDF5 1.10 returned `filter_avail=True` but failed to create a dataset.
These observations are not guarantees for other builds or wheels.
The current filter's C++ configuration and round-trip test also passed when
separately built against HDF5 1.10.7 and 2.2.0 on the test server.

## Default compression

With no `compression_opts`, the filter defaults to ABS error bound `1e-3`
and OpenMP disabled. HDF5 filters require chunked datasets. Run this in the
configured environment:

```python
import h5py
import numpy as np

data = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)

with h5py.File("default.h5", "w") as f:
    f.create_dataset("field", data=data, chunks=(32, 32), compression=40024)

with h5py.File("default.h5", "r") as f:
    restored = f["field"][:]

error = np.max(np.abs(data.astype(np.float64) - restored.astype(np.float64)))
print("max absolute error:", error)
```

## Custom error bounds

`h5py_filter` provides h5py's `compression` and `compression_opts` arguments.
No INI file or manual serialization is needed:

```python
import h5py
import numpy as np
from pyfalcons import h5py_filter

data = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
with h5py.File("custom.h5", "w") as f:
    f.create_dataset(
        "field", data=data, chunks=(32, 32),
        **h5py_filter(error_mode="REL", error_bound=1e-3, openmp=False),
    )

with h5py.File("custom.h5", "r") as f:
    restored = f["field"][:]

error = np.max(np.abs(data.astype(np.float64) - restored.astype(np.float64)))
print("max absolute error:", error)
```

Use `error_mode="ABS"` for an absolute error bound, or `"REL"` for a bound
relative to each chunk's range (maximum minus minimum), not the whole file's
range. Partially filled edge chunks may include HDF5 fill values in that range.
`openmp=False` is the default; see [OpenMP](#openmp) before enabling it.

The wrapper uses Falcons' C++ configuration serialization and passes the
resulting integer tuple as h5py's `compression_opts`. The dataset and chunk
dimensions are set by h5py; the filter derives its dimensions and dtype from
the HDF5 chunk. To read an existing file, h5py needs the plugin but does not
need `pyfalcons` or the original Python configuration.

For advanced settings, create a `falconsConfig` and pass
`**config.h5py_options()` instead. Only fields serialized by `Config::save`
are available through this interface; see the
[pyfalcons guide](../pyfalcons/README.md#hdf5--h5py) for details.

## Data types and dimensions

The basic examples use two-dimensional, native-endian `float32`. On the
test server, `float32` and `float64` two-dimensional chunks both passed
ABS/REL round trips; a four-dimensional `float32` dataset with chunks
`(1, 1, H, W)` also passed a REL round trip. Do not use `float16`: the
current filter misclassifies it as double, risking an out-of-bounds memory
access. These tests do not establish compatibility for every dtype, byte
order or chunk shape. Validate your environment and data before relying on
the result.

## Choosing an ERA5 chunk shape

For an annual `float32` dataset with shape `(1460, 257, 721, 1440)`, the
axes are `(time, variable, latitude, longitude)`. Its logical uncompressed
size is approximately **1.56 TB** (1.42 TiB). h5py takes the dataset shape
and `chunks` in that same order.

HDF5 divides the dataset into chunks, and Falcons compresses each chunk
independently. A read that touches part of a compressed chunk still has to
decompress that chunk. Chunk shape is distinct from the number of DataLoader
workers and the number of OpenMP threads used inside a filter call.

For whole-grid access to individual variables, a starting point is:

```python
shape = (1460, 257, 721, 1440)
chunks = (1, 1, 721, 1440)
```

Each chunk is one variable at one time, or 4,152,960 uncompressed bytes
(3.96 MiB). There are 375,220 chunks in a fully written annual dataset.
Because Falcons removes size-one axes when configuring the filter, each of
these chunks is compressed as a `721 x 1440` **2D** array, not as a 4D array.
For `REL`, the error bound is based on this individual variable/time plane's
range. For example, if its minimum is -20 and maximum is 30, `REL=1e-3`
sets an absolute bound of `0.001 x (30 - (-20)) = 0.05` for that chunk.
`ABS=0.01` instead requests an absolute bound of 0.01 for every chunk.

Other shapes can be useful, depending on the actual training read pattern:

| `chunks` | Raw bytes per full chunk | Typical tradeoff |
|---|---:|---|
| `(1, 1, 721, 1440)` | 3.96 MiB | One complete variable/time plane; efficient for whole-grid reads. |
| `(1, 1, 181, 360)` | 0.25 MiB | Spatial tiles; can avoid reading a whole grid for patch reads, but creates many more chunks. |
| `(4, 1, 721, 1440)` | 15.84 MiB | Four times for one variable; each touched chunk decodes all four, even for a shorter time window. |
| `(1, 4, 721, 1440)` | 15.84 MiB | Four variables for one time; selecting one variable still decodes the whole chunk. |

Grouping times or variables changes Falcons' effective dimensions and can
change both throughput and compression ratio. It also changes the scope of a
REL bound: `(1, 4, 721, 1440)` uses a range across four variables, which can be
inappropriate when they have different units or scales. Edge chunks may be
partially filled; their fill values may also affect the range. For the spatial
tile example, an annual dataset has about 6.0 million chunks rather than
375,220. These examples are starting points, not a universal optimum.

With `(1, 1, 721, 1440)` chunks, reading two times and all 257 variables
touches up to 514 chunks, or about 2.13 GB of uncompressed values. Reading two
times and only 19 variables touches up to 38 chunks, or about 158 MB. The
actual disk traffic depends on compression, caching and HDF5's read pattern.
The [DataLoader example](#reading-with-a-pytorch-dataloader) selects channels
in the HDF5 read. The [conversion example](#converting-an-existing-era5-file)
below shows how to write an existing annual HDF5 file one plane at a time,
without holding the 1.56 TB array in RAM.

## Running with Slurm

Build the plugin beforehand in an environment with the HDF5 development
dependencies described above. h5py users do not need to write a linker command
for Python or for each job; h5py loads the plugin at runtime. The plugin and
h5py still need to use the same actual shared HDF5 library.

For example, submit the following as an `sbatch` script after replacing the
example paths. Add account, partition and GPU request directives as required
by the site. The CPU request does not request GPUs.

```bash
#!/bin/bash
#SBATCH --job-name=falcons-h5py
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
# Add site-specific #SBATCH --account, --partition and GPU options here.

set -e
# Example conda environment; alternatively load the site's modules.
source /shared/example/miniconda3/etc/profile.d/conda.sh
conda activate /shared/example/envs/falcons-h5py

export HDF5_PLUGIN_PATH="/shared/example/Falcons/build/tools/H5Z-Falcons${HDF5_PLUGIN_PATH:+:$HDF5_PLUGIN_PATH}"
srun --export=ALL python train.py
```

The plugin directory and its dependencies must be visible on every compute
node. The export prepends it without discarding existing plugin paths. Set
the plugin path before starting Python and make sure every task and data-loader
worker inherits it. To request four Falcons threads per chunk without OpenMP
environment variables, set `num_threads=4` when creating the dataset as shown
below. The default filter configuration has OpenMP disabled.

Slurm does not expand shell environment variables in `#SBATCH` directives;
the output directive uses Slurm's `%x` and `%j` substitutions instead. For a
multi-node job, use a plugin path visible on every node and the same compatible
software environment on each node. Launching multiple Slurm tasks or data
loader workers is not the same as MPI parallel HDF5; that requires an
MPI-enabled HDF5 and an MPI-enabled h5py build, not ordinary h5py by default.

`h5py.version.info` and `ldd` are troubleshooting tools, not commands needed in
every job. If investigating a failure, run the checks above in the job's
environment on an actual compute node, not only on the login node; adjust the
plugin path in the `ldd` command as needed. `filter_avail(40024) == True`
alone does not prove the plugin and h5py use the same HDF5 library.

## OpenMP

Enable OpenMP when creating the dataset:

```python
import h5py
import numpy as np
from pyfalcons import h5py_filter

data = np.random.default_rng(0).random((721, 1440)).astype(np.float32)
options = h5py_filter(error_mode="REL", error_bound=1e-3,
                      openmp=True, num_threads=4)
with h5py.File("parallel.h5", "w") as f:
    f.create_dataset("field", data=data, chunks=(721, 1440), **options)
```

`num_threads=4` sets the OpenMP team size inside the filter and temporarily
disables dynamic teams for that call. The previous runtime settings are
restored afterward. Neither `OMP_NUM_THREADS` nor `OMP_DYNAMIC` needs to be set.
Build and load the plugin from the same updated Falcons source as the Python
package: an older plugin can ignore the `num_threads` option.
Omitting `num_threads` leaves the calling task's OpenMP settings unchanged;
`openmp=False` disables parallel interpolation and is the default. The plugin
must have OpenMP available, and setting `num_threads` alone does not enable it.
Automatic tuning can select a serial algorithm; short dimensions, thread
limits and runtime restrictions can also yield fewer threads than requested.

To select interpolation directly instead of allowing predictor tuning to
choose Lorenzo, use the advanced configuration:

```python
from pyfalcons import falconsAlgorithm, falconsConfig, falconsErrorBoundMode

config = falconsConfig()
config.errorBoundMode = falconsErrorBoundMode.REL
config.relErrorBound = 1e-3
config.cmprAlgo = falconsAlgorithm.INTERP
config.openmp = True
options = config.h5py_options(num_threads=4)
```

The compressor can still choose a lossless fallback for individual chunks;
such chunks do not use the parallel interpolation decoder.

The filter options are stored when the dataset is created. For parallel
interpolation, each compressed chunk records its actual compression team size,
and the current decoder requests that recorded size. Thus a serially compressed
chunk will not become four-threaded during reading, and a chunk compressed with
four threads must be decoded with a four-thread team. If runtime restrictions
such as `OMP_THREAD_LIMIT` prevent that team from forming, decoding fails.
An OpenMP-enabled plugin is required to decode a parallel chunk.

PyTorch `num_workers=16` starts 16 worker processes. If they simultaneously
decode chunks recorded with four threads, their Falcons teams can use up to
16 x 4 = 64 threads in total, plus the parent and other library threads. The
example Slurm CPU request is a starting point, not a performance guarantee:
Slurm may count logical CPUs, and SMT, affinity, worker placement and other
libraries affect available physical cores. Measure on the allocated hardware.

## Converting an existing ERA5 file

This example targets an ordinary ERA5 file with a root dataset `fields` of
shape `(T, C, H, W)`, such as `(1460, 257, 721, 1440)`. It reads the source
shape automatically and creates a new file with that same shape and dtype.
For the annual example the destination uses `(1, 1, 721, 1440)` chunks.
It copies root and `fields` attributes, and copies all other root objects,
including `global_means` and `global_stds`, without converting them. Only
`fields` is newly compressed with filter 40024. Other objects can retain their
own filter dependencies. This is not a general HDF5 container copier: it does
not handle preservation of object or region references, dimension scales,
links, aliases, or external and virtual datasets. Use a suitable HDF5-aware
workflow for such files.

The plugin removes length-one dimensions from the `(1, 1, H, W)` chunk when
configuring Falcons, giving an effective two-dimensional spatial chunk. This
recipe was checked with a small four-dimensional `float32` file; validate a
representative file with your own plugin, dtype, data and HDF5 environment.
The example accepts only native-endian `float32` or `float64`, requires a
nonempty 4D dataset with both spatial dimensions greater than one, and stops
on non-finite data. In particular it rejects `float16`, which can trigger an
out-of-bounds access in the current plugin.

Run this in the configured environment. Change the two paths and the error
mode and bound as needed; the destination must not already exist.

```bash
python - <<'PY'
import h5py
import numpy as np
from pyfalcons import h5py_filter

source_path = "era5.h5"
target_path = "era5_falcons.h5"
error_mode = "REL"       # Use "ABS" for an absolute error bound.
error_bound = 1e-3
openmp = True
num_threads = 4
options = h5py_filter(error_mode=error_mode, error_bound=error_bound,
                      openmp=openmp, num_threads=num_threads)

def copy_attrs(source, target):
    for name in source.attrs:
        target.attrs.create(name, source.attrs[name], dtype=source.attrs.get_id(name).dtype)

with h5py.File(source_path, "r") as src:
    fields = src.get("fields")
    if not isinstance(fields, h5py.Dataset) or fields.ndim != 4:
        raise ValueError("fields must be a 4D dataset (T, C, H, W)")
    dtype = fields.dtype
    if dtype.kind != "f" or dtype.itemsize not in (4, 8) or not dtype.isnative:
        raise TypeError("fields must be native-endian float32 or float64")
    T, C, H, W = fields.shape
    if T == 0 or C == 0 or H <= 1 or W <= 1:
        raise ValueError("fields must be nonempty with H and W greater than one")
    print("dataset shape:", fields.shape, "chunk shape:", (1, 1, H, W))

    with h5py.File(target_path, "x") as dst:
        copy_attrs(src, dst)
        output = dst.create_dataset(
            "fields", shape=fields.shape, dtype=dtype, chunks=(1, 1, H, W),
            **options,
        )
        copy_attrs(fields, output)
        for name in src:
            if name != "fields":
                src.copy(name, dst, name=name)
        for t in range(T):
            for c in range(C):
                plane = fields[t, c, :, :]
                if not np.isfinite(plane).all():
                    raise ValueError(f"Non-finite values in fields[{t}, {c}]")
                output[t, c, :, :] = plane

with h5py.File(source_path, "r") as src, h5py.File(target_path, "r") as dst:
    original = src["fields"][0, 0, :, :].astype(np.float64)
    restored = dst["fields"][0, 0, :, :].astype(np.float64)
    print("sample chunk range:", np.ptp(original))
    print("sample chunk max absolute error:", np.max(np.abs(original - restored)))
PY
```

The source is opened read-only and `"x"` refuses to overwrite an existing
destination. A failed conversion can leave an incomplete new output file.
The script reads `shape` from the source; dimensions are not passed as
`compression_opts`. For `H=721, W=1440`, each chunk contains one time step,
one variable and one `721 x 1440` spatial plane. This matches whole-plane
reads and writes but can amplify I/O for smaller slices; it is an example,
not an optimal chunk shape for every workload. Reading and writing one plane
at a time avoids loading the full `1460 x 257 x 721 x 1440` array into memory.
Conversion can still take substantial time and needs space for the new file
alongside the source and copied objects.

ABS specifies a fixed absolute error bound. REL specifies a fraction of each
individual chunk's range, not the entire file's range. The printed error and
range cover only the first spatial chunk, not all chunks or a global error
guarantee; validate representative chunks for the intended workload.

## Decompressing a file

h5py invokes the filter automatically when a compressed dataset is read:

```python
import h5py

with h5py.File("era5_falcons.h5", "r") as f:
    plane = f["fields"][0, 0, :, :]  # One decompressed spatial plane.
```

To write an uncompressed HDF5 file without loading the whole array into
memory, use the same one-plane-at-a-time approach:

```python
import h5py

with h5py.File("era5_falcons.h5", "r") as src:
    fields = src["fields"]
    T, C, H, W = fields.shape
    with h5py.File("era5_restored.h5", "x") as dst:
        for name in src.attrs:
            dst.attrs.create(name, src.attrs[name], dtype=src.attrs.get_id(name).dtype)
        output = dst.create_dataset(
            "fields", shape=fields.shape, dtype=fields.dtype, chunks=(1, 1, H, W)
        )
        for name in fields.attrs:
            output.attrs.create(name, fields.attrs[name], dtype=fields.attrs.get_id(name).dtype)
        for name in src:
            if name != "fields":
                src.copy(name, dst, name=name)
        for t in range(T):
            for c in range(C):
                output[t, c, :, :] = fields[t, c, :, :]
```

The destination dataset has no compression filter; the copied root objects
retain any filters they already used. This restores the lossy reconstructed
values, not the original pre-compression values. The same limitations on HDF5
references, links and special datasets described above apply. Both reads
require the Falcons plugin; they do not require the original error settings.

## Reading with a PyTorch DataLoader

Set `HDF5_PLUGIN_PATH` before launching Python. Each worker must be able to
load the plugin. Open the file inside the worker instead of keeping an HDF5
file handle opened by the parent process. For example, this returns two
consecutive times and only the selected channels:

```python
import h5py
import torch
from torch.utils.data import DataLoader, Dataset


class ERA5Dataset(Dataset):
    def __init__(self, path, channels):
        self.path = path
        # h5py fancy indices must be unique and increasing; this is also the
        # order of channels in the returned tensors.
        self.channels = list(channels)
        with h5py.File(path, "r") as f:
            self.length = f["fields"].shape[0] - 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with h5py.File(self.path, "r") as f:
            frames = f["fields"][index:index + 2, self.channels, :, :]
        return torch.from_numpy(frames[0]), torch.from_numpy(frames[1])


if __name__ == "__main__":
    dataset = ERA5Dataset("era5_falcons.h5", channels=range(19))
    loader = DataLoader(dataset, batch_size=1, num_workers=16, pin_memory=True)
    for inputs, targets in loader:
        pass  # Use the tensors in your training loop.
```

Select the actual channel indices required by the model and preserve its
expected order. With `(1, 1, H, W)` chunks, selecting channels inside the
HDF5 read can avoid reading other channels. Whether compression improves
throughput depends on the data, chunking, storage, cache and worker count;
measure it against an uncompressed file with the same access pattern. A file
written using four-thread interpolation can use up to four filter threads per
worker; the worker count does not alter the saved thread count. GPU pinning
requires an appropriate PyTorch accelerator build. See [OpenMP](#openmp) for
the interaction between worker processes and per-chunk threads.

## HDF5 command-line tools and C/C++

### Compression with h5repack

If your HDF5 installation provides both `h5repack-shared` and `h5repack`,
use the shared build so that it can load the plugin. With the default Falcons
configuration:

```bash
h5repack-shared -f UD=40024,0 data.h5 data.falcons.h5
```

To use a Falcons configuration file, inspect the parameters generated by the
current build and use the provided script to run `h5repack`:

```bash
build/tools/H5Z-Falcons/test/print_h5repack_args -c tools/falcons/falcons.config
bash build/tools/H5Z-Falcons/test/h5repack.sh tools/falcons/falcons.config data.h5 data.falcons.h5
```

The script selects `h5repack-shared` if available, otherwise `h5repack`.
These configuration-file tools are not required by the Python wrapper.
Do not use `tools/H5Z-Falcons/test/cdvalueHelper.py`: its current serialized
dimensions are invalid and can terminate Python during dataset creation.

### Decompression with h5repack

```bash
h5repack-shared -f NONE data.falcons.h5 data.falcons_decompressed.h5
```

The input still requires the Falcons plugin. `-f NONE` removes filters from
the output; it restores the lossy reconstructed values, not the original data.

### C/C++ integration

Include `H5Z_Falcons.hpp`, configure a `Falcons::Config`, and call
`set_Falcons_conf_to_H5(dcpl, config)` on the HDF5 dataset creation
property list. Set the chunk dimensions with `H5Pset_chunk`; the filter
derives the effective dimensions and data type from the chunk.
Applications calling this C++ interface must link to `hdf5falcons` and HDF5.

The complete examples are
[`falconsToHDF5.cpp`](test/falconsToHDF5.cpp) and
[`HDF5ToFalcons.cpp`](test/HDF5ToFalcons.cpp).
