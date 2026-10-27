
# H5Z-Falcons Filter Integration

The H5Z-Falcons filter integrates Falcons with HDF5, providing an efficient way to compress and decompress data within HDF5 files.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [HDF5 Executables](#hdf5-executables)
  - [Compression Methods](#compression-methods)
  - [Decompression](#decompression)
- [Integration in C/C++](#integration-in-cc)

## Installation

### Step 1: Build and Install Falcons
Compile and install Falcons with the H5Z-Falcons filter enabled:
```bash
cmake -S . -B build -DBUILD_H5Z_FILTER=ON -DENABLE_AVX2=ON
cmake --build build -j
```

### Step 2: Configure Environment
Add the directory containing the H5Z-Falcons library (`libhdf5falcons.so` or `libhdf5falcons.dylib`) to your `HDF5_PLUGIN_PATH` and `LD_LIBRARY_PATH` environment variables:
```bash
export HDF5_PLUGIN_PATH="$(pwd)/build/tools/H5Z-Falcons${HDF5_PLUGIN_PATH:+:$HDF5_PLUGIN_PATH}"
export LD_LIBRARY_PATH="$(pwd)/build/tools/H5Z-Falcons${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```


## Use H5Z-Falcons in HDF5 Executables

Note, if the HDF5 in your system contains both `h5repack-shared` and `h5repack`, you need to use `h5repack-shared` for filters, as outlined in [H5Z-ZFP Issue #137](https://github.com/LLNL/H5Z-ZFP/issues/137).

### Compression

**Method 1: Compression with default settings (see Falcons/utils/Config.hpp for defaults)**
```bash
h5repack-shared -f UD=40024,0 data.h5 data.falcons.h5
```

**Method 2: Compression with customized settings in configuration file**

Step 1, generate H5Z-Falcons parameters from Falcons configuration file:
```bash
build/tools/H5Z-Falcons/test/print_h5repack_args -c tools/falcons/falcons.config
```
Step 2, compress by h5repack-shared with generated parameters:
```bash
h5repack-shared -f UD=40024,0,11,4077060608,16843009,0,0,4054449152,1348619730,16826431,257,2147483904,2147483648,16777216 data.h5 data.falcons.h5
```

Alternatively, use the script to generate parameters and compress at once:
```bash
bash build/tools/H5Z-Falcons/test/h5repack.sh tools/falcons/falcons.config data.h5 data.falcons.h5
```

### Decompression
```bash
h5repack-shared -f NONE data.falcons.h5 data.falcons_decompressed.h5
```

## Use H5Z-Falcons in code (C/C++)
See examples `falconsToHDF5.cpp` and `HDF5ToFalcons.cpp` for how to use the H5Z-Falcons filter in your C/C++ projects.
