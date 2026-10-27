"""Cython declarations for Falcons compression/decompression API."""

from pyfalcons cimport pyConfig
from libc.stdint cimport int32_t, int64_t
from libc.stddef cimport size_t

cdef extern from "Falcons/api/falcons.hpp" namespace "Falcons":
    size_t Falcons_compress_size_bound[T](const pyConfig.Config &conf) except +

cdef extern from "Falcons/api/falcons.hpp":
    void Falcons_read_config(pyConfig.Config &conf, const char *data, size_t size) except +
    # Compression with pre-allocated buffer
    size_t Falcons_compress[T](const pyConfig.Config &conf, 
                          const T *data, 
                          char *cmpData, 
                          size_t cmpCap) except +
    
    # Decompression with pre-allocated buffer
    # Note: Config is non-const because it gets loaded from compressed data
    void Falcons_decompress[T](pyConfig.Config &conf, 
                         const char *cmpData, 
                         size_t cmpSize, 
                         T *&decData, size_t decCapacity) except +

