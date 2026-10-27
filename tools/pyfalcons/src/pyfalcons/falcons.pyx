# distutils: language = c++
"""Python interface for Falcons compression library."""

from pyfalcons cimport falcons as c_falcons
from pyfalcons cimport pyConfig
cimport cython
from cython.operator cimport dereference
import numpy as np
cimport numpy as cnp
from libc.stdint cimport int8_t, int32_t, int64_t
from libc.stddef cimport size_t
from libc.string cimport memcpy
from typing import Tuple

# Initialize NumPy C API
cnp.import_array()


cdef class falcons:
    """Falcons compression/decompression with zero-copy NumPy API."""
    
    # Supported dtypes
    _SUPPORTED_DTYPES = frozenset([
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.int32),
        np.dtype(np.int64),
    ])
    
    @staticmethod
    def compress(cnp.ndarray data, config, bint copy=False) -> Tuple[cnp.ndarray, float]:
        """
        Compress NumPy array using Falcons.
        
        Parameters
        ----------
        data : numpy.ndarray
            Input data (float32, float64, int32, or int64)
        config : pyConfig or str
            Configuration object or path to config file
        copy : bool, default False
            If False, pass the NumPy buffer directly to Falcons. This is fastest,
            but Falcons overwrites the input array in place. If True, compress a
            private C-contiguous copy so the caller's array is left unchanged.
        
        Returns
        -------
        compressed : numpy.ndarray
            Compressed data as uint8 array
        ratio : float
            Compression ratio (original_size / compressed_size)
        """
        # Validate dtype
        if data.dtype not in falcons._SUPPORTED_DTYPES:
            raise TypeError(
                f"Unsupported dtype: {data.dtype}. "
                f"Supported: float32, float64, int32, int64"
            )
        
        if data.size == 0:
            raise ValueError("Cannot compress an empty array")

        # Handle config parameter
        cdef pyConfig.falconsConfig conf
        if isinstance(config, str):
            # Config file path provided - convert shape to tuple
            shape_tuple = tuple(<Py_ssize_t>data.shape[i] for i in range(data.ndim))
            conf = pyConfig.falconsConfig(*shape_tuple)
            conf.loadcfg(config)
        elif isinstance(config, pyConfig.falconsConfig):
            # Config object provided - set dimensions if needed
            conf = config
            if conf.num_elements != data.size:
                shape_tuple = tuple(<Py_ssize_t>data.shape[i] for i in range(data.ndim))
                conf.setDims(*shape_tuple)
        else:
            raise TypeError(f"config must be pyConfig or str, got {type(config)}")
        
        # Falcons reconstructs values in place during compression. By default we pass
        # the caller's buffer directly; copy=True preserves the old non-mutating behavior.
        if copy:
            data = np.array(data, order='C', copy=True)
        else:
            if not data.flags['C_CONTIGUOUS']:
                raise ValueError(
                    "copy=False requires a C-contiguous array because Falcons reads the raw buffer directly; "
                    "pass copy=True or use np.ascontiguousarray(data)"
                )
            if not data.flags['WRITEABLE']:
                raise ValueError(
                    "copy=False requires a writable array because Falcons compression overwrites input data; "
                    "pass copy=True to leave the input unchanged"
                )

        # Get data pointer. With copy=False this is the caller's buffer; Falcons will mutate it.
        cdef void* data_ptr = <void*> cnp.PyArray_DATA(data)
        cdef size_t original_size = data.nbytes
        
        # Use the C++ bound, which includes headers and small-input overhead.
        cdef size_t buffer_size
        if data.dtype == np.float32:
            buffer_size = c_falcons.Falcons_compress_size_bound[float](conf.conf)
        elif data.dtype == np.float64:
            buffer_size = c_falcons.Falcons_compress_size_bound[double](conf.conf)
        elif data.dtype == np.int32:
            buffer_size = c_falcons.Falcons_compress_size_bound[int32_t](conf.conf)
        else:
            buffer_size = c_falcons.Falcons_compress_size_bound[int64_t](conf.conf)
        cdef cnp.ndarray[cnp.uint8_t, ndim=1] compressed = np.empty(buffer_size, dtype=np.uint8)
        cdef char* compressed_ptr = <char*> cnp.PyArray_DATA(compressed)
        cdef size_t compressed_size = 0
        
        # Compress into pre-allocated buffer
        if data.dtype == np.float32:
            compressed_size = c_falcons.Falcons_compress[float](
                dereference(&conf.conf),
                <float*>data_ptr,
                compressed_ptr,
                buffer_size
            )
        elif data.dtype == np.float64:
            compressed_size = c_falcons.Falcons_compress[double](
                dereference(&conf.conf),
                <double*>data_ptr,
                compressed_ptr,
                buffer_size
            )
        elif data.dtype == np.int32:
            compressed_size = c_falcons.Falcons_compress[int32_t](
                dereference(&conf.conf),
                <int32_t*>data_ptr,
                compressed_ptr,
                buffer_size
            )
        elif data.dtype == np.int64:
            compressed_size = c_falcons.Falcons_compress[int64_t](
                dereference(&conf.conf),
                <int64_t*>data_ptr,
                compressed_ptr,
                buffer_size
            )
        
        if compressed_size == 0:
            raise RuntimeError("Compression failed")
        
        # Resize to actual compressed size (no copy, just view)
        compressed = compressed[:compressed_size]
        
        ratio = original_size / float(compressed_size)
        return compressed, ratio
    
    @staticmethod
    def decompress(cnp.ndarray compressed, dtype, shape, config=None):
        """
        Decompress Falcons-compressed data.
        
        Parameters
        ----------
        compressed : numpy.ndarray
            Compressed data (uint8 array)
        dtype : numpy.dtype or type
            Data type of original data
        shape : tuple
            Shape of the original data
        config : pyConfig or str
            Configuration object or path to config file
        
        Returns
        -------
        decompressed : numpy.ndarray
            Decompressed data with the specified shape
        """
        # Validate compressed data
        if compressed.dtype != np.uint8:
            raise TypeError(f"Compressed data must be uint8, got {compressed.dtype}")
        
        # Validate and normalize dtype
        dtype = np.dtype(dtype)
        if dtype not in falcons._SUPPORTED_DTYPES:
            raise TypeError(
                f"Unsupported dtype: {dtype}. "
                f"Supported: float32, float64, int32, int64"
            )
        
        import operator
        shape = tuple(operator.index(dim) for dim in shape)
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError("Shape dimensions must be positive integers")
        expected_size = 1
        for dim in shape:
            expected_size *= dim

        # Handle config parameter (optional; upstream API is decompress(compressed, dtype, shape))
        cdef pyConfig.falconsConfig conf
        if config is None:
            conf = pyConfig.falconsConfig(*shape)
        elif isinstance(config, str):
            # Config file path provided
            conf = pyConfig.falconsConfig(*shape)
            conf.loadcfg(config)
        elif isinstance(config, pyConfig.falconsConfig):
            # Config object provided - set dimensions if needed
            conf = config
            if conf.num_elements != expected_size:
                conf.setDims(*shape)
        else:
            raise TypeError(f"config must be pyConfig or str, got {type(config)}")
        
        # Ensure compressed data is contiguous
        if not compressed.flags['C_CONTIGUOUS']:
            compressed = np.ascontiguousarray(compressed)
        
        cdef char* compressed_ptr = <char*> cnp.PyArray_DATA(compressed)
        cdef size_t compressed_size = compressed.size
        c_falcons.Falcons_read_config(conf.conf, compressed_ptr, compressed_size)
        if expected_size != conf.num_elements:
            raise ValueError("Requested shape does not match the compressed element count")
        cdef size_t num_elements = conf.num_elements
        
        # Pre-allocate NumPy array for decompressed data
        cdef cnp.ndarray result = np.empty(num_elements, dtype=dtype)
        
        # Get pointer to NumPy array data
        # Pass this to Falcons_decompress - it will decompress directly into our buffer!
        cdef float* float_ptr
        cdef double* double_ptr
        cdef int32_t* int32_ptr
        cdef int64_t* int64_ptr
        
        if dtype == np.float32:
            float_ptr = <float*> cnp.PyArray_DATA(result)
            c_falcons.Falcons_decompress[float](
                conf.conf,
                compressed_ptr,
                compressed_size,
                float_ptr,
                num_elements
            )
        elif dtype == np.float64:
            double_ptr = <double*> cnp.PyArray_DATA(result)
            c_falcons.Falcons_decompress[double](
                conf.conf,
                compressed_ptr,
                compressed_size,
                double_ptr,
                num_elements
            )
        elif dtype == np.int32:
            int32_ptr = <int32_t*> cnp.PyArray_DATA(result)
            c_falcons.Falcons_decompress[int32_t](
                conf.conf,
                compressed_ptr,
                compressed_size,
                int32_ptr,
                num_elements
            )
        elif dtype == np.int64:
            int64_ptr = <int64_t*> cnp.PyArray_DATA(result)
            c_falcons.Falcons_decompress[int64_t](
                conf.conf,
                compressed_ptr,
                compressed_size,
                int64_ptr,
                num_elements
            )
        
        # Data is now in our NumPy array! (zero-copy decompression)
        # Reshape to original dimensions
        return result.reshape(shape), conf
    
    @staticmethod
    def verify(cnp.ndarray src_data, cnp.ndarray dec_data) -> Tuple[float, float, float]:
        """
        Compare decompressed data with original data.
        
        Parameters
        ----------
        src_data : numpy.ndarray
            Original data before compression
        dec_data : numpy.ndarray
            Decompressed data to verify
        
        Returns
        -------
        max_diff : float
            Maximum absolute difference
        psnr : float
            Peak Signal-to-Noise Ratio in dB
        nrmse : float
            Normalized Root Mean Square Error
        """
        # Check shapes match by comparing size and ndim
        if np.shape(src_data) != np.shape(dec_data):
            raise ValueError("Shape mismatch between src_data and dec_data")
        
        if src_data.dtype != dec_data.dtype:
            raise ValueError("Dtype mismatch between src_data and dec_data")
        
        # Calculate data range and difference
        if src_data.size == 0:
            raise ValueError("Cannot verify empty arrays")
        # Subtract integer values exactly before converting residuals to float64.
        # Chunking limits the temporary object arrays for int64 inputs.
        cdef cnp.ndarray source = src_data.astype(np.float64)
        cdef cnp.ndarray diff
        if np.issubdtype(src_data.dtype, np.integer):
            data_range_value = int(np.max(src_data)) - int(np.min(src_data))
            source_flat = src_data.ravel()
            reconstructed_flat = dec_data.ravel()
            diff = np.empty(src_data.size, dtype=np.float64)
            for start in range(0, src_data.size, 65536):
                stop = min(start + 65536, src_data.size)
                diff[start:stop] = (source_flat[start:stop].astype(object) -
                                    reconstructed_flat[start:stop].astype(object)).astype(np.float64)
        else:
            data_range_value = np.max(source) - np.min(source)
            diff = source - dec_data.astype(np.float64)
        cdef double data_range = data_range_value
        cdef double max_diff = np.max(np.abs(diff))
        
        # Calculate MSE and derived metrics
        cdef double mse = np.mean(diff ** 2)
        cdef double nrmse
        cdef double psnr
        if mse == 0:
            nrmse, psnr = 0.0, float('inf')
        elif data_range == 0:
            nrmse, psnr = float('inf'), -float('inf')
        else:
            nrmse = np.sqrt(mse) / data_range
            psnr = -20.0 * np.log10(nrmse)
        
        return max_diff, psnr, nrmse
