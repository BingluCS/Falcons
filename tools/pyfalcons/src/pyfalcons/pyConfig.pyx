# distutils: language = c++
"""Python wrapper for Falcons Config class."""

from builtins import bool as py_bool
from numbers import Integral

from pyfalcons cimport pyConfig
cimport cython
from libc.limits cimport INT_MAX


cdef class falconsConfig:
    """
    Configuration class for Falcons compression.
    
    Enum classes: falconsErrorBoundMode, falconsAlgorithm, falconsInterpAlgorithm
    
    Parameters: Dimension sizes as individual ints, tuple, or list
        falconsConfig(data.shape), falconsConfig(100, 200, 300), etc.
    """

    def __init__(self, *args):
        """Initialize config with optional dimensions."""
        self.conf = Config()
        if args:
            self.setDims(*args)
    
    def setDims(self, *args):
        """Set dimensions. Accepts tuple/list or individual integers."""
        cdef vector[size_t] dims
        
        # Handle both setDims(100, 200, 300) and setDims((100, 200, 300))
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            dims_iter = args[0]
        else:
            dims_iter = args
        
        if not dims_iter:
            raise ValueError("At least one dimension required")
        
        for arg in dims_iter:
            if not isinstance(arg, int) or arg <= 0:
                raise ValueError(f"Dimension must be positive integer, got {arg}")
            dims.push_back(<size_t>arg)
        
        self.conf.setDims(dims.begin(), dims.end())

    def loadcfg(self, cfgpath: str):
        """Load configuration from INI file."""
        cdef string cfgpathStr = cfgpath.encode('utf-8')
        try:
            self.conf.loadcfg(cfgpathStr)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load '{cfgpath}': {e}")

    def h5py_options(self, num_threads=None):
        """Return h5py filter kwargs using the fields saved by Config::save.

        A copy is serialized with placeholder dimensions (1,), which also
        resets blockSize and predDim. The HDF5 filter replaces those fields and
        data type from the dataset and its chunks. An optional positive
        num_threads is appended for the filter's compression runtime and does
        not enable OpenMP. Interpolation settings are not saved by Config::save.
        Combined ABS/REL bounds are not supported for HDF5 round trips.
        """
        cdef Config filter_conf
        cdef size_t byte_count
        cdef size_t word_count
        cdef vector[size_t] placeholder_dims
        cdef vector[uint32_t] cd_values
        cdef unsigned char *buffer

        if num_threads is not None:
            if isinstance(num_threads, py_bool) or not isinstance(num_threads, Integral):
                raise TypeError("num_threads must be a positive integer or None")
            num_threads = int(num_threads)
            if num_threads <= 0 or num_threads > INT_MAX:
                raise ValueError("num_threads must be in the range 1 to C++ INT_MAX")

        if self.conf.errorBoundMode > 5 or self.conf.cmprAlgo > 4:
            raise ValueError("Invalid Falcons error bound mode or compression algorithm")
        if self.conf.errorBoundMode in (4, 5):
            raise ValueError(
                "The current Falcons compressor does not support HDF5 round trips "
                "with ABS_AND_REL or ABS_OR_REL"
            )

        filter_conf = self.conf
        placeholder_dims.push_back(1)
        filter_conf.setDims(placeholder_dims.begin(), placeholder_dims.end())
        byte_count = filter_conf.size_est()
        word_count = (byte_count + sizeof(uint32_t) - 1) // sizeof(uint32_t)

        cd_values.resize(word_count)
        buffer = <unsigned char *>cd_values.data()
        byte_count = filter_conf.save(buffer)
        cd_values.resize((byte_count + sizeof(uint32_t) - 1) // sizeof(uint32_t))
        if num_threads is not None:
            cd_values.push_back(<uint32_t>num_threads)
        return {"compression": 40024, "compression_opts": tuple(cd_values)}
    
    # Read-only properties
    @property
    def dims(self):
        """Get dimensions (read-only)."""
        return tuple(self.conf.dims)
    
    @property
    def num_elements(self):
        """Get total number of elements (read-only)."""
        return self.conf.num
    
    @property
    def ndim(self):
        """Get number of dimensions (read-only)."""
        return self.conf.N
    
    # Error bounds (double)
    @property
    def absErrorBound(self):
        return self.conf.absErrorBound
    
    @absErrorBound.setter
    def absErrorBound(self, double value):
        self.conf.absErrorBound = value
    
    @property
    def relErrorBound(self):
        return self.conf.relErrorBound
    
    @relErrorBound.setter
    def relErrorBound(self, double value):
        self.conf.relErrorBound = value
    
    @property
    def psnrErrorBound(self):
        return self.conf.psnrErrorBound
    
    @psnrErrorBound.setter
    def psnrErrorBound(self, double value):
        self.conf.psnrErrorBound = value
    
    @property
    def l2normErrorBound(self):
        return self.conf.l2normErrorBound
    
    @l2normErrorBound.setter
    def l2normErrorBound(self, double value):
        self.conf.l2normErrorBound = value
    
    # Enum properties (accept enum or int)
    @property
    def errorBoundMode(self):
        """Get/set error bound mode. Use falconsErrorBoundMode enum."""
        return self.conf.errorBoundMode
    
    @errorBoundMode.setter
    def errorBoundMode(self, value):
        if hasattr(value, 'value'):  # Is an enum
            self.conf.errorBoundMode = value.value
        else:
            raise TypeError(f"Use falconsErrorBoundMode enum, got {type(value).__name__}")
    
    @property
    def cmprAlgo(self):
        """Get/set compression algorithm. Use falconsAlgorithm enum."""
        return self.conf.cmprAlgo
    
    @cmprAlgo.setter
    def cmprAlgo(self, value):
        if hasattr(value, 'value'):  # Is an enum
            self.conf.cmprAlgo = value.value
        else:
            raise TypeError(f"Use falconsAlgorithm enum, got {type(value).__name__}")
    
    @property
    def interpAlgo(self):
        """Get/set interpolation algorithm. Use falconsInterpAlgorithm enum."""
        return self.conf.interpAlgo
    
    @interpAlgo.setter
    def interpAlgo(self, value):
        if hasattr(value, 'value'):  # Is an enum
            self.conf.interpAlgo = value.value
        else:
            raise TypeError(f"Use falconsInterpAlgorithm enum, got {type(value).__name__}")
    
    # Boolean predictors
    @property
    def lorenzo(self):
        return self.conf.lorenzo
    
    @lorenzo.setter
    def lorenzo(self, bint value):
        self.conf.lorenzo = value
    
    @property
    def lorenzo2(self):
        return self.conf.lorenzo2
    
    @lorenzo2.setter
    def lorenzo2(self, bint value):
        self.conf.lorenzo2 = value
    
    @property
    def regression(self):
        return self.conf.regression
    
    @regression.setter
    def regression(self, bint value):
        self.conf.regression = value
    
    @property
    def regression2(self):
        return self.conf.regression2
    
    @regression2.setter
    def regression2(self, bint value):
        self.conf.regression2 = value
    
    @property
    def openmp(self):
        return self.conf.openmp
    
    @openmp.setter
    def openmp(self, bint value):
        self.conf.openmp = value
    
    # Integer settings
    @property
    def quantbinCnt(self):
        return self.conf.quantbinCnt
    
    @quantbinCnt.setter
    def quantbinCnt(self, int value):
        self.conf.quantbinCnt = value
    
    @property
    def blockSize(self):
        return self.conf.blockSize
    
    @blockSize.setter
    def blockSize(self, int value):
        self.conf.blockSize = value
    
    @property
    def interpDirection(self):
        return self.conf.interpDirection
    
    @interpDirection.setter
    def interpDirection(self, int value):
        self.conf.interpDirection = value
    
    @property
    def interpAnchorStride(self):
        return self.conf.interpAnchorStride
    
    @interpAnchorStride.setter
    def interpAnchorStride(self, int value):
        self.conf.interpAnchorStride = value
    
    # Double interpolation settings
    @property
    def interpAlpha(self):
        return self.conf.interpAlpha
    
    @interpAlpha.setter
    def interpAlpha(self, double value):
        self.conf.interpAlpha = value
    
    @property
    def interpBeta(self):
        return self.conf.interpBeta
    
    @interpBeta.setter
    def interpBeta(self, double value):
        self.conf.interpBeta = value
    
    def __repr__(self):
        return f"falconsConfig(dims={self.dims}, num_elements={self.num_elements})"
