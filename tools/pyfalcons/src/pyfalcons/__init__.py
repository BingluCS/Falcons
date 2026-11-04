"""Python bindings for Falcons error-bounded lossy compression."""

from pyfalcons.falcons import falcons
from pyfalcons.pyConfig import falconsConfig
from pyfalcons.pyConfigEnums import falconsErrorBoundMode, falconsAlgorithm, falconsInterpAlgorithm
from pyfalcons.h5py_filter import h5py_filter

__all__ = ["falcons", "falconsConfig", "falconsErrorBoundMode", "falconsAlgorithm", "falconsInterpAlgorithm", "h5py_filter"]
