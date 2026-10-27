"""Python bindings for Falcons error-bounded lossy compression."""

from pyfalcons.falcons import falcons
from pyfalcons.pyConfig import falconsConfig
from pyfalcons.pyConfigEnums import falconsErrorBoundMode, falconsAlgorithm, falconsInterpAlgorithm

__all__ = ["falcons", "falconsConfig", "falconsErrorBoundMode", "falconsAlgorithm", "falconsInterpAlgorithm"]
