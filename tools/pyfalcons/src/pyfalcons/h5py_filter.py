"""Build h5py keyword arguments without importing h5py."""

import math

from pyfalcons.pyConfig import falconsConfig
from pyfalcons.pyConfigEnums import falconsErrorBoundMode


def h5py_filter(error_mode="ABS", error_bound=1e-3, openmp=False, num_threads=None):
    """Return Falcons filter kwargs for h5py.create_dataset.

    error_mode is "ABS" or "REL" (case-insensitive), or the corresponding
    falconsErrorBoundMode. error_bound must be finite and non-negative.
    openmp must be explicitly enabled and requires an OpenMP-enabled plugin.
    num_threads optionally sets the OpenMP runtime thread count for compression
    and does not enable OpenMP by itself. HDF5 supplies the actual dimensions
    and data type from each dataset.
    """
    if isinstance(error_mode, str):
        mode = error_mode.strip().upper()
    elif isinstance(error_mode, falconsErrorBoundMode):
        mode = error_mode.name
    else:
        raise TypeError("error_mode must be 'ABS', 'REL', or a corresponding enum")
    if mode not in ("ABS", "REL"):
        raise ValueError("error_mode must be 'ABS' or 'REL'")

    try:
        bound = float(error_bound)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("error_bound must be a finite non-negative number") from exc
    if not math.isfinite(bound) or bound < 0:
        raise ValueError("error_bound must be a finite non-negative number")
    if not isinstance(openmp, bool):
        raise TypeError("openmp must be a bool")

    config = falconsConfig()
    config.errorBoundMode = falconsErrorBoundMode[mode]
    if mode == "ABS":
        config.absErrorBound = bound
    else:
        config.relErrorBound = bound
    config.openmp = openmp
    return config.h5py_options(num_threads=num_threads)
