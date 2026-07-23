# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HRX (amdxdna / ``libhrx``) host-runtime backend for IRON.

The package is split into focused modules:

* :mod:`.discovery`  -- locate ``libhrx.so`` (no ctypes / no dlopen); the cheap
  capability probe used by ``aie.utils.has_hrx``.
* :mod:`._bindings`  -- the C ABI layer: enum/flag constants, ``ctypes`` struct
  mirrors, library ``dlopen``, and the bound ``hrx_*`` entry points.
* :mod:`.context`    -- :class:`~.context.HRXContext`, the process-wide device +
  dispatch-stream singleton (buffers, executables, chained dispatch).
* :mod:`.tensor`     -- :class:`~.tensor.HRXTensor`, a persistent-mapped buffer.
* :mod:`.hostruntime`-- the IRON ``HostRuntime`` implementations.

Importing this package is side-effect-free (no ``dlopen``, no device init); the
library is bound lazily on first :class:`~.context.HRXContext` creation.
"""

from ._bindings import HRXError, _hrx_sync_timeout_s, control_code_from_elf
from .context import HRXContext

__all__ = [
    "HRXContext",
    "HRXError",
    "control_code_from_elf",
]
