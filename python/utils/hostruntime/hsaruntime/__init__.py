# __init__.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""HSA/ROCR (libhsa-runtime64) host-runtime backend for IRON.

The package is split into focused modules:

* :mod:`.discovery`  -- locate ``libhsa-runtime64.so`` (no ctypes / no dlopen);
  the cheap capability probe used by ``aie.utils.has_hsa``.
* :mod:`._bindings`  -- the C ABI layer: enum/flag constants, ``ctypes`` struct
  mirrors, library ``dlopen``, and the bound ``hsa_*`` entry points.
* :mod:`.context`    -- :class:`~.context.HSAContext`, the process-wide device +
  dispatch-queue singleton (region/vmem memory, signals, dispatch, chains).
* :mod:`.tensor`     -- :class:`~.tensor.HSATensor`, a zero-copy vmem buffer.
* :mod:`.hostruntime`-- the IRON ``HostRuntime`` implementations.

Importing this package is side-effect-free (no ``dlopen``, no device init); the
library is bound lazily on first :class:`~.context.HSAContext` creation.
"""

from ._bindings import HSAError, _hsa_sync_timeout_s
from .context import HSAContext

__all__ = ["HSAContext", "HSAError"]
