# __init__.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Tensor factories, device helpers, and re-exports for the IRON runtime."""

from typing import Any

from . import (
    _log_setup,  # noqa: F401  # side effect: configure "aie" logging first
    hostruntime,
)
from .hostruntime import set_current_device
from .hostruntime.hostruntime import HostRuntime as HostRuntime
from .npukernel import NPUKernel as NPUKernel
from .tensor_factory import _NPU_RUNTIME, _probe_hrx, _probe_xrt
from .tensor_factory import arange as arange
from .tensor_factory import ceildiv as ceildiv
from .tensor_factory import full as full
from .tensor_factory import ones as ones
from .tensor_factory import rand as rand
from .tensor_factory import randint as randint
from .tensor_factory import set_tensor_class as set_tensor_class
from .tensor_factory import tensor as tensor
from .tensor_factory import zeros as zeros
from .tensor_factory import zeros_like as zeros_like
from .trace import TraceConfig as TraceConfig

_DefaultNPURuntime = None


def _get_default_npu_runtime():
    global _DefaultNPURuntime
    if _DefaultNPURuntime is not None:
        return _DefaultNPURuntime
    if _NPU_RUNTIME == "hrx":
        from .hostruntime.hrxruntime.hostruntime import CachedHRXRuntime

        _DefaultNPURuntime = CachedHRXRuntime()
    elif _NPU_RUNTIME == "xrt" and _probe_xrt():
        from .hostruntime.xrtruntime.hostruntime import CachedXRTRuntime

        _DefaultNPURuntime = CachedXRTRuntime()
    return _DefaultNPURuntime


def cleanup_npu_runtime() -> None:
    """Release cached NPU runtime resources without initializing the runtime.

    Works for both backends: ``CachedXRTRuntime`` releases hw contexts/insts
    BOs and ``CachedHRXRuntime`` releases loaded XADX executables. If the
    default runtime was never created, this is a no-op (it never forces
    initialization).
    """
    runtime = globals().get("DefaultNPURuntime", _DefaultNPURuntime)
    if runtime is not None:
        runtime.cleanup()


def __getattr__(name: str) -> Any:
    # Return type is ``Any`` deliberately: this serves attributes of unrelated
    # types (the NPU runtime object for ``DefaultNPURuntime`` vs ``bool`` for the
    # ``has_xrt``/``has_hrx`` probes), so a single concrete annotation would
    # mistype callers (e.g. treating ``DefaultNPURuntime`` as possibly ``bool``).
    if name == "DefaultNPURuntime":
        return _get_default_npu_runtime()
    # Public capability flags, probed on first access (see _probe_xrt/_probe_hrx).
    if name == "has_xrt":
        return _probe_xrt()
    if name == "has_hrx":
        return _probe_hrx()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_current_device(*, probe_runtime: bool = True):
    """Get the current NPU device.

    Args:
        probe_runtime: When True, infer the device from the default runtime if
            no explicit device has been bound.  Use False for offline inspection
            paths that must not initialize the runtime.

    Args:
        probe_runtime: When True, fall back to the default NPU runtime if no
            device was explicitly set with ``set_current_device``. When False,
            return only the explicitly selected device and never initialize or
            query the default runtime.

    Returns:
        Device | None: The current device if available, else None.
    """
    if hostruntime._CURRENT_DEVICE is not None:
        return hostruntime._CURRENT_DEVICE

    if not probe_runtime:
        return None

    runtime = _get_default_npu_runtime()
    if runtime:
        return runtime.device()
    else:
        return None


def ensure_current_device(*, probe_runtime: bool = True):
    """Bind and return the device observed by IRON.

    ``get_current_device()`` can infer a device from the runtime without making
    that device explicit. Architecture-sensitive generators need a single
    process-wide device selection so kernel factories, cache hashing, MLIR
    generation, and external-kernel compilation all see the same target.

    Args:
        probe_runtime: Forwarded to ``get_current_device``. Use False for
            offline inspection paths that must not initialize the runtime.

    Returns:
        Device | None: The device that was bound, or ``None`` if no device
        was available and nothing was bound.
    """
    device = get_current_device(probe_runtime=probe_runtime)
    if device is not None:
        set_current_device(device)
    return device
