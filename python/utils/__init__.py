# __init__.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Tensor factories, device helpers, and re-exports for the IRON runtime."""

import logging
import os

import numpy as np

# Prevent "No handlers could be found" warnings when aie is used as a library.
logging.getLogger("aie").addHandler(logging.NullHandler())

# Honour AIE_LOG_LEVEL env var (e.g. DEBUG, INFO, WARNING, ERROR, CRITICAL).
# This must be done before any aie submodule emits a log record.
_log_level_str = os.environ.get("AIE_LOG_LEVEL", "").upper()
if _log_level_str:
    _log_level = getattr(logging, _log_level_str, None)
    if _log_level is not None:
        logging.getLogger("aie").setLevel(_log_level)

_logger = logging.getLogger(__name__)

from .hostruntime.tensor_class import Tensor

try:
    import pyxrt  # pyright: ignore[reportMissingImports]

    has_xrt = True
except ImportError as e:
    _logger.warning(
        "Failed to import PyXRT: %s, proceeding without runtime libraries.", e
    )
    has_xrt = False

# Capability probe for the HRX (libhrx amdxdna) backend. This is the analogue of
# ``import pyxrt`` above for XRT: it only checks whether libhrx.so can be located
# on this host (no dlopen / no device init), so it is cheap and safe to run at
# import time even when HRX is not the selected backend.
try:
    from .hostruntime.hrxruntime.discovery import hrx_available

    has_hrx = hrx_available()
except Exception as e:  # discovery must never break the import of aie.utils
    _logger.debug("HRX discovery probe failed: %s", e)
    has_hrx = False

# Host-runtime backend selection. ``IRON_RUNTIME`` chooses between the XRT and
# HRX host stacks; both consume the identical aiecc artifacts (final.xclbin +
# insts.bin) and only the dispatch path differs. Accepted values:
#   xrt   - force the XRT backend (error later if pyxrt is missing).
#   hrx   - force the HRX backend (error here if libhrx is not found).
#   auto  - (default) prefer XRT when present, else fall back to CPU.
#
# HRX is strictly opt-in: it is selected *only* when IRON_RUNTIME=hrx is set
# explicitly. ``auto`` never selects HRX -- the product contract is "XRT remains
# the default, HRX is opt-in", so an XRT-less host degrades to CPU rather than
# silently switching to HRX. We import the selected backend lazily because the
# HRX package dlopen()s libhrx on first use.
_IRON_RUNTIME = os.environ.get("IRON_RUNTIME", "auto").lower()

# Strict product contract: an unset IRON_RUNTIME defaults to 'auto', but an
# explicitly *invalid* value is a hard error rather than a silent fallback --
# a typo'd backend name must not quietly resolve to something else.
if _IRON_RUNTIME not in ("xrt", "hrx", "auto"):
    raise ImportError(
        f"Invalid IRON_RUNTIME={_IRON_RUNTIME!r}; expected one of xrt|hrx|auto "
        f"(unset defaults to 'auto')."
    )

if _IRON_RUNTIME == "hrx" and not has_hrx:
    raise ImportError(
        "IRON_RUNTIME=hrx was requested but libhrx.so could not be located. "
        "Install HRX to a standard location, or set HRX_DIR/LIBHRX_DIR. "
        "Use IRON_RUNTIME=auto to fall back to XRT/CPU when HRX is absent."
    )

# Resolve 'auto' to a concrete backend with graceful degradation. HRX is never
# auto-selected (opt-in only via IRON_RUNTIME=hrx), so 'auto' is XRT or CPU.
if _IRON_RUNTIME == "auto":
    if has_xrt:
        _IRON_RUNTIME = "xrt"
    else:
        _IRON_RUNTIME = "cpu"

if _IRON_RUNTIME == "hrx":
    from .hostruntime.hrxruntime.tensor import HRXTensor

    DEFAULT_TENSOR_CLASS = HRXTensor
elif _IRON_RUNTIME == "xrt" and has_xrt:
    from .hostruntime.xrtruntime.tensor import XRTTensor

    DEFAULT_TENSOR_CLASS = XRTTensor
else:
    from .hostruntime.tensor_class import CPUOnlyTensor

    DEFAULT_TENSOR_CLASS = CPUOnlyTensor


def ceildiv(a, b):
    """Ceiling division: smallest integer >= a/b."""
    return -(a // -b)


def tensor(*args, **kwargs):
    """
    Create a tensor using the default tensor class.

    Passing a typed ``ndarray`` together with a mismatched ``dtype=``
    kwarg raises :class:`TypeError`.  Matching kwargs are passed through
    unchanged (the underlying tensor backend uses ``dtype`` for buffer
    allocation, so silently stripping it would surprise callers).

    Args:
        *args: Arguments passed to the tensor constructor.  ``args[0]`` is
            either a shape ``tuple`` or an array-like.
        **kwargs: Keyword arguments passed to the tensor constructor.

    Returns:
        Tensor: The created tensor.
    """
    if args and isinstance(args[0], np.ndarray) and "dtype" in kwargs:
        arr_dt = args[0].dtype
        kw_dt = np.dtype(kwargs["dtype"])
        if arr_dt != kw_dt:
            raise TypeError(
                f"iron.tensor: ndarray dtype {arr_dt!r} does not match "
                f"dtype= kwarg {kw_dt!r}.  Cast the array beforehand "
                f"(e.g. arr.astype({kw_dt!r})) or drop the dtype= kwarg."
            )
    return DEFAULT_TENSOR_CLASS(*args, **kwargs)


def ones(*args, **kwargs):
    """
    Create a tensor filled with ones using the default tensor class.

    Args:
        *args: Arguments passed to the ones method.
        **kwargs: Keyword arguments passed to the ones method.

    Returns:
        Tensor: The created tensor.
    """
    return DEFAULT_TENSOR_CLASS.ones(*args, **kwargs)


def zeros(*args, **kwargs):
    """
    Create a tensor filled with zeros using the default tensor class.

    Args:
        *args: Arguments passed to the zeros method.
        **kwargs: Keyword arguments passed to the zeros method.

    Returns:
        Tensor: The created tensor.
    """
    return DEFAULT_TENSOR_CLASS.zeros(*args, **kwargs)


def full(*args, **kwargs):
    """
    Create a tensor filled with a scalar value using the default tensor class.

    Args:
        *args: Arguments passed to the full method (size, fill_value).
        **kwargs: Keyword arguments passed to the full method.

    Returns:
        Tensor: The created tensor.
    """
    return DEFAULT_TENSOR_CLASS.full(*args, **kwargs)


def randint(*args, **kwargs):
    """
    Create a tensor filled with random integers using the default tensor class.

    Args:
        *args: Arguments passed to the randint method.
        **kwargs: Keyword arguments passed to the randint method.

    Returns:
        Tensor: The created tensor.
    """
    return DEFAULT_TENSOR_CLASS.randint(*args, **kwargs)


def rand(*args, **kwargs):
    """
    Create a tensor filled with random values using the default tensor class.

    Args:
        *args: Arguments passed to the rand method.
        **kwargs: Keyword arguments passed to the rand method.

    Returns:
        Tensor: The created tensor.
    """
    return DEFAULT_TENSOR_CLASS.rand(*args, **kwargs)


def arange(*args, **kwargs):
    """
    Create a tensor with a range of values using the default tensor class.

    Args:
        *args: Arguments passed to the arange method.
        **kwargs: Keyword arguments passed to the arange method.

    Returns:
        Tensor: The created tensor.
    """
    return DEFAULT_TENSOR_CLASS.arange(*args, **kwargs)


def zeros_like(*args, **kwargs):
    """
    Create a tensor filled with zeros with the same shape as another tensor using the default tensor class.

    Args:
        *args: Arguments passed to the zeros_like method.
        **kwargs: Keyword arguments passed to the zeros_like method.

    Returns:
        Tensor: The created tensor.
    """
    return DEFAULT_TENSOR_CLASS.zeros_like(*args, **kwargs)


def set_tensor_class(cls):
    """
    Set the default tensor class.

    Args:
        cls: The new default tensor class. Must inherit from Tensor.

    Raises:
        ValueError: If cls does not inherit from Tensor.
    """
    if not issubclass(cls, Tensor):
        raise ValueError(
            f"Tensors must inherit from the Tensor class but {cls} does not."
        )
    global DEFAULT_TENSOR_CLASS
    DEFAULT_TENSOR_CLASS = cls


from .hostruntime import set_current_device
from . import hostruntime
from .hostruntime.hostruntime import HostRuntime
from .trace import TraceConfig
from .npukernel import NPUKernel

if has_xrt:
    from .hostruntime.xrtruntime.hostruntime import CachedXRTRuntime
else:
    CachedXRTRuntime = None


_DefaultNPURuntime = None


def _get_default_npu_runtime():
    global _DefaultNPURuntime
    if _DefaultNPURuntime is not None:
        return _DefaultNPURuntime
    if _IRON_RUNTIME == "hrx":
        from .hostruntime.hrxruntime.hostruntime import CachedHRXRuntime

        _DefaultNPURuntime = CachedHRXRuntime()
    elif _IRON_RUNTIME == "xrt" and has_xrt:
        assert CachedXRTRuntime is not None
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
    if runtime is not None and hasattr(runtime, "cleanup"):
        runtime.cleanup()


def __getattr__(name):
    if name == "DefaultNPURuntime":
        return _get_default_npu_runtime()
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
