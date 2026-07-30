# tensor_factory.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Tensor factories and NPU host-backend selection.

Split out from :mod:`aie.utils` so that :mod:`aie.utils.hostruntime.hostruntime`
(which needs the ``tensor()`` factory) can import it without importing back
through ``aie.utils.__init__`` -- that reverse edge is what used to force
``aie.utils.__init__``'s own imports of ``HostRuntime`` etc. to be deferred
past this module's definitions.
"""

import logging
import os

import numpy as np

from .hostruntime.tensor_class import NpuTensor

_logger = logging.getLogger(__name__)

# Capability probes for the two NPU host backends. Both are memoized and lazy:
# importing ``aie.utils`` no longer eagerly probes either backend. Runtime
# selection below probes only the backend it actually needs (so NPU_RUNTIME=hrx
# never imports pyxrt and NPU_RUNTIME=xrt never runs HRX discovery), and the
# public ``aie.utils.has_xrt`` / ``aie.utils.has_hrx`` attributes are served
# on-demand via module ``__getattr__`` so a bare capability query still works in
# any mode (including the default ``auto``) and pays for at most one probe.
_has_xrt: bool | None = None  # tri-state cache; None => not probed yet
_has_hrx: bool | None = None


def _probe_xrt() -> bool:
    """Whether ``pyxrt`` (the XRT userspace) imports on this host.

    Heavyweight -- importing pyxrt pulls in the XRT stack -- so it runs at most
    once and only when XRT is actually needed or explicitly queried.
    """
    global _has_xrt
    if _has_xrt is None:
        try:
            import pyxrt  # noqa: F401  # pyright: ignore[reportMissingImports]

            _has_xrt = True
        except ImportError as e:
            _logger.warning(
                "Failed to import PyXRT: %s, proceeding without runtime libraries.",
                e,
            )
            _has_xrt = False
    return _has_xrt


def _probe_hrx() -> bool:
    """Whether ``libhrx.so`` can be located on this host.

    Filesystem-only (no dlopen, no device init), but still memoized so repeated
    queries do no extra work.
    """
    global _has_hrx
    if _has_hrx is None:
        try:
            from .hostruntime.hrxruntime.discovery import hrx_available

            _has_hrx = hrx_available()
        except Exception as e:  # discovery must never break importing aie.utils
            _logger.debug("HRX discovery probe failed: %s", e)
            _has_hrx = False
    return _has_hrx


# Host-runtime backend selection. ``NPU_RUNTIME`` chooses between the XRT and
# HRX host stacks; both consume the identical aiecc artifacts (final.xclbin +
# insts.bin) and only the dispatch path differs. Accepted values:
#   xrt   - force the XRT backend (falls back to CPU tensors if pyxrt is missing).
#   hrx   - force the HRX backend (error here if libhrx is not found).
#   auto  - (default) prefer XRT when present, else fall back to CPU.
#
# HRX is strictly opt-in: it is selected *only* when NPU_RUNTIME=hrx is set
# explicitly. ``auto`` never selects HRX -- the product contract is "XRT remains
# the default, HRX is opt-in", so an XRT-less host degrades to CPU rather than
# silently switching to HRX.
#
# NPU_RUNTIME is read *before* any capability probe so a forced backend only
# probes itself: 'hrx' never imports pyxrt, and 'xrt'/'auto' never run HRX
# discovery. Each backend's tensor/runtime module is likewise imported lazily
# (it dlopen()s / imports its own runtime on first use).
_NPU_RUNTIME = os.environ.get("NPU_RUNTIME", "auto").lower()

# Strict product contract: an unset NPU_RUNTIME defaults to 'auto', but an
# explicitly *invalid* value is a hard error rather than a silent fallback --
# a typo'd backend name must not quietly resolve to something else.
if _NPU_RUNTIME not in ("xrt", "hrx", "auto"):
    raise ImportError(
        f"Invalid NPU_RUNTIME={_NPU_RUNTIME!r}; expected one of xrt|hrx|auto "
        f"(unset defaults to 'auto')."
    )

if _NPU_RUNTIME == "hrx" and not _probe_hrx():
    raise ImportError(
        "NPU_RUNTIME=hrx was requested but libhrx.so could not be located. "
        "Install HRX to a standard location, or set HRX_DIR/LIBHRX_DIR. "
        "Use NPU_RUNTIME=auto to fall back to XRT/CPU when HRX is absent."
    )

# Resolve 'auto' to a concrete backend with graceful degradation. HRX is never
# auto-selected (opt-in only via NPU_RUNTIME=hrx), so 'auto' is XRT or CPU.
if _NPU_RUNTIME == "auto":
    _NPU_RUNTIME = "xrt" if _probe_xrt() else "cpu"


def npu_runtime_kind() -> str:
    """DDR-patch ABI: XRT (and CPU) consume the folded firmware ABI; HRX consumes
    the producer-independent (unfolded) insts.bin and adds the AIE DDR aperture
    offset for every arg itself. cl::opt defaults to true, so only pass the
    flag when unfolding is requested.
    """
    return _NPU_RUNTIME


def npu_runtime_folds_ddr_addr_offset() -> bool:
    """DDR-patch ABI: XRT (and CPU) consume the folded firmware ABI; HRX consumes
    the producer-independent (unfolded) insts.bin and adds the AIE DDR aperture
    offset for every arg itself. cl::opt defaults to true, so only pass the
    flag when unfolding is requested.
    """
    return _NPU_RUNTIME != "hrx"


if _NPU_RUNTIME == "hrx":
    from .hostruntime.hrxruntime.tensor import HRXTensor

    DEFAULT_TENSOR_CLASS = HRXTensor
elif _NPU_RUNTIME == "xrt" and _probe_xrt():
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
        NpuTensor: The created tensor.
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
        NpuTensor: The created tensor.
    """
    return DEFAULT_TENSOR_CLASS.ones(*args, **kwargs)


def zeros(*args, **kwargs):
    """
    Create a tensor filled with zeros using the default tensor class.

    Args:
        *args: Arguments passed to the zeros method.
        **kwargs: Keyword arguments passed to the zeros method.

    Returns:
        NpuTensor: The created tensor.
    """
    return DEFAULT_TENSOR_CLASS.zeros(*args, **kwargs)


def full(*args, **kwargs):
    """
    Create a tensor filled with a scalar value using the default tensor class.

    Args:
        *args: Arguments passed to the full method (size, fill_value).
        **kwargs: Keyword arguments passed to the full method.

    Returns:
        NpuTensor: The created tensor.
    """
    return DEFAULT_TENSOR_CLASS.full(*args, **kwargs)


def randint(*args, **kwargs):
    """
    Create a tensor filled with random integers using the default tensor class.

    Args:
        *args: Arguments passed to the randint method.
        **kwargs: Keyword arguments passed to the randint method.

    Returns:
        NpuTensor: The created tensor.
    """
    return DEFAULT_TENSOR_CLASS.randint(*args, **kwargs)


def rand(*args, **kwargs):
    """
    Create a tensor filled with random values using the default tensor class.

    Args:
        *args: Arguments passed to the rand method.
        **kwargs: Keyword arguments passed to the rand method.

    Returns:
        NpuTensor: The created tensor.
    """
    return DEFAULT_TENSOR_CLASS.rand(*args, **kwargs)


def arange(*args, **kwargs):
    """
    Create a tensor with a range of values using the default tensor class.

    Args:
        *args: Arguments passed to the arange method.
        **kwargs: Keyword arguments passed to the arange method.

    Returns:
        NpuTensor: The created tensor.
    """
    return DEFAULT_TENSOR_CLASS.arange(*args, **kwargs)


def zeros_like(*args, **kwargs):
    """
    Create a tensor filled with zeros with the same shape as another tensor using the default tensor class.

    Args:
        *args: Arguments passed to the zeros_like method.
        **kwargs: Keyword arguments passed to the zeros_like method.

    Returns:
        NpuTensor: The created tensor.
    """
    return DEFAULT_TENSOR_CLASS.zeros_like(*args, **kwargs)


def set_tensor_class(cls):
    """
    Set the default tensor class.

    Args:
        cls: The new default tensor class. Must inherit from NpuTensor.

    Raises:
        ValueError: If cls does not inherit from NpuTensor.
    """
    if not issubclass(cls, NpuTensor):
        raise ValueError(
            f"Tensors must inherit from the NpuTensor class but {cls} does not."
        )
    global DEFAULT_TENSOR_CLASS
    DEFAULT_TENSOR_CLASS = cls
