# __init__.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc.
"""Tensor factories, device helpers, and re-exports for the IRON runtime."""

import logging
import os

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
    import pyxrt

    has_xrt = True
except ImportError as e:
    _logger.warning(
        "Failed to import PyXRT: %s, proceeding without runtime libraries.", e
    )
    has_xrt = False

if has_xrt:
    from .hostruntime.xrtruntime.tensor import XRTTensor

    DEFAULT_TENSOR_CLASS = XRTTensor
else:
    from .hostruntime.tensor_class import CPUOnlyTensor

    DEFAULT_TENSOR_CLASS = CPUOnlyTensor


def tensor(*args, **kwargs):
    """
    Create a tensor using the default tensor class.

    Args:
        *args: Arguments passed to the tensor constructor.
        **kwargs: Keyword arguments passed to the tensor constructor.

    Returns:
        Tensor: The created tensor.
    """
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


_DefaultNPURuntime = None


def _get_default_npu_runtime():
    global _DefaultNPURuntime
    if _DefaultNPURuntime is None and has_xrt:
        _DefaultNPURuntime = CachedXRTRuntime()
    return _DefaultNPURuntime


def __getattr__(name):
    if name == "DefaultNPURuntime":
        return _get_default_npu_runtime()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_current_device():
    """
    Get the current NPU device.

    Returns:
        Device | None: The current device if available, else None.
    """
    if hostruntime._CURRENT_DEVICE:
        return hostruntime._CURRENT_DEVICE

    runtime = _get_default_npu_runtime()
    if runtime:
        return runtime.device()
    else:
        return None
