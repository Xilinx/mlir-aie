# __init__.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.
import sys
from .hostruntime.tensor_class import Tensor

try:
    import pyxrt

    has_xrt = True
except ImportError as e:
    print(
        f"Failed to import PyXRT: {e}, proceeding without runtime libraries.",
        file=sys.stderr,
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


from .hostruntime.hostruntime import HostRuntime
from .trace import TraceConfig
from .npukernel import NPUKernel

if has_xrt:
    from .hostruntime.xrtruntime.hostruntime import CachedXRTRuntime

    DefaultNPURuntime = CachedXRTRuntime()
else:
    DefaultNPURuntime = None


def get_current_device():
    """
    Get the current NPU device.

    Returns:
        Device | None: The current device if available, else None.
    """
    return DefaultNPURuntime.device() if DefaultNPURuntime else None
