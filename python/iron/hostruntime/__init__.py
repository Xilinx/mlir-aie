# __init__.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.
import sys

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
    from .xrtruntime.hostruntime import XRTHostRuntime
    from .xrtruntime.tensor import XRTTensor

    DEFAULT_IRON_RUNTIME = XRTHostRuntime()
    DEFAULT_IRON_TENSOR_CLASS = XRTTensor
else:
    from .tensor import CPUOnlyTensor

    DEFAULT_IRON_RUNTIME = None
    DEFAULT_IRON_TENSOR_CLASS = CPUOnlyTensor


def get_current_device():
    return DEFAULT_IRON_RUNTIME.device()


def tensor(*args, **kwargs):
    return DEFAULT_IRON_TENSOR_CLASS(*args, **kwargs)


def ones(*args, **kwargs):
    return DEFAULT_IRON_TENSOR_CLASS.ones(*args, **kwargs)


def zeros(*args, **kwargs):
    return DEFAULT_IRON_TENSOR_CLASS.zeros(*args, **kwargs)


def randint(*args, **kwargs):
    return DEFAULT_IRON_TENSOR_CLASS.randint(*args, **kwargs)


def rand(*args, **kwargs):
    return DEFAULT_IRON_TENSOR_CLASS.rand(*args, **kwargs)


def arange(*args, **kwargs):
    return DEFAULT_IRON_TENSOR_CLASS.arange(*args, **kwargs)


def zeros_like(*args, **kwargs):
    return DEFAULT_IRON_TENSOR_CLASS.zeros_like(*args, **kwargs)


def set_iron_tensor_class(cls):
    if not issubclass(cls, Tensor):
        raise ValueError(
            f"IRON Tensors must inherit from the Tensor class but {cls} does not."
        )
    global DEFAULT_IRON_TENSOR_CLASS
    DEFAULT_IRON_TENSOR_CLASS = cls
