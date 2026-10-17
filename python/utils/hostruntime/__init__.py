# __init__.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Host runtime utilities: device selection, tensor allocation, and numerical helpers."""

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

import numpy as np
from ml_dtypes import bfloat16

from .tensor_class import NpuTensor
from .tensor_class import Tensor as Tensor  # re-export of the old name

if TYPE_CHECKING:
    from aie.iron.device import Device

_CURRENT_DEVICE = None


class DeviceSelection(AbstractContextManager):
    """A selection made by `set_current_device`; leaving its `with` block undoes it.

    Args:
        previous (Device | None): The device that was current before the selection.
    """

    def __init__(self, previous: "Device | None"):
        self.previous = previous

    def __exit__(self, *exc_info) -> None:
        global _CURRENT_DEVICE
        _CURRENT_DEVICE = self.previous


def set_current_device(device: "Device | None") -> DeviceSelection:
    """Set (or clear) the current device.

    The selection takes effect at once. Used as a context manager,
    ``with set_current_device(device):`` restores the previous device on exit,
    also when the block raises.

    Args:
        device (Device | None): The device to set as current. Passing ``None``
            clears the current selection (used by test teardown and to reset
            between designs), so a ``Device | None`` from a resolver can be
            forwarded here directly.

    Returns:
        DeviceSelection: The selection, whose exit restores the previous device.
    """
    global _CURRENT_DEVICE
    selection = DeviceSelection(_CURRENT_DEVICE)
    _CURRENT_DEVICE = device
    return selection


def bfloat16_safe_allclose(dtype, arr1, arr2):
    """Check if two arrays are element-wise equal within a tolerance, handling bfloat16 safely.

    Args:
        dtype: The data type of the arrays.
        arr1: First input array.
        arr2: Second input array.

    Returns:
        bool: True if the arrays are equal within tolerance, False otherwise.
    """
    if dtype == bfloat16:
        if isinstance(arr1, NpuTensor):
            arr1 = np.array(arr1, dtype=np.float16)
        else:
            arr1 = arr1.astype(np.float16)
        if isinstance(arr2, NpuTensor):
            arr2 = np.array(arr2, dtype=np.float16)
        else:
            arr2 = arr2.astype(np.float16)
    return np.allclose(arr1, arr2)
