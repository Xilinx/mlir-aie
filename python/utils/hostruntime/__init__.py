# __init__.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
from typing import TYPE_CHECKING
from ml_dtypes import bfloat16
import numpy as np
from .tensor_class import Tensor

if TYPE_CHECKING:
    from aie.iron.device import Device

_CURRENT_DEVICE = None


def set_current_device(device: "Device"):
    """
    Set the current device.

    Args:
        device (Device): The device to set as current.
    """
    global _CURRENT_DEVICE
    _CURRENT_DEVICE = device


def bfloat16_safe_allclose(dtype, arr1, arr2):
    """
    Check if two arrays are element-wise equal within a tolerance, handling bfloat16 safely.

    Args:
        dtype: The data type of the arrays.
        arr1: First input array.
        arr2: Second input array.

    Returns:
        bool: True if the arrays are equal within tolerance, False otherwise.
    """
    if dtype == bfloat16:
        if isinstance(arr1, Tensor):
            arr1 = np.array(arr1, dtype=np.float16)
        else:
            arr1 = arr1.astype(np.float16)
        if isinstance(arr2, Tensor):
            arr2 = np.array(arr2, dtype=np.float16)
        else:
            arr2 = arr2.astype(np.float16)
    return np.allclose(arr1, arr2)
