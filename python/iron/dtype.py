# dtype.py -*- Python -*-
#
# Copyright (C) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Utilities for converting between short string names and numpy dtype objects."""

import numpy as np
from ml_dtypes import bfloat16

# Mapping from short string names (e.g. 'bf16', 'i32') to numpy/ml_dtypes dtype objects.
dtype_map = {
    "bf16": bfloat16,
    "i8": np.int8,
    "i16": np.int16,
    "f32": np.float32,
    "i32": np.int32,
}


def str_to_dtype(dtype_str: str) -> type:
    """
    Convert a string representation of a data type to its corresponding dtype object.

    Args:
        dtype_str: The string representation of the data type.

    Returns:
        The corresponding numpy/ml_dtypes type object.
    """

    value = None
    try:
        value = dtype_map[dtype_str]
    except KeyError:
        raise ValueError(f"Unrecognized dtype: {dtype_str}")
    return value


def dtype_to_str(dtype: type) -> str:
    """
    Convert a dtype object to its string representation.

    Args:
        dtype: The dtype object to convert.

    Returns:
        The string representation of the dtype.
    """

    for key, value in dtype_map.items():
        if value == dtype:
            return key
    raise ValueError(f"Unrecognized dtype: {dtype}")
