# config.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import numpy as np
from ml_dtypes import bfloat16

dtype_map = {
    "bf16": bfloat16,
    "i8": np.int8,
    "i16": np.int16,
    "f32": np.float32,
    "i32": np.int32,
}


def str_to_dtype(dtype_str):
    """
    Convert a string representation of a data type to its corresponding dtype object.

    Args:
        dtype_str (str): The string representation of the data type.

    Returns:
        dtype: The corresponding dtype object.
    """

    value = None
    try:
        value = dtype_map[dtype_str]
    except KeyError:
        raise ValueError(f"Unrecognized dtype: {dtype_str}")
    return value


def dtype_to_str(dtype):
    """
    Convert a dtype object to its string representation.

    Args:
        dtype: The dtype object to convert.

    Returns:
        str: The string representation of the dtype.
    """

    for key, value in dtype_map.items():
        if value == dtype:
            return key
    raise ValueError(f"Unrecognized dtype: {dtype}")
