#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""Shared helpers for the mobilenet bottleneck modules."""

import os
import numpy as np


def i8(shape):
    """numpy ndarray type alias: int8 with the given shape."""
    return np.ndarray[shape, np.dtype[np.int8]]


def u8(shape):
    """numpy ndarray type alias: uint8 with the given shape."""
    return np.ndarray[shape, np.dtype[np.uint8]]


def load_wts(data_dir, filename, expected_size):
    """Load int8 weights from `data_dir/filename`.

    Raises FileNotFoundError if the file is missing or ValueError if its size
    doesn't match `expected_size` — silent fallback to zero-filled weights
    used to compile a numerically-broken design that ran fine but produced
    wrong output.
    """
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"weight file not found: {path}")
    arr = np.fromfile(path, sep=",", dtype=np.int8)
    if arr.size != expected_size:
        raise ValueError(
            f"{path}: expected {expected_size} int8 elements, got {arr.size}"
        )
    return arr
