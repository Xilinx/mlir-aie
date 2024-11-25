# array.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

import numpy as np
from typing import Sequence


class array:
    def __init__(
        self,
        shape: Sequence[int],
        dtype: np.dtype,
        initial_values: np.ndarray | None = None,
        num_buffs: int | None = 2,
    ):
        self._array = None
        self._dtype = dtype
        self._shape = shape
        self._num_buffs = 2
        if not (num_buffs is None):
            if num_buffs < 1:
                raise ValueError(f"num_buffs must be >= 1, but got {num_buffs}")
            self._num_buffs = num_buffs
        if not (initial_values is None):
            if self._dtype != initial_values.dtype:
                raise ValueError(
                    f"Initial values dtype {initial_values.dtype} does not match given dtype {self._dtype}"
                )
            if self._shape != initial_values.shape:
                raise ValueError(
                    f"Initial values shape {initial_values.shape} does not match given shape {self._shape}"
                )
            self._array = initial_values

    def asnumpy(self):
        if self._array is None:
            self._array = np.zeros(self._shape, self._dtype)
        return self._array


def asarray(arr: np.ndarray, num_buffs: int | None = None):
    return array(arr.shape, arr.dtype, arr, num_buffs)
