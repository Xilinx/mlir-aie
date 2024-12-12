# data.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

import numpy as np
from typing import Sequence

from ...extras.dialects.ext.memref import MemRef  # type: ignore
from ...helpers.util import (
    np_ndarray_type_get_dtype,
    np_ndarray_type_get_shape,
)
from ...helpers.taplib import TensorAccessPattern, TensorTiler2D


class RuntimeData:
    """A handle to I/O data in the Runtime"""

    def __init__(self, arr_type: type[np.ndarray]):
        """Construct a handle to a Runtime buffer.

        Args:
            arr_type (type[np.ndarray]): The type of the I/O data.
        """
        self._arr_type = arr_type
        self._op = None

    @property
    def shape(self) -> Sequence[int]:
        """The shape of the buffer"""
        return np_ndarray_type_get_shape(self._arr_type)

    @property
    def dtype(self) -> np.dtype:
        """The per-element datatype of the buffer"""
        return np_ndarray_type_get_dtype(self._arr_type)

    @property
    def arr_type(self) -> np.ndarray:
        """The tensor type of the buffer."""
        return self._arr_type

    def default_tap(self) -> TensorAccessPattern:
        """A default access pattern for a linear transfer of the buffer."""
        # TODO: what if not two dimensional?
        return TensorTiler2D.simple_tiler(self.shape)[0]

    @property
    def op(self) -> MemRef:
        if self._op is None:
            raise ValueError("Cannot get operation for RuntimeData before it is set.")
        return self._op

    @op.setter
    def op(self, op: MemRef):
        if self._op:
            raise ValueError("Cannot set operation for RuntimeData more than once.")
        self._op = op
