# data.py -*- Python -*-
#
# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import numpy as np
from typing import Sequence

from ...extras.dialects.memref import (  # pyright: ignore[reportMissingImports]
    MemRefValue,
)
from ...helpers.util import (
    NpuDType,
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
    def dtype(self) -> NpuDType:
        """The per-element datatype of the buffer"""
        return np_ndarray_type_get_dtype(self._arr_type)

    @property
    def arr_type(self) -> type[np.ndarray]:
        """The tensor type of the buffer."""
        return self._arr_type

    @property
    def is_scalar(self) -> bool:
        """Whether this runtime argument is a scalar (no shape) rather than a
        tensor. Scalar runtime args (e.g. a runtime ``M``/``K``/``N``) are passed
        to the sequence body as their live SSA value, since they are used in
        arithmetic and ``range_``/``if_`` bounds, not as fill/drain buffers."""
        try:
            return len(np_ndarray_type_get_shape(self._arr_type)) == 0
        except IndexError:
            return True

    def default_tap(self) -> TensorAccessPattern:
        """A default access pattern for a linear transfer of the buffer."""
        # TODO: what if not two dimensional?
        return TensorTiler2D.simple_tiler(self.shape)[0]

    @property
    def op(self) -> MemRefValue:
        if self._op is None:
            raise ValueError("Cannot get operation for RuntimeData before it is set.")
        return self._op

    @op.setter
    def op(self, op: MemRefValue):
        if self._op:
            raise ValueError("Cannot set operation for RuntimeData more than once.")
        self._op = op
