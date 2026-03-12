# data.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

import numpy as np
from typing import Sequence

from ...extras.dialects.memref import MemRefValue  # type: ignore
from ...helpers.util import (
    np_ndarray_type_get_dtype,
    np_ndarray_type_get_shape,
)
from ...helpers.taplib import TensorAccessPattern, TensorTiler2D
from ...ir import Type as MlirType


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
    def op(self) -> MemRefValue:
        if self._op is None:
            raise ValueError("Cannot get operation for RuntimeData before it is set.")
        return self._op

    @op.setter
    def op(self, op: MemRefValue):
        if self._op:
            raise ValueError("Cannot set operation for RuntimeData more than once.")
        self._op = op


class RuntimeScalar:
    """A handle to a scalar runtime parameter (e.g. T.i32()) in the Runtime sequence."""

    def __init__(self, mlir_type: MlirType):
        """Construct a handle to a scalar Runtime parameter.

        Args:
            mlir_type (MlirType): The MLIR type of the scalar (e.g. IntegerType.get_signless(32)).
        """
        self._mlir_type = mlir_type
        self._op = None

    @property
    def mlir_type(self) -> MlirType:
        """The MLIR type of this scalar."""
        return self._mlir_type

    @property
    def op(self):
        """The MLIR SSA value for this scalar, set during resolve."""
        if self._op is None:
            raise ValueError("Cannot get operation for RuntimeScalar before it is set.")
        return self._op

    @op.setter
    def op(self, op):
        self._op = op
