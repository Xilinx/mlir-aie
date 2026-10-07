# data.py -*- Python -*-
#
# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import math
from typing import Sequence, get_origin

import numpy as np

from ... import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ...dialects import memref  # pyright: ignore[reportAttributeAccessIssue]
from ...extras.dialects.memref import (  # pyright: ignore[reportMissingImports]
    MemRefValue,
)
from ...helpers.npdtypes import (
    NpuDType,
    np_ndarray_type_get_dtype,
    np_ndarray_type_get_shape,
)
from ...helpers.taplib import TensorAccessPattern


class RuntimeData:
    """A handle to I/O data in the Runtime."""

    def __init__(self, arr_type: type[np.ndarray]):
        """Construct a handle to a Runtime buffer.

        Args:
            arr_type (type[np.ndarray]): The type of the I/O data.
        """
        self._arr_type = arr_type
        self._op = None

    @property
    def shape(self) -> Sequence[int]:
        """Return the shape of the buffer."""
        return np_ndarray_type_get_shape(self._arr_type)

    @property
    def dtype(self) -> type[NpuDType]:
        """Return the per-element datatype of the buffer."""
        return np_ndarray_type_get_dtype(self._arr_type)

    @property
    def arr_type(self) -> type[np.ndarray]:
        """The tensor type of the buffer."""
        return self._arr_type

    @property
    def is_scalar(self) -> bool:
        """Whether this runtime argument is a scalar (no shape) rather than a tensor.

        Scalar runtime args (e.g. a runtime ``M``/``K``/``N``) are passed
        to the sequence body as their live SSA value, since they are used in
        arithmetic and ``range_``/``if_`` bounds, not as fill/drain buffers.
        """
        if get_origin(self._arr_type) is not np.ndarray:
            # Not an np.ndarray[...] generic alias at all (e.g. bare np.int32).
            return True
        return len(np_ndarray_type_get_shape(self._arr_type)) == 0

    def default_tap(self) -> TensorAccessPattern:
        """Return a default access pattern for a linear transfer of the buffer."""
        return TensorAccessPattern.full(self.shape)

    def __getitem__(self, key) -> TensorAccessPattern:
        """Return the access pattern for a numpy-style slice of this buffer.

        For example, ``flow.fill(a, tap=a[0::2, 1::2, ...])`` selects a transfer
        region. ``TensorAccessPattern.full(shape)[key]`` computes element
        offsets, sizes and strides directly from the shape and key, assuming C-order.
        This returns metadata, not a numpy view or runtime values; indexing
        cannot read the buffer's contents.
        """
        return TensorAccessPattern.full(self.shape)[key]

    def window(self, offset: int, shape: Sequence[int]):
        """Return a contiguous memref window for a runtime-sequence call."""
        if len(self.shape) != 1:
            raise ValueError("RuntimeData.window requires a one-dimensional source.")
        shape = tuple(shape)
        if not shape or any(size < 0 for size in shape):
            raise ValueError(f"RuntimeData.window requires a non-empty static shape: {shape}.")
        size = math.prod(shape)
        if offset < 0 or offset + size > self.shape[0]:
            raise ValueError(
                f"RuntimeData.window [{offset}, {offset + size}) exceeds source "
                f"shape {tuple(self.shape)}."
            )

        source_type = ir.MemRefType(self.op.type)
        view = memref.subview(
            self.op,
            offsets=[offset],
            sizes=[size],
            strides=[1],
        )
        result_type = ir.MemRefType.get(
            shape,
            source_type.element_type,
            memory_space=source_type.memory_space,
        )
        strides = [math.prod(shape[index + 1 :]) for index in range(len(shape))]
        return memref.reinterpret_cast(
            result_type,
            view,
            offsets=[],
            sizes=[],
            strides=[],
            static_offsets=[0],
            static_sizes=shape,
            static_strides=strides,
        )

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
