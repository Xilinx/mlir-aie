# runtimedata.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

import numpy as np
from typing import Sequence

from ...extras.dialects.ext.memref import MemRef
from ...helpers.util import (
    np_ndarray_type_get_dtype,
    np_ndarray_type_get_shape,
)
from ...helpers.taplib import TensorAccessPattern, TensorTiler2D


class RuntimeData:
    def __init__(self, arr_type: type[np.ndarray]):
        self._arr_type = arr_type
        self._op = None

    @property
    def shape(self) -> Sequence[int]:
        return np_ndarray_type_get_shape(self._arr_type)

    @property
    def dtype(self) -> np.dtype:
        return np_ndarray_type_get_dtype(self._arr_type)

    @property
    def arr_type(self) -> np.ndarray:
        return self._arr_type

    def default_tap(self) -> TensorAccessPattern:
        # TODO: what if not two dimensional?
        return TensorTiler2D.simple_tiler(self.shape)[0]

    @property
    def op(self) -> MemRef:
        assert self._op != None
        return self._op

    @op.setter
    def op(self, op: MemRef):
        assert self._op == None
        self._op = op
