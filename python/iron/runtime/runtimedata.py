# runtimedata.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

import numpy as np
from ...extras.dialects.ext.memref import MemRef


class RuntimeData:
    def __init__(self, dtype: type[np.ndarray]):
        self.dtype = dtype
        self._op = None

    @property
    def op(self) -> MemRef:
        assert self._op != None
        return self._op

    @op.setter
    def op(self, op: MemRef):
        assert self._op == None
        self._op = op
