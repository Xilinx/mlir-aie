# binkernel.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

import numpy as np

from ... import ir  # type: ignore
from ...extras.dialects.ext.func import FuncOp
from ...helpers.dialects.ext.func import call
from ...dialects.aie import external_func
from .kernel import Kernel


class BinKernel(Kernel):
    def __init__(
        self,
        name: str,
        bin_name: str,
        arg_types: list[type[np.ndarray] | np.dtype] = [],
    ) -> None:
        self._name = name
        self._bin_name = bin_name
        self._arg_types = arg_types
        self._op: FuncOp | None = None

    @property
    def bin_name(self) -> str:
        return self._bin_name

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if self._op == None:
            self._op = external_func(self._name, inputs=self._arg_types)

    def __call__(self, *args, **kwargs):
        assert self._op, "Need to resolve BinKernel before it can be called"
        call(self._op, args, **kwargs)
