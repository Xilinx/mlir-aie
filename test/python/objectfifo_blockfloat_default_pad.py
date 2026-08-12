# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# Regression: ObjectFifo.resolve must not compute np.dtype(dtype).itemsize for the
# default pad_value=0. That lookup raises "Converting np.generic to a dtype is not
# allowed" on block-float element types like v8bfp16ebs8, so resolving a fifo of
# that type (no padding) must succeed. Guarded by the `if self._pad_value` check.

import aie.iron as iron
import numpy as np
from aie.dialects.aiex import v8bfp16ebs8
from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.device import AnyShimTile, from_name

iron.set_current_device(from_name("npu2", n_cols=1))

TILE = np.ndarray[(16,), np.dtype[v8bfp16ebs8]]

of_in = ObjectFifo(TILE, name="in")
of_out = of_in.cons().forward(name="out")


def sequence(a, c, in_h, out_h):
    in_h.fill(a)
    out_h.drain(c, wait=True)


rt = Runtime(
    sequence,
    [TILE, TILE, of_in.prod(tile=AnyShimTile), of_out.cons(tile=AnyShimTile)],
)

# CHECK: aie.objectfifo
print(Program(iron.get_current_device(), rt).resolve_program())
