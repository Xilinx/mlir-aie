# runtime_add_buffer.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

# RUN: %python %s | FileCheck %s

import numpy as np

from aie.iron import Buffer, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile

tile_ty = np.ndarray[(16,), np.dtype[np.int32]]

of_in = ObjectFifo(tile_ty, name="in")
of_out = ObjectFifo(tile_ty, name="out")


def core_fn(of_in, of_out):
    a = of_in.acquire(1)
    b = of_out.acquire(1)
    for i in range(16):
        b[i] = a[i]
    of_in.release(1)
    of_out.release(1)


worker = Worker(core_fn, [of_in.cons(), of_out.prod()])
staged = Buffer(tile_ty, name="staged", tile=Tile(1, 1))


def sequence(a, b, fifo_in, fifo_out):
    fifo_in.fill(a)
    fifo_out.drain(b, wait=True)


rt = Runtime(sequence, [tile_ty, tile_ty, of_in.prod(), of_out.cons()])
rt.add_buffer(staged)

# CHECK: %[[MT:.*]] = aie.logical_tile<MemTile>(1, 1)
# CHECK: aie.buffer(%[[MT]]) {sym_name = "staged"} : memref<16xi32>
# CHECK: aie.core
# CHECK: aie.runtime_sequence
print(Program(NPU2(), rt, workers=[worker]).resolve_program())
