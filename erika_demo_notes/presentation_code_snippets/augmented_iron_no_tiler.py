# matrix_scalar_add/aie2_iron.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
# fmt: off
import numpy as np

from aie.iron.runtime import Runtime
from aie.iron.dataflow import ObjectFifo
from aie.iron.program import Program
from aie.iron.placers import SequentialPlacer
from aie.iron.worker import Worker
from aie.iron.phys.device import NPU1Col1
from aie.helpers.dialects.ext.scf import _for as range_

# Size of the entire matrix
MATRIX_HEIGHT = 16
MATRIX_WIDTH = 128

# Size of the tile to process
TILE_HEIGHT = 8
TILE_WIDTH = 16

# Types, tile declarations, and AIE data movement with object fifos
matrix_ty = np.ndarray[(MATRIX_HEIGHT, MATRIX_WIDTH), np.dtype[np.int32]]
tile_ty = np.ndarray[(TILE_HEIGHT, TILE_WIDTH), np.dtype[np.int32]]
of_in = ObjectFifo(tile_ty)
of_out = ObjectFifo(tile_ty)

def core_fn(of_in, of_out): # Set up worker
  elem_in = of_in.acquire(1)
  elem_out = of_out.acquire(1)
  for i in range_(TILE_HEIGHT):
    for j in range_(TILE_WIDTH):
      elem_out[i, j] = elem_in[i, j] + 1
  of_in.release(1)
  of_out.release(1)
my_worker = Worker(core_fn, fn_args=[of_in.cons(), of_out.prod()])

rt = Runtime()
with rt.sequence(matrix_ty, matrix_ty) as (in_tensor, out_tensor):
  rt.start(my_worker)
  rt.fill(of_in.prod(), in_tensor, 
      sizes=[1, 1, TILE_HEIGHT, TILE_WIDTH], strides=[0, 0, MATRIX_WIDTH, 1])
  rt.drain(of_out.cons(), out_tensor, wait=True, 
      sizes=[1, 1, TILE_HEIGHT, TILE_WIDTH], strides=[0, 0, MATRIX_WIDTH, 1])

module = Program(NPU1Col1(), rt).resolve_program(SequentialPlacer())
print(module)

# fmt: on
