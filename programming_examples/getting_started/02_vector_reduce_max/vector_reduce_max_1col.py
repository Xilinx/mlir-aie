# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
import argparse
import sys

from aie.iron import (
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    LocalBuffer,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.helpers.util import np_ndarray_type_get_shape
from ml_dtypes import bfloat16
from aie.iron.controlflow import range_

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

devices = {
    "npu": NPU1Col1(),
    "npu2": NPU2Col1()
}
if len(sys.argv) != 2 or sys.argv[1] not in devices:
    print(f"Usage {sys.argv[0]} <{'|'.join(devices.keys())}>")
    sys.exit(1)
device = devices[sys.argv[1]]

in1_size = 524288
out_size = 4
dtype = bfloat16

n_cores = 4
n_mem_elems = 2048
elems_per_core = n_mem_elems // n_cores

in_tensor_size = in1_size // dtype(0).nbytes
out_tensor_size = out_size // dtype(0).nbytes

num_iter = in_tensor_size // n_mem_elems

assert out_size == 4, "Output buffer must be size 4 (4 bytes = 1 integer)."


# --------------------------------------------------------------------------
# In-Array Data Movement
# --------------------------------------------------------------------------

in_ty = np.ndarray[(in_tensor_size,), np.dtype[dtype]]
mem_ty = np.ndarray[(n_mem_elems,), np.dtype[dtype]]
op_ty = np.ndarray[(elems_per_core,), np.dtype[dtype]]
out_ty = np.ndarray[(out_tensor_size,), np.dtype[dtype]]

# Input A and Output C
of_in = ObjectFifo(mem_ty, name="of_in")

in_fifos = []
out_fifos = []

if n_cores > 1:
    of_a_offsets = [
        (np.prod(np_ndarray_type_get_shape(mem_ty)) // n_cores) * i
        for i in range(n_cores)
    ]
else:
    of_a_offsets = [0]

in_fifos = of_in.cons().split(
    of_a_offsets,
    obj_types=[op_ty] * n_cores,
    names=[f"memA{i}" for i in range(n_cores)],
)
for i in range(n_cores):
    out_fifos.append(ObjectFifo(out_ty, name=f"memC{i}"))


# --------------------------------------------------------------------------
# Task each core will run
# --------------------------------------------------------------------------

reduce_max_vector = Kernel(
    f"reduce_max_vector_bfloat16", "reduce_max.cc.o", [op_ty, out_ty, np.int32]
)
compute_max = Kernel(
    f"compute_max_bfloat16", "reduce_max.cc.o", [out_ty, out_ty, out_ty]
)
min_val = np.array([bfloat16(float("-inf"))], dtype=dtype)

def start_core_body(of_in, of_out, reduce_max_vector, compute_max):
    nextC_buffer = LocalBuffer(
        type=np.ndarray[(out_tensor_size,), np.dtype[dtype]],
        initial_value=min_val,
    )
    tmp_buffer = LocalBuffer(
        type=np.ndarray[(out_tensor_size,), np.dtype[dtype]],
        initial_value=min_val,
    )
    elem_out = of_out.acquire(1)
    for _ in range_(num_iter):
        elem_in = of_in.acquire(1)
        reduce_max_vector(elem_in, tmp_buffer, elems_per_core)
        compute_max(nextC_buffer, tmp_buffer, nextC_buffer)
        of_in.release(1)
    elem_out[0] = nextC_buffer[0]
    of_out.release(1)

def core_body(of_in, of_out, in0, reduce_max_vector, compute_max):
    nextC_buffer = LocalBuffer(
        type=np.ndarray[(out_tensor_size,), np.dtype[dtype]],
        initial_value=min_val,
    )
    tmp_buffer = LocalBuffer(
        type=np.ndarray[(out_tensor_size,), np.dtype[dtype]],
        initial_value=min_val,
    )

    for _ in range_(num_iter):
        elem_in = of_in.acquire(1)
        reduce_max_vector(elem_in, tmp_buffer, elems_per_core)
        compute_max(nextC_buffer, tmp_buffer, nextC_buffer)
        of_in.release(1)

    elem_out = of_out.acquire(1)
    elem_in1 = in0.acquire(1)
    compute_max(elem_in1, nextC_buffer, elem_out)
    in0.release(1)
    of_out.release(1)

workers = []
for i in range(n_cores):
    if i != n_cores - 1:
        workers.append(
            Worker(
                core_body,
                fn_args=[
                    in_fifos[i].cons(),
                    out_fifos[i].prod(),
                    out_fifos[i + 1].cons(),
                    reduce_max_vector,
                    compute_max,
                ],
                trace=None,
            )
        )
    else:
        workers.append(
            Worker(
                start_core_body,
                fn_args=[
                    in_fifos[i].cons(),
                    out_fifos[i].prod(),
                    reduce_max_vector,
                    compute_max,
                ],
                trace=None,
            )
        )


# --------------------------------------------------------------------------
# DRAM-NPU data movement and work dispatch
# --------------------------------------------------------------------------

rt = Runtime()
with rt.sequence(in_ty, out_ty) as (a_in, c_out):
    rt.start(*workers)
    rt.fill(of_in.prod(), a_in)
    rt.drain(out_fifos[0].cons(), c_out, wait=True)


# --------------------------------------------------------------------------
# Place and generate MLIR program
# --------------------------------------------------------------------------

program = Program(device, rt)
module = program.resolve_program(SequentialPlacer())
print(module)
