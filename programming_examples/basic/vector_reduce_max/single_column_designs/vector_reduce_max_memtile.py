# single_column_designs/vector_reduce_max_memtile.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker, LocalBuffer
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col2, NPU2Col2
from aie.helpers.util import np_ndarray_type_get_shape
from ml_dtypes import bfloat16
from aie.iron.controlflow import range_

dtype_map = {
    "bf16": bfloat16,
    "i32": np.int32,
}


def my_reduce_max(dev, in1_size, out_size, dtype_str, trace_size):
    n_cores = 4
    n_mem_elems = 2048
    elems_per_core = n_mem_elems // n_cores

    dtype = dtype_map[dtype_str]

    in_tensor_size = in1_size // dtype(0).nbytes
    out_tensor_size = out_size // dtype(0).nbytes

    num_iter = in_tensor_size // n_mem_elems

    assert out_size == 4, "Output buffer must be size 4 (4 bytes = 1 integer)."

    enable_trace = 1 if trace_size > 0 else 0

    # Define tensor types
    in_ty = np.ndarray[(in_tensor_size,), np.dtype[dtype]]
    mem_ty = np.ndarray[(n_mem_elems,), np.dtype[dtype]]
    op_ty = np.ndarray[(elems_per_core,), np.dtype[dtype]]
    out_ty = np.ndarray[(out_tensor_size,), np.dtype[dtype]]
    int_ty = np.ndarray[(out_tensor_size * n_cores,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    in_fifos = []
    out_fifos = []

    of_in = ObjectFifo(mem_ty, name="of_in")
    outC = ObjectFifo(int_ty, name="outC", dims_to_stream=[(1, 2), (1, 1)])
    of_out = ObjectFifo(out_ty, name="of_out")

    if n_cores > 1:
        of_a_offsets = [
            (np.prod(np_ndarray_type_get_shape(mem_ty)) // n_cores) * i
            for i in range(n_cores)
        ]
        of_c_offsets = [(out_tensor_size * i) for i in range(n_cores)]
    else:
        of_a_offsets = [0]
        of_c_offsets = [0]

    in_fifos = of_in.cons().split(
        of_a_offsets,
        obj_types=[op_ty] * n_cores,
        names=[f"memA{i}" for i in range(n_cores)],
    )

    out_fifos = outC.prod().join(
        of_c_offsets,
        obj_types=[out_ty] * n_cores,
        names=[f"memC{i}" for i in range(n_cores)],
    )

    # AIE Core Function declarations
    suffix = "_bfloat16" if dtype_str == "bf16" else ""
    reduce_max_vector = Kernel(
        f"reduce_max_vector{suffix}", "reduce_max.cc.o", [op_ty, out_ty, np.int32]
    )
    reduce_max_scalar = Kernel(
        f"reduce_max_scalar{suffix}", "reduce_max.cc.o", [int_ty, out_ty, np.int32]
    )
    compute_max = Kernel(
        f"compute_max{suffix}", "reduce_max.cc.o", [out_ty, out_ty, out_ty]
    )
    min_val = (
        np.array([bfloat16(float("-inf"))], dtype=dtype)
        if dtype_str == "bf16"
        else np.array([np.iinfo(dtype).min], dtype=dtype)
    )

    # Define a task to run
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
            compute_max(tmp_buffer, nextC_buffer, nextC_buffer)
            of_in.release(1)
        elem_out[0] = nextC_buffer[0]
        of_out.release(1)

    def core_body(
        of_in,
        elemC_out,
        elemA_in,
        of_out,
        reduce_max_vector,
        reduce_max_scalar,
        compute_max,
    ):
        nextC_buffer = LocalBuffer(
            type=np.ndarray[(out_tensor_size,), np.dtype[dtype]],
            initial_value=min_val,
        )
        tmp_buffer = LocalBuffer(
            type=np.ndarray[(out_tensor_size,), np.dtype[dtype]],
            initial_value=min_val,
        )
        elem_out = elemC_out.acquire(1)
        for _ in range_(num_iter):
            elem_in = of_in.acquire(1)
            reduce_max_vector(elem_in, tmp_buffer, elems_per_core)
            compute_max(tmp_buffer, nextC_buffer, nextC_buffer)
            of_in.release(1)
        elem_out[0] = nextC_buffer[0]
        elemC_out.release(1)

        elem_out1 = of_out.acquire(1)
        elem_in1 = elemA_in.acquire(1)
        reduce_max_scalar(elem_in1, elem_out1, n_cores)
        elemA_in.release(1)
        of_out.release(1)

    # Define a worker to run the task on a core
    workers = []
    for i in range(n_cores):
        if i != 0 and n_cores > 1:
            workers.append(
                Worker(
                    start_core_body,
                    fn_args=[
                        in_fifos[i].cons(),
                        out_fifos[i].prod(),
                        reduce_max_vector,
                        compute_max,
                    ],
                    trace=True if i == 1 else None,
                )
            )
        else:
            fifo_args = [
                in_fifos[i].cons(),
                out_fifos[i].prod(),
                outC.cons(),
                of_out.prod(),
                reduce_max_vector,
                reduce_max_scalar,
                compute_max,
            ]
            workers.append(Worker(core_body, fn_args=fifo_args, trace=None))

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (a_in, c_out):
        rt.enable_trace(trace_size)
        rt.start(*workers)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), c_out, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
p.add_argument(
    "-i1s", "--in1_size", required=True, dest="in1_size", help="Input 1 size"
)
p.add_argument("-os", "--out_size", required=True, dest="out_size", help="Output size")
p.add_argument("-dt", "--dtype", required=True, dest="dtype", help="Datatype")
p.add_argument(
    "-t",
    "--trace_size",
    required=False,
    dest="trace_size",
    default=0,
    help="Trace buffer size",
)
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = NPU1Col2()
elif opts.device == "npu2":
    dev = NPU2Col2()
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(opts.device))

in1_size = int(opts.in1_size)
if in1_size % 64 != 0 or in1_size < 512:
    print(
        "In1 buffer size ("
        + str(in1_size)
        + ") must be a multiple of 64 and greater than or equal to 512"
    )
    raise ValueError
out_size = int(opts.out_size)
dtype = str(opts.dtype)
trace_size = int(opts.trace_size)

print(my_reduce_max(dev, in1_size, out_size, dtype, trace_size))
