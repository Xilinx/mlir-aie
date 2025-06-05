# passthrough_kernel/passthrough_kernel_single_col.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import argparse
import sys

from aie.iron import ObjectFifo, Program, Runtime, Worker, Kernel
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col2, NPU2Col1, XCVC1902
from aie.helpers.dialects.ext.scf import _for as range_


def my_passthrough_kernel(
    dev, input_buffer_size, output_buffer_size, trace_buffer_size
):
    input_dtype = np.uint8
    output_dtype = np.uint8

    vector_length = input_buffer_size
    line_length = 1024
    enable_trace = 1 if trace_buffer_size > 0 else 0

    assert input_buffer_size == output_buffer_size
    # Define tensor types
    vector_type = np.ndarray[(vector_length,), np.dtype[np.int32]]
    line_type = np.ndarray[(line_length,), np.dtype[np.int32]]

    # Data movement with ObjectFifos
    input_fifo = ObjectFifo(line_type, name="input")
    intermediate_fifo = ObjectFifo(line_type, name="intermediate")
    output_fifo = ObjectFifo(line_type, name="output")

    # Split input into multiple cores and join outputs back
    num_cores = 4
    split_length = line_length // num_cores

    # Split input into num_cores streams
    split_type = np.ndarray[(split_length,), np.dtype[np.int32]]
    split_offsets_list = [split_length * i for i in range(num_cores)]
    split_input_fifos = input_fifo.cons().split(
        split_offsets_list,
        obj_types=[split_type] * num_cores,
        names=[f"split{i}" for i in range(num_cores)],
    )

    # Join the split streams back together
    joined_output_fifos = intermediate_fifo.prod().join(
        split_offsets_list,
        obj_types=[split_type] * num_cores,
        names=[f"join{i}" for i in range(num_cores)],
    )
    passthrough_kernel = Kernel(
        "passThroughLine",
        "passThrough.cc.o",
        [split_type, split_type, np.int32],
    )

    passthrough_kernel2 = Kernel(
        "passThroughLine2",
        "passThrough.cc.o",
        [line_type, line_type, np.int32],
    )

    def forward_data_worker(input_fifo, output_fifo, passthrough_kernel):
        for _ in range_(sys.maxsize):
            elem_in = input_fifo.acquire(1)
            elem_out = output_fifo.acquire(1)
            passthrough_kernel(elem_in, elem_out, split_length)
            output_fifo.release(1)
            input_fifo.release(1)

    def forward_data_tile0_worker(
        input_fifo,
        output_fifo,
        intermediate_fifo,
        output_fifo2,
        passthrough_kernel,
        passthrough_kernel2,
    ):
        for _ in range_(sys.maxsize):
            elem_in = input_fifo.acquire(1)
            elem_out = output_fifo.acquire(1)
            passthrough_kernel(elem_in, elem_out, split_length)
            output_fifo.release(1)
            input_fifo.release(1)
            elem_in1 = intermediate_fifo.acquire(1)
            elem_out1 = output_fifo2.acquire(1)
            passthrough_kernel2(elem_in1, elem_out1, line_length)
            intermediate_fifo.release(1)
            output_fifo2.release(1)

    # Create workers to forward data through each split path
    worker_list = []
    for i in range(num_cores):
        if i == 0:
            worker_list.append(
                Worker(
                    forward_data_tile0_worker,
                    fn_args=[
                        split_input_fifos[i].cons(),
                        joined_output_fifos[i].prod(),
                        intermediate_fifo.cons(),
                        output_fifo.prod(),
                        passthrough_kernel,
                        passthrough_kernel2,
                    ],
                )
            )
        else:
            worker_list.append(
                Worker(
                    forward_data_worker,
                    fn_args=[
                        split_input_fifos[i].cons(),
                        joined_output_fifos[i].prod(),
                        passthrough_kernel,
                    ],
                )
            )
    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(vector_type, vector_type, vector_type) as (a_in, b_out, _):
        rt.enable_trace(trace_size)
        rt.start(*worker_list)
        rt.fill(input_fifo.prod(), a_in)
        rt.drain(output_fifo.cons(), b_out, wait=True)

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
p.add_argument(
    "-i1s", "--in1_size", required=True, dest="in1_size", help="Input 1 size"
)
p.add_argument("-os", "--out_size", required=True, dest="out_size", help="Output size")
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
    dev = NPU2()
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
trace_size = int(opts.trace_size)

print(my_passthrough_kernel(dev, in1_size, out_size, trace_size))
