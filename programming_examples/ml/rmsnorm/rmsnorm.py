# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.helpers.dialects.ext.scf import _for as range_
from ml_dtypes import bfloat16


def rmsnorm(dev, sequence_length, embedding_dim, trace_size):
    enable_trace = 1 if trace_size > 0 else None

    n_cores = 8

    total_volume = sequence_length * embedding_dim

    dtype = np.ndarray[(total_volume,), np.dtype[bfloat16]]

    rows_per_core = sequence_length // n_cores
    chunk_volume = embedding_dim
    chunk_type = np.ndarray[(chunk_volume,), np.dtype[bfloat16]]

    of_in = [ObjectFifo(chunk_type, name=f"in_{i}") for i in range(n_cores)]
    of_out = [ObjectFifo(chunk_type, name=f"out_{i}") for i in range(n_cores)]

    rms_norm_kernel = Kernel(
        "rms_norm", "rms_norm.o", [chunk_type, chunk_type, np.int32]
    )
    taps_in = []
    taps_out = []
    for i in range(n_cores):
        taps = TensorAccessPattern(
            (sequence_length, embedding_dim),
            offset=embedding_dim * rows_per_core * i,
            sizes=[1, 1, 1, embedding_dim * rows_per_core],
            strides=[0, 0, 0, 1],
        )
        # taps.visualize(
        #     title=f"Core {i} input tap",
        #     show_arsequence_length=True,
        #     plot_access_count=True,
        #     file_path=f"core_{i}_input_tap.png",
        # )
        taps_in.append(taps)

    for i in range(n_cores):
        taps = TensorAccessPattern(
            (sequence_length, embedding_dim),
            offset=embedding_dim * rows_per_core * i,
            sizes=[1, 1, 1, embedding_dim * rows_per_core],
            strides=[0, 0, 0, 1],
        )
        # taps.visualize(
        #     title=f"Core {i} output tap",
        #     show_arsequence_length=True,
        #     plot_access_count=True,
        #     file_path=f"core_{i}_output_tap.png",
        # )
        taps_out.append(taps)

    def core_body(of_in, of_out, rms_norm_kernel):
        for i in range_(rows_per_core):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            rms_norm_kernel(elem_in, elem_out, embedding_dim)
            of_in.release(1)
            of_out.release(1)

    workers = [
        Worker(
            core_body,
            fn_args=[of_in[i].cons(), of_out[i].prod(), rms_norm_kernel],
            trace=None,  # enable_trace
        )
        for i in range(n_cores)
    ]

    rt = Runtime()
    with rt.sequence(dtype, dtype) as (a_in, c_out):
        # rt.enable_trace(trace_size)
        rt.start(*workers)
        for i in range(n_cores):
            rt.fill(of_in[i].prod(), a_in, taps_in[i])
        for i in range(n_cores):
            rt.drain(of_out[i].cons(), c_out, taps_out[i], wait=True)
    return Program(dev, rt).resolve_program(SequentialPlacer())


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
p.add_argument(
    "-s", "--sequence_length", required=True, dest="sequence_length", help="Row size"
)
p.add_argument(
    "-e", "--embedding_dim", required=True, dest="embedding_dim", help="Col size"
)
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
    dev = NPU1()
elif opts.device == "npu2":
    dev = NPU2()
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(opts.device))

sequence_length = int(opts.sequence_length)
embedding_dim = int(opts.embedding_dim)
trace_size = int(opts.trace_size)

print(rmsnorm(dev, sequence_length, embedding_dim, trace_size))
