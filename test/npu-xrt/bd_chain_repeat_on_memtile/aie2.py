# aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

# REQUIRES: ryzen_ai_npu1, valid_xchess_license
# RUN: xchesscc_wrapper aie2 -I %aietools/include -c %S/kernel.cc -o ./kernel.cc.o
# RUN: %python %S/aie2.py npu > ./aie.mlir
# RUN: clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags
# RUN: %python aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=final.xclbin --aie-generate-npu-insts --npu-insts-name=insts.bin ./aie.mlir
# RUN: %run_on_npu1% ./test.exe -x final.xclbin -k MLIR_AIE -i insts.bin | FileCheck %s
# CHECK: PASS!

# This example demonstrates the use of `bd_chain_iter_count` in conjunction with
# `split`, `join`, and `repeat_count` features of objectFifo on a MemTile.
#
# - `bd_chain_iter_count` is the number of times the buffer descriptor (BD) chain iterates for each objectFifo.
# - `repeat_count` is set on the split FIFOs to repeat the data for each consumer.
#
# The code below sets up a pipeline where input data is distributed to two compute tiles,
# each processes its chunk, and the results are joined and repeated as specified.

import numpy as np
import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

import aie.utils.trace as trace_utils


def my_passthrough_kernel(in1_size, out_size):
    in1_dtype = np.uint8
    out_dtype = np.uint8

    N = in1_size // in1_dtype(0).nbytes
    n_cores = 2
    chunk_size = 1024
    elements_per_core_per_chunk = chunk_size // n_cores
    in_num_iterations = 8
    out_num_iterations = 16
    repeat_counter = 2

    N_out = out_size // out_dtype(0).nbytes

    base_out_size = out_size // repeat_counter

    @device(AIEDevice.npu1_2col)
    def device_body():
        full_vector_ty = np.ndarray[(N,), np.dtype[in1_dtype]]
        full_output_vector_ty = np.ndarray[(N_out,), np.dtype[out_dtype]]
        chunk_ty = np.ndarray[(chunk_size,), np.dtype[in1_dtype]]
        core_chunk_ty = np.ndarray[(elements_per_core_per_chunk,), np.dtype[in1_dtype]]

        # AIE Core Function declarations
        passThroughLine = external_func(
            "passThroughLine", inputs=[core_chunk_ty, core_chunk_ty, np.int32]
        )

        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        ComputeTile2 = tile(0, 2)
        ComputeTile3 = tile(0, 3)
        compute_tiles = [ComputeTile2, ComputeTile3]

        tiles_to_trace = compute_tiles + [MemTile, ShimTile]
        trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)

        # AIE-array data movement with object fifos
        of_in = object_fifo(
            "in", ShimTile, MemTile, 2, chunk_ty, bd_chain_iter_count=in_num_iterations
        )

        of_split = []
        for i in range(n_cores):
            split_fifo = object_fifo(
                f"split_{i}",
                MemTile,
                compute_tiles[i],
                2,
                core_chunk_ty,
                bd_chain_iter_count=in_num_iterations,
            )
            split_fifo.set_repeat_count(repeat_counter)
            of_split.append(split_fifo)

        split_offsets = [i * elements_per_core_per_chunk for i in range(n_cores)]
        object_fifo_link([of_in], of_split, [], split_offsets)

        of_join = []
        for i in range(n_cores):
            join_fifo = object_fifo(
                f"join_{i}",
                compute_tiles[i],
                MemTile,
                2,
                core_chunk_ty,
                bd_chain_iter_count=out_num_iterations,
            )
            of_join.append(join_fifo)

        of_out = object_fifo(
            "out",
            MemTile,
            ShimTile,
            2,
            chunk_ty,
            bd_chain_iter_count=out_num_iterations,
        )

        join_offsets = [i * elements_per_core_per_chunk for i in range(n_cores)]
        object_fifo_link(of_join, [of_out], join_offsets, [])

        for i, compute_tile in enumerate(compute_tiles):

            def make_core_fn(idx):
                @core(compute_tile, "kernel.cc.o")
                def core_body():
                    for _ in range_(sys.maxsize):
                        elemOut = of_join[idx].acquire(ObjectFifoPort.Produce, 1)
                        elemIn = of_split[idx].acquire(ObjectFifoPort.Consume, 1)
                        passThroughLine(elemIn, elemOut, elements_per_core_per_chunk)
                        of_split[idx].release(ObjectFifoPort.Consume, 1)
                        of_join[idx].release(ObjectFifoPort.Produce, 1)

                return core_body

            make_core_fn(i)

        @runtime_sequence(full_vector_ty, full_output_vector_ty, full_vector_ty)
        def sequence(inTensor, outTensor, notUsed):
            in_task = shim_dma_single_bd_task(of_in, inTensor, sizes=[1, 1, 1, N])
            dma_start_task(in_task)

            out_task = shim_dma_single_bd_task(
                of_out, outTensor, sizes=[1, 1, 1, N_out], issue_token=True
            )
            dma_start_task(out_task)
            dma_await_task(out_task)
            dma_free_task(in_task)

            trace_utils.gen_trace_done_aie2(ShimTile)


in1_size = 16384
out_size = 32768

with mlir_mod_ctx() as ctx:
    my_passthrough_kernel(in1_size, out_size)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
