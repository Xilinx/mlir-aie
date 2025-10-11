# passthrough_kernel/passthrough_kernel_placed.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

import aie.utils.trace as trace_utils


def my_passthrough_kernel(dev, in1_size, out_size, trace_size):
    in1_dtype = np.uint8
    out_dtype = np.uint8

    # Follow iron design: chunked processing with 2 cores
    N = in1_size // in1_dtype(0).nbytes
    n_cores = 2
    chunk_size = 1024  # Elements processed per iteration in memtile
    elements_per_core_per_chunk = chunk_size // n_cores  # 512 elements per core per chunk
    num_iterations = 8  # Fixed iterations to match iron design
    repeat_counter = 2  # Add repeat counter for additional BD repeat layer
    
    # The output size is the full buffer size (already accounts for repeat)
    N_out = out_size // out_dtype(0).nbytes  # Output elements including repeat
    
    # The base output size should equal input size when divided by repeat_counter
    base_out_size = out_size // repeat_counter

    @device(dev)
    def device_body():
    # define types following iron design
        full_vector_ty = np.ndarray[(N,), np.dtype[in1_dtype]]  # Full input vector
        full_output_vector_ty = np.ndarray[(N_out,), np.dtype[out_dtype]]  # Full output vector (with repeat)
        chunk_ty = np.ndarray[(chunk_size,), np.dtype[in1_dtype]]  # 1024 elements in memtile
        core_chunk_ty = np.ndarray[(elements_per_core_per_chunk,), np.dtype[in1_dtype]]  # 512 elements per core

        # AIE Core Function declarations
        passThroughLine = external_func(
            "passThroughLine", inputs=[core_chunk_ty, core_chunk_ty, np.int32]
        )

        # Tile declarations for 2 cores
        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        ComputeTile2 = tile(0, 2)
        ComputeTile3 = tile(0, 3)
        compute_tiles = [ComputeTile2, ComputeTile3]

        # Set up a packet-switched flow from core to shim for tracing information
        tiles_to_trace = compute_tiles + [MemTile, ShimTile]
        # Always configure packet tracing flow like iron version does
        trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)

        # AIE-array data movement with object fifos - match iron design exactly
        # In iron version, BD chaining only applies to split/join fifos, not input/output
        
        # Input: Shim -> MemTile (no BD chaining, transfers full vector)
        of_in = object_fifo("in", ShimTile, MemTile, 2, chunk_ty)
        
        # Split: MemTile distributes to compute tiles (512 elements each core)
        # Use repeat_count instead of bd_chain_iter_count to see repeat effect
        of_split = []
        for i in range(n_cores):
            split_fifo = object_fifo(f"split_{i}", MemTile, compute_tiles[i], 2, core_chunk_ty)
            split_fifo.set_repeat_count(repeat_counter)
            of_split.append(split_fifo)
        
        # Split link with offsets - each core gets part of the chunks
        split_offsets = [i * elements_per_core_per_chunk for i in range(n_cores)]  # [0, 512]
        object_fifo_link([of_in], of_split, [], split_offsets)
            
        # Join: Collect results from compute tiles to MemTile
        # Use repeat_count instead of bd_chain_iter_count to see repeat effect
        of_join = []
        for i in range(n_cores):
            join_fifo = object_fifo(f"join_{i}", compute_tiles[i], MemTile, 2, core_chunk_ty)
            join_fifo.set_repeat_count(repeat_counter)
            of_join.append(join_fifo)
        
        # Output: MemTile -> Shim (no BD chaining, transfers full vector)
        of_out = object_fifo("out", MemTile, ShimTile, 2, chunk_ty)
        
        # Join link with offsets - cores write to different parts of the output chunks
        join_offsets = [i * elements_per_core_per_chunk for i in range(n_cores)]  # [0, 512]
        object_fifo_link(of_join, [of_out], join_offsets, [])

        # Set up compute tiles using a loop
        for i, compute_tile in enumerate(compute_tiles):
            def make_core_fn(idx):
                @core(compute_tile, "passThrough.cc.o")
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
            if trace_size > 0:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    shim=ShimTile,
                    trace_size=trace_size,
                )

            # Input: Transfer full input buffer
            in_task = shim_dma_single_bd_task(
                of_in, inTensor, sizes=[1, 1, 1, N]
            )
            dma_start_task(in_task)

            # Output: Transfer repeated output buffer (repeat_counter times larger)
            out_task = shim_dma_single_bd_task(
                of_out, outTensor, sizes=[1, 1, 1, N_out], issue_token=True
            )
            dma_start_task(out_task)
            dma_await_task(out_task)
            dma_free_task(in_task)

            trace_utils.gen_trace_done_aie2(ShimTile)


if len(sys.argv) < 4:
    raise ValueError("[ERROR] Need at least 4 arguments (dev, in1_size, out_size)")


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
    dev = AIEDevice.npu1_2col
elif opts.device == "npu2":
    dev = AIEDevice.npu2
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(opts.device))
in1_size = int(opts.in1_size)
if in1_size % 4 != 0 or in1_size < 512:
    print(
        "In1 buffer size ("
        + str(in1_size)
        + ") must be a multiple of 4 and greater than or equal to 512"
    )
    raise ValueError
out_size = int(opts.out_size)
trace_size = int(opts.trace_size)

with mlir_mod_ctx() as ctx:
    my_passthrough_kernel(dev, in1_size, out_size, trace_size)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
