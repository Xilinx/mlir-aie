# 03_time_multiplex_phases/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
#
# Prototype 3: Time-Multiplexed Multi-Phase Pipeline
#
# Demonstrates reusing shimDMA channels across sequential phases:
#   Phase 1: Stream data from DDR -> MemTile -> split to 4 cores (process)
#   Phase 2: Gather results from 4 cores -> MemTile -> join -> DDR
#
# Both phases reuse the same 2 shimDMA channels (1 MM2S + 1 S2MM).
# The runtime_sequence orchestrates the phases using dma_wait() as barriers.
#
# This extends to multi-iteration patterns where the host issues multiple
# batches through the same hardware without reconfiguring the array.
#
# ShimDMA channels used: 1 MM2S + 1 S2MM (reused across iterations)

import numpy as np
import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_


def time_multiplex_phases(dev, n_cores=4, chunk_size=256, n_iterations=4):
    """
    Demonstrates time-multiplexed data movement: send n_iterations batches
    of data through the same shimDMA channels, where each batch is split
    to n_cores, processed, and joined back.
    """
    batch_size = n_cores * chunk_size  # 1024 bytes per batch
    total_in_size = n_iterations * batch_size  # 4096 bytes total input
    total_out_size = n_iterations * batch_size  # 4096 bytes total output
    dtype = np.dtype[np.uint8]

    @device(dev)
    def device_body():
        # Types
        batch_ty = np.ndarray[(batch_size,), dtype]
        chunk_ty = np.ndarray[(chunk_size,), dtype]
        full_in_ty = np.ndarray[(total_in_size,), dtype]
        full_out_ty = np.ndarray[(total_out_size,), dtype]

        # Passthrough kernel
        passthrough_fn = external_func(
            "passThroughLine",
            [chunk_ty, chunk_ty, np.int32],
            link_with="passThrough.cc.o",
        )

        # Tiles
        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        cores = [tile(0, 2 + i) for i in range(n_cores)]

        # --- SPLIT: ShimTile -> MemTile -> individual cores ---
        of_in = object_fifo("in", ShimTile, MemTile, 2, batch_ty)
        of_split = []
        for i in range(n_cores):
            sf = object_fifo(f"split_{i}", MemTile, cores[i], 2, chunk_ty)
            of_split.append(sf)
        split_offsets = [i * chunk_size for i in range(n_cores)]
        object_fifo_link(of_in, of_split, [], split_offsets)

        # --- JOIN: cores -> MemTile -> ShimTile ---
        of_join = []
        for i in range(n_cores):
            jf = object_fifo(f"join_{i}", cores[i], MemTile, 2, chunk_ty)
            of_join.append(jf)
        of_out = object_fifo("out", MemTile, ShimTile, 2, batch_ty)
        join_offsets = [i * chunk_size for i in range(n_cores)]
        object_fifo_link(of_join, of_out, join_offsets, [])

        # --- Core logic: passthrough ---
        for i in range(n_cores):

            def make_core_fn(idx):
                @core(cores[idx])
                def core_body():
                    for _ in range_(sys.maxsize):
                        elem_out = of_join[idx].acquire(
                            ObjectFifoPort.Produce, 1
                        )
                        elem_in = of_split[idx].acquire(
                            ObjectFifoPort.Consume, 1
                        )
                        passthrough_fn(elem_in, elem_out, chunk_size)
                        of_split[idx].release(ObjectFifoPort.Consume, 1)
                        of_join[idx].release(ObjectFifoPort.Produce, 1)

            make_core_fn(i)

        # --- Runtime sequence: time-multiplexed iterations ---
        @runtime_sequence(full_in_ty, full_out_ty)
        def sequence(inTensor, outTensor):
            # Process n_iterations batches, reusing the same shimDMA channels.
            # Each iteration:
            #   1. Send 1 batch (1024 bytes) from DDR to array
            #   2. Wait for results
            #   3. Move to next batch offset
            for it in range(n_iterations):
                in_offset = it * batch_size
                out_offset = it * batch_size

                # Phase 1: Send batch to array
                npu_dma_memcpy_nd(
                    metadata=of_in,
                    bd_id=0,
                    mem=inTensor,
                    offsets=[0, 0, 0, in_offset],
                    sizes=[1, 1, 1, batch_size],
                )
                # Phase 2: Receive results
                npu_dma_memcpy_nd(
                    metadata=of_out,
                    bd_id=1,
                    mem=outTensor,
                    offsets=[0, 0, 0, out_offset],
                    sizes=[1, 1, 1, batch_size],
                )
                # Barrier: wait for this iteration to complete before next
                dma_wait(of_out)


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=False, default="npu", dest="device")
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = AIEDevice.npu1_1col
elif opts.device == "npu2":
    dev = AIEDevice.npu2
else:
    raise ValueError(f"[ERROR] Unknown device: {opts.device}")

with mlir_mod_ctx() as ctx:
    time_multiplex_phases(dev)
    res = ctx.module.operation.verify()
    if res is True:
        print(ctx.module)
    else:
        print(res)
