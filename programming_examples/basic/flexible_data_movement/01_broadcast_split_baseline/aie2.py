# 01_broadcast_split_baseline/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
#
# Prototype 1: ObjectFIFO Broadcast + Split Baseline
#
# Demonstrates two fundamental data distribution patterns through MemTile:
#
#   SPLIT path (different data to each core):
#     DDR -[shimDMA MM2S ch0]-> MemTile -[split via object_fifo_link]-> 4 cores
#     Each core gets a unique 256-byte chunk of 1024-byte input.
#     Each core adds its core_id (1,2,3,4) to differentiate outputs.
#
#   JOIN path (gather results):
#     4 cores -[object_fifo_link join]-> MemTile -[shimDMA S2MM ch0]-> DDR
#
# ShimDMA channels used: 1 MM2S (input) + 1 S2MM (output) = 2 of 4 total
# MemTile channels used: 1 S2MM (from shim) + 4 MM2S (to cores) +
#                         4 S2MM (from cores) + 1 MM2S (to shim) = 10 of 12 total

import numpy as np
import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_


def broadcast_split_baseline(dev, n_cores=4, chunk_size=256):
    """
    Split: 1 shimDMA -> MemTile -> split to n_cores (each gets chunk_size bytes).
    Join:  n_cores -> MemTile -> join -> 1 shimDMA back to DDR.
    """
    total_size = n_cores * chunk_size  # 1024 bytes total
    dtype = np.dtype[np.uint8]

    @device(dev)
    def device_body():
        # Types
        full_ty = np.ndarray[(total_size,), dtype]
        chunk_ty = np.ndarray[(chunk_size,), dtype]

        # External kernel: passthrough (copy input to output)
        add_func = external_func(
            "passThroughLine",
            [chunk_ty, chunk_ty, np.int32],
            link_with="passThrough.cc.o",
        )

        # Tiles
        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        cores = [tile(0, 2 + i) for i in range(n_cores)]

        # --- SPLIT: ShimTile -> MemTile -> individual cores ---
        of_in = object_fifo("in", ShimTile, MemTile, 2, full_ty)

        of_split = []
        for i in range(n_cores):
            split_fifo = object_fifo(
                f"split_{i}", MemTile, cores[i], 2, chunk_ty
            )
            of_split.append(split_fifo)

        # Link with offsets: core i gets bytes [i*chunk_size, (i+1)*chunk_size)
        split_offsets = [i * chunk_size for i in range(n_cores)]
        object_fifo_link(of_in, of_split, [], split_offsets)

        # --- JOIN: cores -> MemTile -> ShimTile ---
        of_join = []
        for i in range(n_cores):
            join_fifo = object_fifo(
                f"join_{i}", cores[i], MemTile, 2, chunk_ty
            )
            of_join.append(join_fifo)

        of_out = object_fifo("out", MemTile, ShimTile, 2, full_ty)

        join_offsets = [i * chunk_size for i in range(n_cores)]
        object_fifo_link(of_join, of_out, join_offsets, [])

        # --- Core logic: passthrough (copy input to output) ---
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
                        add_func(elem_in, elem_out, chunk_size)
                        of_split[idx].release(ObjectFifoPort.Consume, 1)
                        of_join[idx].release(ObjectFifoPort.Produce, 1)

            make_core_fn(i)

        # --- Runtime sequence: move data to/from DDR ---
        @runtime_sequence(full_ty, full_ty)
        def sequence(inTensor, outTensor):
            npu_dma_memcpy_nd(
                metadata=of_in,
                bd_id=0,
                mem=inTensor,
                sizes=[1, 1, 1, total_size],
            )
            npu_dma_memcpy_nd(
                metadata=of_out,
                bd_id=1,
                mem=outTensor,
                sizes=[1, 1, 1, total_size],
            )
            # Output completes after input, so just wait on output
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
    broadcast_split_baseline(dev)
    res = ctx.module.operation.verify()
    if res is True:
        print(ctx.module)
    else:
        print(res)
