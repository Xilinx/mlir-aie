# 05_multi_column_bandwidth/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
#
# Prototype 5: Multi-Column Bandwidth Scaling
#
# Demonstrates using 2 columns to double aggregate shimDMA bandwidth.
# Each column has its own Shim Tile with 2 MM2S + 2 S2MM channels.
#
# Data flow:
#   Column 0: DDR -[Shim(0,0)]-> MemTile(0,1) -> split -> Core(0,2), Core(0,3)
#   Column 1: DDR -[Shim(1,0)]-> MemTile(1,1) -> split -> Core(1,2), Core(1,3)
#   Column 0: Core(0,2), Core(0,3) -> join -> MemTile(0,1) -> Shim(0,0) -> DDR
#   Column 1: Core(1,2), Core(1,3) -> join -> MemTile(1,1) -> Shim(1,0) -> DDR
#
# Total: 4 MM2S + 4 S2MM channels (2 per column)
# Total cores: 4 (2 per column)
# Both columns operate in parallel for doubled throughput.

import numpy as np
import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_


def multi_column_bandwidth(dev, n_cols=2, cores_per_col=2, chunk_size=256):
    """
    Use n_cols columns, each with cores_per_col cores, to scale bandwidth.
    """
    col_data_size = cores_per_col * chunk_size  # 512 bytes per column
    total_size = n_cols * col_data_size  # 1024 bytes total
    dtype = np.dtype[np.uint8]

    @device(dev)
    def device_body():
        full_ty = np.ndarray[(total_size,), dtype]
        col_ty = np.ndarray[(col_data_size,), dtype]
        chunk_ty = np.ndarray[(chunk_size,), dtype]

        passthrough_fn = external_func(
            "passThroughLine",
            [chunk_ty, chunk_ty, np.int32],
            link_with="passThrough.cc.o",
        )

        # Per-column tile arrays
        shim_tiles = [tile(col, 0) for col in range(n_cols)]
        mem_tiles = [tile(col, 1) for col in range(n_cols)]
        core_tiles = [
            [tile(col, 2 + r) for r in range(cores_per_col)]
            for col in range(n_cols)
        ]

        # Per-column ObjectFIFOs
        all_of_in = []
        all_of_split = []
        all_of_join = []
        all_of_out = []

        for col in range(n_cols):
            # Input: Shim -> MemTile
            of_in = object_fifo(
                f"in_{col}", shim_tiles[col], mem_tiles[col], 2, col_ty
            )
            all_of_in.append(of_in)

            # Split: MemTile -> cores
            of_split = []
            for r in range(cores_per_col):
                sf = object_fifo(
                    f"split_{col}_{r}",
                    mem_tiles[col],
                    core_tiles[col][r],
                    2,
                    chunk_ty,
                )
                of_split.append(sf)
            split_offsets = [r * chunk_size for r in range(cores_per_col)]
            object_fifo_link(of_in, of_split, [], split_offsets)
            all_of_split.append(of_split)

            # Join: cores -> MemTile -> Shim
            of_join = []
            for r in range(cores_per_col):
                jf = object_fifo(
                    f"join_{col}_{r}",
                    core_tiles[col][r],
                    mem_tiles[col],
                    2,
                    chunk_ty,
                )
                of_join.append(jf)
            of_out = object_fifo(
                f"out_{col}", mem_tiles[col], shim_tiles[col], 2, col_ty
            )
            join_offsets = [r * chunk_size for r in range(cores_per_col)]
            object_fifo_link(of_join, of_out, join_offsets, [])
            all_of_join.append(of_join)
            all_of_out.append(of_out)

            # Core logic per column
            for r in range(cores_per_col):

                def make_core_fn(c, row):
                    @core(core_tiles[c][row])
                    def core_body():
                        for _ in range_(sys.maxsize):
                            elem_out = all_of_join[c][row].acquire(
                                ObjectFifoPort.Produce, 1
                            )
                            elem_in = all_of_split[c][row].acquire(
                                ObjectFifoPort.Consume, 1
                            )
                            passthrough_fn(
                                elem_in, elem_out, chunk_size
                            )
                            all_of_split[c][row].release(
                                ObjectFifoPort.Consume, 1
                            )
                            all_of_join[c][row].release(
                                ObjectFifoPort.Produce, 1
                            )

                make_core_fn(col, r)

        # --- Runtime sequence: both columns in parallel ---
        @runtime_sequence(full_ty, full_ty)
        def sequence(inTensor, outTensor):
            # Issue DMA transfers for all columns simultaneously
            for col in range(n_cols):
                col_offset = col * col_data_size
                npu_dma_memcpy_nd(
                    metadata=all_of_in[col],
                    bd_id=col * 2,
                    mem=inTensor,
                    offsets=[0, 0, 0, col_offset],
                    sizes=[1, 1, 1, col_data_size],
                )
                npu_dma_memcpy_nd(
                    metadata=all_of_out[col],
                    bd_id=col * 2 + 1,
                    mem=outTensor,
                    offsets=[0, 0, 0, col_offset],
                    sizes=[1, 1, 1, col_data_size],
                )
            # Wait for all columns to complete
            for col in range(n_cols):
                dma_wait(all_of_out[col])


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=False, default="npu", dest="device")
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = AIEDevice.npu1_2col
elif opts.device == "npu2":
    dev = AIEDevice.npu2
else:
    raise ValueError(f"[ERROR] Unknown device: {opts.device}")

with mlir_mod_ctx() as ctx:
    multi_column_bandwidth(dev)
    res = ctx.module.operation.verify()
    if res is True:
        print(ctx.module)
    else:
        print(res)
