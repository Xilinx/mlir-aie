# 04_bd_chain_streaming/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
#
# Prototype 4: BD-Chained Streaming Without Host Intervention
#
# Demonstrates using ObjectFIFO iter_count and repeat_count to stream
# multiple data tiles autonomously. The shimDMA issues a single BD chain
# that iterates automatically, distributing data to cores via MemTile
# split with repeat, and gathering results via join.
#
# Data flow:
#   DDR -[single BD chain, 8 iterations]-> MemTile -[split+repeat]-> 2 cores
#   2 cores -[join]-> MemTile -[single BD chain]-> DDR
#
# The iter_count on ObjectFIFOs controls how many times the BD chain
# iterates on the MemTile side. repeat_count on split FIFOs causes
# each chunk to be sent to each core multiple times.
#
# Key win: host issues 1 DMA command, hardware streams 8 tiles autonomously.
# ShimDMA channels: 1 MM2S + 1 S2MM

import numpy as np
import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_


def bd_chain_streaming(dev, n_cores=2, chunk_size=512, n_iterations=8):
    """
    Stream n_iterations chunks through MemTile split/join using iter_count.
    Each chunk is split to n_cores, processed, and joined back.
    The repeat_count=2 on split FIFOs causes each chunk to be sent twice
    (demonstrating autonomous data repetition).
    """
    repeat_count = 2
    elements_per_core = chunk_size // n_cores
    total_in_size = n_iterations * chunk_size
    total_out_size = n_iterations * chunk_size * repeat_count
    dtype = np.dtype[np.uint8]

    @device(dev)
    def device_body():
        full_in_ty = np.ndarray[(total_in_size,), dtype]
        full_out_ty = np.ndarray[(total_out_size,), dtype]
        chunk_ty = np.ndarray[(chunk_size,), dtype]
        core_chunk_ty = np.ndarray[(elements_per_core,), dtype]

        passthrough_fn = external_func(
            "passThroughLine",
            [core_chunk_ty, core_chunk_ty, np.int32],
            link_with="passThrough.cc.o",
        )

        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        cores = [tile(0, 2 + i) for i in range(n_cores)]

        # --- Input: ShimTile -> MemTile with iter_count ---
        of_in = object_fifo(
            "in", ShimTile, MemTile, 2, chunk_ty,
            iter_count=n_iterations,
        )

        # --- Split: MemTile -> cores with repeat_count ---
        of_split = []
        for i in range(n_cores):
            sf = object_fifo(
                f"split_{i}", MemTile, cores[i], 2, core_chunk_ty,
                iter_count=n_iterations,
            )
            sf.set_repeat_count(repeat_count)
            of_split.append(sf)
        split_offsets = [i * elements_per_core for i in range(n_cores)]
        object_fifo_link(of_in, of_split, [], split_offsets)

        # --- Join: cores -> MemTile -> ShimTile ---
        out_iterations = n_iterations * repeat_count
        of_join = []
        for i in range(n_cores):
            jf = object_fifo(
                f"join_{i}", cores[i], MemTile, 2, core_chunk_ty,
                iter_count=out_iterations,
            )
            of_join.append(jf)

        of_out = object_fifo(
            "out", MemTile, ShimTile, 2, chunk_ty,
            iter_count=out_iterations,
        )
        join_offsets = [i * elements_per_core for i in range(n_cores)]
        object_fifo_link(of_join, of_out, join_offsets, [])

        # --- Core logic ---
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
                        passthrough_fn(
                            elem_in, elem_out, elements_per_core
                        )
                        of_split[idx].release(ObjectFifoPort.Consume, 1)
                        of_join[idx].release(ObjectFifoPort.Produce, 1)

            make_core_fn(i)

        # --- Runtime sequence: single DMA command, hardware iterates ---
        @runtime_sequence(full_in_ty, full_out_ty, full_in_ty)
        def sequence(inTensor, outTensor, notUsed):
            in_task = shim_dma_single_bd_task(
                of_in, inTensor, sizes=[1, 1, 1, total_in_size]
            )
            dma_start_task(in_task)

            out_task = shim_dma_single_bd_task(
                of_out, outTensor,
                sizes=[1, 1, 1, total_out_size],
                issue_token=True,
            )
            dma_start_task(out_task)
            dma_await_task(out_task)
            dma_free_task(in_task)


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
    bd_chain_streaming(dev)
    res = ctx.module.operation.verify()
    if res is True:
        print(ctx.module)
    else:
        print(res)
