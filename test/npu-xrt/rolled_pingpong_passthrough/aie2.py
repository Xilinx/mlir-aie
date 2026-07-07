# rolled_pingpong_passthrough/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# REQUIRES: ryzen_ai_npu1, peano
#
# RUN: %python %S/aie2.py > ./aie2.mlir
# RUN: %aiecc %backend_flags --no-aiesim --aie-generate-npu-insts \
# RUN:        --aie-generate-xclbin --no-compile-host \
# RUN:        --xclbin-name=final.xclbin --npu-insts-name=insts.bin ./aie2.mlir
# RUN: %host_clang %S/test.cpp -o test.exe -std=c++17 -Wall \
# RUN:        %xrt_flags %host_link_flags %test_utils_flags
# RUN: %run_on_npu1% ./test.exe

# Rolled ping-pong passthrough.  The runtime sequence issues N_TILES copies of
# a single input tile using a depth-1 ping-pong.  The scf.for carries the
# in-flight task handle as iter_arg; each iteration starts the next BD while the
# previous is in flight, then frees it.
#
# Golden: output[i] == input[i % TILE_LEN]  (same tile repeated N_TILES times).
#
# This validates the full constant-trip ping-pong stack:
#   aie-unroll-runtime-sequence-loops   → unrolls the loop to straight-line BDs
#   aie-assign-runtime-sequence-bd-ids  → colors them by liveness (ids alternate)
#   aie-dma-tasks-to-npu                → push_queue with concrete bd_ids

import sys

import numpy as np
from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.extras.dialects.scf import yield_

DTYPE = np.int32
TILE_LEN = 256  # elements per tile
N_TILES = 8  # number of tiles streamed; must be >= 2 to exercise ping-pong
TOTAL_OUT = TILE_LEN * N_TILES


def design():
    dev = AIEDevice.npu1
    if len(sys.argv) > 1 and sys.argv[1] == "npu2":
        dev = AIEDevice.npu2

    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            tile_ty = np.ndarray[(TILE_LEN,), np.dtype[DTYPE]]
            in_ty = np.ndarray[(TILE_LEN,), np.dtype[DTYPE]]
            out_ty = np.ndarray[(TOTAL_OUT,), np.dtype[DTYPE]]

            shim = tile(0, 0)
            compute = tile(0, 2)

            # depth=2 for double-buffering on the input side
            of_in = object_fifo("of_in", shim, compute, 2, tile_ty)
            of_out = object_fifo("of_out", compute, shim, 2, tile_ty)

            @core(compute)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                    for i in range_(TILE_LEN):
                        elem_out[i] = elem_in[i]
                    of_in.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)

            @runtime_sequence(in_ty, out_ty)
            def sequence(A, B):
                # Prologue: first input tile.
                init_in = dma_configure_task_for(of_in, issue_token=True)
                with bds(init_in) as bd:
                    with bd[0]:
                        shim_dma_bd(A, sizes=[1, 1, 1, TILE_LEN])
                        EndOp()

                # Output task: collect N_TILES output tiles contiguously.
                # The third dim walks the N_TILES tiles with stride TILE_LEN so
                # tile i lands at offset i*TILE_LEN (a size-N stride-0 wrap dim
                # would overwrite the same tile each time).
                out_task = dma_configure_task_for(of_out, issue_token=True)
                with bds(out_task) as bd:
                    with bd[0]:
                        shim_dma_bd(
                            B,
                            sizes=[1, 1, N_TILES, TILE_LEN],
                            strides=[0, 0, TILE_LEN, 1],
                        )
                        EndOp()

                dma_start_task(init_in, out_task)

                # Rolled ping-pong: issue N_TILES-1 more input BDs while the
                # previous is in flight.  The task handle flows as iter_arg.
                # All input BDs read from the same tile (offset=0) — the output
                # collects N_TILES copies of the input tile.
                for _iv, prev, result in range_(
                    1, N_TILES, iter_args=[init_in.result], insert_yield=False
                ):
                    tile_in = dma_configure_task_for(of_in, issue_token=True)
                    with bds(tile_in) as bd:
                        with bd[0]:
                            shim_dma_bd(A, sizes=[1, 1, 1, TILE_LEN])
                            EndOp()
                    dma_start_task(tile_in)
                    dma_free_task(prev)
                    yield_([tile_in.result])

                dma_await_task(result, out_task)
                dma_free_task(result)

    print(ctx.module)


design()
