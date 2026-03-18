# aie_trace.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
# ===-----------------------------------------------------------------------===#
#
# Python example using the declarative trace API:
# - configure_trace() outside runtime_sequence to set up trace ops
# - start_trace() inside runtime_sequence to activate tracing
#
# Usage:
#   python3 aie_trace.py > aie_trace_from_py.mlir
#
# ===-----------------------------------------------------------------------===#

import sys
import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_

import aie.utils.trace as trace_utils
from aie.utils.trace.events import (
    PortEvent,
    MemTilePortEvent,
    CoreEvent,
    MemEvent,
    ShimTileEvent,
    MemTileEvent,
)


def build_aie_trace():
    tensor_size = 4096
    tile_size = 1024
    num_sub_vectors = 4

    @device(AIEDevice.npu1_1col)
    def device_body():
        tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
        scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]
        tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int32]]

        # External kernel function declaration
        scale = external_func(
            "vector_scalar_mul_aie_scalar",
            inputs=[tile_ty, tile_ty, scalar_ty, np.int32],
            link_with="scale.o",
        )

        # Tile declarations
        shim_noc_tile_0_0 = tile(0, 0)
        mem_tile_0_1 = tile(0, 1)
        tile_0_2 = tile(0, 2)

        # ObjectFIFOs for data movement through memtile
        of_in = object_fifo("in", shim_noc_tile_0_0, mem_tile_0_1, 2, tile_ty)
        of_in_fwd = object_fifo("in_fwd", mem_tile_0_1, tile_0_2, 2, tile_ty)
        object_fifo_link(of_in, of_in_fwd)

        of_factor = object_fifo(
            "infactor", shim_noc_tile_0_0, mem_tile_0_1, 2, scalar_ty
        )
        of_factor_fwd = object_fifo(
            "infactor_fwd", mem_tile_0_1, tile_0_2, 2, scalar_ty
        )
        object_fifo_link(of_factor, of_factor_fwd)

        of_out = object_fifo("out", tile_0_2, mem_tile_0_1, 2, tile_ty)
        of_out_fwd = object_fifo("out_fwd", mem_tile_0_1, shim_noc_tile_0_0, 2, tile_ty)
        object_fifo_link(of_out, of_out_fwd)

        # Core computation
        @core(tile_0_2)
        def core_body():
            for _ in range_(sys.maxsize):
                elem_factor = of_factor_fwd.acquire(ObjectFifoPort.Consume, 1)
                for _ in range_(num_sub_vectors):
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    elem_in = of_in_fwd.acquire(ObjectFifoPort.Consume, 1)
                    scale(elem_in, elem_out, elem_factor, tile_size)
                    of_in_fwd.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)
                of_factor_fwd.release(ObjectFifoPort.Consume, 1)

        # ==================================================================
        # TRACE CONFIGURATION
        # ==================================================================

        # List tiles to trace. Listing the same core tile twice enables
        # tracing both its core and memory.
        tiles_to_trace = [tile_0_2, tile_0_2, mem_tile_0_1, shim_noc_tile_0_0]

        # Event settings are optional; defaults are used if not specified.
        trace_utils.configure_trace(
            tiles_to_trace,
            coretile_events=[
                CoreEvent.INSTR_EVENT_0,
                CoreEvent.INSTR_EVENT_1,
                CoreEvent.INSTR_VECTOR,
                CoreEvent.MEMORY_STALL,
                CoreEvent.STREAM_STALL,
                CoreEvent.LOCK_STALL,
                PortEvent(CoreEvent.PORT_RUNNING_0, WireBundle.DMA, 0, True),
                PortEvent(CoreEvent.PORT_RUNNING_1, WireBundle.DMA, 0, False),
            ],
            coremem_events=[
                MemEvent.DMA_S2MM_0_START_TASK,
                MemEvent.DMA_S2MM_1_START_TASK,
                MemEvent.DMA_MM2S_0_START_TASK,
                MemEvent.DMA_S2MM_0_FINISHED_TASK,
                MemEvent.DMA_S2MM_1_FINISHED_TASK,
                MemEvent.DMA_MM2S_0_FINISHED_TASK,
                MemEvent.DMA_S2MM_0_STREAM_STARVATION,
                MemEvent.DMA_S2MM_1_STREAM_STARVATION,
            ],
            memtile_events=[
                MemTilePortEvent(MemTileEvent.PORT_RUNNING_0, WireBundle.DMA, 0, False),
                MemTilePortEvent(MemTileEvent.PORT_RUNNING_1, WireBundle.DMA, 1, False),
                MemTilePortEvent(MemTileEvent.PORT_RUNNING_2, WireBundle.DMA, 0, True),
                MemTilePortEvent(MemTileEvent.PORT_RUNNING_3, WireBundle.DMA, 1, True),
                MemTilePortEvent(MemTileEvent.PORT_RUNNING_4, WireBundle.DMA, 2, True),
                MemTilePortEvent(MemTileEvent.PORT_RUNNING_5, WireBundle.DMA, 3, True),
                MemTilePortEvent(MemTileEvent.PORT_RUNNING_6, WireBundle.DMA, 4, True),
                MemTilePortEvent(MemTileEvent.PORT_RUNNING_7, WireBundle.DMA, 5, True),
            ],
            shimtile_events=[
                ShimTileEvent.DMA_S2MM_0_START_TASK,
                ShimTileEvent.DMA_S2MM_1_START_TASK,
                ShimTileEvent.DMA_MM2S_0_START_TASK,
                ShimTileEvent.DMA_S2MM_0_FINISHED_TASK,
                ShimTileEvent.DMA_S2MM_1_FINISHED_TASK,
                ShimTileEvent.DMA_MM2S_0_FINISHED_TASK,
                ShimTileEvent.DMA_S2MM_0_STREAM_STARVATION,
                ShimTileEvent.DMA_S2MM_1_STREAM_STARVATION,
            ],
        )

        # ==================================================================
        # RUNTIME SEQUENCE WITH TRACE ACTIVATION
        # ==================================================================

        @runtime_sequence(tensor_ty, scalar_ty, tensor_ty)
        def sequence(A, F, C):
            # Start trace configuration
            trace_utils.start_trace()

            # Configure DMA tasks for input, factor, and output
            in_task = shim_dma_single_bd_task(
                of_in, A, sizes=[1, 1, 1, tensor_size], issue_token=True
            )
            factor_task = shim_dma_single_bd_task(
                of_factor, F, sizes=[1, 1, 1, 1], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                of_out_fwd, C, sizes=[1, 1, 1, tensor_size], issue_token=True
            )

            dma_start_task(in_task, factor_task, out_task)
            dma_await_task(in_task, factor_task, out_task)


with mlir_mod_ctx() as ctx:
    build_aie_trace()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
