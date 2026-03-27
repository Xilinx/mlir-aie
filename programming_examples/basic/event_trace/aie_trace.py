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
# Python equivalent of aie_trace.mlir using declarative trace ops:
# - aie.trace for trace configuration
# - aie.trace.event for specifying events to capture
# - aie.trace.start_config in runtime sequence
#
# This will be incrementally lowered by trace passes
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

        # Trace configuration for compute tile (0,2) - core events
        @trace(tile_0_2, "core_trace")
        def core_trace_body():
            trace_mode(TraceMode.EventTime)
            trace_packet(1, TracePacketType.Core)
            trace_event("INSTR_EVENT_0")
            trace_event("INSTR_EVENT_1")
            trace_event("INSTR_VECTOR")
            trace_event("MEMORY_STALL")
            trace_event("STREAM_STALL")
            trace_event("LOCK_STALL")
            trace_event("PORT_RUNNING_0")
            trace_event("PORT_RUNNING_1")
            trace_port(0, WireBundle.DMA, 0, DMAChannelDir.S2MM)
            trace_port(1, WireBundle.DMA, 0, DMAChannelDir.MM2S)
            trace_start(broadcast=15)
            trace_stop(broadcast=14)

        # Trace configuration for compute tile (0,2) - memory events
        @trace(tile_0_2, "mem_trace")
        def mem_trace_body():
            trace_packet(3, TracePacketType.Mem)
            trace_event("DMA_S2MM_0_START_TASK")
            trace_event("DMA_S2MM_1_START_TASK")
            trace_event("DMA_MM2S_0_START_TASK")
            trace_event("DMA_S2MM_0_FINISHED_TASK")
            trace_event("DMA_S2MM_1_FINISHED_TASK")
            trace_event("DMA_MM2S_0_FINISHED_TASK")
            trace_event("DMA_S2MM_0_STREAM_STARVATION")
            trace_event("DMA_S2MM_1_STREAM_STARVATION")
            trace_start(event="BROADCAST_15")
            trace_stop(event="BROADCAST_14")

        # Trace configuration for mem tile (0,1)
        @trace(mem_tile_0_1, "memtile_trace")
        def memtile_trace_body():
            trace_packet(4, TracePacketType.MemTile)
            trace_event("PORT_RUNNING_0")
            trace_event("PORT_RUNNING_1")
            trace_event("PORT_RUNNING_2")
            trace_event("PORT_RUNNING_3")
            trace_event("PORT_RUNNING_4")
            trace_event("PORT_RUNNING_5")
            trace_event("PORT_RUNNING_6")
            trace_event("PORT_RUNNING_7")
            trace_port(0, WireBundle.DMA, 0, DMAChannelDir.MM2S)
            trace_port(1, WireBundle.DMA, 1, DMAChannelDir.MM2S)
            trace_port(2, WireBundle.DMA, 0, DMAChannelDir.S2MM)
            trace_port(3, WireBundle.DMA, 1, DMAChannelDir.S2MM)
            trace_port(4, WireBundle.DMA, 2, DMAChannelDir.S2MM)
            trace_port(5, WireBundle.DMA, 3, DMAChannelDir.S2MM)
            trace_port(6, WireBundle.DMA, 4, DMAChannelDir.S2MM)
            trace_port(7, WireBundle.DMA, 5, DMAChannelDir.S2MM)
            trace_start(broadcast=15)
            trace_stop(broadcast=14)

        # Trace configuration for shim tile (0,0)
        @trace(shim_noc_tile_0_0, "shim_trace")
        def shim_trace_body():
            trace_packet(2, TracePacketType.ShimTile)
            trace_event("DMA_S2MM_0_START_TASK")
            trace_event("DMA_S2MM_1_START_TASK")
            trace_event("DMA_MM2S_0_START_TASK")
            trace_event("DMA_S2MM_0_FINISHED_TASK")
            trace_event("DMA_S2MM_1_FINISHED_TASK")
            trace_event("DMA_MM2S_0_FINISHED_TASK")
            trace_event("DMA_S2MM_0_STREAM_STARVATION")
            trace_event("DMA_S2MM_1_STREAM_STARVATION")
            trace_start(event="TRUE")
            trace_stop(event="NONE")

        # ==================================================================
        # RUNTIME SEQUENCE WITH TRACE ACTIVATION
        # ==================================================================

        @runtime_sequence(tensor_ty, scalar_ty, tensor_ty)
        def sequence(A, F, C):
            # Start trace configuration
            trace_start_config("core_trace")
            trace_start_config("mem_trace")
            trace_start_config("memtile_trace")
            trace_start_config("shim_trace")

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
