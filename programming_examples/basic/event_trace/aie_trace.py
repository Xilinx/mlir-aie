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
# The passes aie-trace-to-config and aie-inline-trace-config will lower this.
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
        )

        # Tile declarations
        shim_noc_tile_0_0 = tile(0, 0)
        tile_0_2 = tile(0, 2)

        # ObjectFIFOs for data movement
        of_in = object_fifo("in", shim_noc_tile_0_0, tile_0_2, 2, tile_ty)
        of_factor = object_fifo("infactor", shim_noc_tile_0_0, tile_0_2, 2, scalar_ty)
        of_out = object_fifo("out", tile_0_2, shim_noc_tile_0_0, 2, tile_ty)

        # Core computation
        @core(tile_0_2, "scale.o")
        def core_body():
            for _ in range_(sys.maxsize):
                elem_factor = of_factor.acquire(ObjectFifoPort.Consume, 1)
                for _ in range_(num_sub_vectors):
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                    scale(elem_in, elem_out, elem_factor, tile_size)
                    of_in.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)
                of_factor.release(ObjectFifoPort.Consume, 1)

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
            trace_event("PORT_RUNNING_1")
            trace_event("PORT_IDLE_1")
            trace_port(0, WireBundle.DMA, 0, DMAChannelDir.S2MM)
            trace_port(1, WireBundle.DMA, 0, DMAChannelDir.MM2S)
            trace_start(event="BROADCAST_15")
            trace_stop(event="BROADCAST_14")

        # Trace configuration for compute tile (0,2) - memory events
        @trace(tile_0_2, "mem_trace")
        def mem_trace_body():
            trace_mode(TraceMode.EventTime)
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

        # Packet flows to route trace data
        packetflow(
            1,
            tile_0_2,
            WireBundle.Trace,
            0,
            {"dest": shim_noc_tile_0_0, "port": WireBundle.DMA, "channel": 1},
            keep_pkt_header=True,
        )
        packetflow(
            3,
            tile_0_2,
            WireBundle.Trace,
            1,
            {"dest": shim_noc_tile_0_0, "port": WireBundle.DMA, "channel": 1},
            keep_pkt_header=True,
        )
        packetflow(
            2,
            shim_noc_tile_0_0,
            WireBundle.Trace,
            0,
            {"dest": shim_noc_tile_0_0, "port": WireBundle.DMA, "channel": 1},
            keep_pkt_header=True,
        )

        # ==================================================================
        # RUNTIME SEQUENCE WITH TRACE ACTIVATION
        # ==================================================================

        @runtime_sequence(tensor_ty, scalar_ty, tensor_ty)
        def sequence(A, F, C):
            # Trace initialization - applied by lowering passes
            trace_start_config("core_trace")
            trace_start_config("mem_trace")
            trace_start_config("shim_trace")

            # Timer_Control (address 0x34000 = 212992)
            npu_write32(column=0, row=2, address=212992, value=31232)

            # Configure trace buffer descriptor
            npu_writebd(
                bd_id=15,
                buffer_length=8192,
                buffer_offset=0,
                burst_length=64,
                column=0,
                d0_size=0,
                d0_stride=0,
                d0_zero_after=0,
                d0_zero_before=0,
                d1_size=0,
                d1_stride=0,
                d1_zero_after=0,
                d1_zero_before=0,
                d2_size=0,
                d2_stride=0,
                d2_zero_after=0,
                d2_zero_before=0,
                enable_packet=1,
                iteration_current=0,
                iteration_size=0,
                iteration_stride=0,
                lock_acq_enable=0,
                lock_acq_id=0,
                lock_acq_val=0,
                lock_rel_id=0,
                lock_rel_val=0,
                next_bd=0,
                out_of_order_id=0,
                packet_id=0,
                packet_type=0,
                row=0,
                use_next_bd=0,
                valid_bd=1,
            )

            # Patch trace buffer address
            npu_address_patch(addr=119268, arg_idx=4, arg_plus=0)

            # Configure DMA channel for trace
            npu_maskwrite32(
                address=119304, column=0, mask=7936, row=0, value=3840
            )
            npu_write32(address=119308, column=0, row=0, value=2147483663)

            # Start trace control
            npu_write32(address=212992, column=0, row=0, value=32512)
            npu_write32(address=213068, column=0, row=0, value=127)
            npu_write32(address=213000, column=0, row=0, value=127)

            # ==============================================================
            # DATA TRANSFER CONFIGURATION
            # ==============================================================

            in_task = shim_dma_single_bd_task(
                of_in, A, sizes=[1, 1, 1, tensor_size], issue_token=True
            )
            factor_task = shim_dma_single_bd_task(
                of_factor, F, sizes=[1, 1, 1, 1], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                of_out, C, sizes=[1, 1, 1, tensor_size], issue_token=True
            )

            dma_start_task(in_task, factor_task, out_task)
            dma_await_task(in_task, factor_task, out_task)

            # ==============================================================
            # TRACE COMPLETION
            # ==============================================================

            npu_write32(address=213064, column=0, row=0, value=126)
            npu_write32(address=213000, column=0, row=0, value=126)


with mlir_mod_ctx() as ctx:
    build_aie_trace()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
