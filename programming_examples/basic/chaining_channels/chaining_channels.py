# chaining_channels/chaining_channels.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Chaining channels — low-level placed IRON, two compile modes.

The design body intentionally uses low-level IRON (explicit ``flow``,
``@mem`` / ``@memtile_dma``, manual ``npu_writebd`` /
``npu_address_patch`` / ``npu_push_queue`` / ``npu_sync``) — that's the
lesson here.  This file just wraps the design with a small ``main()``
that supports both:

  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH`` — drives
    ``aie.utils.compile.compile_mlir_module`` directly so the Makefile
    matches the @iron.jit ports' shape.
  * emit-MLIR:    ``... --emit-mlir`` — prints the MLIR module to stdout
    (legacy aiecc-on-a-file path).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_
from aie.utils.compile import compile_mlir_module
import aie.utils.trace as trace_utils
from aie.utils.trace.events import (
    MemTileEvent,
    ShimTileEvent,
    MemTilePortEvent,
    ShimTilePortEvent,
    WireBundle,
)


def _build_module(N: int, dev, col: int, enable_trace: int, trace_size: int):
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            # Convert N from bytes to int32 elements
            n_elements = N // 4
            vector_ty = np.ndarray[(n_elements,), np.dtype[np.int32]]

            # Read buffer is 4x larger (4KB)
            n_elements_read = (N * 4) // 4
            vector_ty_read = np.ndarray[(n_elements_read,), np.dtype[np.int32]]

            # Tile declarations
            ShimTile = tile(col, 0)
            MemTile = tile(col, 1)
            ComputeTile2 = tile(col, 2)

            # MemTile buffer and locks for initialized data
            memtile_buff = buffer(
                tile=MemTile,
                datatype=vector_ty,
                name="memtile_buff",
                initial_value=np.arange(1, n_elements + 1, dtype=np.int32),
            )
            # Lock ID 0 is located at address 0xC0000 on NPU2
            memtile_lock = lock(MemTile, lock_id=0, init=0, sym_name="memtile_lock")

            # ComputeTile buffer and locks for read data
            compute_buff = buffer(
                tile=ComputeTile2, datatype=vector_ty_read, name="compute_buff"
            )
            compute_prod_lock = lock(
                ComputeTile2, lock_id=0, init=1, sym_name="compute_prod_lock"
            )
            compute_cons_lock = lock(
                ComputeTile2, lock_id=1, init=0, sym_name="compute_cons_lock"
            )

            # Flow from MemTile to ShimTile for write path
            flow(MemTile, WireBundle.DMA, 0, ShimTile, WireBundle.DMA, 0)

            # Flow from ShimTile to ComputeTile for read path
            flow(ShimTile, WireBundle.DMA, 0, ComputeTile2, WireBundle.DMA, 0)

            # MemTile DMA logic - send initialized data to DDR when triggered
            @memtile_dma(MemTile)
            def memtile_dma_block(block):
                s0 = dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2])
                with block[1]:
                    use_lock(memtile_lock, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(memtile_buff, offset=0, len=n_elements)
                    use_lock(memtile_lock, LockAction.Release, value=0)
                    next_bd(block[1])
                with block[2]:
                    EndOp()

            # ComputeTile DMA logic
            @mem(ComputeTile2)
            def compute_dma_block(block):
                s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
                with block[1]:
                    use_lock(compute_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(compute_buff, offset=0, len=n_elements_read)
                    use_lock(compute_cons_lock, LockAction.Release, value=1)
                    next_bd(block[1])
                with block[2]:
                    EndOp()

            # Compute tile core - toggle locks, discard data
            @core(ComputeTile2)
            def core_body():
                for _ in range_(sys.maxsize):
                    use_lock(compute_cons_lock, LockAction.AcquireGreaterEqual, value=1)
                    # Do nothing with the data, just toggle locks
                    use_lock(compute_prod_lock, LockAction.Release, value=1)

            # Configure tracing on ShimTile (if enabled)
            if enable_trace:
                tiles_to_trace = [ShimTile, MemTile, ComputeTile2]
                trace_utils.configure_trace(
                    tiles_to_trace,
                    memtile_events=[
                        MemTileEvent.LOCK_SEL0_ACQ_GE,
                        MemTilePortEvent(
                            MemTileEvent.PORT_RUNNING_0, WireBundle.South, 3, True
                        ),
                        MemTilePortEvent(
                            MemTileEvent.PORT_TLAST_0, WireBundle.South, 3, True
                        ),
                        MemTileEvent.DMA_MM2S_SEL0_STREAM_BACKPRESSURE,
                        MemTileEvent.DMA_MM2S_SEL0_STALLED_LOCK,
                        MemTileEvent.DMA_MM2S_SEL0_START_TASK,
                        MemTileEvent.DMA_MM2S_SEL0_FINISHED_TASK,
                        MemTileEvent.DMA_MM2S_SEL0_FINISHED_BD,
                    ],
                    shimtile_events=[
                        ShimTileEvent.DMA_S2MM_0_START_TASK,
                        ShimTileEvent.DMA_S2MM_0_FINISHED_TASK,
                        ShimTileEvent.DMA_MM2S_0_START_TASK,
                        ShimTileEvent.DMA_MM2S_0_FINISHED_TASK,
                        ShimTileEvent.DMA_MM2S_0_STALLED_LOCK,
                        ShimTileEvent.DMA_MM2S_0_MEMORY_STARVATION,
                        ShimTilePortEvent(
                            ShimTileEvent.PORT_RUNNING_0, WireBundle.South, 2, True
                        ),
                        ShimTilePortEvent(
                            ShimTileEvent.PORT_RUNNING_1, WireBundle.South, 3, False
                        ),
                    ],
                )

            # Runtime sequence
            @runtime_sequence(vector_ty, vector_ty_read)
            def sequence(A, B):
                # Setup trace buffer (if enabled)
                if enable_trace:
                    trace_utils.start_trace(trace_size=trace_size)

                # Release MemTile lock to trigger DMA
                npu_write32(column=col, row=1, address=0xC0000, value=1)

                # Write BD for S2MM channel 0 (MemTile -> DDR, buffer A)
                npu_writebd(
                    bd_id=0,
                    buffer_length=n_elements,
                    buffer_offset=0,
                    column=col,
                    row=0,
                    enable_packet=0,
                    out_of_order_id=0,
                    packet_id=0,
                    packet_type=0,
                    d0_size=0,
                    d0_stride=0,
                    d0_zero_before=0,
                    d0_zero_after=0,
                    d1_size=0,
                    d1_stride=0,
                    d1_zero_before=0,
                    d1_zero_after=0,
                    d2_size=0,
                    d2_stride=0,
                    d2_zero_before=0,
                    d2_zero_after=0,
                    iteration_current=0,
                    iteration_size=0,
                    iteration_stride=0,
                    lock_acq_enable=1,
                    lock_acq_id=0,
                    lock_acq_val=0,
                    lock_rel_id=1,
                    lock_rel_val=1,
                    next_bd=0,
                    use_next_bd=0,
                    valid_bd=1,
                )
                npu_address_patch(addr=0x1D004, arg_idx=0, arg_plus=0)

                # Write BD for MM2S channel 0 (DDR -> ComputeTile, buffer B)
                npu_writebd(
                    bd_id=1,
                    buffer_length=n_elements_read,
                    buffer_offset=0,
                    column=col,
                    row=0,
                    enable_packet=0,
                    out_of_order_id=0,
                    packet_id=0,
                    packet_type=0,
                    d0_size=0,
                    d0_stride=0,
                    d0_zero_before=0,
                    d0_zero_after=0,
                    d1_size=0,
                    d1_stride=0,
                    d1_zero_before=0,
                    d1_zero_after=0,
                    d2_size=0,
                    d2_stride=0,
                    d2_zero_before=0,
                    d2_zero_after=0,
                    iteration_current=0,
                    iteration_size=0,
                    iteration_stride=0,
                    lock_acq_enable=1,
                    lock_acq_id=1,
                    lock_acq_val=1,
                    lock_rel_id=0,
                    lock_rel_val=0,
                    next_bd=0,
                    use_next_bd=0,
                    valid_bd=1,
                )
                npu_address_patch(addr=0x1D024, arg_idx=1, arg_plus=0)

                # Push BD 0 to S2MM channel 0 queue
                npu_push_queue(
                    column=col,
                    row=0,
                    direction=0,
                    channel=0,
                    bd_id=0,
                    issue_token=False,
                    repeat_count=0,
                )
                # Wait for S2MM channel 0
                # npu_sync(column=col, row=0, direction=0, channel=0, column_num=1, row_num=1)

                # Push BD 1 to MM2S channel 0 queue
                npu_push_queue(
                    column=col,
                    row=0,
                    direction=1,
                    channel=0,
                    bd_id=1,
                    issue_token=True,
                    repeat_count=0,
                )
                # Wait for MM2S channel 0
                npu_sync(
                    column=col, row=0, direction=1, channel=0, column_num=1, row_num=1
                )

    return ctx.module


def _device_for(dev_str: str):
    if dev_str == "npu2":
        return AIEDevice.npu2_1col
    raise ValueError(f"[ERROR] Device name {dev_str!r} is unknown (NPU2 only)")


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Chaining Channels")
    p.add_argument("-d", "--dev", type=str, choices=["npu2"], default="npu2")
    p.add_argument("-n", "--length", type=int, default=1024, help="bytes (>=4)")
    p.add_argument("-c", "--col", type=int, default=0)
    p.add_argument("-t", "--trace", type=int, default=0, help="0 disables tracing")
    p.add_argument("--trace-size", type=int, default=16384)
    p.add_argument(
        "--emit-mlir",
        action="store_true",
        help="print the resolved MLIR module to stdout (legacy aiecc-on-a-file path)",
    )
    p.add_argument("--xclbin-path", type=str, default=None)
    p.add_argument("--insts-path", type=str, default=None)
    return p


def main():
    opts = _make_argparser().parse_args()
    if opts.length % 4 != 0 or opts.length < 4:
        sys.exit(f"--length ({opts.length}) must be a positive multiple of 4")
    module = _build_module(
        N=opts.length,
        dev=_device_for(opts.dev),
        col=opts.col,
        enable_trace=opts.trace,
        trace_size=opts.trace_size,
    )
    if opts.emit_mlir:
        print(module)
        return
    if opts.xclbin_path:
        if not opts.insts_path:
            sys.exit("--xclbin-path requires --insts-path (must be set together)")
        compile_mlir_module(
            mlir_module=str(module),
            xclbin_path=opts.xclbin_path,
            insts_path=opts.insts_path,
            work_dir=str(Path(opts.xclbin_path).resolve().parent),
        )
        return
    # No mode selected: print MLIR (default, matches the original behavior).
    print(module)


if __name__ == "__main__":
    main()
