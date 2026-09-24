# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""tile_dma_task builds a mem tile buffer descriptor from inside the runtime
sequence, so its length and access pattern can come from dispatch-time values.
TileDma, its structural peer, is configured once when the device loads and
cannot vary per dispatch -- which is why an operand held resident in a mem
tile had to give up dynamic shapes before this existed."""

import numpy as np

from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.iron import Buffer, Program, Runtime, tile_dma_task
from aie.iron.runtime.runtime import IronRuntimeError
from aie.iron.device import NPU2Col1, Tile


def emit_dynamic_memtile_task():
    buf_ty = np.ndarray[(4096,), np.dtype[np.int32]]
    host_ty = np.ndarray[(4096,), np.dtype[np.int32]]

    mem_tile = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    buf = Buffer(tile=mem_tile, type=buf_ty, name="resident")

    def sequence(_host, tiles):
        # Both the transfer length and the d1 wrap come from the dispatch-time
        # tile count, so the descriptor differs on every call.
        length = tiles * 512
        task = tile_dma_task(
            mem_tile,
            DMAChannelDir.MM2S,
            0,
            buf,
            sizes=[1, 1, tiles, 512],
            strides=[0, 0, 512, 1],
            transfer_len=length,
            wait=True,
            bd_id=0,
        )
        task.await_()

    rt = Runtime(sequence, [host_ty, np.int64])
    # The buffer is reached only from the sequence body, so the Program has to
    # be told about it (a Worker's fn_args or a TileDma would do it otherwise).
    rt.add_buffer(buf)
    return Program(NPU2Col1(), rt).resolve_program()


# The descriptor is configured on the mem tile's own channel, not through a
# shim DMA allocation, and carries runtime sizes and length. The i64 length is
# range-checked before narrowing so truncation cannot turn an invalid value into
# a different valid transfer.
# CHECK: aiex.npu.assert_bd_field(%[[LEN64:.*]]) {max = 2147483647 : i32} : i64
# CHECK-NEXT: %[[LEN32:.*]] = arith.trunci %[[LEN64]] : i64 to i32
# CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
# CHECK: aie.dma_bd(%{{.*}} : memref<4096xi32> offset = 0 len = %[[LEN32]] sizes = [1, 1, %{{.*}}, 512] strides = [0, 0, 512, 1]) {bd_id = 0 : i32}
# CHECK: aiex.dma_start_task
# CHECK: aiex.dma_await_task
print(emit_dynamic_memtile_task())


def emit_late_add_buffer():
    buf_ty = np.ndarray[(4096,), np.dtype[np.int32]]
    mem_tile = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    buf = Buffer(tile=mem_tile, type=buf_ty, name="late")

    def sequence(_host):
        rt.add_buffer(buf)

    rt = Runtime(sequence, [buf_ty])
    try:
        Program(NPU2Col1(), rt).resolve_program()
    except IronRuntimeError as e:
        print(f"RAISED IronRuntimeError: {e}")


# By the time the sequence body runs the Program has resolved its buffers, so a
# registration from there would be dropped; it is rejected instead.
# CHECK: RAISED IronRuntimeError: Cannot register a Buffer after DMA resolution
emit_late_add_buffer()
