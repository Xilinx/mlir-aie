# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""DmaEndpoint.task builds a mem tile buffer descriptor from inside the runtime
sequence, so its length and access pattern can come from dispatch-time values.
TileDma, its structural peer, is configured once when the device loads and
cannot vary per dispatch -- which is why an operand held resident in a mem
tile had to give up dynamic shapes before this existed."""

import numpy as np

from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.iron import (
    Acquire,
    Bd,
    Buffer,
    DmaEndpoint,
    Flow,
    Lock,
    Program,
    Release,
    Runtime,
)
from aie.helpers.taplib import TensorAccessPattern
from aie.iron.device import NPU2Col1, Tile


def emit_dynamic_memtile_task():
    buf_ty = np.ndarray[(4096,), np.dtype[np.int32]]
    host_ty = np.ndarray[(4096,), np.dtype[np.int32]]

    mem_tile = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    buf = Buffer(tile=mem_tile, type=buf_ty, name="resident")

    def sequence(_host, tiles):
        # Both the transfer length and the d1 wrap come from the dispatch-time
        # tile count, so the descriptor differs on every call.
        bd = Bd(
            buf, tap=TensorAccessPattern((4096,), 0, [tiles, 512], [512, 1]), bd_id=0
        )
        DmaEndpoint(mem_tile, DMAChannelDir.MM2S, 0).task(
            bd, wait=True
        ).start().await_()

    # The buffer is reached only from the sequence body; the task places it.
    rt = Runtime(sequence, [host_ty, np.int64])
    return Program(NPU2Col1(), rt).resolve_program()


# The buffer is declared at device scope, ahead of the sequence that reaches it.
# The descriptor is configured on the mem tile's own channel, not through a
# shim DMA allocation, and carries runtime sizes; the compiler derives the
# transfer length from them.

# CHECK: aie.buffer({{.*}}) {sym_name = "resident"}
# CHECK: aie.runtime_sequence
# CHECK: cf.assert %{{.*}}, "All sizes must be >= 1
# CHECK-NEXT: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
# CHECK-NEXT: aie.dma_bd(%{{.*}} : memref<4096xi32> sizes = [%arg1, 512] strides = [512, 1]) {bd_id = 0 : i32}
# CHECK: } {issue_token = true}
# CHECK-NEXT: aiex.dma_start_task
# CHECK-NEXT: aiex.dma_await_task
print(emit_dynamic_memtile_task())


def emit_shared_length_drain():
    buf_ty = np.ndarray[(4096,), np.dtype[np.int32]]
    shim = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    mem_tile = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    buf = Buffer(tile=mem_tile, type=buf_ty, name="resident")
    out = Flow(mem_tile, shim, src_channel=0, dst_channel=0)

    def sequence(host, tiles):
        tap = TensorAccessPattern((4096,), 0, [tiles, 512], [512, 1])
        out.endpoint(mem_tile).task(Bd(buf, tap=tap)).start().free()
        out.drain(host, tap=tap, wait=True)

    rt = Runtime(sequence, [buf_ty, np.int64])
    rt.add_flow(out)
    return Program(NPU2Col1(), rt).resolve_program()


# The same tap sizes the shim drain.
# CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
# CHECK-NEXT: aie.dma_bd(%{{.*}} : memref<4096xi32> sizes = [%arg1, 512] strides = [512, 1])
# CHECK: aiex.dma_start_task
# CHECK-NEXT: aiex.dma_free_task
# CHECK-NEXT: aiex.dma_configure_task_for
# CHECK-NEXT: aie.dma_bd(%{{.*}} : memref<4096xi32> offset = 0 sizes = [1, 1, %arg1, 512] strides = [0, 0, 512, 1])
print(emit_shared_length_drain())


def emit_i32_dims(chain=False):
    buf_ty = np.ndarray[(4096,), np.dtype[np.int32]]
    shim = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    mem_tile = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    buf = Buffer(tile=mem_tile, type=buf_ty, name="resident")
    out = Flow(mem_tile, shim, src_channel=0, dst_channel=0)

    def sequence(host, tiles):
        tap = TensorAccessPattern((4096,), 0, [tiles, 512], [512, 1])
        bd = Bd(buf, tap=tap)
        out.endpoint(mem_tile).task(*([bd, bd] if chain else [bd])).start()
        if not chain:
            out.drain(host, tap=tap, wait=True)

    rt = Runtime(sequence, [buf_ty, np.int32])
    rt.add_flow(out)
    return Program(NPU2Col1(), rt).resolve_program()


# An i32 dispatch-time scalar is widened once and feeds every descriptor.
# CHECK: aie.runtime_sequence(%arg0: memref<4096xi32>, %arg1: i32)
# CHECK-NEXT: %[[N:.*]] = arith.extsi %arg1 : i32 to i64
# CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
# CHECK-NEXT: aie.dma_bd(%{{.*}} : memref<4096xi32> sizes = [%[[N]], 512] strides = [512, 1])
# CHECK: aiex.dma_configure_task_for
# CHECK-NEXT: aie.dma_bd(%{{.*}} : memref<4096xi32> offset = 0 sizes = [1, 1, %[[N]], 512]
print(emit_i32_dims())

# Each block of a chain holds only its BD.
# CHECK: %[[M:.*]] = arith.extsi %arg1 : i32 to i64
# CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
# CHECK-NEXT: aie.dma_bd(%{{.*}} sizes = [%[[M]], 512]
# CHECK-NEXT: aie.next_bd
# CHECK: aie.dma_bd(%{{.*}} sizes = [%[[M]], 512]
# CHECK-NEXT: aie.end
print(emit_i32_dims(chain=True))


def emit_endpoint_task():
    buf_ty = np.ndarray[(64,), np.dtype[np.int32]]
    shim = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    mem_tile = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    buf = Buffer(type=buf_ty, name="staged")
    out = Flow(mem_tile, shim, name="out")

    def sequence(host):
        out.endpoint(mem_tile).task(buf, runs=2).start().free()
        out.drain(host, wait=True)

    rt = Runtime(sequence, [buf_ty])
    rt.add_flow(out)
    return Program(NPU2Col1(), rt).resolve_program()


# A Flow whose channels the compiler assigns runs the task on its endpoint, and
# a buffer given no tile lands on the endpoint's.
# CHECK: %[[MEM:.*]] = aie.logical_tile<MemTile>
# CHECK: aie.buffer(%[[MEM]]) {sym_name = "staged"}
# CHECK: aiex.dma_configure_task_for @out_src {
# CHECK-NEXT: aie.dma_bd(%staged
# CHECK: } {repeat_count = 1 : i32}
# CHECK: aie.route_endpoint @out_src(%[[MEM]]) DMA
print(emit_endpoint_task())


def emit_resident_replay(uses_ty):
    buf_ty = np.ndarray[(64,), np.dtype[np.int32]]
    shim = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    mem_tile = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    buf = Buffer(tile=mem_tile, type=buf_ty, name="resident")
    empty = Lock(mem_tile, name="empty")
    full = Lock(mem_tile, name="full")
    into = Flow(shim, mem_tile, src_channel=0, dst_channel=0)
    out = Flow(mem_tile, shim, src_channel=0, dst_channel=0)

    def sequence(host, uses):
        empty.set(uses)
        into.endpoint(mem_tile).task(
            Bd(
                buf,
                acquires=[Acquire(empty, value=uses)],
                releases=[Release(full, value=uses)],
            )
        ).start()
        replay = out.endpoint(mem_tile).task(
            Bd(buf, acquires=[Acquire(full)], releases=[Release(empty)])
        )
        replay.start(repeat_count=uses).free()

    rt = Runtime(sequence, [buf_ty, uses_ty])
    rt.add_lock(empty)
    rt.add_flow(into)
    rt.add_flow(out)
    return Program(NPU2Col1(), rt).resolve_program()


# A resident buffer handed out a dispatch-time number of times: the lock values
# of its fill, the re-arm and the replay's start count all take the scalar.
# CHECK: aie.runtime_sequence
# CHECK: aiex.set_lock(%empty, %arg1)
# CHECK: aiex.dma_configure_task(%{{.*}}, S2MM, 0)
# CHECK-NEXT: aie.use_lock(%empty, AcquireGreaterEqual, %arg1)
# CHECK: aie.use_lock(%full, Release, %arg1)
# CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
# CHECK: aiex.dma_start_task(%{{.*}}) repeat %arg1 : i32
print(emit_resident_replay(np.int32))

# A wider scalar is narrowed to the i32 a lock value takes, asserting it fits.
# The start count keeps its width for the push lowering's own guard.
# CHECK: aie.runtime_sequence(%{{.*}}, %arg1: i64)
# CHECK: cf.assert %{{.*}}, "a runtime 32-bit field value must be >= 0"
# CHECK: cf.assert %{{.*}}, "a runtime 32-bit field value must be < 2**31"
# CHECK-NEXT: %[[SET:.*]] = arith.trunci %arg1 : i64 to i32
# CHECK-NEXT: aiex.set_lock(%empty, %[[SET]])
# CHECK: %[[ACQ:.*]] = arith.trunci %arg1 : i64 to i32
# CHECK-NEXT: aie.use_lock(%empty, AcquireGreaterEqual, %[[ACQ]])
# CHECK: aiex.dma_start_task(%{{.*}}) repeat %arg1 : i64
print(emit_resident_replay(np.int64))
