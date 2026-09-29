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
    Bd,
    Buffer,
    DmaEndpoint,
    Flow,
    Program,
    Runtime,
)
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
            buf,
            sizes=[1, 1, tiles, 512],
            strides=[0, 0, 512, 1],
            length=tiles * 512,
            bd_id=0,
        )
        DmaEndpoint(mem_tile, DMAChannelDir.MM2S, 0).task(
            bd, wait=True
        ).start().await_()

    # The buffer is reached only from the sequence body; the task places it.
    rt = Runtime(sequence, [host_ty, np.int64])
    return Program(NPU2Col1(), rt).resolve_program()


# The buffer is declared at device scope, ahead of the sequence that reaches it.
# The descriptor is configured on the mem tile's own channel, not through a
# shim DMA allocation, and carries runtime sizes and length. The i64 length is
# range-checked before narrowing so truncation cannot turn an invalid value into
# a different valid transfer.

# CHECK: aie.buffer({{.*}}) {sym_name = "resident"}
# CHECK: aie.runtime_sequence
# CHECK: aiex.npu.assert_bd_field(%[[LEN64:.*]]) {max = 2147483647 : i32} : i64
# CHECK-NEXT: %[[LEN32:.*]] = arith.trunci %[[LEN64]] : i64 to i32
# CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
# CHECK: aie.dma_bd(%{{.*}} : memref<4096xi32> len = %[[LEN32]] sizes = [1, 1, %{{.*}}, 512] strides = [0, 0, 512, 1]) {bd_id = 0 : i32}
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
        at = dict(sizes=[1, 1, tiles, 512], strides=[0, 0, 512, 1])
        length = tiles * 512
        out.endpoint(mem_tile).task(Bd(buf, length=length, **at)).start().free()
        out.drain(host, transfer_len=length, wait=True, **at)

    rt = Runtime(sequence, [buf_ty, np.int64])
    rt.add_flow(out)
    return Program(NPU2Col1(), rt).resolve_program()


# The same i64 length also sizes the shim drain, narrowed the same way and
# before the task opens, since a BD block admits no arithmetic.
# CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
# CHECK: aie.dma_bd(%{{.*}} : memref<4096xi32> len = %{{.*}} sizes = [1, 1, %{{.*}}, 512]
# CHECK: aiex.dma_start_task
# CHECK-NEXT: aiex.dma_free_task
# CHECK: aiex.npu.assert_bd_field(%[[DLEN64:.*]]) {max = 2147483647 : i32} : i64
# CHECK-NEXT: %[[DLEN32:.*]] = arith.trunci %[[DLEN64]] : i64 to i32
# CHECK-NEXT: aiex.dma_configure_task_for
# CHECK-NEXT: aie.dma_bd(%{{.*}} : memref<4096xi32> offset = 0 len = %[[DLEN32]] sizes = [1, 1, %{{.*}}, 512]
print(emit_shared_length_drain())


def emit_i32_dims(dims=None, chain=False):
    buf_ty = np.ndarray[(4096,), np.dtype[np.int32]]
    shim = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    mem_tile = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    buf = Buffer(tile=mem_tile, type=buf_ty, name="resident")
    out = Flow(mem_tile, shim, src_channel=0, dst_channel=0)

    def sequence(host, tiles):
        sizes = dims(tiles) if dims else [1, 1, tiles, 512]
        bd = Bd(buf, sizes=sizes, strides=[0, 0, 512, 1])
        out.endpoint(mem_tile).task(*([bd, bd] if chain else [bd])).start()
        if not chain:
            out.drain(host, sizes=sizes, strides=[0, 0, 512, 1], wait=True)

    rt = Runtime(sequence, [buf_ty, np.int32])
    rt.add_flow(out)
    try:
        return Program(NPU2Col1(), rt).resolve_program()
    except TypeError as e:
        return f"RAISED TypeError: {e}"


# An i32 dispatch-time scalar feeds the i64 sizes directly, and the omitted
# length defaults to their product: each task widens and multiplies before
# opening, since a BD block admits no arithmetic.
# CHECK: %[[T1:.*]] = arith.extsi %arg1 : i32 to i64
# CHECK: aiex.npu.assert_bd_field
# CHECK-NEXT: %[[L1:.*]] = arith.trunci %{{.*}} : i64 to i32
# CHECK-NEXT: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
# CHECK-NEXT: aie.dma_bd(%{{.*}} : memref<4096xi32> len = %[[L1]] sizes = [1, 1, %[[T1]], 512]
# CHECK: %[[T2:.*]] = arith.extsi %arg1 : i32 to i64
# CHECK: %[[L2:.*]] = arith.trunci %{{.*}} : i64 to i32
# CHECK-NEXT: aiex.dma_configure_task_for
# CHECK-NEXT: aie.dma_bd(%{{.*}} : memref<4096xi32> offset = 0 len = %[[L2]] sizes = [1, 1, %[[T2]], 512]
print(emit_i32_dims())

# A chain's Bds are cast ahead of the task too, so each block holds only the BD.
# CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
# CHECK-NEXT: aie.dma_bd(%{{.*}} sizes = [1, 1, %{{[0-9]+}}, 512]
# CHECK-NEXT: aie.next_bd
# CHECK: aie.dma_bd(%{{.*}} sizes = [1, 1, %{{[0-9]+}}, 512]
# CHECK-NEXT: aie.end
print(emit_i32_dims(chain=True))

# CHECK: RAISED TypeError: A BD field must be an int or an integer SSA value, got float.
print(emit_i32_dims(dims=lambda tiles: [1, 1, 2.0, 512]))


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
