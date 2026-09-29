//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The npu.dma_memcpy_nd twin of ../dma_queue_overflow. Same 14 MM2S pushes on
// one shim channel whose task queue holds 4, same core-tile passthrough, same
// data check -- only the runtime sequence differs, because memcpy_nd is the
// path most designs in tree actually take and enforcement on it was never
// exercised on hardware.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}

    %core_buf = aie.buffer(%tile_0_2) {sym_name = "core_buf"} : memref<256xi32>

    // Input path: shim MM2S ch0 -> core S2MM ch0
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    // Output path: core MM2S ch0 -> shim S2MM ch0
    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)

    aie.packet_flow(0x3) {
      aie.packet_source<%tile_0_0, "TileControl" : 0>
      aie.packet_dest<%tile_0_0, "South" : 0>
    }

    aie.shim_dma_allocation @in0 (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @out0 (%tile_0_0, S2MM, 0)

    // Core tile: continuously looping S2MM->MM2S passthrough
    %lock_in = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "lock_in"}
    %lock_out = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "lock_out"}

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^s2mm, ^mm2s_entry)
    ^s2mm:
      %c1_ul1 = arith.constant 1 : i32
      aie.use_lock(%lock_in, AcquireGreaterEqual, %c1_ul1)
      aie.dma_bd(%core_buf : memref<256xi32> offset = 0 len = 256)
      %c1_ul2 = arith.constant 1 : i32
      aie.use_lock(%lock_out, Release, %c1_ul2)
      aie.next_bd ^s2mm
    ^mm2s_entry:
      %1 = aie.dma_start(MM2S, 0, ^mm2s, ^end)
    ^mm2s:
      %c1_ul3 = arith.constant 1 : i32
      aie.use_lock(%lock_out, AcquireGreaterEqual, %c1_ul3)
      aie.dma_bd(%core_buf : memref<256xi32> offset = 0 len = 256)
      %c1_ul4 = arith.constant 1 : i32
      aie.use_lock(%lock_in, Release, %c1_ul4)
      aie.next_bd ^mm2s
    ^end:
      aie.end
    }

    aie.runtime_sequence(%buf_a: memref<65536xi32>, %output: memref<65536xi32>) {
      // Receive side: one 14-row descriptor covers every transfer. S2MM issues
      // its completion token implicitly, so the trailing wait has something to
      // block on.
      aiex.npu.dma_memcpy_nd(%output[0, 0, 0, 0][1, 1, 14, 2048][0, 0, 2048, 1]) {id = 14 : i64, metadata = @out0} : memref<65536xi32>
      // 14 pushes onto a 4-deep queue, none of them retired: no issue_token and
      // no wait until the very end. Without enforcement the queue overflows and
      // the dropped transfers never reach the core.
      aiex.npu.dma_memcpy_nd(%buf_a[0, 0, 0, 0][1, 1, 1, 2048][0, 0, 0, 1]) {id = 0 : i64, metadata = @in0} : memref<65536xi32>
      aiex.npu.dma_memcpy_nd(%buf_a[0, 0, 0, 0][1, 1, 1, 2048][0, 0, 0, 1]) {id = 1 : i64, metadata = @in0} : memref<65536xi32>
      aiex.npu.dma_memcpy_nd(%buf_a[0, 0, 0, 0][1, 1, 1, 2048][0, 0, 0, 1]) {id = 2 : i64, metadata = @in0} : memref<65536xi32>
      aiex.npu.dma_memcpy_nd(%buf_a[0, 0, 0, 0][1, 1, 1, 2048][0, 0, 0, 1]) {id = 3 : i64, metadata = @in0} : memref<65536xi32>
      aiex.npu.dma_memcpy_nd(%buf_a[0, 0, 0, 0][1, 1, 1, 2048][0, 0, 0, 1]) {id = 4 : i64, metadata = @in0} : memref<65536xi32>
      aiex.npu.dma_memcpy_nd(%buf_a[0, 0, 0, 0][1, 1, 1, 2048][0, 0, 0, 1]) {id = 5 : i64, metadata = @in0} : memref<65536xi32>
      aiex.npu.dma_memcpy_nd(%buf_a[0, 0, 0, 0][1, 1, 1, 2048][0, 0, 0, 1]) {id = 6 : i64, metadata = @in0} : memref<65536xi32>
      aiex.npu.dma_memcpy_nd(%buf_a[0, 0, 0, 0][1, 1, 1, 2048][0, 0, 0, 1]) {id = 7 : i64, metadata = @in0} : memref<65536xi32>
      aiex.npu.dma_memcpy_nd(%buf_a[0, 0, 0, 0][1, 1, 1, 2048][0, 0, 0, 1]) {id = 8 : i64, metadata = @in0} : memref<65536xi32>
      aiex.npu.dma_memcpy_nd(%buf_a[0, 0, 0, 0][1, 1, 1, 2048][0, 0, 0, 1]) {id = 9 : i64, metadata = @in0} : memref<65536xi32>
      aiex.npu.dma_memcpy_nd(%buf_a[0, 0, 0, 0][1, 1, 1, 2048][0, 0, 0, 1]) {id = 10 : i64, metadata = @in0} : memref<65536xi32>
      aiex.npu.dma_memcpy_nd(%buf_a[0, 0, 0, 0][1, 1, 1, 2048][0, 0, 0, 1]) {id = 11 : i64, metadata = @in0} : memref<65536xi32>
      aiex.npu.dma_memcpy_nd(%buf_a[0, 0, 0, 0][1, 1, 1, 2048][0, 0, 0, 1]) {id = 12 : i64, metadata = @in0} : memref<65536xi32>
      aiex.npu.dma_memcpy_nd(%buf_a[0, 0, 0, 0][1, 1, 1, 2048][0, 0, 0, 1]) {id = 13 : i64, metadata = @in0} : memref<65536xi32>
      aiex.npu.dma_wait {symbol = @out0}
    }
  }
}
