//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Regression test for DMA task-queue overflow.
//
// 14 MM2S pushes on one shim channel whose task queue holds 4. The queue does
// not backpressure, so pushes that find it full are dropped, the S2MM never
// receives their data, and the trailing await then blocks forever. Built
// without enforce-queue-depth this design times out on a Strix NPU, 5 runs out
// of 5; with it the compiler waits for a free slot and the design completes.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}

    %core_buf = aie.buffer(%tile_0_2) {sym_name = "core_buf"} : memref<256xi32>

    // Input path: shim MM2S ch0 → core S2MM ch0
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    // Output path: core MM2S ch0 → shim S2MM ch0
    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)

    // Packet flows for issue_token on shim MM2S
    aie.packet_flow(0x3) {
      aie.packet_source<%tile_0_0, "TileControl" : 0>
      aie.packet_dest<%tile_0_0, "South" : 0>
    }

    // Core tile: continuously looping S2MM→MM2S passthrough
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

    // buf_a: 2560 i32 (10 slices of 256 — even-numbered transfers)
    // buf_b: 2560 i32 (10 slices of 256 — odd-numbered transfers)
    // output: 5120 i32 (20 slices of 256)
    aie.runtime_sequence(%buf_a: memref<65536xi32>, %output: memref<65536xi32>) {
      // Receive side: one looping BD covers all 14 transfers.
      %recv = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%output : memref<65536xi32> offset = 0 len = 28672 sizes = [14, 2048] strides = [2048, 1]) {bd_id = 15 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%recv)
      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<65536xi32> offset = 0 len = 2048)
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<65536xi32> offset = 0 len = 2048)
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<65536xi32> offset = 0 len = 2048)
        aie.end
      }
      aiex.dma_start_task(%t2)
      %t3 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<65536xi32> offset = 0 len = 2048)
        aie.end
      }
      aiex.dma_start_task(%t3)
      %t4 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<65536xi32> offset = 0 len = 2048)
        aie.end
      }
      aiex.dma_start_task(%t4)
      %t5 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<65536xi32> offset = 0 len = 2048)
        aie.end
      }
      aiex.dma_start_task(%t5)
      %t6 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<65536xi32> offset = 0 len = 2048)
        aie.end
      }
      aiex.dma_start_task(%t6)
      %t7 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<65536xi32> offset = 0 len = 2048)
        aie.end
      }
      aiex.dma_start_task(%t7)
      %t8 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<65536xi32> offset = 0 len = 2048)
        aie.end
      }
      aiex.dma_start_task(%t8)
      %t9 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<65536xi32> offset = 0 len = 2048)
        aie.end
      }
      aiex.dma_start_task(%t9)
      %t10 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<65536xi32> offset = 0 len = 2048)
        aie.end
      }
      aiex.dma_start_task(%t10)
      %t11 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<65536xi32> offset = 0 len = 2048)
        aie.end
      }
      aiex.dma_start_task(%t11)
      %t12 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<65536xi32> offset = 0 len = 2048)
        aie.end
      }
      aiex.dma_start_task(%t12)
      %t13 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<65536xi32> offset = 0 len = 2048)
        aie.end
      }
      aiex.dma_start_task(%t13)
      aiex.dma_await_task(%recv)
    }
  }
}
