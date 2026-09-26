//===- coverage_dma_op_volume.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 | FileCheck %s
// RUN: sed 's/repeat_count = 0/repeat_count = 1/' %s | not aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" 2>&1 | FileCheck %s --check-prefix=STALL
// RUN: sed 's/loop = false/loop = true/' %s | not aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" 2>&1 | FileCheck %s --check-prefix=STALL

// arbiter_keep_pkt_header_receive.mlir written with aie.dma instead of
// aie.dma_start. Each receiver takes 64 bytes before its lock blocks. Sent
// once, every packet fits and the seven master ports at (0,1) share six
// arbiters; sent twice, or forever, each receiver fills and they cannot.

// CHECK-NOT:   error
// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK-COUNT-7: aie.masterset

// STALL: error: Unable to find a legal routing: at tile (0, 1), no two of
// STALL-SAME: but the switchbox has 6 free.

module {
  aie.device(npu2) {
    %m  = aie.tile(0, 1)
    %t0 = aie.tile(0, 2)
    %t1 = aie.tile(1, 2)
    %t2 = aie.tile(2, 2)
    %t3 = aie.tile(3, 2)
    %t4 = aie.tile(4, 2)
    %t5 = aie.tile(5, 2)
    %t6 = aie.tile(0, 3)

    aie.packet_flow(0) { aie.packet_source<%t0, DMA : 0> aie.packet_dest<%m, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%t1, DMA : 0> aie.packet_dest<%m, DMA : 1> }
    aie.packet_flow(2) { aie.packet_source<%t2, DMA : 0> aie.packet_dest<%m, DMA : 2> }
    aie.packet_flow(3) { aie.packet_source<%t3, DMA : 0> aie.packet_dest<%m, DMA : 3> }
    aie.packet_flow(4) { aie.packet_source<%t4, DMA : 0> aie.packet_dest<%m, DMA : 4> }
    aie.packet_flow(5) { aie.packet_source<%t5, DMA : 0> aie.packet_dest<%m, DMA : 5> }
    aie.packet_flow(6) { aie.packet_source<%m, DMA : 0> aie.packet_dest<%t6, DMA : 1> }

    %b0 = aie.buffer(%t0) : memref<16xi32>
    %sl0 = aie.lock(%t0, 0) {init = 1 : i32}
    %sd0 = aie.lock(%t0, 1) {init = 0 : i32}
    aie.mem(%t0) {
      %one = arith.constant 1 : i32
      %0 = aie.dma(MM2S, 0) {loop = false, repeat_count = 0 : i32} [{
        aie.use_lock(%sl0, AcquireGreaterEqual, %one)
        aie.dma_bd(%b0 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 0, pkt_type = 0>}
        aie.use_lock(%sd0, Release, %one)
      }]
      aie.end
    }
    %b1 = aie.buffer(%t1) : memref<16xi32>
    %sl1 = aie.lock(%t1, 0) {init = 1 : i32}
    %sd1 = aie.lock(%t1, 1) {init = 0 : i32}
    aie.mem(%t1) {
      %one = arith.constant 1 : i32
      %0 = aie.dma(MM2S, 0) {loop = false, repeat_count = 0 : i32} [{
        aie.use_lock(%sl1, AcquireGreaterEqual, %one)
        aie.dma_bd(%b1 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 1, pkt_type = 0>}
        aie.use_lock(%sd1, Release, %one)
      }]
      aie.end
    }
    %b2 = aie.buffer(%t2) : memref<16xi32>
    %sl2 = aie.lock(%t2, 0) {init = 1 : i32}
    %sd2 = aie.lock(%t2, 1) {init = 0 : i32}
    aie.mem(%t2) {
      %one = arith.constant 1 : i32
      %0 = aie.dma(MM2S, 0) {loop = false, repeat_count = 0 : i32} [{
        aie.use_lock(%sl2, AcquireGreaterEqual, %one)
        aie.dma_bd(%b2 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 2, pkt_type = 0>}
        aie.use_lock(%sd2, Release, %one)
      }]
      aie.end
    }
    %b3 = aie.buffer(%t3) : memref<16xi32>
    %sl3 = aie.lock(%t3, 0) {init = 1 : i32}
    %sd3 = aie.lock(%t3, 1) {init = 0 : i32}
    aie.mem(%t3) {
      %one = arith.constant 1 : i32
      %0 = aie.dma(MM2S, 0) {loop = false, repeat_count = 0 : i32} [{
        aie.use_lock(%sl3, AcquireGreaterEqual, %one)
        aie.dma_bd(%b3 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 3, pkt_type = 0>}
        aie.use_lock(%sd3, Release, %one)
      }]
      aie.end
    }
    %b4 = aie.buffer(%t4) : memref<16xi32>
    %sl4 = aie.lock(%t4, 0) {init = 1 : i32}
    %sd4 = aie.lock(%t4, 1) {init = 0 : i32}
    aie.mem(%t4) {
      %one = arith.constant 1 : i32
      %0 = aie.dma(MM2S, 0) {loop = false, repeat_count = 0 : i32} [{
        aie.use_lock(%sl4, AcquireGreaterEqual, %one)
        aie.dma_bd(%b4 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 4, pkt_type = 0>}
        aie.use_lock(%sd4, Release, %one)
      }]
      aie.end
    }
    %b5 = aie.buffer(%t5) : memref<16xi32>
    %sl5 = aie.lock(%t5, 0) {init = 1 : i32}
    %sd5 = aie.lock(%t5, 1) {init = 0 : i32}
    aie.mem(%t5) {
      %one = arith.constant 1 : i32
      %0 = aie.dma(MM2S, 0) {loop = false, repeat_count = 0 : i32} [{
        aie.use_lock(%sl5, AcquireGreaterEqual, %one)
        aie.dma_bd(%b5 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 5, pkt_type = 0>}
        aie.use_lock(%sd5, Release, %one)
      }]
      aie.end
    }

    %mb0 = aie.buffer(%m) : memref<16xi32>
    %mb1 = aie.buffer(%m) : memref<16xi32>
    %mb2 = aie.buffer(%m) : memref<16xi32>
    %mb3 = aie.buffer(%m) : memref<16xi32>
    %mb4 = aie.buffer(%m) : memref<16xi32>
    %mb5 = aie.buffer(%m) : memref<16xi32>
    %p0 = aie.lock(%m, 0) {init = 1 : i32}
    %c0 = aie.lock(%m, 1) {init = 0 : i32}
    %p1 = aie.lock(%m, 2) {init = 1 : i32}
    %c1 = aie.lock(%m, 3) {init = 0 : i32}
    %p2 = aie.lock(%m, 4) {init = 1 : i32}
    %c2 = aie.lock(%m, 5) {init = 0 : i32}
    %p3 = aie.lock(%m, 6) {init = 1 : i32}
    %c3 = aie.lock(%m, 7) {init = 0 : i32}
    %p4 = aie.lock(%m, 8) {init = 1 : i32}
    %c4 = aie.lock(%m, 9) {init = 0 : i32}
    %p5 = aie.lock(%m, 10) {init = 1 : i32}
    %c5 = aie.lock(%m, 11) {init = 0 : i32}
    aie.memtile_dma(%m) {
      %one = arith.constant 1 : i32
      %r0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%p0, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb0 : memref<16xi32> offset = 0 len = 16)
        aie.use_lock(%c0, Release, %one)
      }]
      %r1 = aie.dma(S2MM, 1) [{
        aie.use_lock(%p1, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb1 : memref<16xi32> offset = 0 len = 16)
        aie.use_lock(%c1, Release, %one)
      }]
      %r2 = aie.dma(S2MM, 2) [{
        aie.use_lock(%p2, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb2 : memref<16xi32> offset = 0 len = 16)
        aie.use_lock(%c2, Release, %one)
      }]
      %r3 = aie.dma(S2MM, 3) [{
        aie.use_lock(%p3, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb3 : memref<16xi32> offset = 0 len = 16)
        aie.use_lock(%c3, Release, %one)
      }]
      %r4 = aie.dma(S2MM, 4) [{
        aie.use_lock(%p4, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb4 : memref<16xi32> offset = 0 len = 16)
        aie.use_lock(%c4, Release, %one)
      }]
      %r5 = aie.dma(S2MM, 5) [{
        aie.use_lock(%p5, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb5 : memref<16xi32> offset = 0 len = 16)
        aie.use_lock(%c5, Release, %one)
      }]
      %join = aie.dma(MM2S, 0) [{
        aie.use_lock(%c0, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb0 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 6, pkt_type = 0>}
        aie.use_lock(%p0, Release, %one)
      }, {
        aie.use_lock(%c1, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb1 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 6, pkt_type = 0>}
        aie.use_lock(%p1, Release, %one)
      }, {
        aie.use_lock(%c2, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb2 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 6, pkt_type = 0>}
        aie.use_lock(%p2, Release, %one)
      }, {
        aie.use_lock(%c3, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb3 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 6, pkt_type = 0>}
        aie.use_lock(%p3, Release, %one)
      }, {
        aie.use_lock(%c4, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb4 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 6, pkt_type = 0>}
        aie.use_lock(%p4, Release, %one)
      }, {
        aie.use_lock(%c5, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb5 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 6, pkt_type = 0>}
        aie.use_lock(%p5, Release, %one)
      }]
      aie.end
    }
  }
}
