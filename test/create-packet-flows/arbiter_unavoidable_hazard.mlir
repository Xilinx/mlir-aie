//===- arbiter_unavoidable_hazard.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s

// Flows from one source that can deadlock there are reported, as no routing
// can keep them apart, along with what the router assumed to find it.

// CHECK: warning: Flows can deadlock however they are routed: packet flow (0, 2) DMA:0 -> (0, 3) DMA:0 (id 1) can fill its receiver, and draining that waits on (0, 3) S2MM 1, which receives packet flow (0, 2) DMA:0 -> (0, 3) DMA:1 (id 2). The volume packet flow (0, 2) DMA:0 -> (0, 3) DMA:0 (id 1) carries is unknown, so it is assumed to overrun its receiver. Nothing in the design programs (0, 3) S2MM 0, so it is assumed to wait on anything on its tile. So can 1 other pair of flows.

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.packet_flow(1) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t03, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t03, DMA : 1> }
  }
}

// -----

// Core (0, 3) takes 8 of the 16 words of id 1 only once id 2 has arrived,
// which (0, 2) sends after id 1.

// CHECK: warning: Flows can deadlock however they are routed: packet flow (0, 2) DMA:0 -> (0, 3) DMA:0 (id 1) can fill its receiver, and draining that waits on (0, 3) S2MM 1, which receives packet flow (0, 2) DMA:0 -> (0, 3) DMA:1 (id 2).{{$}}

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.packet_flow(1) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t03, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t03, DMA : 1> }
    %src = aie.buffer(%t02) : memref<16xi32>
    aie.mem(%t02) {
      %0 = aie.dma_start(MM2S, 0, ^first, ^end)
    ^first:
      aie.dma_bd(%src : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.next_bd ^second
    ^second:
      aie.dma_bd(%src : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %ready = aie.lock(%t03, 0) {init = 0 : i32}
    %free = aie.lock(%t03, 1) {init = 1 : i32}
    %done = aie.lock(%t03, 2) {init = 0 : i32}
    %a = aie.buffer(%t03) : memref<8xi32>
    %b = aie.buffer(%t03) : memref<16xi32>
    aie.mem(%t03) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^in0, ^ch1)
    ^in0:
      aie.use_lock(%ready, AcquireGreaterEqual, %one)
      aie.dma_bd(%a : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%done, Release, %one)
      aie.next_bd ^end
    ^ch1:
      %1 = aie.dma_start(S2MM, 1, ^in1, ^end)
    ^in1:
      aie.use_lock(%free, AcquireGreaterEqual, %one)
      aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%ready, Release, %one)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// Once S2MM 0 takes all of id 1 without waiting on id 2, nothing can
// deadlock.

// CHECK-NOT: warning
module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.packet_flow(1) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t03, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t03, DMA : 1> }
    %src = aie.buffer(%t02) : memref<16xi32>
    aie.mem(%t02) {
      %0 = aie.dma_start(MM2S, 0, ^first, ^end)
    ^first:
      aie.dma_bd(%src : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.next_bd ^second
    ^second:
      aie.dma_bd(%src : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %ready = aie.lock(%t03, 0) {init = 0 : i32}
    %free = aie.lock(%t03, 1) {init = 2 : i32}
    %done = aie.lock(%t03, 2) {init = 0 : i32}
    %a = aie.buffer(%t03) : memref<16xi32>
    %b = aie.buffer(%t03) : memref<16xi32>
    aie.mem(%t03) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^in0, ^ch1)
    ^in0:
      aie.use_lock(%free, AcquireGreaterEqual, %one)
      aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%done, Release, %one)
      aie.next_bd ^end
    ^ch1:
      %1 = aie.dma_start(S2MM, 1, ^in1, ^end)
    ^in1:
      aie.use_lock(%free, AcquireGreaterEqual, %one)
      aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%ready, Release, %one)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
