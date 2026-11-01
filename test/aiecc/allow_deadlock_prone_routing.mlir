//===- allow_deadlock_prone_routing.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Core (0, 3) takes 8 of the 16 words of id 1 only once id 2 has arrived,
// which (0, 2) sends after id 1, so the flows can deadlock however they are
// routed. aiecc fails on them unless --allow-deadlock-prone-routing is set.

// RUN: not %aiecc -n --tmpdir %t.err --get-xclbin %s 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: %aiecc -n --tmpdir %t.warn --get-xclbin --allow-deadlock-prone-routing %s 2>&1 | FileCheck %s --check-prefix=WARN

// ERR: error: Flows can deadlock however they are routed:
// ERR-SAME: Set allow-deadlock-prone (aiecc --allow-deadlock-prone-routing) to route them anyway.
// WARN: warning: Flows can deadlock however they are routed:
// WARN-NOT: error

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
