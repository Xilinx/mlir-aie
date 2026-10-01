//===- arbiter_looped_send_volume.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-create-pathfinder-flows %s

// Flows from one source, as in arbiter_unavoidable_hazard.mlir, from a sender
// that loops: S2MM 0 at (0, 3) takes 32 bytes of id 1, then waits for id 2. Each pass of
// the chain sends 32 bytes of id 1 and needs a token of %go, so the chain
// sends as many passes as %go ever holds tokens.

// Nothing but its initial value fills %go: one pass, which fits.

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.packet_flow(1) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t03, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t03, DMA : 1> }
    %go = aie.lock(%t02, 0) {init = 1 : i32}
    %src = aie.buffer(%t02) : memref<16xi32>
    aie.mem(%t02) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^first, ^end)
    ^first:
      aie.use_lock(%go, AcquireGreaterEqual, %one)
      aie.dma_bd(%src : memref<16xi32> offset = 0 len = 8) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.next_bd ^second
    ^second:
      aie.dma_bd(%src : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.next_bd ^first
    ^end:
      aie.end
    }
    %room = aie.lock(%t03, 0) {init = 1 : i32}
    %free = aie.lock(%t03, 1) {init = 1 : i32}
    %done = aie.lock(%t03, 2) {init = 0 : i32}
    %a = aie.buffer(%t03) : memref<8xi32>
    %b = aie.buffer(%t03) : memref<16xi32>
    aie.mem(%t03) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^in0, ^ch1)
    ^in0:
      aie.use_lock(%room, AcquireGreaterEqual, %one)
      aie.dma_bd(%a : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%done, Release, %one)
      aie.next_bd ^in0
    ^ch1:
      %1 = aie.dma_start(S2MM, 1, ^in1, ^end)
    ^in1:
      aie.use_lock(%free, AcquireGreaterEqual, %one)
      aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%room, Release, %one)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// A one-shot receive at (0, 2) adds a token: two passes, which overrun.

module {
  // expected-warning@+1 {{Flows can deadlock however they are routed: packet flow (0, 2) DMA:0 -> (0, 3) DMA:0 (id 1) can fill its receiver}}
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.packet_flow(1) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t03, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t03, DMA : 1> }
    %go = aie.lock(%t02, 0) {init = 1 : i32}
    %src = aie.buffer(%t02) : memref<16xi32>
    %in = aie.buffer(%t02) : memref<8xi32>
    aie.mem(%t02) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^first, ^feed)
    ^first:
      aie.use_lock(%go, AcquireGreaterEqual, %one)
      aie.dma_bd(%src : memref<16xi32> offset = 0 len = 8) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.next_bd ^second
    ^second:
      aie.dma_bd(%src : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.next_bd ^first
    ^feed:
      %1 = aie.dma_start(S2MM, 0, ^fill, ^end)
    ^fill:
      aie.dma_bd(%in : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%go, Release, %one)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %room = aie.lock(%t03, 0) {init = 1 : i32}
    %free = aie.lock(%t03, 1) {init = 1 : i32}
    %done = aie.lock(%t03, 2) {init = 0 : i32}
    %a = aie.buffer(%t03) : memref<8xi32>
    %b = aie.buffer(%t03) : memref<16xi32>
    aie.mem(%t03) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^in0, ^ch1)
    ^in0:
      aie.use_lock(%room, AcquireGreaterEqual, %one)
      aie.dma_bd(%a : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%done, Release, %one)
      aie.next_bd ^in0
    ^ch1:
      %1 = aie.dma_start(S2MM, 1, ^in1, ^end)
    ^in1:
      aie.use_lock(%free, AcquireGreaterEqual, %one)
      aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%room, Release, %one)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// A looping receive adds a token per 16-word BD it fills, and (0, 4) sends it
// 8 words: no BD completes, so one pass again.

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    aie.packet_flow(1) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t03, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t03, DMA : 1> }
    aie.flow(%t04, DMA : 0, %t02, DMA : 0)
    %out = aie.buffer(%t04) : memref<8xi32>
    aie.mem(%t04) {
      %0 = aie.dma_start(MM2S, 0, ^send, ^end)
    ^send:
      aie.dma_bd(%out : memref<8xi32> offset = 0 len = 8)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %go = aie.lock(%t02, 0) {init = 1 : i32}
    %src = aie.buffer(%t02) : memref<16xi32>
    %in = aie.buffer(%t02) : memref<16xi32>
    aie.mem(%t02) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^first, ^feed)
    ^first:
      aie.use_lock(%go, AcquireGreaterEqual, %one)
      aie.dma_bd(%src : memref<16xi32> offset = 0 len = 8) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
      aie.next_bd ^second
    ^second:
      aie.dma_bd(%src : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.next_bd ^first
    ^feed:
      %1 = aie.dma_start(S2MM, 0, ^fill, ^end)
    ^fill:
      aie.dma_bd(%in : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%go, Release, %one)
      aie.next_bd ^fill
    ^end:
      aie.end
    }
    %room = aie.lock(%t03, 0) {init = 1 : i32}
    %free = aie.lock(%t03, 1) {init = 1 : i32}
    %done = aie.lock(%t03, 2) {init = 0 : i32}
    %a = aie.buffer(%t03) : memref<8xi32>
    %b = aie.buffer(%t03) : memref<16xi32>
    aie.mem(%t03) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^in0, ^ch1)
    ^in0:
      aie.use_lock(%room, AcquireGreaterEqual, %one)
      aie.dma_bd(%a : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%done, Release, %one)
      aie.next_bd ^in0
    ^ch1:
      %1 = aie.dma_start(S2MM, 1, ^in1, ^end)
    ^in1:
      aie.use_lock(%free, AcquireGreaterEqual, %one)
      aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%room, Release, %one)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
