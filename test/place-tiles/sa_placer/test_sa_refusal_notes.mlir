//===- test_sa_refusal_notes.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// When the SA placer ends over capacity or illegal, a note per nonzero
// penalty term names the term, its penalty and the tiles it charges, on a
// tile op placed at the first of them.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=1 ctrl-pkt-overlay=true' %s

// Both inputs and the control overlay need three MM2S channels of the one
// shim tile's two.
module @over_capacity {
  // expected-warning@+1 {{SA placer's resource estimate is over capacity (penalty=20)}}
  aie.device(npu2_1col) {
    // expected-note@+1 {{DMA channels penalty=20 on (0, 0)}}
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%s0, {%c}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s1, {%c}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 1 : i64} : memref<16xi32>
      aiex.control_packet {address = 126976 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 1 : i64} : memref<16xi32>
    }
  }
}

// -----

// A core cannot share memory with five others.
module @illegal {
  // expected-error@+1 {{SA placer failed to find a legal placement (cascade and shared memory penalty=300, delegate penalty=0)}}
  aie.device(npu2) {
    // expected-note@+6 {{cascade and shared memory pairing penalty=300 on (0, 3), (2, 3)}}
    %t0 = aie.logical_tile<CoreTile>(?, ?)
    %t1 = aie.logical_tile<CoreTile>(?, ?)
    %t2 = aie.logical_tile<CoreTile>(?, ?)
    %t3 = aie.logical_tile<CoreTile>(?, ?)
    %t4 = aie.logical_tile<CoreTile>(?, ?)
    %t5 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @near1(%t0, {%t1}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @near2(%t0, {%t2}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @near3(%t0, {%t3}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @near4(%t0, {%t4}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @near5(%t0, {%t5}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<8xi32>>
    aie.core(%t0) { aie.end }
    aie.core(%t1) { aie.end }
    aie.core(%t2) { aie.end }
    aie.core(%t3) { aie.end }
    aie.core(%t4) { aie.end }
    aie.core(%t5) { aie.end }
  }
}
