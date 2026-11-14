// RUN: not aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate %s > %t.out 2> %t.err
// RUN: FileCheck %s --input-file=%t.out
// RUN: FileCheck %s --check-prefix=ERR --input-file=%t.err
// RUN: not aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate -debug-only=aie-objectfifo-allocate %s 2>&1 >/dev/null | FileCheck %s --check-prefix=WHY
// REQUIRES: asserts

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The join counterpart of channel_turns_segments.mlir: core r sends x, y and
// c over two output channels, so x and y take turns; their receivers are the
// two segments that fill big's pool on a mem tile, which the one core
// draining big takes whole objects from, so they count as one peer of it.

// CHECK-LABEL: module @join_segments_take_turns
// CHECK-DAG: aie.objectfifo.dma_endpoint @x__prod__dma({{.*}}turns = [@x__prod__dma, @y__prod__dma]}
// CHECK-DAG: aie.objectfifo.dma_endpoint @x__cons__dma({{.*}}turns = [@x__cons__dma, @y__cons__dma]}
module @join_segments_take_turns {
  aie.device(npu2_1col) {
    %sh = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %s = aie.tile(0, 2)
    %r = aie.tile(0, 5)
    aie.objectfifo @x(%r, {%m}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @y(%r, {%m}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @big(%m, {%s}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@x, @y] -> [@big] ([0, 16] [])
    aie.objectfifo @c(%r, {%sh}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%s) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %b = aie.objectfifo.acquire @big(Consume, 1) : memref<32xi32>
        aie.objectfifo.release @big(Consume, 1)
      }
      aie.end
    }
    aie.core(%r) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %x0 = aie.objectfifo.acquire @x(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @x(Produce, 1)
        %y1 = aie.objectfifo.acquire @y(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @y(Produce, 1)
        %v = aie.objectfifo.acquire @c(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @c(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%out : memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @c, id = 0 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @c}
    }
  }
}

// -----

// y before x: the segments take turns the other way round.
// CHECK-LABEL: module @join_segments_take_turns_reversed
// CHECK-DAG: aie.objectfifo.dma_endpoint @x__prod__dma({{.*}}turns = [@y__prod__dma, @x__prod__dma]}
module @join_segments_take_turns_reversed {
  aie.device(npu2_1col) {
    %sh = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %s = aie.tile(0, 2)
    %r = aie.tile(0, 5)
    aie.objectfifo @x(%r, {%m}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @y(%r, {%m}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @big(%m, {%s}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@x, @y] -> [@big] ([0, 16] [])
    aie.objectfifo @c(%r, {%sh}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%s) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %b = aie.objectfifo.acquire @big(Consume, 1) : memref<32xi32>
        aie.objectfifo.release @big(Consume, 1)
      }
      aie.end
    }
    aie.core(%r) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %y0 = aie.objectfifo.acquire @y(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @y(Produce, 1)
        %x1 = aie.objectfifo.acquire @x(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @x(Produce, 1)
        %v = aie.objectfifo.acquire @c(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @c(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%out : memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @c, id = 0 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @c}
    }
  }
}

// -----

// Two parts of x before one of y: x reaches big's second object while the
// first still waits for its y part, and big's pool is one deep.
// WHY: x__prod__dma + y__prod__dma: a pool's segments hold more objects open than it has
// ERR: :[[@LINE+6]]:10: error: 'aie.tile' op number of output DMA channel exceeded!
module @join_segments_hold_too_many {
  aie.device(npu2_1col) {
    %sh = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %s = aie.tile(0, 2)
    %r = aie.tile(0, 5)
    aie.objectfifo @x(%r, {%m}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @y(%r, {%m}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @big(%m, {%s}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@x, @y] -> [@big] ([0, 16] [])
    aie.objectfifo @c(%r, {%sh}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%s) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %b = aie.objectfifo.acquire @big(Consume, 1) : memref<32xi32>
        aie.objectfifo.release @big(Consume, 1)
      }
      aie.end
    }
    aie.core(%r) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %x0 = aie.objectfifo.acquire @x(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @x(Produce, 1)
        %x1 = aie.objectfifo.acquire @x(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @x(Produce, 1)
        %y2 = aie.objectfifo.acquire @y(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @y(Produce, 1)
        %v = aie.objectfifo.acquire @c(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @c(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%out : memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @c, id = 0 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @c}
    }
  }
}

// -----

// The same through two-deep pools: the objects fit, but each object takes
// one part from each segment, so two x turns for one y never balance.
// WHY: x__prod__dma + y__prod__dma: a pool's segments take different counts of turns
// ERR: :[[@LINE+6]]:10: error: 'aie.tile' op number of output DMA channel exceeded!
module @join_segments_uneven {
  aie.device(npu2_1col) {
    %sh = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %s = aie.tile(0, 2)
    %r = aie.tile(0, 5)
    aie.objectfifo @x(%r, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @y(%r, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @big(%m, {%s}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@x, @y] -> [@big] ([0, 16] [])
    aie.objectfifo @c(%r, {%sh}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%s) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %b = aie.objectfifo.acquire @big(Consume, 1) : memref<32xi32>
        aie.objectfifo.release @big(Consume, 1)
      }
      aie.end
    }
    aie.core(%r) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %x0 = aie.objectfifo.acquire @x(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @x(Produce, 1)
        %x1 = aie.objectfifo.acquire @x(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @x(Produce, 1)
        %y2 = aie.objectfifo.acquire @y(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @y(Produce, 1)
        %v = aie.objectfifo.acquire @c(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @c(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%out : memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @c, id = 0 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @c}
    }
  }
}
