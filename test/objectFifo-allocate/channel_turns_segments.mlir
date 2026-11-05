// RUN: not aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate %s > %t.out 2> %t.err
// RUN: FileCheck %s --input-file=%t.out
// RUN: FileCheck %s --check-prefix=ERR --input-file=%t.err
// RUN: not aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate -debug-only=aie-objectfifo-allocate %s 2>&1 >/dev/null | FileCheck %s --check-prefix=WHY
// REQUIRES: asserts

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A mem tile splits each object of big into x and y. The receiver needs x, y
// and c over two input channels, so x and y take turns; their senders are
// big's two segments, which the one core feeding big fills whole objects for.
// CHECK-LABEL: module @segments_take_turns
// CHECK-DAG: aie.objectfifo.dma_endpoint @x__cons__dma({{.*}}turns = [@x__cons__dma, @y__cons__dma]}
// CHECK-DAG: aie.objectfifo.dma_endpoint @x__prod__dma({{.*}}turns = [@x__prod__dma, @y__prod__dma]}
module @segments_take_turns {
  aie.device(npu2_1col) {
    %sh = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %s = aie.tile(0, 2)
    %r = aie.tile(0, 5)
    aie.objectfifo @big(%s, {%m}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @x(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @y(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@big] -> [@x, @y] ([] [0, 16])
    aie.objectfifo @c(%sh, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%s) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %b = aie.objectfifo.acquire @big(Produce, 1) : memref<32xi32>
        aie.objectfifo.release @big(Produce, 1)
      }
      aie.end
    }
    aie.core(%r) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %x0 = aie.objectfifo.acquire @x(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @x(Consume, 1)
        %y1 = aie.objectfifo.acquire @y(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @y(Consume, 1)
        %v = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @c(Consume, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%in : memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @c, id = 0 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @c}
    }
  }
}

// -----

// y before x: the segments take turns the other way round.
// CHECK-LABEL: module @segments_take_turns_reversed
// CHECK-DAG: aie.objectfifo.dma_endpoint @x__prod__dma({{.*}}turns = [@y__prod__dma, @x__prod__dma]}
module @segments_take_turns_reversed {
  aie.device(npu2_1col) {
    %sh = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %s = aie.tile(0, 2)
    %r = aie.tile(0, 5)
    aie.objectfifo @big(%s, {%m}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @x(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @y(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@big] -> [@x, @y] ([] [0, 16])
    aie.objectfifo @c(%sh, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%s) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %b = aie.objectfifo.acquire @big(Produce, 1) : memref<32xi32>
        aie.objectfifo.release @big(Produce, 1)
      }
      aie.end
    }
    aie.core(%r) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %y0 = aie.objectfifo.acquire @y(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @y(Consume, 1)
        %x1 = aie.objectfifo.acquire @x(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @x(Consume, 1)
        %v = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @c(Consume, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%in : memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @c, id = 0 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @c}
    }
  }
}

// -----

// Two parts of x before one of y: x reaches big's second object while the
// first still holds its y part, and big's pool is one deep.
// WHY: x__cons__dma + y__cons__dma: a pool's segments hold more objects open than it has
// ERR: :[[@LINE+6]]:10: error: 'aie.tile' op number of input DMA channel exceeded!
module @segments_hold_too_many {
  aie.device(npu2_1col) {
    %sh = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %s = aie.tile(0, 2)
    %r = aie.tile(0, 5)
    aie.objectfifo @big(%s, {%m}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @x(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @y(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@big] -> [@x, @y] ([] [0, 16])
    aie.objectfifo @c(%sh, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%s) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %b = aie.objectfifo.acquire @big(Produce, 1) : memref<32xi32>
        aie.objectfifo.release @big(Produce, 1)
      }
      aie.end
    }
    aie.core(%r) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %x0 = aie.objectfifo.acquire @x(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @x(Consume, 1)
        %x1 = aie.objectfifo.acquire @x(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @x(Consume, 1)
        %y2 = aie.objectfifo.acquire @y(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @y(Consume, 1)
        %v = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @c(Consume, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%in : memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @c, id = 0 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @c}
    }
  }
}

// -----

// The same through a two-deep pool: the objects fit, but each object holds
// one part for each segment, so two x turns for one y never balance.
// WHY: x__cons__dma + y__cons__dma: a pool's segments take different counts of turns
// ERR: :[[@LINE+6]]:10: error: 'aie.tile' op number of input DMA channel exceeded!
module @segments_uneven {
  aie.device(npu2_1col) {
    %sh = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %s = aie.tile(0, 2)
    %r = aie.tile(0, 5)
    aie.objectfifo @big(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @x(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @y(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@big] -> [@x, @y] ([] [0, 16])
    aie.objectfifo @c(%sh, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%s) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %b = aie.objectfifo.acquire @big(Produce, 1) : memref<32xi32>
        aie.objectfifo.release @big(Produce, 1)
      }
      aie.end
    }
    aie.core(%r) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %x0 = aie.objectfifo.acquire @x(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @x(Consume, 1)
        %x1 = aie.objectfifo.acquire @x(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @x(Consume, 1)
        %y2 = aie.objectfifo.acquire @y(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @y(Consume, 1)
        %v = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @c(Consume, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%in : memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @c, id = 0 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @c}
    }
  }
}

// -----

// big's third segment z goes to another core and does not take the turns, so
// the segments do not count as one peer, and one core feeding two peers
// refuses the turns as before.
// WHY: x__cons__dma + y__cons__dma: a peer's feeding core is not known, or feeds two peers
// ERR: :[[@LINE+6]]:10: error: 'aie.tile' op number of input DMA channel exceeded!
module @segment_outside_turns {
  aie.device(npu2_1col) {
    %sh = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %s = aie.tile(0, 2)
    %r = aie.tile(0, 5)
    %t = aie.tile(0, 3)
    aie.objectfifo @big(%s, {%m}, 1 : i32) : !aie.objectfifo<memref<48xi32>>
    aie.objectfifo @x(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @y(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @z(%m, {%t}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@big] -> [@x, @y, @z] ([] [0, 16, 32])
    aie.objectfifo @c(%sh, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%s) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %b = aie.objectfifo.acquire @big(Produce, 1) : memref<48xi32>
        aie.objectfifo.release @big(Produce, 1)
      }
      aie.end
    }
    aie.core(%r) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %x0 = aie.objectfifo.acquire @x(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @x(Consume, 1)
        %y1 = aie.objectfifo.acquire @y(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @y(Consume, 1)
        %v = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @c(Consume, 1)
      }
      aie.end
    }
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %z0 = aie.objectfifo.acquire @z(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @z(Consume, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%in : memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @c, id = 0 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @c}
    }
  }
}

// -----

// The same pool filled from the shim: the host feeds it, as it did before
// segments counted as one peer, so x and y take turns.
// CHECK-LABEL: module @shim_fed_segment_outside_turns
// CHECK-DAG: aie.objectfifo.dma_endpoint @x__prod__dma({{.*}}turns = [@x__prod__dma, @y__prod__dma]}
module @shim_fed_segment_outside_turns {
  aie.device(npu2_1col) {
    %sh = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %s = aie.tile(0, 2)
    %r = aie.tile(0, 5)
    %t = aie.tile(0, 3)
    aie.objectfifo @big(%sh, {%m}, 1 : i32) : !aie.objectfifo<memref<48xi32>>
    aie.objectfifo @x(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @y(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @z(%m, {%t}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@big] -> [@x, @y, @z] ([] [0, 16, 32])
    aie.objectfifo @c(%sh, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%r) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %x0 = aie.objectfifo.acquire @x(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @x(Consume, 1)
        %y1 = aie.objectfifo.acquire @y(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @y(Consume, 1)
        %v = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @c(Consume, 1)
      }
      aie.end
    }
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %z0 = aie.objectfifo.acquire @z(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @z(Consume, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%in : memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @c, id = 0 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @c}
    }
  }
}
