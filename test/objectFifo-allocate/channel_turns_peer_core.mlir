// RUN: not aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate %s > %t.out 2> %t.err
// RUN: FileCheck %s --input-file=%t.out
// RUN: FileCheck %s --check-prefix=ERR --input-file=%t.err

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The receiving core needs a, b, b and c over two input channels, so a and b
// take turns. The sending core makes them in the same order, so the chain
// walking a, b, b on its side keeps up with it too.
// CHECK-LABEL: module @sender_keeps_up
// CHECK: aie.objectfifo.dma_endpoint @a__prod__dma({{.*}}turns = [@a__prod__dma, @b__prod__dma, @b__prod__dma]}
// CHECK: aie.objectfifo.dma_endpoint @a__cons__dma({{.*}}turns = [@a__cons__dma, @b__cons__dma, @b__cons__dma]}
module @sender_keeps_up {
  aie.device(npu2_1col) {
    %sh = aie.tile(0, 0)
    %s = aie.tile(0, 2)
    %r = aie.tile(0, 5)
    aie.objectfifo @a(%s, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c(%sh, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%s) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %a0 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @a(Produce, 1)
        %b0 = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b(Produce, 1)
        %b1 = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b(Produce, 1)
      }
      aie.end
    }
    aie.core(%r) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %a0 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
        %b0 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b1 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @b(Consume, 1)
        %x = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @c(Consume, 1)
        aie.objectfifo.release @a(Consume, 1)
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

// The same receiver, but the sending core makes b, b, a through pools one
// deep: a chain walking a, b, b would wait on a while the core waits for room
// for its second b. So a and b do not take turns, and the receiver's three
// inputs need three channels.
// ERR: error: 'aie.tile' op number of input DMA channel exceeded!
module @sender_out_of_order {
  aie.device(npu2_1col) {
    %sh = aie.tile(0, 0)
    %s = aie.tile(0, 2)
    %r = aie.tile(0, 5)
    aie.objectfifo @a(%s, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c(%sh, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%s) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %b0 = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b(Produce, 1)
        %b1 = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b(Produce, 1)
        %a0 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @a(Produce, 1)
      }
      aie.end
    }
    aie.core(%r) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %a0 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
        %b0 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b1 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @b(Consume, 1)
        %x = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @c(Consume, 1)
        aie.objectfifo.release @a(Consume, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%in : memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @c, id = 0 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @c}
    }
  }
}
