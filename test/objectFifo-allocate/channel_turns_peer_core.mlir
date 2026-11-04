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

// -----

// The receiver needs a, b, b, b, c through a mem tile, whose MM2S has a and b
// take turns. One core makes both a_up and b_up in that order, so the chain
// keeps up with it through both hops' pools.
// CHECK-LABEL: module @memtile_sender_keeps_up
// CHECK: aie.objectfifo.dma_endpoint @a__prod__dma({{.*}}turns = [@a__prod__dma, @b__prod__dma, @b__prod__dma, @b__prod__dma]}
module @memtile_sender_keeps_up {
  aie.device(npu2_1col) {
    %sh = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %s = aie.tile(0, 2)
    %r = aie.tile(0, 5)
    aie.objectfifo @a_up(%s, {%m}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a_up] -> [@a] ([] [])
    aie.objectfifo @b_up(%s, {%m}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@b_up] -> [@b] ([] [])
    aie.objectfifo @c(%sh, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%s) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %a0 = aie.objectfifo.acquire @a_up(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @a_up(Produce, 1)
        %b0 = aie.objectfifo.acquire @b_up(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b_up(Produce, 1)
        %b1 = aie.objectfifo.acquire @b_up(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b_up(Produce, 1)
        %b2 = aie.objectfifo.acquire @b_up(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b_up(Produce, 1)
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
        %b2 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
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

// As @memtile_sender_keeps_up, but b_up also goes to another tile: a
// broadcast still has one sender, so the core is found through it and the
// turns are taken.
// CHECK-LABEL: module @memtile_sender_broadcasts
// CHECK: aie.objectfifo.dma_endpoint @a__prod__dma({{.*}}turns = [@a__prod__dma, @b__prod__dma, @b__prod__dma, @b__prod__dma]}
module @memtile_sender_broadcasts {
  aie.device(npu2_1col) {
    %sh = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %s = aie.tile(0, 2)
    %r = aie.tile(0, 5)
    %t = aie.tile(0, 3)
    aie.objectfifo @a_up(%s, {%m}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a_up] -> [@a] ([] [])
    aie.objectfifo @b_up(%s, {%m, %t}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@b_up] -> [@b] ([] [])
    aie.objectfifo @c(%sh, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%s) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %a0 = aie.objectfifo.acquire @a_up(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @a_up(Produce, 1)
        %b0 = aie.objectfifo.acquire @b_up(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b_up(Produce, 1)
        %b1 = aie.objectfifo.acquire @b_up(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b_up(Produce, 1)
        %b2 = aie.objectfifo.acquire @b_up(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b_up(Produce, 1)
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
        %b2 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @b(Consume, 1)
        %x = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @c(Consume, 1)
        aie.objectfifo.release @a(Consume, 1)
      }
      aie.end
    }
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c12 = arith.constant 12 : index
      scf.for %i = %c0 to %c12 step %c1 {
        %x = aie.objectfifo.acquire @b_up(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @b_up(Consume, 1)
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

// The same, but the core makes b, b, b, a: with two one-deep pools between it
// and the mem tile, its third b waits for room while the chain waits on a.
// So a and b do not take turns, and the receiver needs three channels.
// ERR: error: 'aie.tile' op number of input DMA channel exceeded!
module @memtile_sender_out_of_order {
  aie.device(npu2_1col) {
    %sh = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %s = aie.tile(0, 2)
    %r = aie.tile(0, 5)
    aie.objectfifo @a_up(%s, {%m}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a_up] -> [@a] ([] [])
    aie.objectfifo @b_up(%s, {%m}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@b_up] -> [@b] ([] [])
    aie.objectfifo @c(%sh, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%s) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %b0 = aie.objectfifo.acquire @b_up(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b_up(Produce, 1)
        %b1 = aie.objectfifo.acquire @b_up(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b_up(Produce, 1)
        %b2 = aie.objectfifo.acquire @b_up(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b_up(Produce, 1)
        %a0 = aie.objectfifo.acquire @a_up(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @a_up(Produce, 1)
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
        %b2 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
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

// a and b reach the mem tile through joins of two cores' halves, so the walk
// back from the peers cannot name one core per peer. Both cores make b three
// times before a while the receiver wants a first, so the ends do not take
// turns.
// ERR: error: 'aie.tile' op number of input DMA channel exceeded!
module @join_peers {
  aie.device(npu2_1col) {
    %sh = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %s1 = aie.tile(0, 2)
    %s2 = aie.tile(0, 3)
    %r = aie.tile(0, 5)
    aie.objectfifo @a1(%s1, {%m}, 1 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo @a2(%s2, {%m}, 1 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo @a(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@a1, @a2] -> [@a] ([0, 512] [])
    aie.objectfifo @b1(%s1, {%m}, 1 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo @b2(%s2, {%m}, 1 : i32) : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo @b(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@b1, @b2] -> [@b] ([0, 512] [])
    aie.objectfifo @c(%sh, {%r}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.core(%s1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %n = arith.constant 4 : index
      scf.for %i = %c0 to %n step %c1 {
        %b1_Produce_0_0 = aie.objectfifo.acquire @b1(Produce, 1) : memref<512xi32>
        aie.objectfifo.release @b1(Produce, 1)
        %b1_Produce_1_0 = aie.objectfifo.acquire @b1(Produce, 1) : memref<512xi32>
        aie.objectfifo.release @b1(Produce, 1)
        %b1_Produce_2_0 = aie.objectfifo.acquire @b1(Produce, 1) : memref<512xi32>
        aie.objectfifo.release @b1(Produce, 1)
        %a1_Produce_0_0 = aie.objectfifo.acquire @a1(Produce, 1) : memref<512xi32>
        aie.objectfifo.release @a1(Produce, 1)
      }
      aie.end
    }
    aie.core(%s2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %n = arith.constant 4 : index
      scf.for %i = %c0 to %n step %c1 {
        %b2_Produce_0_1 = aie.objectfifo.acquire @b2(Produce, 1) : memref<512xi32>
        aie.objectfifo.release @b2(Produce, 1)
        %b2_Produce_1_1 = aie.objectfifo.acquire @b2(Produce, 1) : memref<512xi32>
        aie.objectfifo.release @b2(Produce, 1)
        %b2_Produce_2_1 = aie.objectfifo.acquire @b2(Produce, 1) : memref<512xi32>
        aie.objectfifo.release @b2(Produce, 1)
        %a2_Produce_0_1 = aie.objectfifo.acquire @a2(Produce, 1) : memref<512xi32>
        aie.objectfifo.release @a2(Produce, 1)
      }
      aie.end
    }
    aie.core(%r) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %n = arith.constant 4 : index
      scf.for %i = %c0 to %n step %c1 {
        %a_Consume_0_2 = aie.objectfifo.acquire @a(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @a(Consume, 1)
        %b_Consume_0_2 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b_Consume_1_2 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b_Consume_2_2 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %c_Consume_0_2 = aie.objectfifo.acquire @c(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @c(Consume, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%in : memref<4096xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1]) {metadata = @c, id = 0 : i64, issue_token = true} : memref<4096xi32>
      aiex.npu.dma_wait {symbol = @c}
    }
  }
}

// -----

// Two cores each feed one peer through a mem tile, but s1 makes a only after
// x, which s2 makes after its three b's. Taking turns a, b, b, b, the chain
// waits on a, s1 on x, and s2 for room for its third b, so separate cores
// do not let the ends take turns.
// ERR: error: 'aie.tile' op number of input DMA channel exceeded!
module @own_cores_cross_dep {
  aie.device(npu2_1col) {
    %sh = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %s1 = aie.tile(0, 2)
    %s2 = aie.tile(0, 3)
    %r = aie.tile(0, 5)
    aie.objectfifo @a_up(%s1, {%m}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @a(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@a_up] -> [@a] ([] [])
    aie.objectfifo @b_up(%s2, {%m}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @b(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@b_up] -> [@b] ([] [])
    aie.objectfifo @x(%s2, {%s1}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @c(%sh, {%r}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.core(%s1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %x0 = aie.objectfifo.acquire @x(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @x(Consume, 1)
        %a0 = aie.objectfifo.acquire @a_up(Produce, 1) : memref<1024xi32>
        aie.objectfifo.release @a_up(Produce, 1)
      }
      aie.end
    }
    aie.core(%s2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %b0 = aie.objectfifo.acquire @b_up(Produce, 1) : memref<1024xi32>
        aie.objectfifo.release @b_up(Produce, 1)
        %b1 = aie.objectfifo.acquire @b_up(Produce, 1) : memref<1024xi32>
        aie.objectfifo.release @b_up(Produce, 1)
        %b2 = aie.objectfifo.acquire @b_up(Produce, 1) : memref<1024xi32>
        aie.objectfifo.release @b_up(Produce, 1)
        %x0 = aie.objectfifo.acquire @x(Produce, 1) : memref<1024xi32>
        aie.objectfifo.release @x(Produce, 1)
      }
      aie.end
    }
    aie.core(%r) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %a0 = aie.objectfifo.acquire @a(Consume, 1) : memref<1024xi32>
        %b0 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b1 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b2 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %y = aie.objectfifo.acquire @c(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @c(Consume, 1)
        aie.objectfifo.release @a(Consume, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%in : memref<4096xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1]) {metadata = @c, id = 0 : i64, issue_token = true} : memref<4096xi32>
      aiex.npu.dma_wait {symbol = @c}
    }
  }
}

// -----

// b reaches the mem tile from the shim, which waits on the host alone, and a
// from a core that keeps up with a, b, b, b, so a and b take turns.
// CHECK-LABEL: module @shim_feeds_b
// CHECK: aie.objectfifo.dma_endpoint @a__prod__dma({{.*}}turns = [@a__prod__dma, @b__prod__dma, @b__prod__dma, @b__prod__dma]}
module @shim_feeds_b {
  aie.device(npu2_1col) {
    %sh = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %s1 = aie.tile(0, 2)
    %s2 = aie.tile(0, 3)
    %r = aie.tile(0, 5)
    aie.objectfifo @a_up(%s1, {%m}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @a(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@a_up] -> [@a] ([] [])
    aie.objectfifo @b_up(%sh, {%m}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @b(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@b_up] -> [@b] ([] [])
    aie.objectfifo @c(%sh, {%r}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.core(%s1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %a0 = aie.objectfifo.acquire @a_up(Produce, 1) : memref<1024xi32>
        aie.objectfifo.release @a_up(Produce, 1)
      }
      aie.end
    }
    aie.core(%r) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %a0 = aie.objectfifo.acquire @a(Consume, 1) : memref<1024xi32>
        %b0 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b1 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b2 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %y = aie.objectfifo.acquire @c(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @c(Consume, 1)
        aie.objectfifo.release @a(Consume, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%in : memref<4096xi32>, %in2 : memref<12288xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1]) {metadata = @c, id = 0 : i64, issue_token = true} : memref<4096xi32>
      aiex.npu.dma_memcpy_nd(%in2[0, 0, 0, 0][1, 1, 1, 12288][0, 0, 0, 1]) {metadata = @b_up, id = 1 : i64, issue_token = true} : memref<12288xi32>
      aiex.npu.dma_wait {symbol = @c}
    }
  }
}

// -----

// The core feeding the peers sits two circuits back, through two mem tiles,
// and makes b four times before a: too many for the pools on the way.
// ERR: error: 'aie.tile' op number of input DMA channel exceeded!
module @two_hops_four_b {
  aie.device(npu2) {
    %sh = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %m2 = aie.tile(1, 1)
    %s = aie.tile(0, 2)
    %r = aie.tile(0, 5)
    aie.objectfifo @a_up0(%s, {%m2}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a_up(%m2, {%m}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a_up0] -> [@a_up] ([] [])
    aie.objectfifo @a(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a_up] -> [@a] ([] [])
    aie.objectfifo @b_up0(%s, {%m2}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b_up(%m2, {%m}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@b_up0] -> [@b_up] ([] [])
    aie.objectfifo @b(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@b_up] -> [@b] ([] [])
    aie.objectfifo @c(%sh, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%s) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %b0 = aie.objectfifo.acquire @b_up0(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b_up0(Produce, 1)
        %b1 = aie.objectfifo.acquire @b_up0(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b_up0(Produce, 1)
        %b2 = aie.objectfifo.acquire @b_up0(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b_up0(Produce, 1)
        %b3 = aie.objectfifo.acquire @b_up0(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b_up0(Produce, 1)
        %a0 = aie.objectfifo.acquire @a_up0(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @a_up0(Produce, 1)
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
        %b2 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b3 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
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

// As @two_hops_four_b with three b's, which the pools two circuits deep hold.
// CHECK-LABEL: module @two_hops_out_of_order
// CHECK: aie.objectfifo.dma_endpoint @a__prod__dma({{.*}}turns = [@a__prod__dma, @b__prod__dma, @b__prod__dma, @b__prod__dma]}
module @two_hops_out_of_order {
  aie.device(npu2) {
    %sh = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %m2 = aie.tile(1, 1)
    %s = aie.tile(0, 2)
    %r = aie.tile(0, 5)
    aie.objectfifo @a_up0(%s, {%m2}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a_up(%m2, {%m}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a_up0] -> [@a_up] ([] [])
    aie.objectfifo @a(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a_up] -> [@a] ([] [])
    aie.objectfifo @b_up0(%s, {%m2}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b_up(%m2, {%m}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@b_up0] -> [@b_up] ([] [])
    aie.objectfifo @b(%m, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@b_up] -> [@b] ([] [])
    aie.objectfifo @c(%sh, {%r}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%s) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %b0 = aie.objectfifo.acquire @b_up0(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b_up0(Produce, 1)
        %b1 = aie.objectfifo.acquire @b_up0(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b_up0(Produce, 1)
        %b2 = aie.objectfifo.acquire @b_up0(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b_up0(Produce, 1)
        %a0 = aie.objectfifo.acquire @a_up0(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @a_up0(Produce, 1)
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
        %b2 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
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

// A core sends a and b through a mem tile that broadcasts each to two cores.
// The walk down from the mem tile cannot follow a broadcast, so the core's
// ends do not take turns.
// ERR: error: 'aie.tile' op number of output DMA channel exceeded!
module @down_broadcast {
  aie.device(npu2_1col) {
    %sh = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %r = aie.tile(0, 2)
    %d = aie.tile(0, 3)
    %e = aie.tile(0, 4)
    aie.objectfifo @a_dn(%r, {%m}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @a(%m, {%d, %e}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@a_dn] -> [@a] ([] [])
    aie.objectfifo @b_dn(%r, {%m}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @b(%m, {%d, %e}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@b_dn] -> [@b] ([] [])
    aie.objectfifo @c(%r, {%sh}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.core(%r) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %n = arith.constant 12 : index
      scf.for %i = %c0 to %n step %c1 {
        %a_dn1 = aie.objectfifo.acquire @a_dn(Produce, 1) : memref<1024xi32>
        aie.objectfifo.release @a_dn(Produce, 1)
        %b_dn2 = aie.objectfifo.acquire @b_dn(Produce, 1) : memref<1024xi32>
        aie.objectfifo.release @b_dn(Produce, 1)
        %b_dn3 = aie.objectfifo.acquire @b_dn(Produce, 1) : memref<1024xi32>
        aie.objectfifo.release @b_dn(Produce, 1)
        %b_dn4 = aie.objectfifo.acquire @b_dn(Produce, 1) : memref<1024xi32>
        aie.objectfifo.release @b_dn(Produce, 1)
        %c5 = aie.objectfifo.acquire @c(Produce, 1) : memref<1024xi32>
        aie.objectfifo.release @c(Produce, 1)
      }
      aie.end
    }
    aie.core(%d) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %n = arith.constant 4 : index
      scf.for %i = %c0 to %n step %c1 {
        %b6 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b7 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b8 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b9 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b10 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b11 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b12 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b13 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b14 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %a15 = aie.objectfifo.acquire @a(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @a(Consume, 1)
        %a16 = aie.objectfifo.acquire @a(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @a(Consume, 1)
        %a17 = aie.objectfifo.acquire @a(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @a(Consume, 1)
      }
      aie.end
    }
    aie.core(%e) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %n = arith.constant 4 : index
      scf.for %i = %c0 to %n step %c1 {
        %b18 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b19 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b20 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b21 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b22 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b23 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b24 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b25 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %b26 = aie.objectfifo.acquire @b(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @b(Consume, 1)
        %a27 = aie.objectfifo.acquire @a(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @a(Consume, 1)
        %a28 = aie.objectfifo.acquire @a(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @a(Consume, 1)
        %a29 = aie.objectfifo.acquire @a(Consume, 1) : memref<1024xi32>
        aie.objectfifo.release @a(Consume, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%out : memref<12288xi32>) {
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 12288][0, 0, 0, 1]) {metadata = @c, id = 0 : i64, issue_token = true} : memref<12288xi32>
      aiex.npu.dma_wait {symbol = @c}
    }
  }
}
