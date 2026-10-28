// RUN: not aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate %s 2>&1 | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Too few input channels, and no ends can take turns: c is acquired under a
// branch on the counter of the core's only loop, which no loop around it
// unrolls, so the core needs its objects in no fixed order, and a window over
// a (three held, one released a step) beside a one-deep b would stall. The
// turns a, b, a, b hand the core a's first two objects and b's first, and
// b's second waits for a slot the core frees only after a's third arrives.

// CHECK: error: 'aie.tile' op number of input DMA channel exceeded! requires at least 3 S2MM channels, but capacity is 2
module @window_beside_one_deep {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %m = aie.tile(0, 1)
    %t = aie.tile(0, 2)
    aie.objectfifo @a_in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a(%m, {%t}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a_in] -> [@a] ([] [])
    aie.objectfifo @b_in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%m, {%t}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@b_in] -> [@b] ([] [])
    aie.objectfifo @c_in(%s1, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@c_in] -> [@c] ([] [])
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      scf.for %i = %c0 to %c64 step %c1 {
        %a0, %a1, %a2 = aie.objectfifo.acquire @a(Consume, 3) : memref<16xi32>, memref<16xi32>, memref<16xi32>
        %b0 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        %first = arith.cmpi eq, %i, %c0 : index
        scf.if %first {
          %c0_ = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
          aie.objectfifo.release @c(Consume, 1)
        }
        aie.objectfifo.release @b(Consume, 1)
        aie.objectfifo.release @a(Consume, 1)
      }
      aie.objectfifo.release @a(Consume, 2)
      aie.end
    }
  }
}

// -----

// As above, with a loop around the steps, but the branch on c reads a value
// the core computes, so no unrolling folds it.

// CHECK: error: 'aie.tile' op number of input DMA channel exceeded! requires at least 3 S2MM channels, but capacity is 2
module @branch_on_data {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %m = aie.tile(0, 1)
    %t = aie.tile(0, 2)
    aie.objectfifo @a_in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a(%m, {%t}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a_in] -> [@a] ([] [])
    aie.objectfifo @b_in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%m, {%t}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@b_in] -> [@b] ([] [])
    aie.objectfifo @c_in(%s1, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@c_in] -> [@c] ([] [])
    %flag = aie.buffer(%t) : memref<1xi32>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %zero = arith.constant 0 : i32
      scf.for %image = %c0 to %c64 step %c1 {
        scf.for %step = %c0 to %c8 step %c1 {
          %a0, %a1, %a2 = aie.objectfifo.acquire @a(Consume, 3) : memref<16xi32>, memref<16xi32>, memref<16xi32>
          %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
          %v = memref.load %flag[%c0] : memref<1xi32>
          %set = arith.cmpi ne, %v, %zero : i32
          scf.if %set {
            %c = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
            aie.objectfifo.release @c(Consume, 1)
          }
          aie.objectfifo.release @b(Consume, 1)
          aie.objectfifo.release @a(Consume, 1)
        }
      }
      aie.objectfifo.release @a(Consume, 2)
      aie.end
    }
  }
}
