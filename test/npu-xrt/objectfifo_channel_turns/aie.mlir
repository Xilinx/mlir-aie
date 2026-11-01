//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core reads three fifos from one mem tile but has two input channels, so
// allocation lets two of them take turns on one, in the order the core
// acquires them. Each step the core takes two objects of a and one each of b
// and c, and writes a0 + a1 + 10 * b + 100 * c.

module {
  aie.device(NPUDEVICE) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %t = aie.tile(0, 2)

    aie.objectfifo @in_a(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in_a] -> [@a] ([] [])

    aie.objectfifo @in_bc(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @b(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in_bc] -> [@b, @c] ([] [0, 16])

    aie.objectfifo @out(%t, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %k10 = arith.constant 10 : i32
      %k100 = arith.constant 100 : i32
      scf.for %i = %c0 to %c8 step %c1 {
        %a0, %a1 = aie.objectfifo.acquire @a(Consume, 2) : memref<16xi32>, memref<16xi32>
        %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        %c = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
        %y = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
        scf.for %j = %c0 to %c16 step %c1 {
          %va0 = memref.load %a0[%j] : memref<16xi32>
          %va1 = memref.load %a1[%j] : memref<16xi32>
          %vb = memref.load %b[%j] : memref<16xi32>
          %vc = memref.load %c[%j] : memref<16xi32>
          %sa = arith.addi %va0, %va1 : i32
          %sb = arith.muli %vb, %k10 : i32
          %sc = arith.muli %vc, %k100 : i32
          %s1 = arith.addi %sa, %sb : i32
          %s2 = arith.addi %s1, %sc : i32
          memref.store %s2, %y[%j] : memref<16xi32>
        }
        aie.objectfifo.release @a(Consume, 2)
        aie.objectfifo.release @b(Consume, 1)
        aie.objectfifo.release @c(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence(%in_a : memref<256xi32>, %in_bc : memref<256xi32>, %out : memref<128xi32>) {
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 128][0, 0, 0, 1]) {metadata = @out, id = 2 : i64, issue_token = true} : memref<128xi32>
      aiex.npu.dma_memcpy_nd(%in_a[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {metadata = @in_a, id = 0 : i64} : memref<256xi32>
      aiex.npu.dma_memcpy_nd(%in_bc[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {metadata = @in_bc, id = 1 : i64} : memref<256xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}
