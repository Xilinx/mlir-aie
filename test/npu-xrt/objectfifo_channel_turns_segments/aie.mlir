//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Core s fills big, which a mem tile splits into x and y for core r. Core r
// also reads c from the shim, three inputs on two channels, so x and y take
// turns on one: their senders are big's two segments, fed by core s. Each
// step s writes big = 3 * in + 1 and r writes x + 10 * y + 100 * c.

module {
  aie.device(NPUDEVICE) {
    %sh = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %s = aie.tile(0, 2)
    %r = aie.tile(0, 5)

    aie.objectfifo @in(%sh, {%s}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @big(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @x(%m, {%r}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @y(%m, {%r}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@big] -> [@x, @y] ([] [0, 16])
    aie.objectfifo @c(%sh, {%r}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%r, {%sh}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%s) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %k1 = arith.constant 1 : i32
      %k3 = arith.constant 3 : i32
      scf.for %i = %c0 to %c8 step %c1 {
        %a = aie.objectfifo.acquire @in(Consume, 1) : memref<32xi32>
        %b = aie.objectfifo.acquire @big(Produce, 1) : memref<32xi32>
        scf.for %j = %c0 to %c32 step %c1 {
          %v = memref.load %a[%j] : memref<32xi32>
          %t = arith.muli %v, %k3 : i32
          %u = arith.addi %t, %k1 : i32
          memref.store %u, %b[%j] : memref<32xi32>
        }
        aie.objectfifo.release @in(Consume, 1)
        aie.objectfifo.release @big(Produce, 1)
      }
      aie.end
    }

    aie.core(%r) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %k10 = arith.constant 10 : i32
      %k100 = arith.constant 100 : i32
      scf.for %i = %c0 to %c8 step %c1 {
        %x = aie.objectfifo.acquire @x(Consume, 1) : memref<16xi32>
        %y = aie.objectfifo.acquire @y(Consume, 1) : memref<16xi32>
        %c = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
        scf.for %j = %c0 to %c16 step %c1 {
          %vx = memref.load %x[%j] : memref<16xi32>
          %vy = memref.load %y[%j] : memref<16xi32>
          %vc = memref.load %c[%j] : memref<16xi32>
          %sy = arith.muli %vy, %k10 : i32
          %sc = arith.muli %vc, %k100 : i32
          %s1 = arith.addi %vx, %sy : i32
          %s2 = arith.addi %s1, %sc : i32
          memref.store %s2, %o[%j] : memref<16xi32>
        }
        aie.objectfifo.release @x(Consume, 1)
        aie.objectfifo.release @y(Consume, 1)
        aie.objectfifo.release @c(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence(%in : memref<256xi32>, %cin : memref<128xi32>, %out : memref<128xi32>) {
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 128][0, 0, 0, 1]) {metadata = @out, id = 2 : i64, issue_token = true} : memref<128xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {metadata = @in, id = 0 : i64} : memref<256xi32>
      aiex.npu.dma_memcpy_nd(%cin[0, 0, 0, 0][1, 1, 1, 128][0, 0, 0, 1]) {metadata = @c, id = 1 : i64} : memref<128xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}
