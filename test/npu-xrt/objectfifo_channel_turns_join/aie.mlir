//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Core r sends x, y and c over two output channels, so x and y take turns on
// one: their receivers are the two segments of a mem tile pool that joins
// them into big for core s. Each step r reads in and writes x = in + 1,
// y = 2 * in and c = 5 * in - 7; s writes 3 * big + 1.

module {
  aie.device(NPUDEVICE) {
    %sh = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %s = aie.tile(0, 2)
    %r = aie.tile(0, 5)

    aie.objectfifo @in(%sh, {%r}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @x(%r, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @y(%r, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @big(%m, {%s}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@x, @y] -> [@big] ([0, 16] [])
    aie.objectfifo @c(%r, {%sh}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%s, {%sh}, 2 : i32) : !aie.objectfifo<memref<32xi32>>

    aie.core(%r) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %k1 = arith.constant 1 : i32
      %k2 = arith.constant 2 : i32
      %k5 = arith.constant 5 : i32
      %k7 = arith.constant 7 : i32
      scf.for %i = %c0 to %c8 step %c1 {
        %a = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
        %x = aie.objectfifo.acquire @x(Produce, 1) : memref<16xi32>
        scf.for %j = %c0 to %c16 step %c1 {
          %v = memref.load %a[%j] : memref<16xi32>
          %u = arith.addi %v, %k1 : i32
          memref.store %u, %x[%j] : memref<16xi32>
        }
        aie.objectfifo.release @x(Produce, 1)
        %y = aie.objectfifo.acquire @y(Produce, 1) : memref<16xi32>
        scf.for %j = %c0 to %c16 step %c1 {
          %v = memref.load %a[%j] : memref<16xi32>
          %u = arith.muli %v, %k2 : i32
          memref.store %u, %y[%j] : memref<16xi32>
        }
        aie.objectfifo.release @y(Produce, 1)
        %c = aie.objectfifo.acquire @c(Produce, 1) : memref<16xi32>
        scf.for %j = %c0 to %c16 step %c1 {
          %v = memref.load %a[%j] : memref<16xi32>
          %t = arith.muli %v, %k5 : i32
          %u = arith.subi %t, %k7 : i32
          memref.store %u, %c[%j] : memref<16xi32>
        }
        aie.objectfifo.release @c(Produce, 1)
        aie.objectfifo.release @in(Consume, 1)
      }
      aie.end
    }

    aie.core(%s) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %k1 = arith.constant 1 : i32
      %k3 = arith.constant 3 : i32
      scf.for %i = %c0 to %c8 step %c1 {
        %b = aie.objectfifo.acquire @big(Consume, 1) : memref<32xi32>
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<32xi32>
        scf.for %j = %c0 to %c32 step %c1 {
          %v = memref.load %b[%j] : memref<32xi32>
          %t = arith.muli %v, %k3 : i32
          %u = arith.addi %t, %k1 : i32
          memref.store %u, %o[%j] : memref<32xi32>
        }
        aie.objectfifo.release @big(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence(%in : memref<128xi32>, %cout : memref<128xi32>, %out : memref<256xi32>) {
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {metadata = @out, id = 2 : i64, issue_token = true} : memref<256xi32>
      aiex.npu.dma_memcpy_nd(%cout[0, 0, 0, 0][1, 1, 1, 128][0, 0, 0, 1]) {metadata = @c, id = 1 : i64, issue_token = true} : memref<128xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 128][0, 0, 0, 1]) {metadata = @in, id = 0 : i64} : memref<128xi32>
      aiex.npu.dma_wait {symbol = @c}
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}
