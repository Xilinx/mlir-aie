//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core reads a, b and c over two input channels, so a and b take turns on
// one. A mem tile holds 40 rows of a, more than its descriptors can take one
// a row, so descriptors walk the pool whole; the core takes two rows of a for
// each of b, so two of them walk a's pool in turn, each every other row. The
// core takes a row of a before its loop and one of b after it, so the channel
// repeats the loop's turns alone. Per image, out row 0 is a[0] + 4 c[0], row r
// in 1..39 is a[2r-1] + 2 a[2r] + 3 b[r-1] + 4 c[r], and row 40 is
// a[79] + 3 b[39].

module {
  aie.device(NPUDEVICE) {
    %s = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %m = aie.tile(0, 1)
    %t = aie.tile(0, 2)

    aie.objectfifo @a_in(%s, {%m}, [2 : i32, 40 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @a(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@a_in] -> [@a] ([] [])
    aie.objectfifo @b_in(%s1, {%m}, [2 : i32, 3 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @b(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@b_in] -> [@b] ([] [])
    aie.objectfifo @c(%s1, {%t}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out(%t, {%s}, 2 : i32) : !aie.objectfifo<memref<64xi32>>

    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c40 = arith.constant 40 : index
      %c64 = arith.constant 64 : index
      %c2_i32 = arith.constant 2 : i32
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %images = arith.constant 0xFFFFFFFF : index
      scf.for %n = %c0 to %images step %c1 {
        %a0 = aie.objectfifo.acquire @a(Consume, 1) : memref<64xi32>
        %c0_ = aie.objectfifo.acquire @c(Consume, 1) : memref<64xi32>
        %o0 = aie.objectfifo.acquire @out(Produce, 1) : memref<64xi32>
        scf.for %j = %c0 to %c64 step %c1 {
          %va = memref.load %a0[%j] : memref<64xi32>
          %vz = memref.load %c0_[%j] : memref<64xi32>
          %w = arith.muli %vz, %c4_i32 : i32
          %v = arith.addi %va, %w : i32
          memref.store %v, %o0[%j] : memref<64xi32>
        }
        aie.objectfifo.release @a(Consume, 1)
        aie.objectfifo.release @c(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
        scf.for %r = %c1 to %c40 step %c1 {
          %x0, %x1 = aie.objectfifo.acquire @a(Consume, 2) : memref<64xi32>, memref<64xi32>
          %y = aie.objectfifo.acquire @b(Consume, 1) : memref<64xi32>
          %z = aie.objectfifo.acquire @c(Consume, 1) : memref<64xi32>
          %o = aie.objectfifo.acquire @out(Produce, 1) : memref<64xi32>
          scf.for %j = %c0 to %c64 step %c1 {
            %va = memref.load %x0[%j] : memref<64xi32>
            %vb = memref.load %x1[%j] : memref<64xi32>
            %vy = memref.load %y[%j] : memref<64xi32>
            %vz = memref.load %z[%j] : memref<64xi32>
            %w1 = arith.muli %vb, %c2_i32 : i32
            %w2 = arith.muli %vy, %c3_i32 : i32
            %w3 = arith.muli %vz, %c4_i32 : i32
            %t1 = arith.addi %va, %w1 : i32
            %t2 = arith.addi %t1, %w2 : i32
            %t3 = arith.addi %t2, %w3 : i32
            memref.store %t3, %o[%j] : memref<64xi32>
          }
          aie.objectfifo.release @a(Consume, 2)
          aie.objectfifo.release @b(Consume, 1)
          aie.objectfifo.release @c(Consume, 1)
          aie.objectfifo.release @out(Produce, 1)
        }
        %yl = aie.objectfifo.acquire @b(Consume, 1) : memref<64xi32>
        %al = aie.objectfifo.acquire @a(Consume, 1) : memref<64xi32>
        %ol = aie.objectfifo.acquire @out(Produce, 1) : memref<64xi32>
        scf.for %j = %c0 to %c64 step %c1 {
          %va = memref.load %al[%j] : memref<64xi32>
          %vy = memref.load %yl[%j] : memref<64xi32>
          %w = arith.muli %vy, %c3_i32 : i32
          %v = arith.addi %va, %w : i32
          memref.store %v, %ol[%j] : memref<64xi32>
        }
        aie.objectfifo.release @a(Consume, 1)
        aie.objectfifo.release @b(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence(%ain : memref<5120xi32>, %bcin : memref<5120xi32>, %out : memref<2624xi32>) {
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 2624][0, 0, 0, 1]) {metadata = @out, id = 2 : i64, issue_token = true} : memref<2624xi32>
      aiex.npu.dma_memcpy_nd(%ain[0, 0, 0, 0][1, 1, 1, 5120][0, 0, 0, 1]) {metadata = @a_in, id = 0 : i64} : memref<5120xi32>
      aiex.npu.dma_memcpy_nd(%bcin[0, 0, 0, 0][1, 1, 1, 2560][0, 0, 0, 1]) {metadata = @b_in, id = 1 : i64} : memref<5120xi32>
      aiex.npu.dma_memcpy_nd(%bcin[0, 0, 0, 2560][1, 1, 1, 2560][0, 0, 0, 1]) {metadata = @c, id = 3 : i64} : memref<5120xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}
