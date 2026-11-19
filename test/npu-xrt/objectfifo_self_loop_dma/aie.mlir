//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core sends @loop to itself through its tile's DMAs, acquiring 1 object at a
// time to produce and 3 at once to consume, so the producing end's pool is 2
// deep and the consuming end's 4. Each round writes 3 objects from one input
// object, then reads them back and writes their weighted sum out. 16 rounds
// move 48 objects through @loop: the producing pool wraps 24 times and every
// consuming window after the first straddles that pool's wrap.

module {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @in_fifo(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @loop(%core, {%core}, 4 : i32) {transport = #aie.transport<dma>} : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out_fifo(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<64xi32>>

    aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %rounds = arith.constant 16 : index
      %off0 = arith.constant 65536 : i32
      %off1 = arith.constant 131072 : i32
      %off2 = arith.constant 196608 : i32
      %w1 = arith.constant 3 : i32
      %w2 = arith.constant 9 : i32

      scf.for %r = %c0 to %rounds step %c1 {
        %x = aie.objectfifo.acquire @in_fifo(Consume, 1) : memref<64xi32>

        %p0 = aie.objectfifo.acquire @loop(Produce, 1) : memref<64xi32>
        scf.for %i = %c0 to %c64 step %c1 {
          %v = memref.load %x[%i] : memref<64xi32>
          %s = arith.addi %v, %off0 : i32
          memref.store %s, %p0[%i] : memref<64xi32>
        }
        aie.objectfifo.release @loop(Produce, 1)

        %p1 = aie.objectfifo.acquire @loop(Produce, 1) : memref<64xi32>
        scf.for %i = %c0 to %c64 step %c1 {
          %v = memref.load %x[%i] : memref<64xi32>
          %s = arith.addi %v, %off1 : i32
          memref.store %s, %p1[%i] : memref<64xi32>
        }
        aie.objectfifo.release @loop(Produce, 1)

        %p2 = aie.objectfifo.acquire @loop(Produce, 1) : memref<64xi32>
        scf.for %i = %c0 to %c64 step %c1 {
          %v = memref.load %x[%i] : memref<64xi32>
          %s = arith.addi %v, %off2 : i32
          memref.store %s, %p2[%i] : memref<64xi32>
        }
        aie.objectfifo.release @loop(Produce, 1)

        aie.objectfifo.release @in_fifo(Consume, 1)

        %b0, %b1, %b2 = aie.objectfifo.acquire @loop(Consume, 3) : memref<64xi32>, memref<64xi32>, memref<64xi32>
        %o = aie.objectfifo.acquire @out_fifo(Produce, 1) : memref<64xi32>
        scf.for %i = %c0 to %c64 step %c1 {
          %v0 = memref.load %b0[%i] : memref<64xi32>
          %v1 = memref.load %b1[%i] : memref<64xi32>
          %v2 = memref.load %b2[%i] : memref<64xi32>
          %t1 = arith.muli %v1, %w1 : i32
          %t2 = arith.muli %v2, %w2 : i32
          %s01 = arith.addi %v0, %t1 : i32
          %s = arith.addi %s01, %t2 : i32
          memref.store %s, %o[%i] : memref<64xi32>
        }
        aie.objectfifo.release @out_fifo(Produce, 1)
        aie.objectfifo.release @loop(Consume, 3)
      }
      aie.end
    }

    aie.runtime_sequence(%in : memref<1024xi32>, %out : memref<1024xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c1024 = arith.constant 1024 : i64
      aiex.npu.dma_memcpy_nd (%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c1024][%c0,%c0,%c0,%c1]) { metadata = @out_fifo, id = 1 : i64 } : memref<1024xi32>
      aiex.npu.dma_memcpy_nd (%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c1024][%c0,%c0,%c0,%c1]) { metadata = @in_fifo, id = 0 : i64, issue_token = true } : memref<1024xi32>
      aiex.npu.dma_wait { symbol = @out_fifo }
    }
  }
}
