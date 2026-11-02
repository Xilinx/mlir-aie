//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A mem tile holds 40 rows, more than a channel's buffer descriptors, so its
// pool walks them with one descriptor a side stepping a row a run. 80 rows
// go through, twice round the pool; the core adds each row's index.

module {
  aie.device(NPUDEVICE) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %t = aie.tile(0, 2)

    aie.objectfifo @in(%s, {%m}, [2 : i32, 40 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @held(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in] -> [@held] ([] [])
    aie.objectfifo @out(%t, {%s}, 2 : i32) : !aie.objectfifo<memref<64xi32>>

    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c80 = arith.constant 80 : index
      scf.for %i = %c0 to %c80 step %c1 {
        %x = aie.objectfifo.acquire @held(Consume, 1) : memref<64xi32>
        %y = aie.objectfifo.acquire @out(Produce, 1) : memref<64xi32>
        %k = arith.index_cast %i : index to i32
        scf.for %j = %c0 to %c64 step %c1 {
          %v = memref.load %x[%j] : memref<64xi32>
          %w = arith.addi %v, %k : i32
          memref.store %w, %y[%j] : memref<64xi32>
        }
        aie.objectfifo.release @held(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence(%in : memref<5120xi32>, %unused : memref<5120xi32>, %out : memref<5120xi32>) {
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 5120][0, 0, 0, 1]) {metadata = @out, id = 1 : i64, issue_token = true} : memref<5120xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 5120][0, 0, 0, 1]) {metadata = @in, id = 0 : i64} : memref<5120xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}
