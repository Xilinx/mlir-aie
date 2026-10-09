//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two packet-switched objectFifos fed from the shim, neither transfer naming a
// header: @in_a through aiex.npu.dma_memcpy_nd, @in_b through
// aiex.dma_configure_task_for. Both take the header allocation assigned.

module {
  aie.device(NPUDEVICE) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)

    aie.objectfifo @in_a(%t00, {%t02}, 2 : i32) {packet} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_b(%t00, {%t03}, 2 : i32) {packet} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_a(%t02, {%t00}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_b(%t03, {%t00}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%t02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %k = arith.constant 1 : i32
      scf.for %i = %c0 to %c4 step %c1 {
        %x = aie.objectfifo.acquire @in_a(Consume, 1) : memref<16xi32>
        %y = aie.objectfifo.acquire @out_a(Produce, 1) : memref<16xi32>
        scf.for %j = %c0 to %c16 step %c1 {
          %v = memref.load %x[%j] : memref<16xi32>
          %w = arith.addi %v, %k : i32
          memref.store %w, %y[%j] : memref<16xi32>
        }
        aie.objectfifo.release @in_a(Consume, 1)
        aie.objectfifo.release @out_a(Produce, 1)
      }
      aie.end
    }
    aie.core(%t03) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %k = arith.constant 2 : i32
      scf.for %i = %c0 to %c4 step %c1 {
        %x = aie.objectfifo.acquire @in_b(Consume, 1) : memref<16xi32>
        %y = aie.objectfifo.acquire @out_b(Produce, 1) : memref<16xi32>
        scf.for %j = %c0 to %c16 step %c1 {
          %v = memref.load %x[%j] : memref<16xi32>
          %w = arith.addi %v, %k : i32
          memref.store %w, %y[%j] : memref<16xi32>
        }
        aie.objectfifo.release @in_b(Consume, 1)
        aie.objectfifo.release @out_b(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence(%a : memref<64xi32>, %b : memref<64xi32>, %ya : memref<64xi32>, %yb : memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%ya[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @out_a, id = 2 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%yb[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @out_b, id = 3 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @in_a, id = 0 : i64} : memref<64xi32>
      %tb = aiex.dma_configure_task_for @in_b {
        aie.dma_bd(%b : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%tb)
      aiex.npu.dma_wait {symbol = @out_a}
      aiex.npu.dma_wait {symbol = @out_b}
      aiex.dma_free_task(%tb)
    }
  }
}
