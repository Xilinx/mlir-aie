//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Three designs packed into one device, each run by its own runtime sequence,
// one dispatch at a time. Their six shim ends need three MM2S and three S2MM
// channels on a shim tile that has two of each; as no two members are ever
// in flight together, a and b take turns on MM2S 0 and S2MM 0, told apart by
// packet headers, and c keeps channel 1 of each.

module {
  aie.device(NPUDEVICE) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    %c = aie.tile(0, 4)
    aie.objectfifo @in_a(%s, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_a(%a, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_b(%s, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_b(%b, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_c(%s, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_c(%c, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
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
    aie.core(%b) {
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
    aie.core(%c) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %k = arith.constant 3 : i32
      scf.for %i = %c0 to %c4 step %c1 {
        %x = aie.objectfifo.acquire @in_c(Consume, 1) : memref<16xi32>
        %y = aie.objectfifo.acquire @out_c(Produce, 1) : memref<16xi32>
        scf.for %j = %c0 to %c16 step %c1 {
          %v = memref.load %x[%j] : memref<16xi32>
          %w = arith.addi %v, %k : i32
          memref.store %w, %y[%j] : memref<16xi32>
        }
        aie.objectfifo.release @in_c(Consume, 1)
        aie.objectfifo.release @out_c(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence @run_a(%in : memref<64xi32>, %out : memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @out_a, id = 1 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @in_a, id = 0 : i64} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @out_a}
    }
    aie.runtime_sequence @run_b(%in : memref<64xi32>, %out : memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @out_b, id = 1 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @in_b, id = 0 : i64} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @out_b}
    }
    aie.runtime_sequence @run_c(%in : memref<64xi32>, %out : memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @out_c, id = 1 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @in_c, id = 0 : i64} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @out_c}
    }
  }
}
