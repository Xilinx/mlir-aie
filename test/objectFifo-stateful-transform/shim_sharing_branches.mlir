//===- shim_sharing_branches.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-objectFifo-stateful-transform %s > %t.out 2> %t.err
// RUN: FileCheck %s --input-file=%t.out
// RUN: FileCheck %s --check-prefix=ERR --input-file=%t.err

// Three shim drains on a tile with two S2MM channels: out_a and out_b take
// turns only when out_b's drain is surely awaited before out_a's next one.

// Straight-line: out_b is issued and awaited every dispatch, so it takes
// turns with out_a.
// CHECK-LABEL: module @straight
// CHECK-DAG:   aie.shim_dma_allocation @out_a__shim_alloc(%{{.*}}, S2MM, [[CH:[0-9]]])
// CHECK-DAG:   aie.shim_dma_allocation @out_b__shim_alloc(%{{.*}}, S2MM, [[CH]])
module @straight {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    %c = aie.tile(0, 4)
    aie.objectfifo @in_a(%s, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_a(%a, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_b(%s, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_b(%b, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_c(%c, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c1 step %c1 {
        %got = aie.objectfifo.acquire @in_a(Consume, 1) : memref<16xi32>
        %put = aie.objectfifo.acquire @out_a(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @in_a(Consume, 1)
        aie.objectfifo.release @out_a(Produce, 1)
      }
      aie.end
    }
    aie.core(%b) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c1 step %c1 {
        %got = aie.objectfifo.acquire @in_b(Consume, 1) : memref<16xi32>
        %put = aie.objectfifo.acquire @out_b(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @in_b(Consume, 1)
        aie.objectfifo.release @out_b(Produce, 1)
      }
      aie.end
    }
    aie.core(%c) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c1 step %c1 {
        %put = aie.objectfifo.acquire @out_c(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @out_c(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%in : memref<64xi32>, %out : memref<64xi32>, %flag : i32) {
      %zero = arith.constant 0 : i32
      %cond = arith.cmpi ne, %flag, %zero : i32
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @out_a, id = 1 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_a, id = 0 : i64} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @out_a}
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_b, id = 2 : i64} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @out_b, id = 3 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @out_b}
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @out_c, id = 4 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @out_c}
    }
  }
}

// -----

// out_b is drained every dispatch but awaited only when %flag is set: an
// await under a branch takes nothing, so out_b keeps a channel of its own.
// ERR: :[[@LINE+3]]:10: error: 'aie.tile' op number of input DMA channel exceeded!
module @await_under_if {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    %c = aie.tile(0, 4)
    aie.objectfifo @in_a(%s, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_a(%a, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_b(%s, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_b(%b, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_c(%c, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c1 step %c1 {
        %got = aie.objectfifo.acquire @in_a(Consume, 1) : memref<16xi32>
        %put = aie.objectfifo.acquire @out_a(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @in_a(Consume, 1)
        aie.objectfifo.release @out_a(Produce, 1)
      }
      aie.end
    }
    aie.core(%b) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c1 step %c1 {
        %got = aie.objectfifo.acquire @in_b(Consume, 1) : memref<16xi32>
        %put = aie.objectfifo.acquire @out_b(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @in_b(Consume, 1)
        aie.objectfifo.release @out_b(Produce, 1)
      }
      aie.end
    }
    aie.core(%c) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c1 step %c1 {
        %put = aie.objectfifo.acquire @out_c(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @out_c(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%in : memref<64xi32>, %out : memref<64xi32>, %flag : i32) {
      %zero = arith.constant 0 : i32
      %cond = arith.cmpi ne, %flag, %zero : i32
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @out_a, id = 1 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_a, id = 0 : i64} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @out_a}
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_b, id = 2 : i64} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @out_b, id = 3 : i64, issue_token = true} : memref<64xi32>
      scf.if %cond {
        aiex.npu.dma_wait {symbol = @out_b}
      }
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @out_c, id = 4 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @out_c}
    }
  }
}

// -----

// out_b is drained only when %flag is set and awaited after the branch: no
// await can take an issue that may not run, so out_b keeps a channel too.
// ERR: :[[@LINE+3]]:10: error: 'aie.tile' op number of input DMA channel exceeded!
module @issue_under_if {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    %c = aie.tile(0, 4)
    aie.objectfifo @in_a(%s, {%a}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_a(%a, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_b(%s, {%b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_b(%b, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_c(%c, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c1 step %c1 {
        %got = aie.objectfifo.acquire @in_a(Consume, 1) : memref<16xi32>
        %put = aie.objectfifo.acquire @out_a(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @in_a(Consume, 1)
        aie.objectfifo.release @out_a(Produce, 1)
      }
      aie.end
    }
    aie.core(%b) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c1 step %c1 {
        %got = aie.objectfifo.acquire @in_b(Consume, 1) : memref<16xi32>
        %put = aie.objectfifo.acquire @out_b(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @in_b(Consume, 1)
        aie.objectfifo.release @out_b(Produce, 1)
      }
      aie.end
    }
    aie.core(%c) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c1 step %c1 {
        %put = aie.objectfifo.acquire @out_c(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @out_c(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%in : memref<64xi32>, %out : memref<64xi32>, %flag : i32) {
      %zero = arith.constant 0 : i32
      %cond = arith.cmpi ne, %flag, %zero : i32
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @out_a, id = 1 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_a, id = 0 : i64} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @out_a}
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in_b, id = 2 : i64} : memref<64xi32>
      scf.if %cond {
        aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @out_b, id = 3 : i64, issue_token = true} : memref<64xi32>
      }
      aiex.npu.dma_wait {symbol = @out_b}
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @out_c, id = 4 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @out_c}
    }
  }
}
