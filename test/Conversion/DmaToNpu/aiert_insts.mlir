//===- aiert_insts.mlir -----------------------------------------*- MLIR -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-npu %s | FileCheck %s
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 118816 : ui32} : memref<8xi32>
// CHECK: aiex.npu.write32 {address = 119300 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483649 : ui32}
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 118784 : ui32} : memref<8xi32>
// CHECK: aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 0 : ui32}

module {
  aie.device(npu1_4col) {
    memref.global "public" @of_toMem : memref<32xi32>
    memref.global "public" @of_fromMem : memref<32xi32>
    func.func @sequence(%in : memref<4x2x8xi32>, %buf : memref<32xi32>, %out : memref<64xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %c4 = arith.constant 4 : i64
      %c8 = arith.constant 8 : i64
      %c16 = arith.constant 16 : i64
      %c32 = arith.constant 32 : i64
      aiex.npu.dma_memcpy_nd (0, 0, %out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c32][%c0,%c0,%c0, %c1]) { metadata = @of_toMem, id = 1 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_memcpy_nd (0, 0, %in[%c0,%c2,%c0,%c0][%c1,%c2,%c2,%c8][%c0,%c16,%c8, %c1]) { metadata = @of_fromMem, id = 0 : i64, issue_token = false } : memref<4x2x8xi32>
      return
    }
    aie.shim_dma_allocation @of_fromMem (MM2S, 0, 0)
    aie.shim_dma_allocation @of_toMem (S2MM, 0, 0)
  }
}
